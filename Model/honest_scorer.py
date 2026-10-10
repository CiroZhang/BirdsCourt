"""The final, from-scratch court-candidate scorer: picks the best candidate
out of MonoTrack's own uncapped candidate pool using THREE independently-
trained scorers, each reading only real image evidence -- never MonoTrack's
own classical score, directly or indirectly.

  - Scorer C (edge alignment): classical Canny edge map + per-line
    hit-fraction against it, 12 features (one per real court line
    segment), logistic regression head.
  - Scorer D (orientation alignment): local image-gradient ORIENTATION at
    sampled points along each line, checked against the line's own
    expected direction -- catches false edges that merely happen to be
    nearby but don't run the right way. Same 12-segment structure,
    logistic regression head.
  - Scorer F (pixel/geometry CNN): a small conv net over (image + rendered
    line mask), trained end to end as a single logistic-regression-style
    binary classifier (correct/incorrect vs ground truth).

Final score = fixed weighted combination (weights chosen via differential
evolution on Train only, see the project writeup): 0.095*C + 0.218*D +
0.836*F (each scorer's own logit passed through its own sigmoid first).

Scored under the project's dihedral-permutation-aware corner metric (same
convention as scripts/monotrack_test.py), this combo alone reaches 100%
Test / 99.5% Train pool-top-1 accuracy (<15px mean 4-corner error) --
beating vanilla MonoTrack's own classical score (93.5%/6.79px mean) and the
older reranker that's allowed to read MonoTrack's score as an input
feature (97.8%/5.96px mean). No hand-tuned consensus/tolerance heuristic
is needed for this result; the plain combo already achieves it, so that's
what this module implements. See OVERNIGHT_SUMMARY_2026-10-09.md's "MAJOR
CORRECTION" section for the full derivation and ablations.
"""
import os
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights", "honest_scorer")

CORNERS = ["P1_TL", "P2_BL", "P3_BR", "P4_TR"]

# The 12 real court line segments, as point-name pairs -- matches
# MonoTrack's own TennisCourtModel::drawModel exactly (confirmed from its
# C++ source): 4 outer box + 2 singles sidelines + 2 front service lines +
# 2 center-line halves + 2 back doubles-service lines. Net line excluded.
LINE_SEGMENTS = [
    ("P1_TL", "P2_BL"), ("P2_BL", "P3_BR"), ("P3_BR", "P4_TR"), ("P4_TR", "P1_TL"),
    ("P5_singlesTL", "P6_singlesBL"), ("P7_singlesBR", "P8_singlesTR"),
    ("P9_serviceFarL", "P10_serviceFarR"), ("P11_serviceNearL", "P12_serviceNearR"),
    ("P13_serviceFarCenter", "P21_baseFarCenter"), ("P14_serviceNearCenter", "P22_baseNearCenter"),
    ("P17_singlesFarL", "P18_singlesFarR"), ("P19_singlesNearL", "P20_singlesNearR"),
]

BEST_W = (0.095, 0.218, 0.836)  # (C, D, F) combo weights

N_SAMPLES = 15
_TS = np.linspace(0.08, 0.92, N_SAMPLES)
C_TOL_PX = 5.0
D_ANG_TOL = 20 * np.pi / 180
CNN_IMG_SIZE = 224


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------- Scorer C
def _edge_dist_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    inv = cv2.bitwise_not(edges)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def _edge_features(candidates, w, h, dist_map):
    avail = [(a, b) for a, b in LINE_SEGMENTS if a in candidates[0] and b in candidates[0]]
    n = len(candidates)
    feats = np.zeros((n, len(avail)), dtype=np.float32)
    for si, (a, b) in enumerate(avail):
        pa = np.array([c[a] for c in candidates], dtype=np.float64)
        pb = np.array([c[b] for c in candidates], dtype=np.float64)
        pts_line = pa[:, None, :] * (1 - _TS)[None, :, None] + pb[:, None, :] * _TS[None, :, None]
        xi = np.round(pts_line[..., 0]).astype(np.int64)
        yi = np.round(pts_line[..., 1]).astype(np.int64)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi_c, yi_c = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        d = dist_map[yi_c, xi_c]
        feats[:, si] = (valid & (d <= C_TOL_PX)).mean(axis=1)
    return feats


# ---------------------------------------------------------------- Scorer D
def _gradient_maps(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx)
    mag_thresh = float(np.percentile(mag, 80))
    return ang, mag, mag_thresh


def _orientation_features(candidates, w, h, ang_map, mag_map, mag_thresh):
    avail = [(a, b) for a, b in LINE_SEGMENTS if a in candidates[0] and b in candidates[0]]
    n = len(candidates)
    feats = np.zeros((n, len(avail)), dtype=np.float32)
    for si, (a, b) in enumerate(avail):
        pa = np.array([c[a] for c in candidates], dtype=np.float64)
        pb = np.array([c[b] for c in candidates], dtype=np.float64)
        line_dir = pb - pa
        line_ang = np.arctan2(line_dir[:, 1], line_dir[:, 0])
        pts_line = pa[:, None, :] * (1 - _TS)[None, :, None] + pb[:, None, :] * _TS[None, :, None]
        xi = np.round(pts_line[..., 0]).astype(np.int64)
        yi = np.round(pts_line[..., 1]).astype(np.int64)
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi_c, yi_c = np.clip(xi, 0, w - 1), np.clip(yi, 0, h - 1)
        local_ang = ang_map[yi_c, xi_c]
        local_mag = mag_map[yi_c, xi_c]
        perp_ang = line_ang[:, None] + np.pi / 2
        diff = np.abs(((local_ang - perp_ang + np.pi / 2) % np.pi) - np.pi / 2)
        aligned = valid & (diff < D_ANG_TOL) & (local_mag > mag_thresh)
        feats[:, si] = aligned.mean(axis=1)
    return feats


# ---------------------------------------------------------------- Scorer F
class SmallGeometryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.classifier(h).squeeze(-1)


def _cnn_scores(candidates, img, model, device, batch_size=256):
    h, w = img.shape[:2]
    img_r = cv2.resize(img, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype(np.float32) / 255.0
    scores = np.zeros(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        batch_x = []
        for c in candidates[start:end]:
            mask = np.zeros((h, w), dtype=np.uint8)
            for a, b in LINE_SEGMENTS:
                if a not in c or b not in c:
                    continue
                pa, pb = c[a], c[b]
                cv2.line(mask, (int(round(pa[0])), int(round(pa[1]))),
                          (int(round(pb[0])), int(round(pb[1]))), 255, thickness=3)
            mask_r = cv2.resize(mask, (CNN_IMG_SIZE, CNN_IMG_SIZE)).astype(np.float32) / 255.0
            batch_x.append(np.concatenate([img_r, mask_r[..., None]], axis=2))
        batch_t = torch.from_numpy(np.stack(batch_x)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            logits = model(batch_t)
        scores[start:end] = logits.cpu().numpy()
    return scores


# ---------------------------------------------------------------- top-level
_clf_c = None
_clf_d = None
_cnn_model = None
_device = None


def _load():
    global _clf_c, _clf_d, _cnn_model, _device
    if _clf_c is None:
        with open(os.path.join(WEIGHTS_DIR, "clf_c.pkl"), "rb") as fh:
            _clf_c = pickle.load(fh)
    if _clf_d is None:
        with open(os.path.join(WEIGHTS_DIR, "clf_d.pkl"), "rb") as fh:
            _clf_d = pickle.load(fh)
    if _cnn_model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cnn_model = SmallGeometryCNN().to(_device)
        _cnn_model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "geometry_cnn_best.pt"),
                                               map_location=_device, weights_only=True))
        _cnn_model.eval()
    return _clf_c, _clf_d, _cnn_model, _device


def score_candidates(image_path, candidates):
    """candidates: list of {point_name: (x, y)} dicts (MonoTrack's own
    uncapped candidate pool for this image -- see court_detection.py's
    _run_detect(..., dump_candidates=True)).
    Returns a (len(candidates),) array of combo scores (higher = better;
    NOT probabilities, just the fixed weighted sum of 3 sigmoid scores)."""
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    clf_c, clf_d, cnn_model, device = _load()
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    dist_map = _edge_dist_map(img)
    c_feat = _edge_features(candidates, w, h, dist_map)
    c_score = sigmoid(clf_c.decision_function(c_feat))

    ang, mag, mag_thresh = _gradient_maps(img)
    d_feat = _orientation_features(candidates, w, h, ang, mag, mag_thresh)
    d_score = sigmoid(clf_d.decision_function(d_feat))

    f_logits = _cnn_scores(candidates, img, cnn_model, device)
    f_score = sigmoid(f_logits)

    wc, wd, wf = BEST_W
    return wc * c_score + wd * d_score + wf * f_score


def pick_best(image_path, candidates):
    """Returns the single best candidate dict (argmax combo score) -- drop-in
    replacement for reranker.rerank_candidates(), with zero reuse of
    MonoTrack's own classical score anywhere in the computation."""
    scores = score_candidates(image_path, candidates)
    return candidates[int(np.argmax(scores))]
