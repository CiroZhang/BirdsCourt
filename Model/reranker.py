"""Learned reranker over MonoTrack's own candidate pool -- picks a better
candidate than the classical argmax specifically on hard/adversarial inputs
(occlusion, partial frame, decoys); see the paper's hard-case table. Needs
weights/reranker_model.pt + weights/pca_artifacts.npz (both included in this
repo). Used by court_detection.py when a trained checkpoint is present.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT_DIM = 12
WARP_SIZE = 128
MARGIN = 1.5
COURT_W, COURT_L = 6.1, 13.41
TEMPLATE_CORNERS = np.array([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]], dtype=np.float32)
CORNERS = ["P1_TL", "P4_TR", "P3_BR", "P2_BL"]

LINE_SEGMENTS_M = [
    ((0, 0), (COURT_W, 0)), ((0, COURT_L), (COURT_W, COURT_L)),
    ((0, 0), (0, COURT_L)), ((COURT_W, 0), (COURT_W, COURT_L)),
    ((0.46, 0), (0.46, COURT_L)), ((5.64, 0), (5.64, COURT_L)),
    ((0, 4.72), (COURT_W, 4.72)), ((0, 8.685), (COURT_W, 8.685)),
    ((0, 0.76), (COURT_W, 0.76)), ((0, 12.65), (COURT_W, 12.65)),
    ((3.05, 0), (3.05, 4.72)), ((3.05, 8.685), (3.05, COURT_L)),
]


def _build_canonical_homography():
    src = np.array([
        [-MARGIN, -MARGIN], [COURT_W + MARGIN, -MARGIN],
        [COURT_W + MARGIN, COURT_L + MARGIN], [-MARGIN, COURT_L + MARGIN],
    ], dtype=np.float32)
    dst = np.array([[0, 0], [WARP_SIZE, 0], [WARP_SIZE, WARP_SIZE], [0, WARP_SIZE]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


_CANONICAL_H = _build_canonical_homography()
_LINE_SEGMENTS_WARP_PX = []
for (ax, ay), (bx, by) in LINE_SEGMENTS_M:
    pa = _CANONICAL_H @ np.array([ax, ay, 1.0]); pa = pa[:2] / pa[2]
    pb = _CANONICAL_H @ np.array([bx, by, 1.0]); pb = pb[:2] / pb[2]
    _LINE_SEGMENTS_WARP_PX.append((pa, pb))


def _warp_candidate(img, corners_xy):
    cand_px = np.array(corners_xy, dtype=np.float32)
    h_world_to_img = cv2.getPerspectiveTransform(TEMPLATE_CORNERS, cand_px)
    h_canonical_to_img = h_world_to_img @ np.linalg.inv(_CANONICAL_H)
    return cv2.warpPerspective(img, h_canonical_to_img, (WARP_SIZE, WARP_SIZE),
                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)


def _template_alignment_score(warped_gray, n_samples_per_line=10, perp_offset=4):
    h, w = warped_gray.shape
    hits, total = 0, 0
    for pa, pb in _LINE_SEGMENTS_WARP_PX:
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        length = np.hypot(dx, dy)
        if length < 1e-3:
            continue
        perp = np.array([-dy, dx]) / length
        for t in np.linspace(0.1, 0.9, n_samples_per_line):
            pt = pa + t * (pb - pa)
            x, y = pt
            if not (0 <= x < w and 0 <= y < h):
                continue
            side_a, side_b = pt + perp * perp_offset, pt - perp * perp_offset
            if not (0 <= side_a[0] < w and 0 <= side_a[1] < h and 0 <= side_b[0] < w and 0 <= side_b[1] < h):
                continue
            line_val = warped_gray[int(y), int(x)]
            bg_val = (warped_gray[int(side_a[1]), int(side_a[0])] + warped_gray[int(side_b[1]), int(side_b[0])]) / 2
            total += 1
            if line_val > bg_val + 15:
                hits += 1
    return hits / total if total else 0.0


class _FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x).flatten(1)


class _SetRanker(nn.Module):
    def __init__(self, n_feat, embed_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_feat, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
        )
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        h = self.embed(x)
        consensus = h.mean(dim=0, keepdim=True).expand(h.shape[0], -1)
        return self.score_head(torch.cat([h, consensus], dim=1)).squeeze(-1)


_extractor = None
_model = None
_pca = None


def _load(model_path):
    global _extractor, _model, _pca
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _extractor is None:
        _extractor = _FeatureExtractor().to(device)
    if _pca is None:
        art = np.load(os.path.join(HERE, "weights", "pca_artifacts.npz"))
        _pca = {"components": art["pca_components"], "mean": art["pca_mean"],
                "f_mean": art["f_mean"], "f_std": art["f_std"]}
    if _model is None:
        n_feat = _pca["f_mean"].shape[1]
        _model = _SetRanker(n_feat).to(device)
        _model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _model.eval()
    return _extractor, _model, _pca, device


def rerank_candidates(image_path, candidates, model_path):
    """candidates: list of {point_name: (x,y)} dicts (MonoTrack's own
    candidate pool, rank-0 = classical argmax's pick -- see court_detection.py).
    Returns the single best candidate's point dict, per the trained model."""
    extractor, model, pca, device = _load(model_path)

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    warps, grays = [], []
    corners_list = []
    for c in candidates:
        corners = [c[name] for name in CORNERS]
        corners_list.append(corners)
        w = _warp_candidate(img, corners)
        warps.append(w)
        grays.append(cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32))

    batch = np.stack(warps).astype(np.float32) / 255.0
    batch = (batch - 0.5) / 0.5
    batch_t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        raw_context = extractor(batch_t).cpu().numpy()
    reduced = ((raw_context - pca["mean"]) @ pca["components"].T).astype(np.float32)

    # classical score proxy: MonoTrack's own candidate ordering (rank-0 =
    # best by its internal pixel-overlap score), z-scored across the pool --
    # matches how the model was trained (see pc_remote_scripts/).
    n = len(candidates)
    scores = np.arange(n, 0, -1, dtype=np.float32)
    s_mean, s_std = scores.mean(), scores.std() + 1e-6

    feats = []
    for i, (corners, gray) in enumerate(zip(corners_list, grays)):
        rank_frac = i / max(1, n - 1)
        ta = _template_alignment_score(gray)
        corners_norm = []
        for (cx, cy) in corners:
            corners_norm.extend([cx / img_w, cy / img_h])
        feats.append([(scores[i] - s_mean) / s_std, rank_frac, ta] + corners_norm)
    feats = np.array(feats, dtype=np.float32)
    feats = np.concatenate([feats, reduced], axis=1)
    feats = (feats - pca["f_mean"]) / pca["f_std"]

    feats_t = torch.from_numpy(feats.astype(np.float32)).to(device)
    with torch.no_grad():
        model_scores = model(feats_t).cpu().numpy()
    best_idx = int(np.argmax(model_scores))
    return candidates[best_idx]
