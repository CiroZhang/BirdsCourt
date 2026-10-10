"""Court corner (and full 22-point court-line) detection for a single
badminton broadcast/amateur photo. Wraps the compiled MonoTrack binary
(monotrack_line_detection/, build it first with
monotrack_line_detection/build.sh) for UNCAPPED candidate generation
(every candidate that passes the basic validity gate -- see the comment
in monotrack_line_detection/src/main.cpp), then either:

  - picks the classical top candidate (zero-downside vs. the original
    net-scored MonoTrack), or
  - scores the full candidate pool with candidate_scorer.py's three
    independently-trained scorers (edge alignment, gradient orientation,
    pixel/geometry CNN) and picks the best by their fixed-weight combo --
    the current, recommended method. Reaches 100% Test / 99.5% Train
    pool-top-1 accuracy (<15px), beating vanilla MonoTrack (93.5%) and the
    older MonoTrack-score-reliant reranker (97.8%) -- see RESULTS.md.
    Never reuses MonoTrack's own classical score as an input signal,
    anywhere in the computation.

An older, MonoTrack-score-reliant reranker (`legacy_reranker/`) is kept
for reference but is no longer the default -- see
`legacy_reranker/README.md` for why it was superseded.

Net-pole position is NOT part of this module's output (MonoTrack's own net
detection is unreliable by design -- see net_detection.py, which reprojects
it from the calibrated camera instead).
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_BIN = os.path.join(HERE, "monotrack_line_detection", "src", "build", "detect")
WEIGHTS_DIR = os.path.join(HERE, "weights")

POINT_NAMES = [
    "P1_TL", "P2_BL", "P3_BR", "P4_TR", "P5_singlesTL", "P6_singlesBL", "P7_singlesBR", "P8_singlesTR",
    "P9_serviceFarL", "P10_serviceFarR", "P11_serviceNearL", "P12_serviceNearR",
    "P13_serviceFarCenter", "P14_serviceNearCenter", "P15_netL", "P16_netR",
    "P17_singlesFarL", "P18_singlesFarR", "P19_singlesNearL", "P20_singlesNearR",
    "P21_baseFarCenter", "P22_baseNearCenter",
]
CORNERS = ["P1_TL", "P4_TR", "P3_BR", "P2_BL"]


def _check_binary():
    if not os.path.exists(DETECT_BIN):
        raise RuntimeError(
            f"MonoTrack binary not found at {DETECT_BIN} -- build it first: "
            f"bash {os.path.join(HERE, 'monotrack_line_detection', 'build.sh')}"
        )


def _run_detect(image_path, work_dir, dump_candidates=False):
    """Runs the compiled binary on a single image (via a throwaway 3-frame
    .avi, which is what its video-reading interface expects) and returns the
    parsed point dicts: (winner_points, candidate_pts_dicts or None)."""
    avi = os.path.join(work_dir, "in.avi")
    pts = os.path.join(work_dir, "out.pts")
    overlay = os.path.join(work_dir, "overlay.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", image_path, "-frames:v", "3",
         "-c:v", "mjpeg", "-q:v", "3", avi, "-loglevel", "error"],
        check=True, timeout=30, capture_output=True,
    )

    args = [DETECT_BIN, avi, pts, overlay]
    cands_dir = None
    if dump_candidates:
        cands_dir = os.path.join(work_dir, "candidates")
        args.append(cands_dir)
    subprocess.run(args, capture_output=True, text=True, timeout=120)

    def parse_pts(path):
        lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        p = {}
        for name, line in zip(POINT_NAMES, lines):
            x, y = line.split(";")
            x, y = float(x), float(y)
            if x == x and y == y:  # drop NaN (net-free build can't give poles)
                p[name] = (x, y)
        return p

    if not os.path.exists(pts):
        raise RuntimeError(
            "MonoTrack found no valid court-line candidates for this image "
            "(its own classical line-pixel detector requires high-contrast "
            "painted lines -- heavy occlusion, very low contrast, or an "
            "extreme camera angle can all cause this)."
        )
    winner = parse_pts(pts)

    candidates = None
    if dump_candidates and cands_dir and os.path.isdir(cands_dir):
        candidates = []
        for f in sorted(os.listdir(cands_dir), key=lambda s: (len(s), s)):
            if not f.startswith("cand_"):
                continue
            candidates.append(parse_pts(os.path.join(cands_dir, f)))

    return winner, candidates


def detect(image_path, use_scorer="auto"):
    """Returns a dict of {point_name: (x, y)} for the 22 court-line points
    (see POINT_NAMES) -- no net points, see net_detection.py for that.

    use_scorer: "auto" (use candidate_scorer.py if its weights/ are present,
    else fall back to classical), True (require it, raise if weights
    missing), False (always classical, even if weights are present)."""
    _check_binary()

    scorer_weights_dir = os.path.join(HERE, "weights", "candidate_scorer")
    have_scorer_weights = all(
        os.path.exists(os.path.join(scorer_weights_dir, f))
        for f in ("clf_c.pkl", "clf_d.pkl", "geometry_cnn_best.pt")
    )
    want_scorer = use_scorer is True or (use_scorer == "auto" and have_scorer_weights)
    if use_scorer is True and not have_scorer_weights:
        raise RuntimeError(f"use_scorer=True but weights missing from {scorer_weights_dir}")

    work_dir = tempfile.mkdtemp(prefix="court_detection_")
    try:
        winner, candidates = _run_detect(image_path, work_dir, dump_candidates=want_scorer)
        if not want_scorer or not candidates or len(candidates) < 2:
            return winner

        import candidate_scorer
        return candidate_scorer.pick_best(image_path, candidates)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def detect_corners(image_path, use_scorer="auto"):
    """Convenience wrapper: just the 4 outer corners, in TL,TR,BR,BL order."""
    points = detect(image_path, use_scorer=use_scorer)
    return [points[c] for c in CORNERS]
