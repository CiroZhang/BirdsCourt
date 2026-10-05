"""Court corner (and full 22-point court-line) detection for a single
badminton broadcast/amateur photo. Wraps the compiled net-free MonoTrack
binary (monotrack_line_detection/, build it first with
monotrack_line_detection/build.sh) for candidate generation, then either:

  - picks the classical top candidate (zero-downside vs. the original
    net-scored MonoTrack -- see the paper's Table 1/2), or
  - reranks the full candidate pool with the trained model in weights/, if
    present -- helps specifically on hard/adversarial inputs (occlusion,
    partial frame, decoys); see the paper's hard-case table. On typical
    unobstructed real photos the classical pick is usually already correct,
    so the reranker mostly matters when you expect difficult input.

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


def detect(image_path, use_reranker="auto"):
    """Returns a dict of {point_name: (x, y)} for the 22 court-line points
    (see POINT_NAMES) -- no net points, see net_detection.py for that.

    use_reranker: "auto" (use it if weights/ has a trained checkpoint, else
    fall back to classical), True (require it, raise if weights missing),
    False (always classical, even if weights are present)."""
    _check_binary()

    reranker_path = os.path.join(WEIGHTS_DIR, "reranker_model.pt")
    want_reranker = use_reranker is True or (use_reranker == "auto" and os.path.exists(reranker_path))
    if use_reranker is True and not os.path.exists(reranker_path):
        raise RuntimeError(f"use_reranker=True but no checkpoint at {reranker_path}")

    work_dir = tempfile.mkdtemp(prefix="court_detection_")
    try:
        winner, candidates = _run_detect(image_path, work_dir, dump_candidates=want_reranker)
        if not want_reranker or not candidates or len(candidates) < 2:
            return winner

        from reranker import rerank_candidates
        return rerank_candidates(image_path, candidates, reranker_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def detect_corners(image_path, use_reranker="auto"):
    """Convenience wrapper: just the 4 outer corners, in TL,TR,BR,BL order."""
    points = detect(image_path, use_reranker=use_reranker)
    return [points[c] for c in CORNERS]
