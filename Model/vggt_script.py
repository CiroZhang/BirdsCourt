"""VGGT (Wang et al. 2025, github.com/facebookresearch/vggt) wrapper: gets a
horizontal-FOV estimate (used as a focal-length hint for net_detection.py's
camera calibration) and, optionally, per-candidate planarity/confidence
(used by the reranker, see court_detection.py). Single-image mode -- VGGT's
cross-view attention degenerates harmlessly to self-attention for N=1.

Checks BirdsCourtData/Real Data/VGGT Outputs/<basename>.json first -- all
229 images in our real dataset already have this precomputed (ground-truth
corners used as the "candidate"), so most callers never need a GPU or the
VGGT weights at all. Only a genuinely new image falls through to a live
VGGT pass, which needs a CUDA GPU and will download the ~5GB facebook/VGGT-1B
checkpoint on first use.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(
    os.path.dirname(HERE), "BirdsCourtData", "Real Data", "VGGT Outputs"
)

_model = None
_device = None
_dtype = None


def _load_model():
    """Lazily loads VGGT onto the GPU, downloading weights via huggingface_hub
    on first use if not already cached. Only called when the cache above
    misses, so most pipeline runs on the existing dataset never hit this."""
    global _model, _device, _dtype
    if _model is not None:
        return _model, _device, _dtype

    import torch
    from vggt.models.vggt import VGGT

    if not torch.cuda.is_available():
        raise RuntimeError(
            "VGGT needs a CUDA GPU and this image isn't in the precomputed "
            f"cache ({CACHE_DIR}). Either run this on a GPU machine, or "
            "skip the VGGT focal hint (net_detection.py works without it, "
            "just with somewhat worse pole-height accuracy -- see the "
            "paper's Table on net position)."
        )

    _device = "cuda"
    _dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    _model = VGGT.from_pretrained("facebook/VGGT-1B").to(_device)
    _model.eval()
    return _model, _device, _dtype


def _cache_path(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(CACHE_DIR, base + ".json")


def _sample_quad_points(corners_xy, n_per_side=6):
    """corners_xy: 4 (x,y) in TL,TR,BR,BL order. Bilinear-samples a grid
    covering the quadrilateral interior (works for any convex quad)."""
    corners = np.array(corners_xy, dtype=np.float64)
    pts = []
    for a in np.linspace(0.1, 0.9, n_per_side):
        for b in np.linspace(0.1, 0.9, n_per_side):
            top = corners[0] * (1 - a) + corners[1] * a
            bot = corners[3] * (1 - a) + corners[2] * a
            pts.append(top * (1 - b) + bot * b)
    return np.array(pts)


def _run_vggt_live(image_path, candidates):
    """candidates: list of {"corners": [(x,y)x4 in TL,TR,BR,BL order]}.
    Returns {"hfov_deg": float, "candidates": [{"cand_idx", "vggt_planarity_ratio", "vggt_mean_conf"}, ...]}."""
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    import cv2

    model, device, dtype = _load_model()

    orig = cv2.imread(image_path)
    orig_h, orig_w = orig.shape[:2]

    images = load_and_preprocess_images([image_path]).to(device)
    resized_h, resized_w = images.shape[-2:]
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            images_b = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_b.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_b, ps_idx)

    extrinsic_np = extrinsic[0].float().cpu().numpy()
    intrinsic_np = intrinsic[0].float().cpu().numpy()
    depth_map_np = depth_map[0].float().cpu().numpy()
    depth_conf_np = depth_conf[0].float().cpu().numpy()
    point_map = unproject_depth_map_to_point_map(depth_map_np, extrinsic_np, intrinsic_np)[0]
    conf_map = depth_conf_np[0]

    fx = float(intrinsic_np[0, 0, 0])
    hfov_deg = float(np.degrees(2 * np.arctan(resized_w / (2 * fx))))

    sx, sy = resized_w / orig_w, resized_h / orig_h
    cand_results = []
    for idx, c in enumerate(candidates):
        corners_resized = [(x * sx, y * sy) for x, y in c["corners"]]
        sample_pts = _sample_quad_points(corners_resized)
        pix = np.round(sample_pts).astype(int)
        pix[:, 0] = np.clip(pix[:, 0], 0, resized_w - 1)
        pix[:, 1] = np.clip(pix[:, 1], 0, resized_h - 1)

        pts3d = point_map[pix[:, 1], pix[:, 0]]
        confs = conf_map[pix[:, 1], pix[:, 0]]

        centroid = pts3d.mean(axis=0)
        centered = pts3d - centroid
        try:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
            normal = vt[-1]
            residuals = centered @ normal
            planarity_rms = float(np.sqrt(np.mean(residuals ** 2)))
            scale = float(s[0]) + 1e-8
            planarity_ratio = planarity_rms / scale
        except Exception:
            planarity_ratio = None

        cand_results.append({
            "cand_idx": c.get("cand_idx", idx),
            "vggt_planarity_ratio": planarity_ratio,
            "vggt_mean_conf": float(np.mean(confs)),
        })

    return {"hfov_deg": hfov_deg, "candidates": cand_results}


def get_vggt_output(image_path, candidates=None, use_cache=True):
    """Main entry point. `candidates`: optional list of {"corners": [...]}
    (e.g. the court_detection.py output's 4 outer corners) -- only needed if
    you want per-candidate planarity/confidence, not just the focal hint.

    Checks the precomputed VGGT Outputs cache first (matched by image
    basename); only runs VGGT live if that misses. Set use_cache=False to
    force a fresh run (e.g. if you want planarity for different candidates
    than the cached ground-truth-corner version)."""
    if use_cache:
        cache_path = _cache_path(image_path)
        if os.path.exists(cache_path):
            return json.load(open(cache_path, encoding="utf-8"))

    if candidates is None:
        candidates = []
    return _run_vggt_live(image_path, candidates)
