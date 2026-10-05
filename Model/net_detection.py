"""Net position for a single badminton photo: NOT MonoTrack's own net
detection (its net-pole pixel detection is unreliable -- a net is a
low-contrast mesh, not a painted line). Instead, calibrates a camera from
the already-reliable 22 court-line points (see court_detection.py) and
reprojects the net's known real-world position through it. See the paper's
net-position table: raw MonoTrack 14.8% success@15px -> this method 88.8%.
"""
import os

from monotrack_line_detection.calabration import court_detection
import vggt_script

NET_NAMES = ["P15_netL", "P16_netR", "poleL_top", "poleR_top"]
ALL_NAMES = list(court_detection.COURT_POINTS_3D.keys())
NON_NET_NAMES = [n for n in ALL_NAMES if n not in NET_NAMES]


def detect_net(image_path, court_points, img_width, img_height, use_vggt_hint=True):
    """court_points: the dict from court_detection.detect() (22 court-line
    points, no net). Returns {point_name: (x, y)} for P15_netL, P16_netR,
    poleL_top, poleR_top -- the 2 ground-level net-endpoint points (already
    near-perfect, since they're close to the court plane) and the 2 pole-top
    points (the genuinely hard part this method actually fixes).

    use_vggt_hint: if True (default), gets a focal-length hint from VGGT
    (checking the precomputed cache first -- see vggt_script.py) and uses it
    as a soft prior during calibration. Significantly reduces the rare but
    large pole-height errors caused by a flat/far-away camera being
    ambiguous with a close/wide one from court-plane points alone. Without
    it, calibration still works (RANSAC + multi-seed least-squares), just
    with somewhat lower pole-height accuracy on that minority of images."""
    cd = court_detection(width=img_width, height=img_height)
    cd.set_detections({n: court_points[n] for n in NON_NET_NAMES if n in court_points})

    focal_hint_hfov_deg = None
    focal_prior_weight = 0.0
    if use_vggt_hint:
        try:
            vdata = vggt_script.get_vggt_output(image_path)
            focal_hint_hfov_deg = vdata["hfov_deg"]
            focal_prior_weight = 0.15
        except Exception:
            pass  # no GPU and no cached VGGT output for this image -- fall back to no hint

    cd.fit_self_calibrated(
        inlier_thresh_px=20.0, n_iters=150,
        focal_hint_hfov_deg=focal_hint_hfov_deg, focal_prior_weight=focal_prior_weight,
    )

    return {name: tuple(cd.reproject(name)) for name in NET_NAMES}
