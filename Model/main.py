"""BirdsCourt: full court + net detection for a single badminton photo.

    python3 main.py path/to/photo.jpg
    python3 main.py path/to/photo.jpg --overlay out.jpg --no-vggt

Pipeline (see RESULTS.md for validated accuracy numbers):
  1. court_detection.py  -- MonoTrack uncapped candidate generation, scored
     by honest_scorer.py's three independent scorers (if weights/ is
     present) -- never reuses MonoTrack's own classical score.
  2. net_detection.py    -- reprojects net position from a camera calibrated
     off the court points above (VGGT-assisted if available), instead of
     trusting MonoTrack's own unreliable net-pixel detection.

First run needs the MonoTrack binary built (monotrack_line_detection/build.sh)
and, for new images not already in BirdsCourtData/{Train,Test}/VGGT Outputs/, a
CUDA GPU to run VGGT live (weights auto-download via huggingface_hub on first
use). Everything else runs on CPU.
"""
import argparse
import json
import sys

import cv2

import court_detection
import net_detection


def run(image_path, use_scorer="auto", use_vggt_hint=True):
    """Returns {point_name: (x, y)} for all 22 court-line points + the 4
    net points (P15_netL, P16_netR, poleL_top, poleR_top) -- 26 total."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"couldn't read image: {image_path}")
    h, w = img.shape[:2]

    court_points = court_detection.detect(image_path, use_scorer=use_scorer)
    net_points = net_detection.detect_net(image_path, court_points, w, h, use_vggt_hint=use_vggt_hint)

    return {**court_points, **net_points}


def draw_overlay(image_path, points, out_path):
    img = cv2.imread(image_path)
    for (x, y) in points.values():
        cv2.circle(img, (int(round(x)), int(round(y))), 6, (0, 0, 0), 2)
        cv2.circle(img, (int(round(x)), int(round(y))), 4, (61, 220, 132), -1)
    cv2.imwrite(out_path, img)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to a badminton court photo")
    parser.add_argument("--overlay", help="also save a dot-overlay visualization to this path")
    parser.add_argument("--no-scorer", action="store_true", help="skip honest_scorer.py even if its weights are present (classical pick only)")
    parser.add_argument("--no-vggt", action="store_true", help="skip the VGGT focal hint for net position (classical-only calibration)")
    args = parser.parse_args()

    points = run(
        args.image,
        use_scorer=False if args.no_scorer else "auto",
        use_vggt_hint=not args.no_vggt,
    )
    print(json.dumps({k: list(v) for k, v in points.items()}, indent=2))

    if args.overlay:
        draw_overlay(args.image, points, args.overlay)
        print(f"wrote overlay -> {args.overlay}", file=sys.stderr)


if __name__ == "__main__":
    main()
