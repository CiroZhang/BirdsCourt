"""Runs vanilla (net-included) MonoTrack on the 46-image Test/ set and
scores the 4 outer corners (dihedral-permutation-aware, same convention as
everywhere else in this project)."""
import json
import os
import shutil
import subprocess
import time
from multiprocessing import Pool

import numpy as np

REPO = "/scratch/ciro/BirdsCourt_repo"
TEST_DIR = os.path.join(REPO, "BirdsCourtData", "Real Data", "Test")
IMG_DIR = os.path.join(TEST_DIR, "Image")
ANN_DIR = os.path.join(TEST_DIR, "Annotation")
DETECT_BIN = "/scratch/ciro/monotrack_net_included_src/build/detect"
WORK_ROOT = "/scratch/ciro/vanilla_test_work"
N_WORKERS = 4

POINT_NAMES = [
    "P1_TL", "P2_BL", "P3_BR", "P4_TR", "P5_singlesTL", "P6_singlesBL", "P7_singlesBR", "P8_singlesTR",
    "P9_serviceFarL", "P10_serviceFarR", "P11_serviceNearL", "P12_serviceNearR",
    "P13_serviceFarCenter", "P14_serviceNearCenter", "P15_netL", "P16_netR",
    "P17_singlesFarL", "P18_singlesFarR", "P19_singlesNearL", "P20_singlesNearR",
    "P21_baseFarCenter", "P22_baseNearCenter", "poleL_top", "poleR_top",
]
CORNERS = ["P1_TL", "P4_TR", "P3_BR", "P2_BL"]
PERMS = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1]]


def corner_err(pred, gt):
    cand = [pred.get(c) for c in CORNERS]
    truth = [gt.get(c) for c in CORNERS]
    if any(v is None for v in cand) or any(v is None for v in truth):
        return None
    best = None
    for perm in PERMS:
        errs = [np.hypot(cand[perm[i]][0] - truth[i][0], cand[perm[i]][1] - truth[i][1]) for i in range(4)]
        e = np.mean(errs)
        if best is None or e < best:
            best = e
    return best


def process_one(fname):
    base = os.path.splitext(fname)[0]
    img_path = os.path.join(IMG_DIR, fname)
    gt = json.load(open(os.path.join(ANN_DIR, base + ".json")))["points"]
    work = os.path.join(WORK_ROOT, base)
    os.makedirs(work, exist_ok=True)
    avi = os.path.join(work, "in.avi")
    pts = os.path.join(work, "out.pts")
    overlay = os.path.join(work, "overlay.jpg")
    try:
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", img_path, "-frames:v", "3",
                         "-c:v", "mjpeg", "-q:v", "3", avi, "-loglevel", "error"],
                        check=True, timeout=30, capture_output=True)
        result = subprocess.run([DETECT_BIN, avi, pts, overlay], capture_output=True, text=True, timeout=400)
        if not os.path.exists(pts):
            return {"base": base, "status": "fail_no_candidates", "stderr": result.stderr[-300:]}
        lines = [l.strip() for l in open(pts) if l.strip()]
        pred = {}
        for name, line in zip(POINT_NAMES, lines):
            x, y = line.split(";")
            x, y = float(x), float(y)
            if x == x and y == y:
                pred[name] = (x, y)
        e = corner_err(pred, gt)
        return {"base": base, "err": e}
    except Exception as ex:
        return {"base": base, "status": f"error: {ex}"}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    os.makedirs(WORK_ROOT, exist_ok=True)
    images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"{len(images)} test images", flush=True)
    t0 = time.time()
    results = []
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, images)):
            results.append(res)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(images)}] ({(time.time()-t0)/60:.1f}min)", flush=True)

    out_path = os.path.join(REPO, "Results", "vanilla_monotrack_test_set_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    errs = [r["err"] for r in results if r.get("err") is not None]
    failed = [r for r in results if r.get("err") is None]
    print(f"\nn_ok={len(errs)}, n_failed={len(failed)}")
    for r in failed:
        print(" FAIL", r["base"], r.get("status"))
    errs = np.array(errs)
    print(f"vanilla MonoTrack: mean={errs.mean():.2f}px median={np.median(errs):.2f}px "
          f"succ@5={100*(errs<5).mean():.1f}% succ@10={100*(errs<10).mean():.1f}% succ@15={100*(errs<15).mean():.1f}%")


if __name__ == "__main__":
    main()
