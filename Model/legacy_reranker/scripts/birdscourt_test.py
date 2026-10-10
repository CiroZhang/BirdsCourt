"""Evaluates both 'Ours (classical)' and 'Ours (full pipeline, retrained on
train-set only)' on the held-out Test/ folder (46 images, never touched
during training). Multiprocessing across images."""
import json
import os
import sys
import time
from multiprocessing import Pool

# NOTE: path depth updated after this script moved from Model/scripts/ to
# Model/legacy_reranker/scripts/ when the reranker was superseded by
# ../../honest_scorer.py -- one more dirname() than the original version.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(REPO, "Model")
LEGACY_DIR = os.path.join(MODEL_DIR, "legacy_reranker")
RERANKER_WEIGHTS = os.path.join(LEGACY_DIR, "weights", "reranker_model.pt")
TEST_DIR = os.path.join(REPO, "BirdsCourtData", "Test")
IMG_DIR = os.path.join(TEST_DIR, "Image")
ANN_DIR = os.path.join(TEST_DIR, "Annotation")
N_WORKERS = 4
OPENCV_PYPATH = "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/opencv/4.10.0/lib/python3.11/site-packages"

CORNERS = ["P1_TL", "P4_TR", "P3_BR", "P2_BL"]
PERMS = [[0, 1, 2, 3], [1, 0, 3, 2], [3, 2, 1, 0], [2, 3, 0, 1]]
POLE_NAMES = ["poleL_top", "poleR_top"]


def best_perm_corner_err(pred, gt):
    import numpy as np
    cand = [pred[c] for c in CORNERS]
    truth = [gt[c] for c in CORNERS]
    best = None
    for perm in PERMS:
        errs = [np.hypot(cand[perm[i]][0] - truth[i][0], cand[perm[i]][1] - truth[i][1]) for i in range(4)]
        e = np.mean(errs)
        if best is None or e < best:
            best = e
    return best


def best_perm_pole_err(pred, gt):
    import numpy as np
    normal = [np.hypot(pred[n][0] - gt[n][0], pred[n][1] - gt[n][1]) for n in POLE_NAMES]
    swapped = [np.hypot(pred["poleL_top"][0] - gt["poleR_top"][0], pred["poleL_top"][1] - gt["poleR_top"][1]),
               np.hypot(pred["poleR_top"][0] - gt["poleL_top"][0], pred["poleR_top"][1] - gt["poleL_top"][1])]
    return normal if sum(normal) < sum(swapped) else swapped


def process_one(fname):
    os.environ["PYTHONPATH"] = OPENCV_PYPATH + ":" + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, OPENCV_PYPATH)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, LEGACY_DIR)
    import court_detection
    import vggt_features
    from reranker import rerank_candidates
    from monotrack_line_detection.camera_calibration import court_detection as CourtDetectionCalib
    import cv2

    NON_NET = [n for n in CourtDetectionCalib.COURT_POINTS_3D
               if n not in ("P15_netL", "P16_netR", "poleL_top", "poleR_top")]
    NET_NAMES = ["P15_netL", "P16_netR", "poleL_top", "poleR_top"]

    base = os.path.splitext(fname)[0]
    img_path = os.path.join(IMG_DIR, fname)
    gt = json.load(open(os.path.join(ANN_DIR, base + ".json")))["points"]
    result = {"base": base}
    try:
        im = cv2.imread(img_path)
        h, w = im.shape[:2]

        classical_pts = court_detection.detect(img_path, use_scorer=False)
        result["classical_corner_err"] = best_perm_corner_err(classical_pts, gt)

        # Specifically the OLD reranker (not honest_scorer.py, which is now
        # court_detection.detect()'s default) -- this script's whole point
        # is reproducing the legacy reranker's own reported numbers.
        import shutil
        import tempfile
        work_dir_candidates = tempfile.mkdtemp(prefix="birdscourt_test_")
        try:
            _, candidates = court_detection._run_detect(img_path, work_dir_candidates, dump_candidates=True)
            full_pts = rerank_candidates(img_path, candidates, RERANKER_WEIGHTS) if candidates and len(candidates) >= 2 else classical_pts
        finally:
            shutil.rmtree(work_dir_candidates, ignore_errors=True)
        result["full_corner_err"] = best_perm_corner_err(full_pts, gt)

        cd = CourtDetectionCalib(width=w, height=h)
        cd.set_detections({n: full_pts[n] for n in NON_NET if n in full_pts})
        focal_hint, weight = None, 0.0
        try:
            vdata = vggt_features.get_vggt_output(img_path)
            focal_hint, weight = vdata["hfov_deg"], 0.15
        except Exception:
            pass
        cd.fit_self_calibrated(inlier_thresh_px=20.0, n_iters=150,
                                focal_hint_hfov_deg=focal_hint, focal_prior_weight=weight)
        net_pts = {name: tuple(cd.reproject(name)) for name in NET_NAMES}
        if all(n in gt for n in POLE_NAMES):
            result["pole_err"] = best_perm_pole_err(net_pts, gt)
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def main():
    out_path = os.path.join(REPO, "Results", "test_set_eval.json")
    images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    done_bases = set()
    results = []
    if os.path.exists(out_path):
        results = json.load(open(out_path))
        done_bases = {r["base"] for r in results}
        print(f"resuming: {len(done_bases)} already done", flush=True)
    todo = [f for f in images if os.path.splitext(f)[0] not in done_bases]
    print(f"{len(images)} test images, {len(todo)} remaining", flush=True)

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, todo)):
            results.append(res)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            if (i + 1) % 5 == 0:
                print(f"[{i+1}/{len(todo)}] ({(time.time()-t0)/60:.1f}min)", flush=True)

    import numpy as np
    classical_errs = [r["classical_corner_err"] for r in results if "classical_corner_err" in r]
    full_errs = [r["full_corner_err"] for r in results if "full_corner_err" in r]
    pole_errs = [e for r in results if "pole_err" in r for e in r["pole_err"]]
    errored = [r for r in results if "error" in r]

    print(f"\nn={len(results)}, errored={len(errored)}")
    for r in errored:
        print(" ERROR", r["base"], r["error"])
    ce = np.array(classical_errs)
    fe = np.array(full_errs)
    pe = np.array(pole_errs)
    print(f"classical: mean={ce.mean():.2f}px succ@5={100*(ce<5).mean():.1f}% succ@10={100*(ce<10).mean():.1f}% succ@15={100*(ce<15).mean():.1f}%")
    print(f"full pipeline: mean={fe.mean():.2f}px succ@5={100*(fe<5).mean():.1f}% succ@10={100*(fe<10).mean():.1f}% succ@15={100*(fe<15).mean():.1f}%")
    print(f"pole-top: mean={pe.mean():.2f}px succ@15={100*(pe<15).mean():.1f}%  (n_points={len(pe)})")


if __name__ == "__main__":
    main()
