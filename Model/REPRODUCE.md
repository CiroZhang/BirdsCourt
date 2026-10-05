# Reproducing the reported numbers and weights

This file is the map from "a number in the paper" to "the exact script, data,
and split that produced it." Everything here runs locally, no cloud compute
required (the original runs used Killarney/PC GPUs for speed, not because
they're required).

## 1. The shipped reranker weights (`weights/`)

`weights/reranker_model.pt` + `weights/pca_artifacts.npz` are produced by
`train_final_model.py`, trained on 881 images combined (450 synthetic/flip +
431 real crop/perspective hard-case set), **no held-out fold** -- this is the
deploy artifact, not an evaluation. Rerun it:

```
python3 train_final_model.py
```

Needs the two training jsonls (context_feat + template_alignment already
extracted per candidate) at the paths set in that script's `ORIGINAL_PATH` /
`HARD_PATH` constants -- see the script's own docstring for exactly what's in
each and where they live.

## 2. The reported reranker accuracy (paper Table 3, hard-real subset)

**Number**: 10.13px / 96.3% (classical) -> 6.85px / 97.4% (+ reranker),
mean corner error / success@15px, on the 431-image real crop/perspective
hard-case set.

**Script**: `court_reranker/pc_remote_scripts/grouped_5fold_eval.py`
(in the research repo, not duplicated here since it also needs the 55MB
training jsonl).

**Methodology**: 5-fold cross-validation, but folds are assigned by SOURCE
PHOTO (229 unique real photos, each contributing up to 2 augmented crop/
perspective variants) rather than by individual image -- so no two variants
of the same photo are ever split across train and validation. Every one of
the 431 images gets exactly one out-of-sample prediction, concatenated
across all 5 folds. This fixes a real leakage risk in an earlier, simpler
per-image split that had inflated the result to 5.82px/97.7% (see
MANIFEST.md's 2026-10-05 entry for the full before/after comparison and the
three failed "zero real data" alternatives that were tried first).

**Reproducible split**: every run with `seed=0` produces the identical fold
assignment and per-image predictions, saved to
`../BirdsCourtData/grouped_5fold_split.json` (source photo -> fold
assignment, plus every image's classical vs. reranker error and the overall
summary) -- lives with the dataset since it's a partition of it, not model
code. So the reported number is independently auditable without rerunning
anything. The canonical copy (written fresh on rerun) lives at
`court_reranker/pc_remote_scripts/grouped_5fold_split.json` in the research
repo -- copy it to `../BirdsCourtData/` after rerunning. Rerun:

```
cd "court_reranker/pc_remote_scripts"
python3 grouped_5fold_eval.py
```

## 3. The reported net-calibration accuracy (paper's net paragraph)

**Number**: pole-top 35.5px/14.8% (raw MonoTrack) -> 14.4px/87.9%
(+ reprojection), net-quad IoU 0.805 -> 0.914, ground-endpoint 7.6px, on the
99-image hand-verified real-net set.

**Script**: `court_reranker/net_calibration/validate_net99.py`. Uses
already-fit camera parameters (`net99_camera_fits.json` -- K, R, t per image,
fit via `court_detection.fit_self_calibrated` with no VGGT hint, per
MANIFEST.md's conclusion that no-VGGT is the final reported configuration)
and reprojects against the current `BirdsCourtData/Real Data/Annotation`
ground truth. Both the pole-top and ground-endpoint pairs are scored with
best-of-2 left/right permutation matching (the same dihedral-symmetry
convention used for the 22 court-line corners elsewhere in this project --
net points have the identical left/right labeling ambiguity).

The raw (uncalibrated) MonoTrack baseline (35.5px/14.8%/0.805) requires
MonoTrack's net-INCLUDED build, a different binary than this repo's shipped
net-free one (`monotrack_line_detection/`, which deliberately drops net
scoring -- see the main readme's Method section) -- that baseline is cited
from the original validated run logged in MANIFEST.md, not re-derived by
this script.

A copy of the result is right here at `net99_validation_result.json`; the
canonical copy (written fresh on rerun) lives at
`court_reranker/net_calibration/net99_validation_result.json`. Rerun:

```
cd "court_reranker/net_calibration"
python3 validate_net99.py
```

## 4. The main real-image comparison (paper Table 2, 229-image set)

No training or split involved -- this is purely the classical (net-free)
candidate selection, no reranker. Fully covered by this repo's own
`court_detection.py` with `use_reranker=False`, run over
`BirdsCourtData/Real Data/Image/`.
