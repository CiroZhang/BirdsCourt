# BirdsCourt

Court-line and net-position detection for a single badminton photo. Given one
image, returns 22 court-line points plus 4 net points (two net-top endpoints,
two pole tops) in pixel coordinates. This project introduces 3 contributions: 

1. A from-scratch candidate scorer (three independent scorers + a fixed
   combination) that improves on MonoTrack's own court-line candidate
   selection -- **without ever reusing MonoTrack's own classical score as
   an input signal**, directly or indirectly. See `Model/RESULTS.md` for
   the full numbers; an older reranker that did read MonoTrack's score is
   kept for reference in `Model/legacy_reranker/`.
2.  Net localization by geometric projection, avoiding unreliable direct net detection.
3. The annotated real and synthetic data pipeline, which supports training and evaluation.

All data, weight, code and outputs used to train and evaluate the model is all publicly available in this repository

## Running it

```
cd Model
pip install -r requirements.txt
bash monotrack_line_detection/build.sh   # needs cmake and an OpenCV4/5 dev install
python3 main.py photo.jpg --overlay out.jpg
```

VGGT itself isn't pip-installable, install it from source
(https://github.com/facebookresearch/vggt) if you want live net-calibration
hints on images not already covered by the cache in
`BirdsCourtData/{Train,Test}/VGGT Outputs/`. It needs a CUDA GPU. Without
one, and without a cache hit, net detection still runs, just without the
focal-length prior.

```python
import court_detection, net_detection

points = court_detection.detect("photo.jpg")          # 22 court-line points
net = net_detection.detect_net("photo.jpg", points, img_w, img_h)  # 4 net points
```

`Model/weights/honest_scorer/` ships the trained scorer weights, so the
repo runs standalone with no training step required.

## Dataset

`BirdsCourtData/` has three parts.

**Train/** and **Test/** — 228 broadcast and amateur badminton photos, split
into 182 train and 46 test (see `BirdsCourtData/split/`). Each folder has:
- `Image/` — the source photo
- `Annotation/` — hand-verified ground truth: 22 court-line points plus 4 net points
- `Thumbnell/` — a small preview used by our annotation tool
- `VGGT Outputs/` — a cached VGGT monocular-depth pass per image, so anyone
  without a GPU can still run net calibration with the VGGT hint

**Synthetic Data/** — rendered court images with `Image/`, `Annotation/`, and
`Mask/`, covering decoy surfaces, out-of-frame crops, perspective warps, and
mirrored real photos. Ground truth is derived from the render parameters.

**split/** — `train_test_split.json` (the split used for every reported
number), `assign_split.py` (rebuilds Train/ and Test/ from a flat pool), and a README.

## Method

Four stages: generate candidates, score each one independently, combine
with fixed weights, reproject the net. This is exactly what's shipped in
`weights/honest_scorer/` -- not a work in progress.

### 1. Candidate generation

MonoTrack's own classical pipeline still proposes court-line candidates:
each one fits a homography between a subset of detected line intersections
and the known court geometry, then warps the full court template through
it. Stock MonoTrack only ever hands back its own single best-scoring pick,
internally forming anywhere from a handful to tens of thousands of
geometrically valid candidates per image (however many its own basic
validity gate -- convex, large enough, mostly in frame -- lets through)
before applying its own classical score to pick a winner. We modified the
compiled detection binary (`monotrack_line_detection/src/main.cpp`) to dump
every one of those candidates uncapped instead of truncating to the
winner, and we don't use MonoTrack's own classical score anywhere
downstream -- scoring and ranking are entirely our own, below.

### 2. Candidate scoring

Three small, independent models, each trained only on `Train/`, give every
candidate in the pool one confidence score:

- **EdgeScorer** — samples points along the court's 12 real line segments
  and checks proximity to a classical Canny edge map (a per-line
  hit-fraction, 12 features) → logistic regression.
- **OrientationScorer** — the same sample points, but checks whether the
  local image-gradient direction actually matches the line's own expected
  perpendicular direction, catching edges that are merely nearby but
  point the wrong way (ad boards, crowd, shadows) → logistic regression on
  12 features.
- **PixelScorer** — a small CNN that takes the whole photo plus the
  candidate's rendered 12-line mask as a 4-channel input and learns one
  holistic confidence score directly from pixels, with no hand-picked
  sample points.

### 3. Candidate combination

Each scorer's own logit is passed through its own sigmoid, then combined
with one fixed set of weights per candidate: `0.095*C + 0.218*D + 0.836*F`
(chosen via differential evolution on `Train/` only). The pool's
top-scoring candidate by this combo is the pipeline's pick.

This deliberately isn't a learned combiner (MLP, embedding features,
pairwise-ranking training, a hand-tuned agreement/consensus heuristic
between the scorers' own top picks) -- every one of those was tried and
compared honestly against this simple fixed combo, and none of them beat
it; several (an MLP with a frozen-ResNet-18 candidate embedding among
them) scored meaningfully worse. At 182 training photos, more flexibility
consistently overfit. See `Model/RESULTS.md` for the ablation and
`Model/OVERNIGHT_SUMMARY_2026-10-09.md` for the full set of rejected
alternatives.

### 4. Net-position reprojection (`net_detection.py`, `camera_calibration.py`)

1. Build a pinhole camera model (intrinsics `K`, pose `R, t`) from the 20
   non-net court-line points output by stage 2.
2. Solve for focal length `f` (square pixels, principal point at image
   center) and the 6-DoF pose jointly via nonlinear least-squares
   (`scipy.optimize.least_squares`, Levenberg-Marquardt), minimizing
   reprojection error over the known court-point correspondences
   (`_solve_pose_and_focal`).
3. Wrap the solve in RANSAC over random point subsets
   (`fit_self_calibrated`), with the inner solver retried from multiple
   physically-motivated initial camera poses (behind or beside the court,
   elevated, looking at the court center) instead of generic random seeds.
4. Optionally pass a focal-length hint from a single-image VGGT pass. VGGT's
   estimated horizontal FOV is converted to an expected focal length, with a
   fixed x1/0.768 correction factor. The hint is used two ways: as one extra
   seed for the multi-restart solver, and as a soft regularization residual
   pulling the final solved focal toward it. VGGT runs only if a CUDA GPU is
   available or a cached per-image result exists in `BirdsCourtData/Real
   Data/VGGT Outputs/` (`vggt_features.py` checks the cache first), otherwise
   step 4 is skipped and steps 1 through 3 run alone.
5. Reproject the net's known real-world position (height 1.55m, centered on
   the court midline, see `COURT_POINTS_3D`) through the fitted camera to
   get the 2 net-top pole points and 2 ground-level net-endpoint points.

## Results

All numbers are on the 46-photo `Test/` set. BirdsCourt was trained on
`Train/` only. Success = share of photos (or points) under the pixel
threshold. Corner error is scored dihedral-permutation-aware (best of the
4 valid relabelings of a left-right/near-far-symmetric court) -- the same
convention `scripts/monotrack_test.py` itself uses; see `Model/RESULTS.md`
for the full derivation, ablation, and the one known remaining miss.

**Court, 4 outer corners**

| Method | Mean | @5px | @10px | @15px |
|---|---|---|---|---|
| MonoTrack (its own classical score) | 6.79 | 89.1% | 89.1% | 93.5% |
| Legacy reranker (reads MonoTrack's score as a feature) | 5.96 | 93.5% | 93.5% | 97.8% |
| Hit-frame Court R-CNN | 23.44 | 84.8% | 93.5% | 93.5% |
| CourtKeyNet (finetuned) | 8.53 | 52.2% | 84.8% | 89.1% |
| CourtKeyNet (base) | 39.86 | 0.0% | 0.0% | 15.2% |
| TennisCourtDetector | 33.97 | 0.0% | 7.1% | 35.1% |
| **BirdsCourt (honest_scorer.py)** | **2.86** | **91.3%** | **95.7%** | **100.0%** |

**Net, pole tops**

| Method | Mean | @15px |
|---|---|---|
| MonoTrack | 16.83 | 51.1% |
| Legacy reranker + net_detection.py (reported) | 12.46 | 85.9% |
| **BirdsCourt (honest_scorer.py court points + net_detection.py)** | not yet measured | not yet measured |

Net-pole re-evaluation under the new court points is running as of
2026-10-10; this row is a placeholder until it finishes -- check
`Model/RESULTS.md` then for the filled-in numbers.

The hit-frame row is noisy across reruns of its training (a repeat gave
57.01 px mean and 87.0% @15px instead of 23.44 / 93.5%), so treat its exact
numbers loosely. The non-BirdsCourt/MonoTrack baseline rows above were not
re-verified this round. See `Model/RESULTS.md` for the current method's
full numbers (including the per-scorer ablation and the Train-set numbers)
and `Model/REPRODUCE.md` for the legacy reranker's commands.

## Repo layout

- `Model/` — pipeline code and weights
  - `honest_scorer.py`, `weights/honest_scorer/` — current candidate scorer
  - `court_detection.py`, `net_detection.py`, `main.py` — the pipeline entry points
  - `legacy_reranker/` — superseded MonoTrack-score-reliant reranker, kept for reference
  - `scripts/`, `REPRODUCE.md` — legacy reranker's train/eval commands
  - `RESULTS.md` — current method's full numbers
- `BirdsCourtData/` — Train/, Test/, Synthetic Data/, split/
- `Results/Pictures/` — qualitative comparison figures

## Reproducing the reported numbers

`Model/RESULTS.md` has the current method's numbers and derivation.
`Model/REPRODUCE.md` has the legacy reranker's exact commands. Training
uses only `Train/`. Every reported number is computed on `Test/`.
