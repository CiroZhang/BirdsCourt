# BirdsCourt

Court-line and net-position detection for a single badminton photo. Given one
image, returns 22 court-line points plus 4 net points (two net-top endpoints,
two pole tops) in pixel coordinates. This project introduces 3 contributions: 

1. A learned reranker that improves Monotrack's court-line candidate selection. 
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
hints on images not already covered by the cache in `BirdsCourtData/Real
Data/VGGT Outputs/`. It needs a CUDA GPU. Without one, and without a cache
hit, net detection still runs, just without the focal-length prior.

```python
import court_detection, net_detection

points = court_detection.detect("photo.jpg")          # 22 court-line points
net = net_detection.detect_net("photo.jpg", points, img_w, img_h)  # 4 net points
```

`Model/weights/` ships the trained reranker checkpoint, so the repo runs
standalone with no training step required.

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

Four stages: generate candidates, score each one independently, rank them
with a learned combiner, reproject the net. (This describes the pipeline
as it's currently being built; the shipped checkpoint in `weights/` is the
previous-generation reranker described further down until this is finished
and swapped in.)

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

### 3. Candidate ranking

A learned combiner picks the final candidate. For every candidate it
builds one feature vector from: the three scorers' own scores above, the 4
outer-corner pixel positions (normalized by image size), and a compressed
per-candidate image feature — the candidate is warped into a canonical
top-down view via its own homography, embedded with a frozen ResNet-18
(ImageNet weights), then reduced to a small number of dimensions. A
sanity-check term is also planned, to catch degenerate/outlier candidates
the scorers above don't reliably flag on their own — still in development,
so left out of this description for now. A small MLP maps the combined
vector to one final score per candidate, trained with a pairwise ranking
loss (a candidate under 15px corner error should outscore one over), and
the pool's top-scoring candidate is the pipeline's pick.

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
threshold.

**Court, 4 outer corners**

| Method | Mean | @5px | @10px | @15px |
|---|---|---|---|---|
| MonoTrack | 6.79 | 89.1% | 89.1% | 93.5% |
| Hit-frame Court R-CNN | 23.44 | 84.8% | 93.5% | 93.5% |
| CourtKeyNet (finetuned) | 8.53 | 52.2% | 84.8% | 89.1% |
| CourtKeyNet (base) | 39.86 | 0.0% | 0.0% | 15.2% |
| TennisCourtDetector | 33.97 | 0.0% | 7.1% | 35.1% |
| **BirdsCourt** | **5.84** | **93.5%** | **93.5%** | **97.8%** |

**Court, all 22 points** (only MonoTrack and BirdsCourt produce all 22;
the other baselines output 4–6 points)

| Method | Mean | @5px | @10px | @15px |
|---|---|---|---|---|
| MonoTrack | 4.97 | 92.7% | 93.3% | 94.2% |
| **BirdsCourt** | **4.34** | **95.9%** | **96.5%** | **97.4%** |

**Net, pole tops** (92 points)

| Method | Mean | @5px | @10px | @15px |
|---|---|---|---|---|
| MonoTrack | 16.83 | 40.2% | 45.7% | 51.1% |
| **BirdsCourt** | **12.41** | **47.8%** | **77.2%** | **87.0%** |

The hit-frame row is noisy across reruns of its training (a repeat gave
57.01 px mean and 87.0% @15px instead of 23.44 / 93.5%), so treat its exact
numbers loosely. See `Model/REPRODUCE.md` for the commands.

## Repo layout

- `Model/` — pipeline code, weights, `scripts/` (train and eval), `REPRODUCE.md`
- `BirdsCourtData/` — Train/, Test/, Synthetic Data/, split/
- `Results/Pictures/` — qualitative comparison figures

## Reproducing the reported numbers

`Model/REPRODUCE.md` has the exact commands and the results table. Training
uses only `Train/`. Every reported number is computed on `Test/`.
