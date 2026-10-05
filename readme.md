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

`BirdsCourtData/` has two parts.

**Real Data** (`BirdsCourtData/Real Data/`) — 229 broadcast and amateur
badminton photos, each with:
- `Image/` — the source photo
- `Annotation/` — hand-verified ground truth: 22 court-line points plus 4 net
  points, a `source` tag, and a `verified` flag
- `Thumbnell/` — a small preview used by our annotation tool
- `VGGT Outputs/` — a cached VGGT monocular-depth pass per image, computed once so
  anyone without a GPU can still run net calibration with the VGGT hint

**Synthetic Data** (`BirdsCourtData/Synthetic Data/`) — 2,963 rendered court
images with `Image/`, `Annotation/`, and `Mask/`, covering decoy surfaces,
simulated out-of-frame crops, perspective warps, and mirrored real photos.
Ground truth here is derived automatically from the render/warp parameters,
so it costs nothing to label. This is the main source of reranker training
data, since it's cheap to generate hard cases at scale (occlusion, partial
frame, decoy lines) that real footage rarely provides.

## Method

Three stages: generate court-corner candidates, pick one, reproject the net.

### 1. Candidate generation

MonoTrack's own classical line-pixel detector proposes 10 to 30 candidate
court-line hypotheses per image. Each candidate fits a homography between
detected line intersections and the known court geometry, then warps the
full court template through it. The net term is removed from this stage's
internal scoring (net pixels aren't used to build or score candidates at
all), net position is handled separately in stage 3.

### 2. Candidate selection

**Learned reranker** (`reranker.py`), used when `weights/` has a checkpoint.
For every candidate in the pool:

1. classical rank and score, z-normalized within the pool.
2. four outer-corner pixel positions, normalized by image size.
3. warp the candidate into a canonical top-down view via its own homography
   (`_warp_candidate`), embed it with a frozen ResNet-18 (ImageNet weights),
   PCA-reduce the 512-d embedding to 12 dims (`weights/pca_artifacts.npz`).
4. `_template_alignment_score` — sample points along each of the court's 12
   real line segments at their predicted position in the canonical warp,
   check whether those pixels are brighter than pixels just off to the side.
5. (court stage only) two VGGT-derived features: planarity of the
   candidate's quadrilateral under VGGT's monocular depth estimate, and
   VGGT's per-point confidence.

These feed a DeepSets-style set ranker (`SetRanker`). Each candidate's
feature vector goes through a 2-layer MLP embedding, embeddings are
mean-pooled into one permutation-invariant pool-level vector, that pool
vector is concatenated back onto each candidate's own embedding, then a
final 2-layer head scores each candidate. Trained with a pairwise ranking
loss (softplus hinge on score differences between candidates under vs. over
15px corner error).

Training data combines the synthetic decoy set, hand-identified real
MonoTrack failures, random-crop and perspective-warp augmentations of the
verified real images, and horizontal mirroring. `train_final_model.py`
trains the shipped checkpoint on all of it combined, with no held-out fold.

### 3. Net-position reprojection (`net_detection.py`, `calabration.py`)

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
   Data/VGGT Outputs/` (`vggt_script.py` checks the cache first), otherwise
   step 4 is skipped and steps 1 through 3 run alone.
5. Reproject the net's known real-world position (height 1.55m, centered on
   the court midline, see `COURT_POINTS_3D`) through the fitted camera to
   get the 2 net-top pole points and 2 ground-level net-endpoint points.

## Repo layout

- `Model/` — the deployable pipeline (see above)
- `BirdsCourtData/` — the dataset (see above)
- `Results/` — qualitative comparison figures
