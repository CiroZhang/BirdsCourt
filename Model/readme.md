# BirdsCourt

Court-line and net-position detection for a single badminton photo. Given one
image, returns 22 court-line points plus 4 net points (two net-top endpoints,
two pole tops) in pixel coordinates.

```
python3 main.py photo.jpg --overlay out.jpg
```

## Setup

```
pip install -r requirements.txt
bash monotrack_line_detection/build.sh   # needs cmake + an OpenCV4/5 dev install
```

VGGT itself isn't pip-installable — install from source
(https://github.com/facebookresearch/vggt) if you want live net-calibration
hints on images that aren't already in `BirdsCourtData/Real Data/VGGT
Outputs/`. It also needs a CUDA GPU; without one (and without a cache hit),
net detection still runs, just without the focal-length prior.

## Usage

```python
import court_detection, net_detection

points = court_detection.detect("photo.jpg")          # 22 court-line points
net = net_detection.detect_net("photo.jpg", points, img_w, img_h)  # 4 net points
```

or end-to-end via `main.py`:

```
python3 main.py photo.jpg --overlay out.jpg
python3 main.py photo.jpg --no-reranker   # force classical court pick
python3 main.py photo.jpg --no-vggt       # classical-only net calibration
```

## Files

- `monotrack_line_detection/` — C++ candidate detector (`src/`, build via
  `build.sh`) and `calabration.py` (the self-calibration camera class).
- `court_detection.py` / `reranker.py` — court-point detection + learned
  reranking.
- `vggt_script.py` / `net_detection.py` — VGGT focal hint (cache-first) +
  net reprojection.
- `main.py` — CLI entry point.
- `weights/` — the trained reranker checkpoint (`reranker_model.pt` +
  `pca_artifacts.npz`), trained on the full combined real+synthetic hard-case
  dataset (no held-out fold — this is the deployable model, not a crossval
  fold).

---

## Method

Three stages: generate court-corner candidates, pick one, reproject the net.

### 1. Candidate generation

MonoTrack's own classical line-pixel detector proposes 10–30 candidate
court-line hypotheses per image. Each candidate fits a homography between
detected line intersections and the known court geometry, then warps the
full court template through it. The net term is removed from this stage's
internal scoring (net pixels aren't used to build or score candidates at
all) — net position is handled separately in stage 3.

### 2. Candidate selection

**Classical:** take MonoTrack's highest-scoring candidate by pixel overlap
(net-free score).

**Learned reranker** (`reranker.py`), used when `weights/` has a checkpoint.
For every candidate in the pool:

1. classical rank/score, z-normalized within the pool.
2. four outer-corner pixel positions, normalized by image size.
3. warp the candidate into a canonical top-down view via its own homography
   (`_warp_candidate`), embed it with a frozen ResNet-18 (ImageNet weights),
   PCA-reduce the 512-d embedding to 12 dims (`weights/pca_artifacts.npz`).
4. `_template_alignment_score`: sample points along each of the court's 12
   real line segments at their predicted position in the canonical warp,
   check whether those pixels are brighter than pixels just off to the side.
5. (court stage only) two VGGT-derived features: planarity of the
   candidate's quadrilateral under VGGT's monocular depth estimate, and
   VGGT's per-point confidence.

These feed a DeepSets-style set ranker (`SetRanker`): each candidate's
feature vector goes through a 2-layer MLP embedding, embeddings are
mean-pooled into one permutation-invariant pool-level vector, that pool
vector is concatenated back onto each candidate's own embedding, then a
final 2-layer head scores each candidate. Trained with a pairwise ranking
loss (softplus hinge on score differences between candidates under vs. over
15px corner error).

Training data: a synthetic decoy set, hand-identified real MonoTrack
failures, random-crop/perspective-warp augmentations of verified real
images, and horizontal mirroring. `train_final_model.py` trains the shipped
checkpoint on all of it combined, no held-out fold.

### 3. Net-position reprojection (`net_detection.py`, `calabration.py`)

1. Build a pinhole camera model (intrinsics `K`, pose `R, t`) from the 20
   non-net court-line points output by stage 2.
2. Solve for focal length `f` (square pixels, principal point at image
   center) and the 6-DoF pose jointly via nonlinear least-squares
   (`scipy.optimize.least_squares`, Levenberg-Marquardt) minimizing
   reprojection error over the known court-point correspondences
   (`_solve_pose_and_focal`).
3. Wrap the solve in RANSAC over random point subsets
   (`fit_self_calibrated`), with the inner solver retried from multiple
   physically-motivated initial camera poses (behind/beside the court,
   elevated, looking at the court center) instead of generic random seeds.
4. Optionally pass a focal-length hint from a single-image VGGT pass: VGGT's
   estimated horizontal FOV converted to an expected focal length, with a
   fixed ×1/0.768 correction factor. Used as (a) one extra seed for the
   multi-restart solver, and (b) a soft regularization residual pulling the
   final solved focal toward the hint. VGGT runs only if a CUDA GPU is
   available or a cached per-image result exists in `BirdsCourtData/Real
   Data/VGGT Outputs/` (`vggt_script.py` checks the cache first); otherwise
   step 4 is skipped and steps 1–3 run alone.
5. Reproject the net's known real-world position (height 1.55m, centered on
   the court midline — `COURT_POINTS_3D`) through the fitted camera to get
   the 2 net-top pole points and 2 ground-level net-endpoint points.
