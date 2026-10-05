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

The pipeline is three independent stages: generate court-corner candidates
classically, pick the best one (classically or learned), then recover net
position by reprojection rather than detection.

### 1. Candidate generation (unchanged from MonoTrack)

A classical line-pixel detector (Hough-style line finding over edge/color
cues, inherited from MonoTrack) proposes on the order of 10–30 candidate
court-line hypotheses per image. Each candidate is constructed by fitting a
homography between a small set of detected line intersections and the known
court geometry, then warping the full court template through it — so every
candidate is geometrically self-consistent by construction (it's always "a
valid court, from *some* camera pose"). The only thing that varies between
candidates is *which* pose is actually correct; a wrong candidate usually
comes from locking onto the wrong real-world lines (an ad board edge, a
baseline from the wrong side of the net, a decoy rectangle).

The net is deliberately excluded from this stage's own scoring. A badminton
net is a low-contrast mesh, not a painted line, so MonoTrack's net-pixel
detector is unreliable enough that letting it vote on which candidate wins
can override an otherwise-correct court fit. We drop it from candidate
selection entirely and instead recover net position afterward (stage 3).

### 2. Candidate selection

**Classical (default, zero-downside):** take MonoTrack's own highest-scoring
candidate by raw pixel-overlap against the warped template, just without the
net term contaminating the score.

**Learned reranker (`reranker.py`, used automatically when `weights/` has a
checkpoint):** re-scores the *entire* candidate pool with a small model
instead of trusting the classical argmax. For every candidate we compute:

1. its classical rank/score, z-normalized within the candidate pool for that
   image (so the feature is comparable across images of different candidate
   counts);
2. its four outer-corner pixel positions, normalized by image size — gives
   the model direct access to the candidate's shape and perspective skew;
3. a **frozen ResNet-18** embedding (ImageNet weights, no fine-tuning) of the
   candidate warped into a canonical top-down view via its own homography
   (`_warp_candidate` in `reranker.py`) — a real court warps into a clean,
   consistent green rectangle with the right line pattern; a decoy (an ad
   board, a bleacher rail, a wrongly-matched baseline) warps into something
   visibly off, which a frozen ImageNet backbone already has useful texture
   priors for even with no court-specific training. This 512-d embedding is
   PCA-reduced to 12 components (fit once, saved in `weights/pca_artifacts.npz`)
   before being fed to the ranker;
4. a classical, non-learned **template-alignment score**
   (`_template_alignment_score`): for each of the court's 12 real line
   segments, sample points along where that line *should* land in the
   canonical warp and check whether the pixels there are actually brighter
   than the pixels just off to either side — i.e. "do painted lines actually
   exist at the positions this candidate's geometry predicts," independent
   of whatever pixel-overlap score the classical detector already computed;
5. (court-stage only, not net-stage) a **VGGT** monocular-depth-derived pair
   of features: the planarity of the candidate's quadrilateral under VGGT's
   single-image 3D point estimate, and VGGT's own per-point confidence —
   catches cases where a candidate's 2D geometry looks plausible but its
   implied 3D surface clearly isn't flat.

These features feed a small **DeepSets-style set ranker**
(`_SetRanker`/`SetRanker` — two-layer MLP embedding per candidate, mean-pooled
into a permutation-invariant "consensus" summary of the whole pool, then
concatenated back onto each candidate's own embedding before a final 2-layer
scoring head). The consensus term is the whole point of using a set model
instead of scoring each candidate in isolation: it lets the network learn
"do independent hypotheses in this pool agree with each other," a signal a
fixed per-candidate formula structurally cannot represent. Training uses a
**pairwise ranking loss** — softplus hinge over (positive − negative) score
differences, where positive = corner error under 15px, negative = over —
since what matters is relative ordering of candidates *within* an image, not
an absolute score comparable across images, and most candidate pools have
exactly one correct hypothesis and several wrong ones.

Training data combines a synthetic decoy-stress-test set, hand-identified
real MonoTrack failures, and real-photo hard cases generated via simulated
out-of-frame crops and perspective warps of verified real images (exact,
automatically-derived ground truth — no manual re-labeling needed), plus
zero-labeling-cost horizontal mirroring (a badminton court is left-right
symmetric, so a mirrored broadcast frame is a physically valid alternate
camera view). The shipped checkpoint (`train_final_model.py`) is trained on
all of this combined, with no held-out fold — it's the deployable model, not
a cross-validation fold.

### 3. Net-position reprojection (`net_detection.py`, `calabration.py`)

Rather than trust any detector's raw net-pixel output, we calibrate a
pinhole camera (intrinsics `K`, pose `R, t`) from the 20 already-reliable,
non-net court-line points produced by stage 2, then reproject the net's
*known* real-world position (net height 1.55m, centered at the court
midline — see `COURT_POINTS_3D`) through that camera. The net-top pole
positions are the genuinely hard part this fixes (they're off the court
plane, so a poorly-conditioned camera estimate shows up there most); the two
ground-level net-endpoint points were already close to correct even under
raw detection, since they sit on the same plane as the rest of the court.

Camera fitting (`_solve_pose_and_focal`) solves for a single shared focal
length `f` (square pixels, principal point fixed at image center) and the
6-DoF pose jointly, via nonlinear least-squares (`scipy.optimize.least_squares`,
Levenberg-Marquardt) minimizing reprojection error over the known
court-point correspondences — no external camera prior is required in
principle, since 20 points is enough redundancy to make focal length
jointly recoverable. In practice this is wrapped in **RANSAC** over random
subsets of the detected points (`fit_self_calibrated`) for robustness to any
one bad corner, and the inner solver itself retries from multiple
physically-motivated initial guesses — camera positions behind/beside the
court at realistic broadcast/amateur heights, looking toward the court
center — rather than generic random seeds, since pure random (rotation,
translation) guesses empirically don't reliably land in the correct basin
of a nonlinear, non-convex reprojection-error landscape, and any fit placing
the camera at or below court height is rejected outright as unphysical.

A known failure mode of this setup is that the 20 fitting points are
*coplanar* (they're all on the court surface), which gives the optimization
a near-flat cost valley along the focal-length/depth direction — a far-away,
telephoto camera and a close, wide-angle one can both nearly explain a flat
point set, even though they imply very different net-pole heights once
reprojected off-plane. To break this ambiguity we optionally add a focal
hint from a single-image VGGT pass: VGGT's own estimated horizontal FOV for
the image, converted to an expected focal length, with an empirical
**×1/0.768 bias correction** (VGGT measurably and consistently
underestimates focal length on this kind of broadcast/amateur footage —
confirmed across dozens of independently-verified fits). This hint is used
two ways — as one additional candidate seed for the multi-restart solver,
and as a **soft regularization residual** pulling the final solved focal
toward the hint rather than only seeding near it (seeding alone was tried
and doesn't prevent the optimizer sliding away to an equally-low-cost but
wrong far/telephoto solution on the flat point set; a genuine pull on the
objective does help). A hard bound on focal length was also tried and
rejected — it interacts badly with the other unbounded pose parameters
under bounded optimization and converges to worse fits even when the true
answer sits well inside the bound. The VGGT hint is strictly additive: a
bad or missing hint just loses to a better unhinted seed under the same
lowest-cost selection rule, it never corrupts an already-good fit. VGGT
itself is only queried if a CUDA GPU is available, or a cached per-image
result already exists in `BirdsCourtData/Real Data/VGGT Outputs/`
(`vggt_script.py` checks the cache first) — without either, net detection
still runs via classical self-calibration alone, just with somewhat lower
pole-height accuracy on the minority of images where the planar ambiguity
actually bites.
