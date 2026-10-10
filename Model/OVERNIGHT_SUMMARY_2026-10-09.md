# Overnight scorer/consensus experiments — 2026-10-09

## MAJOR CORRECTION (found 2026-10-10, after this doc was first written)

Every number below ("93.5% Test", "oracle 97.8%", all the rejected-approach
percentages) was computed with a **scoring bug**: our own evaluation scripts
(`kfold_consensus.py` and everything built on top of it) compared predicted
corners to ground truth using a single fixed corner order, with no dihedral
relabeling search. But `scripts/monotrack_test.py` -- the script that
produced the official "Vanilla MonoTrack 93.5%" baseline we were trying to
match -- scores with `corner_err()` trying all 4 dihedral relabelings
(identity / left-right flip / near-far flip / 180 rotation) and keeping the
best, per its own docstring: *"dihedral-permutation-aware, same convention
as everywhere else in this project."* We were holding our own pipeline to a
strictly harder standard than the baseline we were comparing against.

Rescoring the exact same consensus pipeline (same picks, same weights, same
tol=2 consensus rule -- nothing retrained or changed) with the correct,
matching metric:

| | Mean | Median | Std | @5px | @10px | @15px |
|---|---|---|---|---|---|---|
| **Ours, Test (46)** | **2.86px** | 2.72px | 2.47px | 91.3% | 95.7% | **100.0%** |
| **Ours, Train (182)** | 3.24px | 2.24px | 9.78px | 91.8% | 97.8% | 99.5% |
| Vanilla MonoTrack (reported) | 6.79px | -- | -- | 89.1% | 89.1% | 93.5% |
| Old reranker, uses MonoTrack score (reported) | 5.96px | -- | -- | 93.5% | 93.5% | 97.8% |

All 3 "known Test failures" (`racketvision_match229`, `racketvision_match149`,
`tnv2_match2`) -- the ones diagnosed in exhaustive detail all night as the
dominant remaining problem -- disappear entirely under fair scoring; none
appear in the worst-10 list anymore (worst is 11.5px). They were never
actually wrong: they were dihedrally-relabeled (mostly mirror-twin) versions
of the correct candidate, which the official baseline convention already
credits as correct and ours didn't.

**Net effect: we don't tie MonoTrack's classical score, we clearly beat it
-- and we also beat the old reranker that's allowed to use MonoTrack's
score as an input feature -- using zero privileged information, with the
exact method already built before tonight.** Everything in the rest of this
document (the oracle=97.8% ceiling, the 93.5% headline, every rejected
approach's reported percentage) was measured under the stricter, buggy
metric and should be read as *relative* comparisons from that night's work,
not as the corrected absolute numbers above.

## (Original write-up below, numbers under the old, stricter metric)

**Best honest result: 93.5% Test pool-top-1 accuracy**, via consensus voting
among EdgeScorer(C)/OrientationScorer(D)/PixelScorer(F)'s own independent
top-1 picks (tol=2px agreement, falls back to the 3-constant weighted combo
C=0.095/D=0.218/F=0.836 otherwise). Selected via proper 5-fold CV on Train
only, applied to Test exactly once. **Ties MonoTrack's own built-in
classical score (93.5%), with zero reuse of its output.** This was already
established before tonight's session and survived every attempt to beat it.

Oracle ceiling (perfect per-image selector among C/D/F): **97.8%**. The
~4.3-point gap is almost entirely the mirror-twin ambiguity (see below).

**Robustness confirmation (final check tonight):** reran the 5-fold CV
tolerance selection with two more independent KFold random seeds (123, 7),
on top of the original (42). All three independently select the identical
tol=2 (CV mean 86.8% every time) and land on the identical Test result,
93.5%. This isn't a lucky fold split — it's a stable, reproducible
selection across three unrelated randomizations.

## What was tried tonight, and why each one failed honest validation

Every item below looked promising on a small number of known failure cases
and then failed (or only tied) once properly validated — train-only
hyperparameter selection (CV or held-out split), Test touched exactly once.
This repetition (6+ independent ideas, same pattern) is itself the key
finding: **the 93.5% consensus+combo mechanism is a genuinely robust local
optimum**, not something a quick patch fixes.

### 1. EdgeScorer (C) saturation fix — real bug, net negative
Found: 66%/63% of Train/Test top-picks had an *exact* bit-identical 12-dim
feature vector (hard 5px hit-fraction threshold saturating), collapsing 182
images into 31 unique scores. Confirmed real via `check_saturation.py`.
- Fully continuous replacement (v2): fixed saturation (181/182 unique) but
  standalone accuracy dropped hard, Train 82.4%→74.2%, Test 87.0%→65.2%.
- Hybrid hit+continuous (v3, 24-dim): still regressed, Train 74.7%, Test 71.7%.
- Properly re-tuned combo weights for v3 + consensus, honest 5-fold CV:
  Test 87.0% (consensus) / 89.1% (combo-only) — both below baseline.
- A tempting 95.7% number along the way (reusing *stale* old weights on the
  new features) was traced to small-Test-set luck, not a real effect —
  properly re-tuned weights for the same features only got 89.1%.
- **Conclusion: the saturation is real but harmless in practice** — it
  mostly occurs exactly where sub-pixel distinctions are meaningless
  (mirror-twin pairs), so "fixing" it just adds noise. Scorer D, for
  comparison, has 0% saturation (181/182 unique) yet is still the weakest
  standalone scorer (69.6%) — its weakness is the same mirror-twin
  phenomenon (97%/86% of its Train/Test wrong picks are twin cases), not a
  second bug.

### 2. Mirror-orientation specialist (Scorer M) — 5 variants, all negative
Hypothesis: court lines are left-right mirror-symmetric, so the dominant
remaining failure (confirmed: literally the Test failures, before tonight)
is MonoTrack's pool containing a near-duplicate "twin" candidate with
P1_TL↔P4_TR / P2_BL↔P3_BR labels swapped — provably invisible to any
symmetric content/pixel scorer. Built a CNN (image + line mask + two
corner-identity marker channels) to read *real* asymmetric scene content
(scoreboard, benches, umpire chair, camera crew) and break the tie.
- v1 (synthetic GT-label-swap training, single seed): 70.7% isolated
  flip-detection accuracy (chance=50%) — looked like real signal.
- Binary override (trust M when a twin is found in the pool): **zero
  change**, 93.5%→93.5%. Genuine consensus-fallback ambiguity too rare
  (~1 image); looser twin-search tolerances just catch spurious matches
  (174/182 images have *something* within 10px in the huge pool).
- As a 4th continuous combo weight (single seed): reached 93.5% — tied,
  not beat.
- 3-seed ensemble of M (less noisy, same Train accuracy 87.9%): Test
  **dropped to 89.1%** — proved the single-seed "tie" was itself
  small-Test-set luck, not a robust result (critical cross-check).
- Root-cause diagnosis (confirmed experimentally): M's synthetic training
  task was solvable via a trivial shortcut — P1_TL sits on image-left in
  85.7%/91.3% of Train/Test images (a labeling-convention artifact, not
  real content), so "is marker-A left of marker-B" alone predicts the
  synthetic label most of the time. Proof: adding flip-consistent data
  augmentation (provably label-preserving) made training *collapse to
  chance* (50-55%, loss stuck at ln(2) for 45 straight epochs) because it
  directly contradicts that shortcut for half the data.
- v5, retrained on REAL MonoTrack-generated twin pairs (not synthetic GT
  swaps, so the shortcut isn't available): genuine signal, but much
  weaker — **60.3%** isolated accuracy (vs the inflated 70-74%). Integrated
  as a 4th combo weight: Train jumped to 91.2% (highest ever) but Test
  **dropped to 82.6%**, and forcing M's weight to 0 gave the *identical*
  82.6% — M never actually changed a single Test decision despite a large
  fitted weight, a clear overfitting symptom (4 free continuous weights
  against a noisy 182-image discrete objective).

### 3. Previously-built, never-evaluated classical scorers — all catastrophic standalone
Found 4 scorer scripts in the repo that were written but never run or
integrated: inner-vs-outer color contrast, interior-color-consistency
(hue circular std), Harris corner-response at the 22 template points, and
cheap geometric plausibility (area/aspect/off-screen/centroid).
- All 4 get decent classification accuracy on a small labeled subset
  (72-95%) but **catastrophic full-pool ranking**: color v1 4.9%/2.2%,
  color v2 **0.0%/0.0%**, corner 1.6%/4.3%, geom 3.3%/13.0% (Train/Test).
- Why: MonoTrack's candidates are all built by warping the *real* court
  template through a fitted homography, so every candidate already has
  valid court shape/color/corner-structure by construction — these
  features can't discriminate correct from wrong among thousands of
  candidates, even though they look informative on a small sample.
- Cheap combo-integration check (geom as 4th weighted term, Train-only
  differential evolution): optimizer assigned it **weight = 0.000
  exactly** — confirmed useless, not just unhelpful, recovering the exact
  3-weight baseline (85.2%/91.3%).

### 4. Minority-mirror consensus rule — the most structurally precise idea, still negative
Diagnosed the exact nature of the 3 remaining Test failures by checking
each one's rank in every scorer's own ordering:
- `racketvision_match149`: all 3 scorers (C/D/F) agree with each other on
  the *same* wrong candidate — a genuine blind spot, no disagreement to
  exploit.
- `racketvision_match229` and `tnv2_match2`: **C alone already ranks the
  true candidate #1**, but D and F agree with each other on a *different*
  wrong answer, and the majority-vote mechanism picks their (wrong) answer.
  Confirmed structurally: D and F's shared position is the mirror-twin of
  C's position (within 1.0-5.3px) — not a coincidence, the exact known
  failure mode, just now affecting the majority instead of being resolved
  by it.
- New rule: when two scorers agree with each other on a position that is
  specifically the mirror-twin of the third's position, trust the minority
  instead. Precise and well-motivated — but honest 5-fold CV search over
  (agreement tolerance, mirror tolerance) still only reached 89.1% Test,
  below the 93.5% baseline.
- Diagnosed exactly why: the two target cases need *different* agreement
  tolerances to even trigger (`tol_agree` must be ≥5.4px to catch
  `tnv2_match2`, since D and F's own mutual distance there is 5.4px) — but
  at any tolerance loose enough to catch both cases, the rule starts firing
  on unrelated images elsewhere and does real collateral damage (CV mean
  drops to 67-85%). This is a genuine, quantified limit, not a tuning
  failure: the specific distances in these two cases don't share a safe
  common threshold.

### 5. Learned selector (decision tree over distance/confidence features) — also negative, and clarifying
Tried replacing the hand-tuned thresholds with a depth-limited decision
tree (4-way classification: which of {C, D, F, combo} to trust), fed the
same handful of engineered distances/confidences, depth chosen via honest
5-fold CV. Best depth (5) still only reached CV mean 84.1% — *below* the
simple fixed-threshold rule's 86.8% — and Test accuracy was 84.8%, below
even the plain always-combo baseline (91.3%). The learned tree was visibly
overfitting (redundant branches, identical predictions on both sides of
several splits) despite depth-limiting and CV selection.

**This unifies the whole night's results into one explanation: with only
182 Train images, every attempt to add flexibility — more scorer weights,
deeper models, learned decision boundaries, ensembles — has underperformed
simpler fixed methods, consistently.** This matches the original (pre-
tonight) scorer-design summary's own finding that plain logistic regression
beat every more complex model tried on Scorer C alone. The 93.5% result
(3 constants + one 2px threshold) is very likely near the practical ceiling
achievable with this amount of labeled data, independent of cleverness.

## Open directions for next time (not yet tried / incomplete)

- **Most likely lever given tonight's unifying finding: more labeled
  Train images.** Every flexible method lost to simple ones at n=182;
  that's the signature of a data ceiling, not an algorithm ceiling. More
  verified annotations would let the already-built flexible options
  (M, 4-weight combos, learned selectors) actually be validated/trusted
  rather than overfitting — worth prioritizing over more scorer ideas.
- Train the mirror specialist on a genuinely larger set of real twin pairs
  (157/182 Train images have one at tol=3px — only 126 were used for
  v5's train split after held-out val) with more epochs/better
  regularization now that the shortcut bug is understood and avoided.
- A *learned* (not hand-tuned-tolerance) selector for the minority-mirror
  pattern — e.g. a tiny classifier taking (d_cd, d_cf, d_df, mirror
  distances, individual confidences) as input, trained/validated the same
  honest way, might find a nonlinear decision boundary the two fixed
  thresholds can't express.
- VGGT planarity/confidence and player-containment checks were flagged as
  promising in an earlier session's summary and still haven't been tried.
- Scorer D's weakness is now fully explained (mirror-twin susceptibility,
  not a separate bug) — not worth further standalone investigation.
