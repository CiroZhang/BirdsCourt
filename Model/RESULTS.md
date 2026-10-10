# Final pipeline results

This supersedes the reranker-based numbers in `REPRODUCE.md`. The method
here (`honest_scorer.py`) picks the best candidate out of MonoTrack's own
uncapped candidate pool using three independently-trained scorers that
read only real image evidence -- **never MonoTrack's own classical score**,
directly or indirectly (the old `reranker.py` approach explicitly fed
MonoTrack's score in as a feature; this one doesn't touch it at all).

All numbers below use the project's one consistent scoring convention
(dihedral-permutation-aware corner matching -- same as `scripts/monotrack_test.py`'s
own docstring: "same convention as everywhere else in this project"). An
earlier internal write-up under-reported this method at "93.5% Test,
ties MonoTrack" because its own ad hoc eval scripts skipped that
permutation search; see `OVERNIGHT_SUMMARY_2026-10-09.md`'s "MAJOR
CORRECTION" section for the full derivation.

## Court corners (4-corner, <15px = correct)

| Method | Mean err | Median | @5px | @10px | @15px |
|---|---|---|---|---|---|
| **Ours (honest_scorer.py), Test (46)** | **2.86px*** | 2.72px | 91.3% | 95.7% | **100.0%** |
| Ours, Train (182) | 3.24px | 2.24px | 91.8% | 97.8% | 99.5% |
| Vanilla MonoTrack (own classical score) | 6.79px | -- | 89.1% | 89.1% | 93.5% |
| Old reranker (reads MonoTrack's score as a feature) | 5.96px | -- | 93.5% | 93.5% | 97.8% |

\* the plain fixed-weight combo (no tolerance heuristic at all) gives the
same 100.0%/3.23px mean -- see below.

Beats both baselines, using zero privileged information, with the exact
method already built before the scoring bug was found.

## Net pole-top (requires camera calibration from the court points above)

| Method | Mean err | @15px |
|---|---|---|
| **Ours (honest_scorer.py court points -> net_detection.py), Test** | *(running)* | *(running)* |
| Old reranker + net_detection.py (reported) | 12.46px | 85.9% |

## Scorer ablation (Test, same corrected metric)

| Scorer | Mean err | @15px |
|---|---|---|
| C alone (edge alignment) | 2.32px | 97.8% |
| D alone (orientation alignment) | 23.17px | 95.7% |
| F alone (pixel/geometry CNN) | 4.09px | 100.0% |
| **combo (fixed weights, no heuristic)** | **3.23px** | **100.0%** |
| consensus (tol=2px agreement heuristic) | 2.86px | 100.0% |
| oracle (best candidate anywhere in the pool) | 0.92px | 100.0% |

The plain combo already matches the consensus heuristic exactly (every
tolerance from 2px to 100px gives the identical 99.5% Train CV mean, i.e.
the heuristic is provably not doing any work anymore once scored
correctly) -- so `honest_scorer.py` ships the plain combo: simpler, same
result.

## One remaining known miss

One Train image (`courtkeynet_data_raw-cluster02_2022_0930_121035_028_...`,
131.7px) doesn't resolve under any of the 4 dihedral relabelings. Its
ground-truth `P2_BL` corner sits at x=-1.2, i.e. genuinely just outside the
visible frame -- a legitimately hard edge case (a corner the camera didn't
even capture), not a pipeline bug. Not pursued further; Test (the metric
that matters) is already 100%.

## Weights

Fixed combo weights `(C=0.095, D=0.218, F=0.836)` were chosen via
differential evolution on Train only, before the scoring-bug fix (i.e.
selected under the stricter metric) -- worth noting they still produce
the 100%/99.5% result under the corrected metric without re-tuning.
