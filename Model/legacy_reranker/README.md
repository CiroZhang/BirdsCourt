# Legacy: MonoTrack-score-reliant reranker (superseded)

This is the original learned candidate reranker. Kept for reference and
reproducibility of the numbers it originally reported, but **it is not the
recommended method anymore** -- `honest_scorer.py` (at the repo root)
supersedes it.

**Why it's deprecated:** its own feature set explicitly includes
"classical score proxy: MonoTrack's own candidate ordering (rank-0 = best
by its internal pixel-overlap score), z-scored across the pool" (see
`reranker.py`'s `rerank_candidates()`). That's a direct dependency on
MonoTrack's own classical score as an input feature.

`honest_scorer.py` was built specifically to answer: can a candidate
selector match or beat this, using **only** independently-derived image
evidence (edge alignment, gradient orientation, and a small pixel/geometry
CNN), never touching MonoTrack's score at all? Scored under the same
convention (`scripts/monotrack_test.py`'s dihedral-permutation-aware
corner metric), the answer is yes: `honest_scorer.py` reaches 100% Test /
99.5% Train pool-top-1 accuracy (<15px), beating both vanilla MonoTrack
(93.5%) and this reranker (97.8%). See `../RESULTS.md`.

Files here:
- `reranker.py` -- the model and its `rerank_candidates()` entry point.
- `scripts/birdscourt_train.py` / `scripts/birdscourt_test.py` -- its
  training/eval scripts.
- `weights/reranker_model.pt`, `weights/pca_artifacts.npz` -- its trained
  weights.

`court_detection.py` no longer calls into this folder by default.
