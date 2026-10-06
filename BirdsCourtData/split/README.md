# Real-image train/test split

- `train_test_split.json`: the split used for every reported number.
  182 train, 46 test. Seed 0. Lists the photo base names in `train_set` and `test_set`.
- `assign_split.py`: builds `../Train/` and `../Test/` from a flat pool, using the JSON.

## Rules

- The reranker trains on `Train/` only. Its augmented derivatives (flip, crop, perspective)
  of test photos are excluded from training.
- All reported numbers are computed on `Test/` only.
- Four photos were forced into test: the struggle cases vanilla MonoTrack gets wrong
  (`videobadminton_07`, `bfmd_new_PETRONAS...QF`, `courtkeynet..frame206`, `courtkeynet..frame97`).
- One photo (`racketvision_match163`, extreme camera angle) is excluded entirely.

## Reproduce the folders

1. Build a flat pool with `Image/`, `Annotation/`, `Thumbnell/`, `VGGT Outputs/` holding all 228 photos.
2. `python3 assign_split.py /path/to/pool`
