# Reproducing the reported results (legacy reranker)

**Superseded -- see `RESULTS.md` for the current, recommended method and
numbers.** This file documents the older reranker-based pipeline (now in
`legacy_reranker/`), kept for archival reproducibility of its reported
numbers. It uses MonoTrack's own classical score as an input feature;
the current method (`honest_scorer.py`) doesn't, and scores higher.

## Layout

    BirdsCourt/
      Model/                       pipeline code and weights
        court_detection.py         court corners (MonoTrack + honest_scorer)
        honest_scorer.py           CURRENT candidate scorer -- see ../RESULTS.md
        net_detection.py           net poles via camera calibration
        vggt_features.py           cached VGGT focal/planarity features
        camera_calibration.py      pinhole camera fit (inside monotrack_line_detection/)
        weights/honest_scorer/     current scorer's weights
        legacy_reranker/           superseded reranker (uses MonoTrack's own score)
          reranker.py                learned candidate reranker
          weights/                   reranker_model.pt, pca_artifacts.npz
          scripts/
            birdscourt_train.py      train the reranker on Train only
            birdscourt_test.py       full pipeline on Test only
        scripts/
          monotrack_test.py        vanilla MonoTrack on Test only
      BirdsCourtData/
        Train/                     182 photos: Image, Annotation, Thumbnell, VGGT Outputs
        Test/                      46 photos, same folders
        Synthetic Data/            rendered decoy/crop/mask data
        split/                     train_test_split.json, assign_split.py, README.md
      Results/Pictures/

## Setup

    pip install -r requirements.txt
    bash monotrack_line_detection/build.sh     # net-free build used by court_detection.py
    # monotrack_test.py needs the net-included MonoTrack binary built from its own source

## 1. Train the reranker (Train only)

    python3 legacy_reranker/scripts/birdscourt_train.py

Reads `BirdsCourtData/split/train_test_split.json` and the two training jsonls
(context_feat + template_alignment precomputed per candidate). Writes
`legacy_reranker/weights/reranker_model.pt` and `legacy_reranker/weights/pca_artifacts.npz`.
(Paths inside these scripts still assume their old location -- see
`legacy_reranker/README.md`.)

## 2. Evaluate (Test only, 46 photos)

    python3 scripts/monotrack_test.py              # vanilla MonoTrack, net-included build
    python3 legacy_reranker/scripts/birdscourt_test.py    # legacy full pipeline (reranker + net calibration)

Each script writes per-photo results to `Results/`.

## Results on the 46-photo Test set

Corner error in px. Success = share of photos under the threshold.

| Method | Mean | @5px | @10px | @15px |
|---|---|---|---|---|
| Vanilla MonoTrack | 6.79 | 89.1% | 89.1% | 93.5% |
| Ours, full pipeline (reranker) | 5.96 | 93.5% | 93.5% | 97.8% |
| CourtKeyNet (finetuned) | 8.53 | -- | -- | 89.1% |
| CourtKeyNet (base) | 39.86 | -- | -- | 15.2% |

Net pole-top on the same Test set: 12.46 px mean, 85.9% success@15px.

Fresh re-run from this cleaned repo (Train-only reranker, 46 Test photos,
0 errors): full pipeline 5.84 px mean, 93.5% / 93.5% / 97.8% at 5 / 10 / 15 px;
net pole-top 12.41 px mean, 87.0% success@15px (92 points). The success rates
match the table; the mean and pole-top numbers differ from the table by
0.12 px and 0.05 px / 1.1 points, most likely from the RANSAC randomness in the net calibration (not yet checked by repeating with a fixed seed).

## Notes

- The CourtKeyNet rows come from an earlier run on our cluster. Its code and
  weights are not in this repo, so those two rows are reported, not reproducible here.
- The scripts in `scripts/` still hold the cluster paths they ran with. Point
  the path constants at this layout before running.
- The reranker is trained on Train derivatives only. Test photos never enter training.
