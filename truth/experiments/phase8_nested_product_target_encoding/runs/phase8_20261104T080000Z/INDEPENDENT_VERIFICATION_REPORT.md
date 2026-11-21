# Independent verification and seed-averaged OOF diagnostic

## Verification

- Train rows were loaded by ID and TARGET only; test data were not opened.
- All `171,202` OOF IDs are unique and TARGET-aligned to the train file.
- Recomputed all `18` seed/variant pooled OOF AUCs from saved predictions; each matches `SEED_METRICS.csv` within 1e-12.
- All outer fit/validation maps and inner map/application folds are recorded disjoint; no outer validation TARGET is reported as used for encoding.
- Synthetic perturbation test: changing only application-row labels leaves their encoding and count unchanged: `True`.
- Overlapping target-map fit/application indices are rejected: `True`.
- Source/data hashes are recorded in `INDEPENDENT_AUDIT_PROVENANCE.csv`.

## Fixed seed-average diagnostic

Each applicant's three predictions are averaged across the three fold seeds. For each of those three models, the applicant is held out, so this is a valid cross-fit average prediction diagnostic; it is **not** a full-data deployed-model score.

- Raw + Phase 3 product seed-averaged OOF AUC: `0.764195071`.
- 8-key nested target encoding seed-averaged OOF AUC: `0.769695603` (delta `+0.005500532`).
- 8-key plus counts: `0.769844976` (delta `+0.005649905`).
- 10-key: `0.769559052` (delta `+0.005363981`).
- Descriptive paired stratified row-bootstrap 95% interval for the 8-key seed-averaged delta: `[+0.004318504, +0.006582593]`, over 500 resamples. Labels are historically exposed; this interval is not independent confirmation and does not adjust for feature/key search.

The seed-average improves over the mean of per-seed pooled AUC when its rank aggregation helps, but seed repeats use the same applicants and are correlated. It should not be represented as three independent datasets.
