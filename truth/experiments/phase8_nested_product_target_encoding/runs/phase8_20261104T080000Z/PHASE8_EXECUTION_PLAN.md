# Phase 8 preregistered experiment plan

## Question

Does correctly nested, smoothed target encoding of the hypothesized loan-product keys add signal beyond the fold-safe raw-plus-Phase-3 product model? Claude reported a large lift on a weaker HistGradientBoosting baseline. This run tests the claim on LightGBM and repeats five-fold CV across seeds `(42, 7, 2026)`.

## Data and rule boundary

- Read only the provided `train.csv`; do not open or use test predictors, test IDs, or test labels.
- No external data or hand labels.
- The supplied rule audit says target encoding is not expressly prohibited but is **unresolved in scope**. This experiment is research evidence only and does not authorize submission.
- Every target map is fit on labels from a strict subset of the current outer-training fold. Outer-validation labels are never used for feature construction, stopping, or model fitting.

## Frozen model and candidate definitions

- Five stratified folds per seed, seeds `(42, 7, 2026)`.
- Raw baseline and raw + exact Phase 3 K/Q/P block are matched controls.
- Primary augmentation: raw + Phase 3 block + smoothed target means for the eight predeclared keys `Q3,Q4,Q5,KQ3,KQ4,P4,P5,AG3`.
- Nested training encodings: inner 5-fold OOF within each outer-fit partition, smoothing alpha `30.0`. Outer-validation encodings use maps fit on the complete outer-fit partition.
- Comparisons: `Q4` only; 8 keys; 8 means plus their log-count controls; 10 keys (adds `P3,KQ5`).
- `P` keys use fixed offset `130.5` and `KQ` keys use fixed step `0.0066` solely as hypotheses from the supplied discussion. They are not verified fees/terms. Key rounding uses integerized decimal ratios; no target-dependent key construction.
- Shared boosting rounds are selected on an inner split of outer-fit rows using raw baseline features only, then held constant across variants. No hyperparameter search, seed bagging, test-time inference, or holdout scoring is performed.

## Prespecified outputs

Report per-seed pooled OOF AUC, five fold AUCs and paired deltas vs raw and raw-plus-Phase-3, key support/cardinality, prediction hashes, encoding overlap checks, and exact inputs/source hashes. Repeated folds reuse the same labeled dataset and do not constitute independent confirmation.
