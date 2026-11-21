# Validation framework specification

## Common discovery protocol

- Primary metric: ROC-AUC.
- Initial comparison folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, saved by ID in `FOLD_ASSIGNMENTS.csv` (SHA-256 `a8455332203da5234e3eb180cc1852b95b5dccdc627ceb7f6d22e8d0f5cd5cfb`).
- All candidates in a comparison receive the same row/fold assignments.
- Every learned transform, imputer, category vocabulary, bin boundary, feature selection, clusterer, and supervised fit is learned only on the current outer fit data. No test covariates participate.
- Inner early stopping uses an inner stratified split of outer-training labels; refit on all outer-fit rows for exactly the selected iterations; score each outer validation row once.
- OOF file must contain unique ID, target, fold, one finite prediction per row, and exact current data ID/label/fold alignment. Hash predictions and source.
- Report pooled OOF AUC, five fold AUCs, mean/std fold AUC, fold deltas, positive-fold count, prediction correlation and ranking disagreement for model comparisons.
- Training validation labels may be used for descriptive error analysis only after prediction generation; they cannot select features/weights for the same reported score.
- Fixed blends must have weights frozen before evaluation. Any blend selection requires nested selection or a separate untouched labeled cohort.
- Confirmation seed/cohort is kept sealed; no tuning after inspecting its results. A different random seed over these same labels is not independent confirmation.

## Current exposure and limitations

The saved seed-42 folds and every label in this training table have been examined in earlier research. The Phase 6 baseline is reproducible and usable for paired exploratory comparisons, but not a fresh independent estimate. A truly independent confirmation requires a previously unseen labeled cohort or an organizer-approved validation split that has not informed feature/model design.

## Prohibited or disabled

No test-to-test coupling, pooled train/test fitting, unresolved target encoding, external data, manual label/score edits, or leaderboard-driven tuning. Test data are read only for header/ID integrity here.
