# Research and evaluation protocol

## Frozen data and compliance boundary

Only the exact organizer files whose SHA-256 values appear in
`OFFICIAL_RULES_SNAPSHOT.md` are admitted. `sample_submission.csv` is used to
validate identifiers and output schema. Model features exclude `SK_ID_CURR`
and `TARGET`; the target is accessed only as the supervised training response.

The training/test split is never combined to fit a vocabulary, imputer,
frequency, statistic, residual model, scaler, embedding, or model. Row-wise
formula features may be evaluated independently on each record, but all
learned transforms are fit only on the active training portion. No applicant
matching or label transfer is permitted.

## Evaluation design

- Fixed first evaluation seed: `314159`.
- Five-fold shuffled stratified CV; the same seed/fold assignments are used
  for every candidate in the first comparison.
- Each row receives an OOF prediction from a model that was not fit on that
  row's `TARGET`; prediction arrays are saved with `SK_ID_CURR`.
- Report per-fold AUC, mean/std fold AUC, and pooled OOF AUC separately.
  Fold mean is primary because pooling predictions from separate fold models
  may be sensitive to between-fold score scales.
- Report paired fold deltas and count how many of five folds improve. Do not
  call a one-fold or sub-fold change a reliable gain.
- Fixed equal-weight ensemble results are diagnostics. No ensemble weight is
  selected on the same OOF labels that are used to report candidate AUC.
- Candidate configurations are written before their run and kept unchanged;
  no leaderboard score is used as a training label or validation metric.

## Evaluation-exposure limitation

The project history contains many full-training-set experiments, and the
recorded V28 validation split overlaps earlier development/benchmark rows.
The seed above is a new fixed split for this work, but a different seed alone
cannot establish that it is historically untouched. Therefore the first
results are controlled paired experiments, not a fully independent
generalization estimate. Once the first candidate is selected, future changes
must use nested inner validation or a new locked holdout that is not inspected
until the final comparison. That holdout must still be described as
post-selection if the history audit cannot establish it was previously
unseen.

## GPU and reproducibility

CatBoost and XGBoost are configured for the detected RTX 4050 GPU; LightGBM
uses CPU because GPU support is not assumed. Device tests are run explicitly,
and any device mismatch/fallback is recorded. Seeds, library versions,
hardware, input hashes, source hashes, fold results, and output hashes are
retained for each candidate. GPU kernels may not be bit-for-bit deterministic
across library/driver versions; reproducibility is reported to the level
supported by the recorded environment.

## Decision rule

Promote an improvement only if it:

1. passes the compliance and leakage tests;
2. has consistent paired fold gains or a clearly justified robust trade-off;
3. does not depend on test labels, hand-coded row decisions, or ambiguous
   train/test pooling; and
4. can be rerun from the saved code/configuration and admitted inputs.

No experiment can guarantee a public leaderboard increase. The public
leaderboard is a subset, while final scoring uses the complete test set.
