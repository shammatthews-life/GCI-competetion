# Phase 16 — nested CatBoost parameter selection

## Question

Can fold-local CatBoost hyperparameter selection improve ranking over the
Phase 15 three-feature-block, eight-key native-category model, without
selecting settings on each outer validation fold?

## Frozen data and feature pipeline

- Training rows: `D:\gci\inpuy\train.csv`.
- Outer folds: the Phase 13 five-fold stratified assignment, seed `20261113`.
- Features: Phase 13 raw, rowwise and all eight product-key features; Phase 14
  deterministic numeric enhancements; Phase 15 deterministic date-offset
  lifecycle features.
- Model: GPU CatBoost, logloss, AUC evaluation metric, one-hot threshold 10,
  CTR complexity 1, border count 128, and identical folds.
- Reference: saved Phase 15 primary OOF prediction on the same fold assignment.

The product-key constants and CatBoost native CTR eligibility remain
unresolved. This is a separate research track, not a competition eligibility
claim.

## Nested selection

Each outer training partition is split into three stratified inner folds.
Every candidate receives inner OOF predictions over the complete outer
training partition. The candidate with the greatest pooled inner OOF AUC is
selected; it is then refit on the complete outer training partition and
predicts only that outer validation partition. Outer labels are not used in
selection, stopping, or fit.

Four parameter combinations are fixed before execution:

| Name | Depth | Learning rate | Iterations | L2 leaf regularization | Random strength |
|---|---:|---:|---:|---:|---:|
| historical_d6_lr005_i700 | 6 | 0.050 | 700 | 5.0 | 1.0 |
| shallower_d5_lr005_i850 | 5 | 0.050 | 850 | 5.0 | 1.0 |
| deeper_d7_lr004_i800 | 7 | 0.040 | 800 | 8.0 | 1.0 |
| regularized_d6_lr0035_i1000 | 6 | 0.035 | 1,000 | 8.0 | 0.5 |

Within each outer fold, candidates use fixed but distinct deterministic
seeds; the selected model uses a separate deterministic seed. No changes to
the candidate grid are permitted after examining outer-fold results.

## Evaluation and limits

Primary metric is pooled outer-fold ROC-AUC. Also report per-fold AUC,
paired fold deltas vs Phase 15, and a 1,000-resample class-stratified paired
bootstrap interval conditional on these saved predictions. The bootstrap
does not account for prior label exposure or model-selection uncertainty.

All labels have been repeatedly used in earlier experiments. Therefore even
properly nested fold selection is research evidence, not independent
confirmation. No test rows, external data, submissions, or champion changes
are allowed. A score increase does not resolve competition eligibility of
the product constants or CatBoost CTRs.
