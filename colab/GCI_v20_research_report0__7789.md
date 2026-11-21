# GCI v20 — Tariff-distance challenger and conservative leaderboard probe

Date: 20 October 2026

## Decision

This is an **experimental leaderboard probe**, not a validated champion replacement. The v15 OOF/cache bundle is unavailable in this runtime, so the new challenger cannot be tested as a matched v15 OOF blend. Its public score is unknown until submitted. Keep v15 as fallback (user-reported public score 0.77891).

## New hypothesis

Loan-product decoding already helps the project. We add label-free features measuring (a) distance of the annuity/credit tariff ratio to the nearest frequent ratio center, (b) frequency support of the observed rounded ratio, and (c) distance/support for credit-to-goods ratio. Frequent centers are derived from supplied train+test predictors only; no target labels are used to define them.

## Validation results

Both runs use all 171,202 training rows, 3-fold stratified CV, LightGBM, 450 estimators, and independent split seeds. The simple baseline and tariff-distance model are evaluated on the same folds within each run.

| Seed | Baseline OOF AUC | Tariff-distance OOF AUC | Gain | Fold gains |
|---:|---:|---:|---:|---|
| 20261026 | 0.756518 | 0.759753 | +0.003236 | +0.002816, +0.003034, +0.003870 |
| 20261027 | 0.755399 | 0.757979 | +0.002579 | +0.003512, +0.002053, +0.002196 |

The feature addition wins on all 6/6 held-out folds across the two independent seeds. The 50:50 rank blend of the baseline and tariff model was inferior to the tariff-only model in both runs, so it was rejected.

## Submission construction

- Preserve the current v15 test ranking as the 99.5% component.
- Add 0.5% of the averaged tariff-distance challenger rank, averaged across two independently seeded 3-fold model ensembles.
- This small weight is deliberately conservative because the previous 5% v18 probe scored 0.77882, below v15's 0.77891.
- The 0.5% choice is a cautious probe, **not** a weight optimized on matched champion OOF; there is no proof it improves leaderboard score.

## File checks

- Rows: 61500
- Columns: `SK_ID_CURR,TARGET`
- Unique IDs: True
- IDs in sample order: True
- All predictions finite: True
- Rank correlation between challenger and v15: 0.948165
- Mean absolute change in rank-percentile score: 0.00033167
- SHA-256: `ff62ef29af29b4833b3dd0b08e80d2880f1c26e174eace07111144c1e6858d44`

## Next acceptance test

To claim a real improvement, recover `gci_cache_final` and compare the candidate branch against v15 using identical outer folds. Require the existing gate: pooled OOF gain >= +0.00015, at least 4/5 fold gains positive, and paired-bootstrap 95% lower bound > 0. Without this, v20 must remain labeled a probe.
