# Phase 54 — fold-local q-fingerprint experiment

## Hypothesis

The annuity/credit ratio may encode discrete loan-product schedules. A tree can use repeated rounded ratio values as categorical predictors, while a continuous ratio alone encourages monotone threshold splits. This run tests that representation without decoding fees or transferring labels between applicants.

## Protocol

- Train/test rows: 171202 / 61500.
- Five stratified folds, seed `20261113`; fixed `900` rounds and no outer-fold early stopping.
- Paired control and candidate use the same LightGBM settings and folds.
- The only candidate-specific features are row-local `annuity / credit` keys rounded to 3, 4, and 5 decimals.
- Category dictionaries are fit on each training fold only; unseen validation/test keys are missing.
- No custom target encoding, group/value counts, applicant matching, train/test pooling, fee constants, cached predictions, or rank blending.

## Results

| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |
|---|---:|---:|---|---:|
| Baseline | 0.759105515 | 0.759141612 | 0.763026, 0.764707, 0.756379, 0.756045, 0.755550 | 0.245785542 |
| q-fingerprint candidate | 0.753230864 | 0.753297111 | 0.758711, 0.759691, 0.745648, 0.751464, 0.750972 | 0.248769847 |

- Paired AUC delta: `-0.005874651`; positive folds `0/5`.
- Paired OOF bootstrap interval: `[-0.007954601, -0.003712842]` (conditional on these predictions).

## Interpretation

This is the clean test of whether quantized ratio categories add value beyond row-local numeric finance features in this learner. A positive OOF delta would support the representation, not the source notebook's target-encoded or pooled-statistic pipeline. A negative/neutral delta rejects these q-key encodings for this setup. All labels have prior project exposure, so this is not an untouched confirmation or a leaderboard score.

- Runtime seconds: `410.40`.
- Source SHA-256: `64e74b91f5110bb722d4f6237a29f206b8d8d1e9601b134cdc3561641de8a6e1`.
- Full artifact hashes: `PHASE54_SHA256_MANIFEST.json`.
