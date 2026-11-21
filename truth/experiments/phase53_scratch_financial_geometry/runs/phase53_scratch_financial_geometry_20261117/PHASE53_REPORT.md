# Phase 53 — independent scratch financial-geometry experiment

## Protocol

- Rows: 171202 train / 61500 test.
- Stratified folds: 5, seed `20261113`.
- Paired control: original predictors versus the same model with deterministic, row-local features.
- Fixed CatBoost iterations; no outer-fold early stopping or selection on validation labels.
- No target encoding, applicant linking, cross-row aggregates, test-pool statistics, or outside files.
- The test file is used only for fold-model inference and output-ID verification.

## Results

| Candidate | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |
|---|---:|---:|---|---:|
| Raw-feature baseline | 0.752305736 | 0.752328334 | 0.755285, 0.757689, 0.750763, 0.749023, 0.748882 | 0.247768430 |
| Financial-geometry features | 0.759685265 | 0.759725317 | 0.762688, 0.766527, 0.755908, 0.755581, 0.757923 | 0.245708820 |

- Pooled AUC delta: `+0.007379529`; positive folds: `5/5`.
- Log-loss delta (candidate minus baseline): `-0.002059610`.
- Paired OOF bootstrap interval: `[+0.006189928, +0.008573897]`.
- Bootstrap uncertainty is conditional on these OOF vectors and is not an independent confirmation.

## Interpretation

The candidate is a controlled feature-family comparison, not a leaderboard promise. The formulas use only fields from the same application: amounts are converted to ratios/log scales; age and employment durations are expressed in years; and external-score geometry is represented by row-wise moments, pairwise differences, products, and ratios. No loan term or interest-rate assumptions are imposed.

The five fold models' probabilities are averaged arithmetically to form the single research submission. No CSV-to-CSV rank blend is used. ROC-AUC is invariant to strictly increasing score transformations, so the submission is not rank-normalized or post-calibrated.

All labels have prior project exposure. Do not describe this result as an untouched validation score or promote the output solely because it wins this comparison.

## Artifact hashes and runtime

- Train SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Source SHA-256: `10b98d608ca7a10a1d239443a4d219e3b8fba51024db4c9c44750ff60a81db2e`.
- Runtime seconds: `632.26`.
- Full paths and hashes: `PHASE53_SHA256_MANIFEST.json`.
