# Phase 21 — focused full-data validation

## Status and protocol

This is a focused, full-training-data follow-up on the same five stratified folds
as Phase 15 (`random_state=20261113`). Each candidate is compared with the
reproduced Phase 15 GPU CatBoost single-seed baseline on identical rows. It is
not independent confirmation: these labels have been examined in prior phases.
No test data, external data, submissions, or champion files were read or changed.

The Phase 15 Track B baseline uses its original 99-column feature representation,
including native CatBoost category statistics and historical product keys. Their
competition eligibility remains unresolved. Track A comparisons use GPU
CatBoost with one-hot categories only (`one_hot_max_size=256`) and no historical
product-key categories; a second Track A model adds fold-local frequencies of
four direct, constant-free ratios. All models use 700 iterations, depth 6,
learning rate 0.05, and no outer-fold early stopping.
For each learning-curve fraction, product-key frequency features are recomputed
from that fraction's selected training rows and applied to the unchanged outer
validation fold.

## Main findings

- Reproduced single-seed full-data AUC: `0.767472835`;
  saved Phase 15 reference: `0.767510928`.
- Historical Phase 15 three-seed OOF AUC:
  `0.768824708`.
  Those component OOF vectors used different fold assignments, so this is
  context only; the new 1/3/5/10/20 seed curve holds the same folds fixed.
- Best result in this preregistered follow-up: `seed_average_20` at
  `0.768650650`. This is a research result on previously
  examined labels, not a promotion or score claim.
- See `PHASE21_LEARNING_CURVE.csv`, `PHASE21_SEED_SCALING.csv`,
  `PHASE21_SPECIALIST_SUMMARY.csv`, and `PHASE21_MODEL_RESULTS.csv` for
  fold-level detail.

## Specialist error and calibration analysis

Full-data five-fold specialist means:

```
                                  hard_route_auc  fixed_50_50_auc  calibrated_50_50_auc
gate                                                                                   
contract_type                           0.765127         0.767666              0.767676
contract_x_external_completeness        0.759450         0.766523              0.766523
external_completeness                   0.763833         0.767358              0.767358
```

Hard routing, the fixed 50/50 global/expert mixture, and the contract-type
Platt-calibrated mixture are evaluated on each outer fold. Calibration
parameters are fitted only to inner-fold OOF expert scores within each outer
training partition. The sampled positive-negative pair counts showing where
specialists reversed global ordering errors are in `PHASE21_ERROR_CORRECTIONS.json`.
Complete row-level predictions, IDs, targets, and fold assignments are saved in
`PHASE21_OOF_PREDICTIONS.csv` for independent AUC, correlation, and error analysis.
`PHASE21_PAIRED_BOOTSTRAP.csv` gives 300-replicate row-stratified intervals
conditional on these OOF vectors; they do not capture training-fold
variability or the effects of earlier model selection.

## Eligibility, blocked directions, and data inventory

- Only supplied `train.csv` and the saved Phase 15 train OOF reference were read.
- All heavy model fits use CatBoost GPU. The tiny two-parameter Platt mapping is
  fitted on CPU with deterministic numerical optimization; no CPU classifier
  or general model fallback is used.
- GPU monotonic constraints were tested and rejected by CatBoost as
  unimplemented for GPU.
- A one-fold GPU `YetiRankPairwise` screen scored
  `0.710221857`. Its default PFound
  metric is CPU-computed, so the objective was excluded from the full suite
  both for the weak result and to avoid CPU work.
- The logistic-regression stack was not run because no CUDA logistic solver is
  installed; CPU fallback is intentionally prohibited.
- Nested target encodings/hierarchical target-statistic shrinkage are deferred
  pending an organizer ruling. No TARGET encodings are used here.
- No `FLAG_DOCUMENT_*` columns exist in the supplied training schema.
- Test product-key coverage was not inspected to preserve the established
  no-test-data development protocol.
- A genuinely untouched labeled holdout is unavailable after prior phases;
  this run cannot establish independent generalization.

## Completion

Run summary: `5041.0` seconds, with
`241` GPU model fits including smoke test,
on `171202` rows across
`2` process session(s). GPU telemetry is checkpointed per fold, but an interrupted process may have an unrecorded tail:
`2382` samples at two-second intervals.
Completed folds: `[1, 2, 3, 4, 5]`.
Completed fold predictions are atomically checkpointed and resumable only when
input, code, and feature-schema hashes match.
Train SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
