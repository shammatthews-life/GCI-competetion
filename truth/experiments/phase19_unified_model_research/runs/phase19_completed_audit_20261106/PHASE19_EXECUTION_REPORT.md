# Phase 19 — unified model recovery, validation, and discovery

**Status:** Historical OOF recovery and matched source audit complete. A new five-fold CPU neural candidate was also completed. Research-only; no champion or submission was changed.

## Unified score position

| Model | Pooled OOF ROC-AUC | Seed mean AUC | Prediction-average AUC |
|---|---:|---:|---:|
| phase9_native8_seed_average | 0.769839127 | 0.768239947 | 0.769839127 |
| phase15_three_seed_average | 0.768824708 | 0.767512684 | 0.768824708 |
| phase9_native8_seed2026 | 0.768764364 | — | 0.768764364 |
| phase18_nested_blend | 0.768668866 | — | 0.768668866 |
| phase9_native8_seed7 | 0.768000057 | — | 0.768000057 |
| phase9_native8_seed42 | 0.767955420 | — | 0.767955420 |
| phase15_seed20261114 | 0.767946055 | — | — |
| phase16_nested_tuned_catboost | 0.767652809 | — | 0.767652809 |
| phase15_seed20261113 | 0.767510928 | — | — |
| phase11_catboost_keys | 0.767399318 | — | 0.767399318 |
| phase15_seed20261115 | 0.767081070 | — | — |
| phase13_E_frequency_ohe | 0.766037689 | — | 0.766037689 |
| phase17_nested_te_lightgbm | 0.765290753 | — | 0.765290753 |
| phase13_C_raw_rowwise | 0.759619220 | — | 0.759619220 |
| phase19_mlp_constant_free | 0.742694381 | — | 0.742694381 |

Phase 9's reported mean seed AUC is `0.768239947`, while its recomputed rowwise mean-prediction AUC is `0.769839127`.
Phase 15's mean seed AUC is `0.767512684`, while the saved/recomputed three-seed prediction average is `0.768824708`.
The saved Phase 19 MLP run is a historical CPU-only experiment (its summary records `gpu_available=false`, runtime `493.03` seconds, and pooled AUC `0.742694381`); it was not rerun or represented as a GPU result.
The seed-averaged OOF vectors do not have a single native fold assignment. Fold metrics label the common Phase 13/15 primary partition as descriptive for those aggregates; native seed-fold results are separately retained.

## Recovered model differences and decision

Phase 9 and Phase 15 are not an exact matched replication. Their feature builders differ; Phase 15 adds numeric/time features, while Phase 9 used its earlier raw/rowwise builder. Both use GPU CatBoost native categorical statistics and eight keys. Phase 9 averages three OOF vectors from split seeds 42/7/2026; Phase 15 averages seeds 20261113/14/15. The Phase 15 time-feature matched lift was only +0.000280 (paired interval included zero), and Phase 16 nested tuning added +0.000142 (interval included zero). The records do not isolate one cause for the difference between aggregate scores.

The highest-scoring conservative no-product-key reference in this inventory is the Phase 13 C benchmark at `0.759619220`. The Phase 19 MLP scored lower and does not change this position. Eligibility remains provisional and subject to organizer interpretation.
The Phase 15 three-seed average remains the best complete result from Phase 15; Phase 9's 0.769839 retrospective rowwise average is higher but is not a controlled reproduction or independently validated champion. No basis was found to claim 0.780–0.785.

## Validation, limitations, and next step

All requested source OOF files were checked for row count, applicant IDs, target alignment, and native fold IDs where saved or reconstructable. Correlations use only exactly aligned OOF vectors. Full fold definitions, model settings, features, eligibility, and reproducibility status are in `PHASE19_MODEL_COMPARISON.csv`; fold detail is in `PHASE19_FOLD_METRICS.csv`.
No new adaptive blend was selected. All historical labels have been repeatedly examined, so nested selection and paired row bootstrap do not provide independent confirmation. The local data contains no bureau/history tables; external data is not allowed. Stop tuning on the same OOF labels unless a genuinely untouched labeled evaluation becomes available.

See `PHASE19_ERROR_ANALYSIS.md`, `PHASE19_FEATURE_ABLATION.csv`, `PHASE19_PREDICTION_CORRELATIONS.csv`, and `PHASE19_ELIGIBILITY_AUDIT.md`.
