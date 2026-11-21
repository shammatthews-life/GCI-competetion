# Phase 3C execution report — model intelligence and OOF research

**Status:** score reproduction, fixed-fold model benchmarking, OOF ranking analysis, and the fold-safe segmentation/nonlinear/missingness/fixed-blend discovery phase have completed.

## Reproduced references

Train OOF scores are recalculated by applicant ID against the preserved five folds. The previous Phase 3 value remains an earlier discovery score; Phase 3B is the controlled product comparison. They are not interchanged.

| OOF vector | Pooled AUC | Mean fold AUC | Fold SD | Status |
|---|---:|---:|---:|---|
| reference__corrected_baseline | 0.760658 | 0.760729 | 0.003770 | historical_reference |
| reference__phase3_discovery | 0.762826 | 0.762902 | 0.003929 | historical_reference |
| reference__phase3b_controlled_product | 0.762444 | 0.762508 | 0.004160 | historical_reference |
| lightgbm__raw | 0.760878 | 0.760908 | 0.004187 | new_fixed_capacity_discovery |
| lightgbm__raw_plus_product | 0.762491 | 0.762511 | 0.004501 | new_fixed_capacity_discovery |
| lightgbm__raw_plus_nonredundant | 0.762157 | 0.762192 | 0.004627 | new_fixed_capacity_discovery |
| catboost__raw | 0.759332 | 0.759393 | 0.003716 | new_fixed_capacity_discovery |
| catboost__raw_plus_product | 0.762173 | 0.762210 | 0.003812 | new_fixed_capacity_discovery |
| catboost__raw_plus_nonredundant | 0.762047 | 0.762078 | 0.003834 | new_fixed_capacity_discovery |
| xgboost__raw | 0.760557 | 0.760597 | 0.003901 | new_fixed_capacity_discovery |
| xgboost__raw_plus_product | 0.761642 | 0.761678 | 0.004192 | new_fixed_capacity_discovery |
| xgboost__raw_plus_nonredundant | 0.761432 | 0.761471 | 0.004393 | new_fixed_capacity_discovery |
| hist_gradient_boosting__raw | 0.758541 | 0.758597 | 0.003711 | new_fixed_capacity_discovery |
| hist_gradient_boosting__raw_plus_product | 0.759660 | 0.759733 | 0.003620 | new_fixed_capacity_discovery |
| hist_gradient_boosting__raw_plus_nonredundant | 0.760181 | 0.760230 | 0.003121 | new_fixed_capacity_discovery |
| random_forest__raw | 0.744717 | 0.744766 | 0.004535 | new_fixed_capacity_discovery |
| random_forest__raw_plus_product | 0.748815 | 0.748852 | 0.003851 | new_fixed_capacity_discovery |
| random_forest__raw_plus_nonredundant | 0.747805 | 0.747849 | 0.004980 | new_fixed_capacity_discovery |

## Fixed-fold model benchmark

- Best new benchmark configuration by pooled OOF AUC: **lightgbm__raw_plus_product**, AUC **0.762491**.
- In-run LightGBM raw comparator: **0.760878**.
- Best configuration's fold wins over LightGBM raw: **5/5**.
- Gap to requested 0.790 target: **0.027509**.
- Versus the Phase 3B controlled reference (0.762444): **+0.000047**.
- Versus the earlier Phase 3 discovery (0.762826): **-0.000335**.

All model configurations were fixed in the runner. Boosting iterations were selected on inner splits using only raw baseline features, then fixed across feature variants for that outer fold. Imputation and one-hot encoding were fit inside the relevant training partition. Results on previously exposed folds are discovery evidence—not independent confirmation. Candidate choice based on these results is itself exploratory.

The historical 0.762826 and the Phase 3B 0.762444 are not a like-for-like score trend: they came from different model/feature training procedures. The new controlled LightGBM product model is close to Phase 3B, but does not exceed the earlier discovery score. It would be inaccurate to call the score increase or decrease an independent generalization result.

### Runtime and measured memory

The benchmark took approximately 55.2 minutes. Runtime by model and fold is in `MODEL_RUNTIME_RESULTS.csv`. Peak working set was measured in isolated full-fold-1 reruns using the raw-plus-product configuration; it is a separate representative profile, not a per-variant memory trace of the long benchmark:

| Model | Fit/predict seconds | Process peak working set (GB) |
|---|---:|---:|
| lightgbm | 6.3 | 0.83 |
| catboost | 139.9 | 0.68 |
| xgboost | 15.8 | 1.02 |
| hist_gradient_boosting | 19.1 | 1.18 |
| random_forest | 33.4 | 1.04 |

## Ranking-error diagnostics

`OOF_ERROR_ANALYSIS.csv` preserves applicant IDs, targets, folds, selected raw financial/external-score fields, product features, and OOF probability/rank fields. Rank percentiles and deciles are calculated within each fold to avoid mixing fold-specific score scales. `RANKING_FAILURE_ANALYSIS.csv` reports fold-specific default rates and prediction distributions by rank decile. The 0.5 cutoff is reported only as a separate probability diagnostic; it is not the primary ranking-error criterion.

The top-1%-rank default rate for the selected benchmark is 0.4525; this is descriptive on reused folds and does not establish operational calibration or a deployable threshold.

## Applicant segmentation and feature-family probes

- **Segmentation:** best segment feature test: `segment_k4` AUC 0.762432, delta -0.000059, 2/5 folds improved. Mean agreement between alternate K-means initializations: k=4: mean ARI 0.997, k=6: mean ARI 0.997, k=8: mean ARI 0.945. Cluster profiles and training-only cluster default rates are descriptive; validation labels were used only to report segment-specific AUC.
- **Nonlinear bins:** `bin:AMT_ANNUITY` AUC 0.762665, delta +0.000174, 3/5 folds improved.
- **Interactions:** `INTER_EXT_MEAN_X_ANNUITY_TO_INCOME` AUC 0.762462, delta -0.000029, 3/5 folds improved.
- **Missingness summaries:** `family:all_missingness` AUC 0.762522, delta +0.000031, 2/5 folds improved.
- **Combined screened features:** `family:nonlinear_plus_interactions` AUC 0.762059, delta -0.000432, 1/5 folds improved.
- Fold-specific bin cutpoints and applicant segment assignments were fit on each outer training partition. Validation-label default rates are reporting diagnostics only, not a feature-selection input outside the exploratory comparisons.

## Fixed-weight ensemble discovery

`blend_lgb_cat_50_50` AUC 0.763957; 5/5 folds exceeded the LightGBM product model (best single model AUC 0.762491).

The best fixed blend has pooled AUC **0.763957**, leaving **0.026043** to 0.790. Versus the previous Phase 3 discovery it differs by **+0.001131**. The blends use equal weights and no outer-fold label fitting; however, selecting the best among several blends after observing these reused-fold scores is adaptive discovery, not unbiased confirmation. Fold-level blend results, fixed weights, and prediction correlations are reported separately. No learned ensemble weights are claimed.

## Decision and limitations

Do not replace the champion or claim the 0.79 target. Do not treat this benchmark winner as independently validated. Before model promotion, evaluate the frozen candidate on genuinely unseen labeled applicants. The fixed folds have been repeatedly studied; another random split of these same labels cannot undo researcher exposure.

The competition test set was not used for training or AUC. No test labels, external data, pooled train/test statistics, target encoding, or submission generation were used. Ensemble blends used fixed equal weights; no weights were fitted on outer-fold labels.

## Artifacts

See `MODEL_BENCHMARK_RESULTS.csv`, `MODEL_FOLD_RESULTS.csv`, `MODEL_RUNTIME_RESULTS.csv`, `MODEL_MEMORY_PROFILE.csv`, `MODEL_BENCHMARK_OOF_PREDICTIONS.csv`, `OOF_ERROR_SUMMARY.csv`, segment/nonlinear/missingness reports, and the ensemble tables. `OUTPUT_MANIFEST.csv` contains SHA-256 checksums for the present files in this milestone.
