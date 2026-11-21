# Exploratory hidden-key screen

- Started/completed UTC: 2026-11-05T09:13:14.736994+00:00 / 2026-11-05T09:25:32.228107+00:00
- Train SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; Phase 13 OOF SHA-256: `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`.
- Rows: 171,202; fixed folds: 5; candidate count: 1,968.
- Baseline: Phase 13 complete raw+rowwise C; pooled OOF AUC `0.759619`.
- Cross-fitted category-rate smoothing: 30; blend weights screened: 0.05, 0.10, 0.20.
- Candidate generation uses the prior Phase 11 search helper over numeric singles, pair sums/differences and both ratio orientations at 3/4/5 significant digits, smooth-bin controls, low-cardinality crosses, and selected numeric-bin/category crosses. Eight product keys were added as positive controls.
- For each candidate, fold-held-out category-rate estimates were computed from labels in the other four folds only. This is target-dependent exploratory research; its rule eligibility is unresolved and it is not part of the Phase 13 label-free benchmark.
- Quantile-based smooth/numeric-bin candidate boundaries are descriptive screen definitions estimated from train predictors before cross-fitting, including held-out-fold predictor values. They use no labels or test data, but are not fold-local preprocessing and cannot be promoted without a fold-local rerun.
- The OOF rows have already been examined in prior phases and thousands of candidates were screened. The winning blend is adaptively selected and is not confirmatory evidence.
- No test file, external data, holdout, submission, or champion was used.

## Strongest result

- `product_key:PRODUCT_KEY_P4` (predeclared_product_key): category-rate OOF AUC 0.680344; fixed 10% blend AUC 0.762727 (delta +0.003107); distinct categories 775.

## Highest numeric-pair results

| Candidate | Categories | Target-rate AUC | 10% blend delta vs C |
|---|---:|---:|---:|
| pair:AMT_ANNUITY:ratio:AMT_CREDIT:sig3 | 661 | 0.674582 | +0.002542 |
| pair:AMT_CREDIT:ratio:AMT_ANNUITY:sig4 | 2,754 | 0.670257 | +0.002461 |
| pair:AMT_CREDIT:ratio:AMT_ANNUITY:sig3 | 409 | 0.673865 | +0.002399 |
| pair:AMT_ANNUITY:ratio:AMT_CREDIT:sig4 | 4,402 | 0.663644 | +0.002277 |
| pair:AMT_CREDIT:ratio:AMT_ANNUITY:sig5 | 11,348 | 0.652035 | +0.001949 |
| pair:AMT_ANNUITY:ratio:AMT_CREDIT:sig5 | 15,024 | 0.645304 | +0.001767 |
| pair:AMT_ANNUITY:ratio:AMT_GOODS_PRICE:sig4 | 6,201 | 0.641536 | +0.001621 |
| pair:AMT_GOODS_PRICE:ratio:AMT_ANNUITY:sig4 | 3,997 | 0.644822 | +0.001488 |
| pair:AMT_ANNUITY:ratio:AMT_GOODS_PRICE:sig5 | 19,670 | 0.634276 | +0.001479 |
| pair:AMT_GOODS_PRICE:ratio:AMT_ANNUITY:sig5 | 15,063 | 0.635922 | +0.001433 |

## Interpretation and limitations

This is a machine-scale hypothesis screen, not proof of a latent identifier, contractual product tier, or competition eligibility. High-cardinality candidate labels are derived from supplied training predictors; their target-rate values use only other folds' labels. Adaptive comparison across many candidates inflates the best observed OOF result. Treat this as a prioritization signal only; any candidate would need a preregistered nested evaluation and explicit rules review.

See `HIDDEN_KEY_CANDIDATE_RESULTS.csv` for every candidate and `FREQUENCY_PRESERVING_NULLS.csv` for 40 shuffled controls. The exact Phase 13 key rate vectors are in `PRODUCT_KEY_CROSSFIT_RATE_OOF.csv`.
