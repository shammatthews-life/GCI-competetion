# Phase 14 label-free model-development report

## Status and scope

- Run: `phase14_20261105T114015Z`
- Start UTC: `2026-11-05T11:40:15.600247+00:00`
- End UTC: `2026-11-05T12:16:17.167118+00:00`
- Rows: 171,202; positives: 13,821; primary folds: 5; seed: 20261113.
- Training SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Phase 13 reference OOF SHA-256: `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`.
- Runtime: 36.0 minutes; peak process working set: 1000.63 MiB.
- Models: LightGBM 4.7.0, CatBoost 1.2.10; Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1.

This is nested out-of-fold model development on previously exposed training labels. It is **not** an untouched confirmation: prior project work evaluated these labels across all applicants, so another random split cannot restore label-level independence. The results are exploratory model-development evidence only. No test rows, external data, submissions, or champion files were used or opened.

The Phase 13 feature definitions and primary folds were frozen. Hidden-screen target-rate candidates were excluded. All eight keys were retained in the research track; an additional constant-free track used only Q3/Q4/Q5/AG3. P4/P5/KQ3/KQ4 remain unresolved for competition eligibility because their historical constants lack verified provenance. The local materials do not certify any model or specific engineered formula.

## Preregistered candidate configurations

- Phase 13 E baseline: full raw, rowwise and all-eight-key feature representation; fold-fitted frequency-supported one-hot.
- Numeric enhancement block: log1p amounts/age/employment, external-score summaries/pair terms, financial per-family-member ratios, selected interactions, and missingness/support summaries. All are deterministic rowwise transforms from supplied columns; no labels or test statistics define features.
- Constant-free track: same raw and numeric-enhancement blocks, with only Q3/Q4/Q5/AG3 keys.
- Model families: CPU LightGBM with two fixed configurations and CPU CatBoost with one fixed configuration. LightGBM configuration selection and CatBoost early stopping use three inner folds within each outer-fit partition; the outer models are refit on all outer-fit rows using the median selected iteration.
- CatBoost uses raw categorical columns and product-key strings with `one_hot_max_size=10000`. The runner asserts every fit-partition category count is below that threshold, so the category features use one-hot handling rather than target-derived CTRs. Fit-fold key-frequency counts are added as numeric columns.
- Fixed blend: equal-weight average of the all-eight-key augmented LightGBM and CatBoost OOF predictions. Blend weights were not fit on labels.

## Results

| Variant | Pooled OOF AUC | Delta vs Phase 13 E | Positive folds vs E |
|---|---:|---:|---:|
| Phase 13 E reference | 0.766038 | +0.000000 | — |
| Phase 14 augmented LightGBM, all eight keys | 0.765719 | -0.000318 | 2/5 |
| Phase 13 D reference | 0.765270 | -0.000767 | — |
| Fixed 50:50 augmented LightGBM/CatBoost blend | 0.764886 | -0.001152 | 0/5 |
| Augmented LightGBM, constant-free four-key set | 0.764053 | -0.001984 | 0/5 |
| Phase 13 C reference | 0.759619 | -0.006418 | — |
| Augmented CatBoost, one-hot categorical handling | 0.759529 | -0.006508 | 0/5 |

No new candidate exceeded Phase 13 E or the historical Phase 9 `0.768240`. The best new model, augmented LightGBM at `0.765719`, was `-0.000318` versus E (paired bootstrap 95% interval `[-0.001154, +0.000563]`) and `-0.002521` versus the historical Phase 9 value. The equal-weight blend also underperformed E; no blend weights were fitted.

The ledger also records a higher exploratory Phase 2 score (`0.771103`), but that result is not eligible or certified: it used unresolved fee/product constants and supervised encodings. Thus `0.768240` is the historical Phase 9 reference requested for this objective, not the highest raw research score in the entire ledger.

See `MODEL_RESULTS.csv`, `FOLD_LEVEL_METRICS.csv`, and `PAIRED_BOOTSTRAP.csv` for every candidate. Bootstrap intervals condition on fitted OOF predictions and do not account for model-selection or prior label exposure. Subgroup rows are descriptive fold-level diagnostics with age/income cut points fit on each outer training partition; they are not fairness certification or independent subgroup validation.

Fold-paired subgroup deltas are also saved in `SUBGROUP_STABILITY_PAIRED.csv`, with their cross-fold summaries in `SUBGROUP_STABILITY_SUMMARY.csv`. Across 12 valid subgroup levels and five folds, augmented LightGBM was positive in 19/60 subgroup-fold comparisons versus E; its mean of subgroup mean deltas was `-0.000307`, with subgroup means ranging from `-0.001173` to `+0.001398`. No broad, consistent subgroup improvement emerged. The aggregate is descriptive only and does not adjust for subgroup size or multiple comparisons.

## Integrity and outcome limits

Input IDs, targets, row order, OOF predictions, and the Phase 13 fold assignment are programmatically checked before fitting. Every candidate must predict each row once out of fold; the run aborts on incomplete or non-finite coverage. Inner tuning uses only outer-fit labels. No test information or target-derived feature candidates enter the pipeline.

The score target does not override provenance/eligibility. Even if a candidate exceeds `0.768240`, that would not establish a competition-ready model or independent confirmation. A genuinely untouched labeled cohort or organizer-approved external evaluation protocol is unavailable in this workspace; obtaining one is required before claiming confirmation.

## Decision

Do not promote a champion or submit. If a candidate improves the matched Phase 13 E result reproducibly, preserve it as research and resolve the four constant-based key formulas before considering competition eligibility. No target-derived hidden-key feature is recommended for this next step.
