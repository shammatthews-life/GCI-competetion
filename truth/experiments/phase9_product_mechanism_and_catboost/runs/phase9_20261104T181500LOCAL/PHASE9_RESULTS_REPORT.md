# Phase 9 product-mechanism and model-family report

## Completion

Train-only work completed with CPU LightGBM (the installed build has no GPU tree learner) and GPU CatBoost, across three five-fold seeds, with no test-file access. Full results are in `AGGREGATE_COMPARISON.csv`, `SEED_METRICS.csv`, `LIGHTGBM_FOLD_RESULTS.csv`, and `CATBOOST_FOLD_RESULTS.csv`.

## H1/H2/H3 product-key discrimination

LightGBM candidates compare the exact raw+Phase-3 product baseline against nested target means for Q3/Q4/Q5; cash-only Q4; exact (credit,annuity), (goods,credit), and (goods,credit,annuity) keys; K-only; Q4+K; and an unsupervised ordinal Q4 feature. All map features are nested inside each outer training fold; alpha=30.0; no outer validation TARGET is used for maps or stopping.

Interpret results as prediction evidence only. Q4/Q5 lifting more than Q3 would be consistent with greater product-key resolution, but will not prove discrete contract products or rule out every histogram effect. Ratio keys outperforming exact amount keys would support portable ratio identity rather than row/entity duplication or loan-size buckets. K-only and Q4+K test whether financing multiplier adds distinct signal. CatBoost native categorical results test whether native ordered category statistics recover similar information without hand-built target maps.

## Independently verified results

The independent audit recalculated all 45 pooled seed/model AUCs and 225 fold AUCs from the saved ID-aligned OOF predictions; every result matched within `1e-10`. All 135 nested-encoding integrity checks passed.

| Model / candidate | Mean pooled OOF AUC | Delta vs same-family baseline | Positive folds |
|---|---:|---:|---:|
| LightGBM product baseline | 0.762503 | — | — |
| LightGBM Q3 target mean | 0.764703 | +0.002200 | 15/15 |
| LightGBM Q4 target mean | 0.765577 | +0.003074 | 15/15 |
| LightGBM Q5 target mean | 0.764648 | +0.002145 | 15/15 |
| LightGBM cash-only Q4 | 0.765701 | +0.003199 | 15/15 |
| LightGBM exact (credit, annuity) | 0.762906 | +0.000403 | 9/15 |
| LightGBM exact (goods, credit) | 0.762086 | -0.000416 | 4/15 |
| LightGBM exact (goods, credit, annuity) | 0.762916 | +0.000414 | 11/15 |
| LightGBM K-only | 0.762194 | -0.000308 | 6/15 |
| LightGBM Q4×K | 0.763152 | +0.000649 | 11/15 |
| LightGBM unsupervised ordinal Q4 | 0.762499 | -0.000003 | 7/15 |
| CatBoost GPU product baseline | 0.762027 | — | — |
| CatBoost GPU native Q4 | 0.766511 | +0.004483 | 15/15 |
| CatBoost GPU native eight keys | **0.768240** | **+0.006213** | 15/15 |
| CatBoost GPU eight keys + category crosses | 0.768032 | +0.006005 | 15/15 |

**Interpretation:** Q4 resolution was the best single LightGBM ratio key; moving from 4 to 5 decimals reduced the average score, so “finer is always better” was not supported. Exact amount keys and K alone were weak, and the unsupervised ordinal Q4 feature was neutral. Native CatBoost categories recovered a larger within-family gain than the LightGBM target-mean variants. Adding the two proposed demographic/work category crosses did not improve on CatBoost's eight product keys. These comparisons do not prove the products are contractual identities or that pricing causes default.

### Subgroup and mechanism checks

Using Phase 8's already cross-fit OOF predictions (not a fresh or independent cohort), the mean three-seed gain was +0.0068 AUC for pensioners and +0.0064 for the oldest age quartile, versus +0.0052 for the youngest. Cash loans gained +0.0059; revolving loans were essentially flat (-0.0001). This broadly supports concentration of the predictive lift in cash, older, and pensioner groups, but it does not reproduce Claude's specific subgroup magnitudes and is post-hoc.

Across seeds, 128 Q4 keys had at least 300 rows. Their baseline Pearson-residual z-score correlations were consistently negative with mean age (-0.579 to -0.584) and pensioner share (-0.525 to -0.531). Only 3.1–5.5% of keys exceeded |z|>3. The observations are descriptive; applicants within a product key are clustered and repeated CV seeds reuse labels, so these are not independent significance tests.

Within exact cash-loan (goods price, credit) groups, 6,792 adjacent annuity gaps were 130.5 among 27,255 total gaps (24.9%); the listed placebo gaps were rare or absent. This supports a repeated numerical gap, **not** a fee assignment, rate, term, or loan-product explanation. The reported gap count is reproduced here under the stated exact-pair definition, while the denominator differs from the original message's narrower gap count.

## CatBoost

CatBoost was run on the NVIDIA GPU with inner-only iteration selection. It compares raw+product baseline, native Q4, native eight product keys, and eight keys plus organization×occupation and income-type×education crosses. No CatBoost metric is directly comparable to the prior HistGradientBoosting claim; use within-family paired deltas.

## Mechanism diagnostics

- `SUBGROUP_OOF_DIAGNOSTICS.csv`: Phase 8 held-out predictions compared within cash/revolving, pensioner, age quartiles, and external-score availability groups. This checks the claimed subgroup pattern; these are post-hoc diagnostics.
- `PRODUCT_RESIDUAL_HETEROGENEITY.csv` and per-seed Q4 tables: groups with at least 300 rows and OOF residual heterogeneity. These do not support causal pricing claims and have no naive p-values.
- `FEE_GAP_PLACEBO_AUDIT.csv`: exact (goods,credit)-cell adjacent annuity gaps compared with prespecified nearby alternatives. This audits the narrow 130.5 gap claim, not fee multiplicity.

## Rules and confirmation

The official local materials leave target encoding unresolved; nested validation is necessary but does not establish competition permission. All results reuse the same historically exposed training labels. No independent confirmation, 0.77 guarantee, model promotion, or submission is claimed.
