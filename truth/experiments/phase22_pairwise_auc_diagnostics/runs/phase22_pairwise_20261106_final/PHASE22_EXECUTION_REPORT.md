# Phase 22 — pairwise AUC error decomposition

## Scope

This is a post-hoc diagnostic of saved out-of-fold predictions. It trains no models and reads only `train.csv`, Phase 15 OOF predictions, and Phase 18 OOF predictions. It does not read test data. These labels have been examined in earlier phases; findings are exploratory, not independent confirmation.

## Rank-pair accounting

- Rows: 171,202; positives: 13,821 (8.073%); negatives: 157,381.
- Positive-negative pairs: 2,175,162,801.
- Phase 15 baseline OOF AUC: 0.767510928.
- Baseline incorrectly ranked pairs: 505,701,581.5 (23.249%).

| Candidate | OOF AUC | Delta AUC | Net pair change | Net gain from bottom-half positives |
|---|---:|---:|---:|---:|
| phase18_catboost_component | 0.767510928 | +0.000000000 | +0.0 | +0.0 |
| phase17_nested_te_lightgbm | 0.765290753 | -0.002220175 | -4,829,241.5 | +6,750,027.5 |
| phase18_nested_blend | 0.768668866 | +0.001157938 | +2,518,703.5 | +1,742,886.5 |

The pair matrix groups both positives and negatives into deciles of the Phase 15 score. Each cell reports all positive-negative comparisons in that pair of score regions, so its contribution is weighted by the number of pairs, not by the number of rows.

### Where baseline ranking errors occur

| Positive rows' baseline score region | Positive rows | Baseline wrong pairs | Share of all baseline wrong pairs |
|---|---:|---:|---:|
| bottom half | 2,256 | 234,901,340.5 | 46.45% |
| top half | 11,565 | 270,800,241.0 | 53.55% |

### Candidate error corrections by positive-risk decile

Positive-risk deciles are defined by the baseline score. `Net pair change` is candidate concordant pairs minus baseline concordant pairs over every negative; it is a net count (pairs corrected minus pairs newly broken), not the gross number corrected.

| Candidate | Pos. risk decile | Positives | Baseline pair AUC | Candidate pair AUC | Net pair change |
|---|---:|---:|---:|---:|---:|
| phase18_catboost_component | 1 | 173 | 0.066654 | 0.066654 | 0.0 |
| phase18_catboost_component | 2 | 313 | 0.164718 | 0.164718 | 0.0 |
| phase18_catboost_component | 3 | 430 | 0.272810 | 0.272810 | 0.0 |
| phase18_catboost_component | 4 | 584 | 0.375928 | 0.375928 | 0.0 |
| phase18_catboost_component | 5 | 756 | 0.480817 | 0.480817 | 0.0 |
| phase18_catboost_component | 6 | 1,044 | 0.583216 | 0.583216 | 0.0 |
| phase18_catboost_component | 7 | 1,376 | 0.684205 | 0.684205 | 0.0 |
| phase18_catboost_component | 8 | 1,785 | 0.783273 | 0.783273 | 0.0 |
| phase18_catboost_component | 9 | 2,675 | 0.878958 | 0.878958 | 0.0 |
| phase18_catboost_component | 10 | 4,685 | 0.970040 | 0.970040 | 0.0 |
| phase17_nested_te_lightgbm | 1 | 173 | 0.066654 | 0.098984 | 880,247.0 |
| phase17_nested_te_lightgbm | 2 | 313 | 0.164718 | 0.192089 | 1,348,313.5 |
| phase17_nested_te_lightgbm | 3 | 430 | 0.272810 | 0.299445 | 1,802,503.5 |
| phase17_nested_te_lightgbm | 4 | 584 | 0.375928 | 0.393027 | 1,571,637.0 |
| phase17_nested_te_lightgbm | 5 | 756 | 0.480817 | 0.490460 | 1,147,326.5 |
| phase17_nested_te_lightgbm | 6 | 1,044 | 0.583216 | 0.582397 | -134,554.5 |
| phase17_nested_te_lightgbm | 7 | 1,376 | 0.684205 | 0.681968 | -484,429.5 |
| phase17_nested_te_lightgbm | 8 | 1,785 | 0.783273 | 0.775590 | -2,158,296.5 |
| phase17_nested_te_lightgbm | 9 | 2,675 | 0.878958 | 0.868348 | -4,466,882.5 |
| phase17_nested_te_lightgbm | 10 | 4,685 | 0.970040 | 0.964160 | -4,335,106.0 |
| phase18_nested_blend | 1 | 173 | 0.066654 | 0.072153 | 149,715.0 |
| phase18_nested_blend | 2 | 313 | 0.164718 | 0.169774 | 249,037.5 |
| phase18_nested_blend | 3 | 430 | 0.272810 | 0.279760 | 470,336.5 |
| phase18_nested_blend | 4 | 584 | 0.375928 | 0.381449 | 507,441.0 |
| phase18_nested_blend | 5 | 756 | 0.480817 | 0.483896 | 366,356.5 |
| phase18_nested_blend | 6 | 1,044 | 0.583216 | 0.584803 | 260,771.5 |
| phase18_nested_blend | 7 | 1,376 | 0.684205 | 0.686996 | 604,564.5 |
| phase18_nested_blend | 8 | 1,785 | 0.783273 | 0.784201 | 260,876.5 |
| phase18_nested_blend | 9 | 2,675 | 0.878958 | 0.878585 | -157,077.5 |
| phase18_nested_blend | 10 | 4,685 | 0.970040 | 0.969777 | -193,318.0 |

## Interpretation boundary

A large share of errors in the bottom-half positive-score region would support investigating low-scored positives, but would not prove that adding more features there improves total AUC. The full 10x10 matrix should guide any later, predeclared candidate design. No model selection or blend tuning is performed here.

Per-model calibration does not change AUC if it is a single strictly increasing transform applied to every row. It can change a blend's cross-model ordering, which is why the Phase 18 mixture is compared directly by pairwise outcomes.
