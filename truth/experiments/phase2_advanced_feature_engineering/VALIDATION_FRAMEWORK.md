# Validation framework

The raw baseline feature builder and LightGBM configuration are imported from the preserved EXP85 source snapshot. The comparison uses the same fixed outer folds for both variants. Each fold selects a boosting iteration count on a stratified inner split of the outer training rows, refits on all outer training rows, and predicts the untouched outer validation rows.

- Execution timestamp UTC: 2026-11-03T08:38:11Z
- Train shape: (171202, 34)
- Test shape: (61500, 33)
- Outer folds: 5, shuffle=True, seed=42
- Inner split: stratified 90/10, seed 20261102 + fold number
- Preprocessing: existing raw-row builder and existing LightGBM frame helper; category levels fitted on the training partition only
- Missing numeric values: retained for LightGBM's native missing-value handling
- Validation rows are excluded from early stopping and all feature selection.
- Test data is schema-checked but not used to fit or select this experiment.

## Results

                        variant  mean_fold_auc  std_fold_auc  pooled_oof_auc  delta_pooled_vs_baseline
                   raw_baseline       0.760729      0.003770        0.760658                  0.000000
baseline_plus_existing_formulas       0.760452      0.003521        0.760372                 -0.000287

## Interpretation

The prior run that stopped on outer folds is retained only as historical output. Its metrics are not considered valid OOF evidence. The previously named top-three set is not repeated because it was selected using full-training labels. This run tests the full predeclared formula list only.