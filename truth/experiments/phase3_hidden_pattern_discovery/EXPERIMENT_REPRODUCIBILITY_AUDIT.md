# Experiment reproducibility audit

- Exact train/test input paths are pinned and SHA-256 hashes are in `INPUT_DATA_HASHES.txt`.
- Train schema is 34 columns including TARGET; test schema is 33 predictors; predictor columns and IDs are validated.
- All saved historical OOF predictions were aligned to train rows by SK_ID_CURR; target arrays and five fixed fold assignments were verified.
- Historical pooled AUCs and five fold AUCs were recalculated from saved predictions. Historical model training itself was not rerun in this verification step.
- The complete v17 reuse register has 23 physical data rows and was read directly. Its CSV quoting is malformed, so generic CSV parsing fails; it was not used as a model input.
- Feature transforms use supplied row-level values plus fold-fit fee, lattice, and frequency parameters; no feature parameter was fit on outer validation rows or test rows.
- Inner split early stopping does not use outer validation labels. Model uses the preserved raw baseline LightGBM configuration and helper.
- Random seeds: outer folds inherited from saved seed-42 assignments; inner split seed is 20261102 + fold; LightGBM seed is baseline seed + fold.
- Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, LightGBM 4.7.0; exact versions in environment.json.
- Reproducing discovery metrics still requires running this script against the hashed files. CPU timing and LightGBM nondeterminism may cause tiny numerical differences.
- The fixed 5 folds are reused for comparability and are not an untouched confirmation set.
