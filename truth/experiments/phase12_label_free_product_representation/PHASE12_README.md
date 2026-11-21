# Phase 12: label-free product representation

This phase tests whether the Phase 11 product-key signal is retained by
representations that do not calculate target statistics. It uses only the
provided training CSV and preserves the exact Phase 11 key formulas, including
their missing-value sentinels and historical rounding.

The implementation is standalone. It does not import Phase 9, Phase 10, or
EXP85 feature builders, does not read test data, and does not create
submissions. `phase12_feature_definitions.py` independently reconstructs the
eight keys. The fixed fee offset (`130.5`) and financing step (`0.0066`) are
preserved for formula reproducibility, but their provenance is not established;
results using them remain eligibility-pending.

The primary experiment uses one fixed stratified five-fold assignment for all
representations. Hyperparameter selection and early stopping are confined to
three inner folds within each outer training partition. Fold-fitted one-hot
encoders ignore unseen apply-fold categories. Frequency support is measured
from each fit partition only. No target encoding, target-derived category
statistics, test fitting, external data, or cross-partition pooled statistics
are used.

Run formula tests:

```powershell
python -m unittest discover -s . -p 'test_phase12*.py' -v
```

Run the full primary experiment from this directory:

```powershell
python run_phase12.py
```

Every invocation creates a new timestamped folder under `runs/`; an existing
run is never overwritten. The full run is CPU LightGBM because the installed
LightGBM build is CPU-only. Results are research evidence, not independent
confirmation, organizer approval, a champion change, or a competition-ready
submission.
