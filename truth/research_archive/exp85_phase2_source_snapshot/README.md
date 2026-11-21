# EXP85 Phase 2 source snapshot

This directory contains byte-for-byte copies of the three available source
scripts used as the starting point for the TRUTH early-stopping experiment:

- `project/source/run_independent_baseline.py`
- `project/experiments/phase_01_fee_product_features/run_experiment.py`
- `project/experiments/phase_02_fold_safe_key_encodings/run_experiment.py`

The original `EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb` named in the handoff was
not found in the audited workspace roots. These files are the available
frozen-release source, not a reconstruction of that missing notebook.

The copies are preserved unchanged. The separate experiment implementation
under `experiments/exp85_phase2_inner_early_stopping/` trains only from the
provided raw train/test/sample files and does not consume archived predictions
or model caches.
