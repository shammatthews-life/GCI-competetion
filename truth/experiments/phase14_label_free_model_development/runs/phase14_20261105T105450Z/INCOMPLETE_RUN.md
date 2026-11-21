# Phase 14 XGBoost attempt — incomplete, excluded from results

- Status: stopped by the operator before any complete OOF candidate was produced.
- The only completed outer-fold candidate logged was fold 1 for `lgbm_all8_augmented`. Its prediction was not saved or used for any metric or model-selection decision.
- The single-configuration XGBoost run on the high-dimensional sparse key matrix remained disproportionately slow. It was stopped to replace it with a fixed one-hot CatBoost comparator that can avoid target-derived categorical statistics.
- This directory contains only `EXECUTION_LOG.txt`; there is no OOF file or score artifact. Do not treat this partial attempt as an experiment result.
- Original runner snapshot: `run_phase14_xgb_attempt.py`, 35,848 bytes, SHA-256 `e9c10119b18c8c6be2081a2cc4ba41eb643d8c93188d630aff3b612649ffa8ae`.
- Feature-builder SHA-256 at the time of this run: `7b094f3166cebb93af53ebfc7c56d950f059252abe5a9eaca0e3e3bfe730f931`.
- Preserved execution-log SHA-256: `a433c0090f5fb55785001ba368591222c4f177618f9f2c9ee16fa4a4895657d3`.
