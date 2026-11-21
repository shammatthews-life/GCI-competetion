# Phase 14 preliminary run — incomplete, excluded from results

- Status: stopped by the operator before any complete OOF candidate was produced.
- The only completed outer-fold fits logged were fold 1 for `lgbm_all8_e` and `lgbm_all8_augmented`. Their predictions were not saved or used for any metric or model-selection decision.
- The run entered the two-configuration XGBoost inner-CV workload, whose measured runtime was disproportionate for the full nested comparison. It was stopped so the final run could use a single predeclared XGBoost configuration and omit a redundant LightGBM refit of Phase 13 E.
- This directory contains only `EXECUTION_LOG.txt`; there is no OOF file or score artifact. Do not treat this partial attempt as an experiment result.
- Original runner snapshot: `run_phase14_incomplete_initial.py`, 35,763 bytes, SHA-256 `3378f31597a39487015e77512d8961faced8650295f1f0267d3d3e5fc9510325`.
- Feature-builder SHA-256 at the time of this run: `7b094f3166cebb93af53ebfc7c56d950f059252abe5a9eaca0e3e3bfe730f931`.
- Preserved execution-log SHA-256: `917d00d205e7fd0f665ad5bf1f71f1ebd88282f99d583a04ebe2b7d442881017`.
