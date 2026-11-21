# Phase 4 Stage 1 inventory and initial GPU detection

## Status

Completed a metadata/source inventory pass and static review of priority notebooks and screenshots; no notebook was executed and no model was trained. The remaining notebook families, full feature-reuse matrix, and data-dictionary/provenance checks are not complete.

## Inventory coverage

Scanned these local roots: C:\Users\SHARON\Downloads, D:\exp colab, D:\gci\truth, D:\gci\rules and info. Excluded generated caches and the current Phase 4 output folder from the recursive input inventory.

- Notebook files discovered: 15.
- Notebook parse failures: 0.
- Screenshot/image files discovered: 14.
- Other source/report/result files inventoried and hashed: 194.
- Notebook inventory includes cell counts, execution counts, saved outputs/errors, imports, paths, seed/fold mentions, model names and AUC text matches.
- Ten formula screenshots were visually transcribed in `SCREENSHOT_FORMULA_TRIAGE.md`; `tesseract` is not installed, so this is not automated OCR. Four other images were classified as unrelated/duplicate. Formula semantics and score claims still require source/data corroboration.
- A related local source candidate, `C:\Users\SHARON\Downloads\solution_v5_gci_research.py`, was discovered (140,761 bytes). It has a 20-entry model registry and extensive CV/blending code. Its authorship/identity as Claude's exact seven-model source is unverified; its OOF vectors and supporting run artifacts were not found in the scanned inventory.

## Initial GPU finding (detection only)

- Device/driver and framework capability metadata are recorded in `GPU_ENVIRONMENT_AUDIT.json`.
- A visible NVIDIA device does not prove a Python estimator can train on it. PyTorch CUDA availability is only one probe and does not determine LightGBM/CatBoost/XGBoost capability.
- No GPU model benchmark was run in this stage. No CPU fallback or full training was started.
- Complete source review and a small representative capability fit are still gates before the requested full model reproduction.

## Next actions

1. Inspect remaining v18/v19 and research notebooks/source cell-by-cell; complete the feature/source reuse matrix.
2. Triage hashes and provenance for caches and prediction files before trusting historical metrics.
3. Perform model-specific GPU capability tests on a small, bounded representative dataset and record real device use.
4. Verify formula semantics against the official dictionary and establish permitted fold/data scope before any Phase 4 feature/model experiment.
