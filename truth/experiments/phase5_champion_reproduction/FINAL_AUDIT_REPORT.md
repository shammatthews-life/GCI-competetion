# Phase 5 status: source recovery and champion-comparator gate

## Result

Static inspection is complete for the executed V16 notebook, its source-only counterpart, the EXP85 runner, the frozen champion submission, the local data, and the available component caches. The pre-run gate **blocked model training** because the exact final EXP85 OOF vector and its fold assignments are missing. This is not a partial AUC result: no candidate model ran, and no AUC improvement is claimed.

## Verified evidence

- Raw inputs: `train.csv` 171,202 x 34 and `test.csv` 61,500 x 33; hashes match the recorded local inputs.
- V16 executed and source-only notebook copies have identical canonical code-cell source hash `0d6a0cc7be62873b66a759ffcd05d3d64e0b8e6b4dc8550990dbd829f9d09ef1`; only execution state/output differs. The executed file has 11/11 code cells with execution counts.
- EXP85 frozen submission and historical reference are byte-identical: SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- V17/V55 upstream model bundles and multiple OOF component arrays exist, but the final test synthesis requires a test-only Anchor74 plus an eight-engine suite. Component OOF arrays do not constitute an exact final champion OOF.
- The V16 `0.77649` and other values, if displayed in saved outputs, are recorded in `V16_SAVED_OUTPUT_EVIDENCE.csv` as claims, not reproduced results.

## Why no AUC comparison was run

The frozen EXP85 test submission is available and byte-identical to the historical reference, but the exact final 85/15 Anchor74+eight-engine OOF vector and its fold map are not available. The source's Anchor74 contribution is a test-only transductive prediction; available V17/V55/XGB OOF arrays are upstream components and do not by themselves reproduce the final champion's OOF. Using Phase 2/3C or component OOF as a substitute would violate the requested comparator.

The official competition log's verified `0.80041` submission hash differs from the frozen EXP85 file hash. The runner's `0.800850` comment and the user-provided V16 `0.78222` screenshot/name are not independently validated score-to-artifact links.

## Next required inputs

1. Exact final EXP85 champion OOF predictions from the same 85/15 synthesis, with the OOF fold assignment and member-level lineage.
2. The exact Anchor74 OOF representation corresponding to the frozen test anchor, including how its test-to-test coupling is represented out of fold.
3. V16 cache files (`resid16.pkl`, view caches, model-member NPZs, CatBoost fold NPZs) with a provenance manifest binding them to raw data hashes, source, feature schema, and folds.
4. Written organizer clarification for pooled test-covariate fitting and test-to-test score coupling before any competition submission using those methods.

## Deliverables

See `SOURCE_CODE_AUDIT.md`, `CHAMPION_SOURCE_AUDIT.md`, `V16_VALIDATION_AUDIT.md`, `V16_FEATURE_EXTRACTION.csv`, `V16_MODEL_CONFIGURATION.csv`, `FEATURE_ALIAS_AUDIT.csv`, `MISSING_ARTIFACTS.csv`, `FOLD_LEVEL_CV_RESULTS.csv`, and `BASELINE_VS_FEATURES_RESULTS.csv`. Blocked metric fields are intentionally empty rather than fabricated.
