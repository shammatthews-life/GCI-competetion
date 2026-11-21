# EXP85 champion OOF recovery update

## Status

**EXP85 champion OOF recovery: NOT RECOVERED.** The exact final Anchor74 + eight-engine champion OOF vector, row-to-fold assignments, and matching Anchor74 OOF are still unavailable.

## Evidence inspected

- Frozen submission `D:\gci\champion\submission_exp85_solo_champion.csv` and historical reference `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv` are test-only artifacts, not OOF.
- `D:\gci\gemini\dependent\exp85_notebook_run\cache\rebuilt_eight_engine_oof.npy` is a finite `(171202, 8)` upstream component matrix (SHA-256 `f2fa3b20528ace4596fa412cc2d571ca94f85f6bfdaf91b0006433b71d4fc134`); it is not the final macro-blended prediction and has no row IDs or fold sidecar.
- `D:\gci\champion\01_EXP85_RECOVERY\source\reproduce_exact_synthesis.py` applies the final 85:15 Anchor74/suite synthesis to the test Anchor74 and component OOFs. It does not create an equivalent Anchor74 OOF; substituting the test Anchor74 for training OOF is invalid.
- The EXP02 runner is explicitly a raw-data surrogate and disclaims historical Anchor74 reproduction. It groups by applicant entities and uses `StratifiedGroupKFold`; it does not train a new model in this audit.

## Can the surrogate produce valid fold-safe OOF?

The raw five-family surrogate path assigns every train row to an outer validation fold, uses fixed iterations without early stopping, and recomputes graph label calibration inside each fold from outer-fit labels only. The source asserts fold coverage, fit/validation disjointness, group isolation, and finite predictions. These properties make the **raw base-family surrogate OOF** structurally fold-safe with respect to its outer labels, subject to static code review and reproducible execution.

However, its feature builder computes label-free ranks, group/frequency statistics, ridge expectation features, and entity structure over concatenated train+test covariates. The local materials do not resolve permission for pooled/transductive fitting. Its 85:15 blend also combines the grouped surrogate OOF with historical suite OOF caches whose exact fold sidecars are unavailable; that hybrid blend is **not demonstrated fold-safe** and is not an EXP85 reproduction. The cached suite components can include validation-training overlap relative to the different group folds.

The two-stream meta OOF code cross-fits logistic regression, but that cannot repair unknown fold provenance or possible upstream overlap in its cached suite input.
