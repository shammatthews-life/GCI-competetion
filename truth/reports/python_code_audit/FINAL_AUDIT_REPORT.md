# TRUTH — Available Python code audit

## Executive decision

**Overall competition-rule status: UNVERIFIED, not a finding that the project is ineligible.** The only available rule text is a local secondary transcription in the old champion archive. Its cited primary competition brief/portal was not present in the allowed read-only source roots, and the official portal could not be independently verified here. Transductive methods, target encoding, interactions, and test-covariate statistics are therefore not declared prohibited; where the text is silent, they are marked ambiguous or unverified.

Recommended candidate for a new solution: start from the raw-only Phase 2 baseline and its nested fold-safe target-encoding helper, then build only from verified supplied train/test files. Do not reuse the historical submission synthesis as training evidence. Keep transductive test statistics and identity-based train-label transfer disabled until organizers clarify them. This is a code reuse recommendation, not a score claim or guarantee of reaching public AUC 0.79000.

## Scope and isolation

- Git root: `D:\gci\truth`; independent empty Git repository; no remote configured.
- Read-only source roots: `D:\gci\champion`, `D:\gci\gemini`, `D:\gci\codex`, `D:\gci\past_85`, `D:\gci\5 independent`, `D:\gci\name5_dependent_code`, `D:\gci\pre info`.
- Available files inspected: **637** (585 `.py`, 52 notebooks).
- Exact source hashes unique: **560**; repeated copies: **77** additional paths.
- No source files or data were copied from or modified in the old repositories.
- The two exact previously identified missing-file names could not be established from permitted documentation; no guessed exclusions were made.

## Source audit results

See `PYTHON_FILE_INVENTORY.csv` for every path, byte size, hash, syntax, imports, definitions, artifact strings, validation/feature signals and notebook metadata. `PYTHON_DEPENDENCY_MAP.md` traces per-file imports and input-artifact references.

Syntax compilation passed for **633** sources and failed for **4**. All 2 notebook parse failures and 2 `.py` failures are listed in the execution report. Notebook cells were compiled independently after removing IPython magic lines. The `.py` files are incomplete extraction/dump artifacts. Syntax passing does not prove runtime imports, data paths, results, or eligibility.

**Test discovery:** 49 test-like Python files found. They were not executed because legacy imports may trigger side effects and many tests require data/artifacts; safe static parse/compilation was used instead.

## Code lineage and practical decisions

| Implementation family | Run with available files? | Feature reproduction | Validation trust | Suggested decision |
|---|---|---|---|---|
| Frozen Phase 2 raw-data baseline | Likely, if local raw data and Python dependencies match; not executed here. | Rowwise features are directly implemented; a historical reference is optionally read only for post-hoc similarity, not as a training input. | Stratified 5-fold OOF and fold-specific fit/validation are visible. Early stopping is on the current validation fold; do not use same folds for extensive experiment selection. | REUSABLE_AFTER_MODIFICATION |
| Phase 1/2 fee features and nested target encoding | Requires raw data and a referenced Phase 1 run/baseline outputs for full experiment. | Raw formula source present; target-key creation uses train/test covariates; encodings use training labels in disjoint inner/outer folds. | Strong helper-level evidence and focused label-invariance test; full run/outputs unverified. | REUSABLE_AFTER_MODIFICATION |
| Hasan historical feature laboratory / screening | Some scripts run only with expected raw data/dependencies. | Formula code present; train+test rank/statistics are transductive; screening methodology must be fold-local. | Individual features can be reproduced; historical metrics are not reproduced by this audit. | REUSABLE_AFTER_MODIFICATION |
| V15 monolithic research solution | Source is available; complete execution is configuration/artifact/dependency dependent. | Large in-script feature laboratory and fold-safe TE branches. | Many modes, tuning, validation-based decisions and optional pseudo-labels; exact promoted branch and outer-fold independence need audit. | PARTIALLY_REPRODUCIBLE |
| `5 independent` legacy scripts | Source present, but direct inspection shows Feather/NPZ/PKL/anchor inputs; they are not raw-data-only standalone training routes. | Depends on precomputed feature/model arrays; historical output cannot be reproduced from these entry points alone. | CV code exists in some scripts; validity is not enough to repair missing upstream provenance. | PARTIALLY_REPRODUCIBLE |
| `name5_dependent_code` replacement scripts | Raw train/test versions exist; not executed. | New approximations rather than historical parity; some concatenate train/test covariates or use identity-key test adjustments. | Stratified folds are present in inspected runners; blend and early-stopping choices require outer validation. | REUSABLE_AFTER_MODIFICATION |
| Historical EXP85 submission reconstruction/synthesis | No, not as a self-contained raw-data pipeline; several historical arrays/caches are missing. | Synthesis formula can replay available artifacts; missing neural streams and upstream predictions block exact reproduction. | Similarity to champion is not independent training or OOF evidence. | BLOCKED |
| Incomplete extracted `.py` / notebook snapshots | No as currently parsed. | Unknown/incomplete source text. | Not assessable. | BLOCKED |

The exact per-file reusable/blocked evidence is in `REUSABLE_COMPONENTS.csv` and `BLOCKED_COMPONENTS.csv`.

## Rule compliance matrix

The matrix distinguishes policy claims from implementation evidence. The archived `OFFICIAL_RULES_SNAPSHOT.md` states only supplied train/test data, prohibits manual hand-labeling, requires a two-column 61,500-row submission and says honors submissions must be reproducible. It cites `Competition/README.ipynb` and the Omnicampus portal but provides no primary-source URL, rule numbers, capture date, or organizer quotation record. These exact official statements are therefore **not independently verified** in this audit.

The archived `DATA_POLICY_AUDIT.md` also records raw train/test SHA-256 values that do not match the current files at `D:\gci\inpuy`: its train/test hashes are `129858df3c08354c25f448b488734e7bbbe6256ce850e0388172c72b22b2b1ff` / `9aaee24c4dc2794d4d6ee179f8263595ddb3e3437ce1cb793b827e70ea5fbca9`; current hashes are `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1` / `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. That historic audit cannot certify the current dataset bytes without an explained version transition.

| Technique | Classification |
|---|---|
| External datasets/network data | UNVERIFIED; the secondary snapshot says prohibited. |
| Manual hand-labeling | UNVERIFIED absence; the secondary snapshot says prohibited. |
| Test-label access | UNVERIFIED; no source-only proof of all cached artifacts. |
| Transductive/test-covariate statistics | AMBIGUOUS — ORGANIZER CLARIFICATION REQUIRED. |
| Target encoding | UNVERIFIED in organizer rules; fold safety is separate and code-specific. |
| Feature interactions/derived features | UNVERIFIED; no restriction was established. |
| CV/OOF and validation selection | UNVERIFIED implementation-by-implementation. |
| Pretrained models/licensing | UNVERIFIED; verify model/dependency lineage in chosen pipeline. |
| Ensemble/calibration/postprocessing | UNVERIFIED; no rule text found; OOF selection protocol required. |
| Submission format/reproducibility | UNVERIFIED primary authority; secondary snapshot describes requirements. |

No technique is declared universally non-compliant solely because it uses unlabeled test covariates, feature interactions, or target encoding. No project-wide ineligibility finding is made.

## Can available code contribute to an AUC 0.79000+ solution?

Potentially as reusable starting components, but this audit cannot establish performance. The strongest clean-room direction is a minimal raw-data baseline, one fold-safe feature addition at a time, a frozen external OOF protocol, and explicit source/data hashes. Do not target 0.79000 as a promised outcome; score must be demonstrated by valid evaluation and the official leaderboard.

## Required next actions

1. Obtain the actual GCI World 2026 rule text from an organizer-controlled source, including data, transductive learning, pseudo-labeling, identity linkage, pretrained models and reproducibility clauses.
2. Provide the exact two missing filenames previously identified so they can be excluded explicitly; until then all available files remain inventoried.
3. Select one raw-only implementation and run it only inside TRUTH after verifying raw data hashes, requirements, local imports and all inputs. Preserve the old champion and research roots read-only.
4. Use held-out outer folds for final model/blend selection; do not re-optimize weights on validation labels used to claim the final result.

## Reports

- `PYTHON_FILE_INVENTORY.csv` — all source paths and static metadata.
- `PYTHON_DEPENDENCY_MAP.md` — imports, paths, and artifact relationships.
- `FEATURE_ENGINEERING_AUDIT.csv` — per-file feature-family evidence.
- `LEAKAGE_AND_VALIDATION_AUDIT.md` — fold, encoding, transductive, and leakage review.
- `COMPETITION_COMPLIANCE_MATRIX.csv` — policy classification with authority limitations.
- `EXECUTION_AND_REPRODUCIBILITY_AUDIT.md` — safe checks and blockers.
- `REUSABLE_COMPONENTS.csv` / `BLOCKED_COMPONENTS.csv` — practical source decisions.
- `SHA256_MANIFEST.json` — hashes of inspected sources and generated reports.
