# EXP85 Phase 2 baseline recovery and audit

## Status

Recovered as a verified standalone notebook snapshot. This is an engineering baseline, not a compliance clearance or a verified public leaderboard result. No training, submission, commit, or push was performed for this recovery.

## Recovery and provenance

- Original local package: `D:\gci\_gci_push_tmp\champion\EXP85_PHASE2_COMPLETE_SUBMISSION\`
- Git remote: `https://github.com/shammatthews-life/competetion-one.git`
- Canonical source commit: `a77b9ba113f1260250b436d1d756fdb83b1b1f9f`, committed `2026-11-01T22:01:02+05:30`
- Canonical notebook: [EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb](./EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb), 6,706,327 bytes, SHA-256 `ab73f6f39a0fee803fc946de5bd64a4219f9b59a35663c8510b9c002570069f0`. This matches the notebook entry in the original package manifest.
- Recovered to `D:\gci\truth\baselines\exp85_phase2_complete_submission_baseline\`.
- Separate next-experiment working copy: `D:\gci\truth\experiments\exp85_phase2_next_experiment_working_copy\EXP85_PHASE2_NEXT_EXPERIMENT_WORKING_COPY.ipynb`, byte-identical to the canonical baseline and independently hashed. It is an unmodified starter, not a corrected or approved experiment.
- The local source worktree notebook was already modified: 6,703,420 bytes, SHA-256 `4e0024c002036fed829c9e790315df69b8274ec12810cb4ba01f8823b9fca4c0`, last written `2026-11-01T16:50:11.570475Z`. Its embedded source hash fields do not match its embedded source text. The committed snapshot does match its embedded hashes and package-manifest entry; that version was copied. Neither source copy was edited.
- The notebook embeds 24 Python source files, all three JSON configs, and frozen Phase 2 comparison evidence. No sibling `.py` or config files are required to restore the notebook pipeline. The supporting README, model card, implementation reports, requirements, and original package hash manifest are preserved beside it.
- The feature-level source and transformation register is [FEATURE_MANIFEST.csv](./FEATURE_MANIFEST.csv).
- The original package manifest also lists run outputs, model files, predictions, and a submission. Those files were not copied. The frozen OOF/test reference evidence remains embedded in the recovered notebook and is used by its pipeline only for reproduction comparisons, not as model-training features or labels. Its provenance must remain attached to any future use.
- No EXP02 financial-behaviour code or results, friend-code artifacts, external data, model weights, candidate submission, or separate prediction arrays were added to this baseline directory.

## Result evidence

| Evidence | ROC-AUC | Interpretation |
|---|---:|---|
| Frozen seed-42 reference | 0.7715614423 | OOF reference, not public leaderboard |
| Raw-data package run `phase2_complete_raw_20261101T211310` | 0.7715805073 | OOF; delta `+0.0000190650` vs frozen reference |
| Isolated standalone-notebook run in the implementation report | 0.7715614961 | OOF; delta `+0.0000000538` vs frozen reference |
| Previously stated `0.77617` | — | Not evidenced as the public score for this exact notebook/version; the package report explicitly says `NOT EVALUATED` for official competition score |

The reported OOF values are development-validation metrics, not public scores. The package report's reproduction pass establishes similarity to a frozen run, not independent validation quality or a leaderboard score.

## Pipeline summary

- Inputs: supplied `train.csv`, `test.csv`, and `sample_submission.csv`. The recorded raw-input hashes match the files currently at `D:\gci\inpuy`.
- Features: row-wise transformations of provided fields, including credit/annuity/income/family/external-score/bureau-derived features, plus decoded fee/product features. The fee decoder uses constants `129.5` and `0.0132`.
- Fee frequency counts pool unlabeled train and test covariates. No target is used to calculate these counts.
- Validation: five-fold stratified outer CV, seed 42. For target encoding, five inner folds use seed 7; outer-fit rows receive inner-OOF encodings, and outer-validation/test encodings use outer-fit labels only. The encoder checks fit/apply separation.
- Models: LightGBM CPU and CatBoost GPU 0, with model seeds `42 + fold`; both use early stopping. The ensemble is a fixed 50:50 arithmetic mean of raw class probabilities, with no post-processing.
- Reproducibility: the notebook records source/config/input hashes and explicit device requirements. Its GPU path fails closed rather than silently switching CatBoost to CPU.

## Leakage, validation, and rule findings

1. **Early-stopping contamination of the reported OOF estimate.** Both LightGBM and CatBoost use the outer validation fold for early stopping and then score predictions on that same fold. The validation labels therefore influence the chosen iteration. This is a validation-quality flaw, not by itself proof of a competition-rule violation. Treat the reported OOF AUC as development evidence, not an unbiased held-out estimate. Correct by selecting iterations on an inner split of each outer-fit fold and refitting on the full outer-fit fold before scoring the untouched outer fold.
2. **Target encoding.** The inspected encoder is nested and does not use a row's own target for its training encoding; outer validation labels do not enter the encoding maps. The supplied rules do not explicitly resolve target encoding, so its rule status remains unverified even though the implementation's fold isolation is sound.
3. **Other covariate-only validation exposure.** The fee decoder computes frequency counts from all train and test covariates before outer folds are built. LightGBM category vocabularies are then derived from the full training feature frame, which includes outer-validation and other-fold covariates. Neither step uses those rows' labels, but feature construction is not strictly isolated to each outer-fit fold. Treat the fee pooling as **AMBIGUOUS — organizer clarification required** and record the vocabulary construction as a transductive validation limitation.
4. **Hard-coded assumptions.** The package calls `129.5` and `0.0132` archived hypotheses but does not establish their original derivation. The Merton features also fix `mu`/`sigma` pairs at `(0.02, 0.20)` and `(0.03, 0.25)`. The target-encoding smoothing values and 50:50 blend are fixed in config, but their selection history is absent. These values are not automatically prohibited, but their derivation or selection provenance is **UNVERIFIED**.
5. **External data and manual prediction changes.** No outside dataset, manual label, or test-row score adjustment was found in this pipeline. It loads the three supplied CSVs and generates model probabilities. The embedded frozen predictions are comparison references, not model inputs.
6. **Ensemble provenance.** The code applies a fixed 50:50 blend and does not optimize the weight in the training run. The historical reason for choosing that weight is not established by this package.

The rule references above follow the reviewed competition materials summarized in [OFFICIAL_RULES_AUDIT.md](../../reports/final_competition_audit/OFFICIAL_RULES_AUDIT.md): README notebook markdown cell 4 (external data and hand-labeling), and tutorial PDF slide 22 (feature engineering from existing fields). The same audit records transductive statistics and target-encoding rule scope as unresolved.

## Safe execution checks

- Python syntax: 24 package source/script/test files parsed successfully; all embedded Python code cells parsed successfully.
- Notebook: valid JSON, 89 cells / 43 code cells, no saved cell outputs or execution counts.
- Imports: all 11 production source modules imported without invoking training.
- Unit tests: `python -B -m unittest discover -s tests -p 'test_*.py' -v` passed, 22 tests. These are isolated unit tests; no model training was run as part of this recovery.
- Environment: Python 3.12.10 and installed NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, LightGBM 4.7.0, CatBoost 1.2.10, and SciPy 1.18.1 match the package requirements. `nvidia-smi` sees an NVIDIA GeForce RTX 4050 Laptop GPU. These checks indicate the runtime prerequisites are present; the recovered notebook was not run again.

## Disposition

**Engineering: `REUSABLE_AFTER_MODIFICATION`. Compliance: NOT CLEARED.**

Reusable after review: raw-data validation, row-wise transformations, nested target encoding, fold construction, model setup, blending, and submission checks. Before treating the working copy as a trusted next experiment, fix outer-fold early stopping, fit validation-sensitive preprocessing on outer-fit rows only where applicable, document the constants and blend-weight provenance, and obtain an organizer ruling on transductive feature fitting and target encoding. The existing corrected early-stopping work under `D:\gci\truth\experiments\exp85_phase2_inner_early_stopping\` was left untouched.

Best next experiment: a separate standalone, validation-corrected EXP85 replication that changes only early stopping, preserves the same raw inputs/features/folds/seeds/blend, reports per-fold results, and does not use EXP02. Do not claim that it reaches `0.79000+` until a corrected and rule-cleared evaluation demonstrates that result.
