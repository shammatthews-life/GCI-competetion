# Static audit: `solution_v5_gci_research.py`

## Scope and provenance

- Audited file: `C:\Users\SHARON\Downloads\solution_v5_gci_research.py`
- Size: 140,761 bytes.
- SHA-256: `4f0f1e9208f99ef2350af05f9c3da98eaae922de729295f2e0c410186924ae74`.
- The source header identifies it as a Home Credit v5 research solution. It does not identify an author or prove that it is the missing Claude seven-model implementation.
- Its `MODELS` registry contains 20 model names, not seven (`MODELS`, source lines 1235-1256). No associated OOF vectors, model checkpoints, or run manifest were located in the scanned inventory. The reported 0.7647 AUC is therefore not reproduced by this source audit.
- This was static inspection only. The script was not imported or executed; no training was started.

## Execution and data behavior

- Input is `sys.argv[1]` (default `"."`); the script reads `train.csv` and `test.csv` from that folder (lines 49-58).
- Import-time behavior creates the configured output directory. `OUT_DIR` defaults to relative `output_v4` (lines 53-64). A safe execution must set an isolated absolute `OUT_DIR`.
- `MODE` defaults to `blend`; the blend mode expects existing model OOF/test prediction files and does not train models itself.
- `SMOKE=1` samples rows from train and truncates test; it is not a full-data score.
- The source globally suppresses warnings (`warnings.filterwarnings("ignore")`, line 40), which would hide relevant compatibility/fallback warnings.

## Validation and leakage findings

| Area | Source evidence | Assessment |
|---|---|---|
| Global feature construction | `build()` concatenates target-free train and test before feature engineering (line 487). | Every fold's validation and test covariates contribute to pooled transforms. This is transductive; eligibility depends on the competition rules and is not established here. |
| Pooled statistics | `winsor()` takes the input quantile (lines 122-124); `fe_base`, `fe_product`, `fe_freq`, `fe_group` and group percentiles are run by the pooled `build()` (lines 142, 360, 395, 442, 458, 478-519). | Quantile cutoffs, product peaks/frequencies, category frequencies, peer statistics and percentiles are not fitted within each outer training fold. Validation rows participate in their own transform statistics. |
| External-score imputation | `fe_ext_impute()` does five-fold cross-fitting over rows passed to it (lines 224-263); `build()` passes the pooled train+test feature matrix (lines 498-505). | Although cross-fitted internally and independent of `TARGET`, each outer validation row can be included in models that generate other pooled rows' features. This does not satisfy strict outer-fold isolation. |
| Target encoding | `target_encode()` creates an inner stratified split for training-row encodings and uses the outer-fold training labels to encode validation/test rows (lines 556-577); `fold_frames()` invokes it on each outer fold (lines 1118-1135). | This component is materially better isolated from direct target leakage. Category keys are still built from pooled, label-free rows. |
| Early stopping/model selection | `run_model()` passes `Xva, y[va]` to the model fit and reports AUC on those same rows (lines 1477-1488). LightGBM, XGBoost and CatBoost use that validation data for early stopping/best-iteration selection (lines 1288-1319). | The reported fold is not an untouched validation estimate. Its labels select iteration count and then score the same predictions; OOF terminology does not remove this bias. |
| Blend evaluation | `cv_score()` re-splits saved base OOF rows to fit/evaluate blend rules (lines 2081-2086), while `blend_models()` selects a rule and also reports same-row `auc_oof` (lines 2089-2112). | `auc_oof` for fitted blends is in-sample. The row-level meta-CV is not a fully nested stacked CV: base-model OOF predictions used to train the meta-fit can depend on labels from the meta-validation rows through their base-model training folds. |
| Cache identity | `FE_KEY` uses feature-code signature, flags, and train/test row counts (lines 75-90); fold checkpoint config hashes the model/configuration (lines 1466-1474). | Dataset content hashes, fold-index hashes, and full environment identity are not included. Same-size replacement data can reuse stale feature caches/checkpoints. |

## Model, GPU, and mathematical notes

- The XGBoost configuration uses `tree_method="hist"` without `device="cuda"` (line 1302); LightGBM has no GPU device setting in `LGB_BASE`/`fit_lgb` (lines 1260-1294); CatBoost has no `task_type="GPU"` in its parameters (lines 1310-1319). This script is not configured to use the detected GPU.
- `AMT_CREDIT / AMT_ANNUITY` is called an implied number of payments/term, and `INTEREST_PROXY` is based on a rounded quotient (lines 170-180). Principal and payment alone do not identify both rate and term; the proxy must not be presented as an observed term or interest rate without independent contract assumptions and validation.
- The source contains comments claiming measured feature gains and marking some experiment configurations accepted (for example `fe_product()` and `EXPERIMENTS`). Those comments/configuration labels are not run evidence and cannot validate the stated deltas.
- AUC methodology and data/feature provenance require correction before any score from this script can be compared with the preserved champion.

## Safe next steps

1. Establish whether the competition permits pooled unlabeled train/test transforms; if not, replace each pooled learned transform with training-fold-fitted transforms.
2. Move early stopping/model selection into inner folds; preserve outer validation labels strictly for final scoring.
3. Use nested base-model and blend fitting for unbiased ensemble assessment.
4. Bind caches and checkpoints to dataset and fold hashes, record the environment, and stop suppressing warnings in the audit runner.
5. Recover an authoritative source/OOF manifest if the user intends this file to stand in for Claude's seven-model solution.
6. Only then run bounded model-specific GPU capability checks and comparative training.
