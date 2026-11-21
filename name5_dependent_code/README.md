# Independent raw-data reconstructions

Each required replacement is self-contained: it reads only `../inpuy/train.csv`
and `../inpuy/test.csv`, uses `TARGET` only from the training data, and writes
its own submission to `outputs/` when run.  It neither imports another
replacement nor loads a historical artifact.  Install dependencies with
`pip install -r requirements.txt`; run any script from this directory.  The
default is 5-fold training. `--fast` is a 3-fold smoke-test mode, not a score
claim. XGBoost scripts also accept `--device cuda` when a supported GPU exists.

## Original dependency audit and elimination

| Original script | Historical artifacts read | Creator source found | Replacement approach |
|---|---|---|---|
| `build_precision_transductive_apex_champion.py` | asymmetric-tail CSV; `hasan_expanded_zoo.npz` | `build_asymmetric_convex_gated_champion.py`; `train_catboost_and_deep_lgb.py`, `train_hasan_master_zoo.py`, `screen_and_select_features.py`, `build_hasan_600_features.py` | Trains five fresh tree-model prediction families, reconstructs the rank/probit discrepancy gate, graph adjustment and tail-stage intent in memory. The old anchor prediction cannot be reproduced without its historical models. |
| `train_catboost_and_deep_lgb.py` | selected-550 Feather matrices; base-zoo NPZ | `screen_and_select_features.py`; `train_hasan_master_zoo.py`; `build_hasan_600_features.py` | Rebuilds raw application features in memory and trains the requested CatBoost + deep LightGBM families. It does not load the base zoo. |
| `train_exp15_gpu_xgboost.py` | v16 Feather frames; residual/view Pickles; partner Feather; encoder Pickles; v17 bundle | `build_v16_features.py`; `run_v3_standalone_v17.py`; `exp14_partner_expansion_features.py`; `build_v16_encodings.py`; `train_full_v16_suite.py` | Rebuilds base/enriched raw features in memory, trains EXP15-style XGBoost and a fresh LightGBM standing-model alternative, then chooses an OOF-only rank blend. Historical residual/view and v17 learned predictions cannot be reconstructed exactly without retraining their unavailable upstream suites. |
| `train_v58_clean_gpu_0783_champion.py` | v55 bundle; XGBoost arrays; v43 submission | V55 and 10-fold arrays: `champion/exp85_solo_runner.py`; no writer for the exact v43 filename was found anywhere in the repository | Trains seven genuine raw-data model members (three LightGBM, ExtraTrees, CatBoost, two XGBoost), fits OOF-only weights, then fits the original-style solvency adjustment on OOF data. v43 is replaced by the learned raw-data ensemble and is not represented as a fake anchor. |
| `train_full_v16_suite.py` | eight v16 Feather frames; residual/view/encoder Pickles; v16/v6/v8 bundles | `build_v16_features.py`; `build_v16_views.py`; `build_v16_encodings.py`; `train_v16_and_optimized.py`; `train_models_v6_multiseed.py`; `train_models_v8_knn.py` | Reconstructs raw feature handling internally, retrains the documented five LightGBM member configurations and their original member weights. The historical residual/view and v6/v8 learned bundles are unavailable as raw deterministic source products, so they are explicitly excluded rather than silently fabricated. |

`build_hasan_600_features.py` is available and was audited as the source of the
old 550-feature chain. Its long hand-crafted feature laboratory is not copied
verbatim into all five scripts because the original downstream ensembles also
depend on historical artifacts whose exact learned predictions are unavailable.
Instead, each independent script uses explicit raw application, categorical,
missingness, external-source, affordability and temporal features and trains
its declared models from scratch. This is a legitimate raw-data alternative,
not an assertion of score or prediction equivalence.

## Input/output and validation

Inputs are exactly the official train/test CSVs. Each script verifies the
`SK_ID_CURR,TARGET` schema, test-row count, finite/non-null predictions and
the `[0,1]` target range before writing. No test target, external data,
historical submission, cache, NPZ/NPY/Feather/Pickle, checkpoint, network
download, or embedded prediction is used.

## Verification status

Syntax and static artifact checks were run on all five replacements. Fast
end-to-end runs completed for the CatBoost/deep-LGB, EXP15 XGBoost and full-v16
replacements. The v58 run was stopped during its real ExtraTrees training after
the task was redirected; it was not claimed successful. No full default
five-fold run has been claimed. The original exact V43 generator is missing
from the project source, and old learned model predictions cannot be recovered
from code alone.
