# Phase 5 evidence-recovery source audit

Audit time: 2026-11-04T05:35:53.266344+00:00

## Data and scope

- Train: `D:\gci\inpuy\train.csv`; shape (171202, 34); SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Test: `D:\gci\inpuy\test.csv`; shape (61500, 33); SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- `TARGET` is present in train and absent from test. This audit used train labels only to score already-saved arrays. No test labels were read.
- No model was fitted, no feature selection was performed, and no submission or notebook was modified.

## V16 sources and configured caches

- Executed notebook: `D:\gci\pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb`; saved outputs report fixed-blend AUC 0.77649.
- Source-only counterpart: `D:\exp colab\GCI_v16_everything_merged.ipynb`.
- The notebook extraction configures `gci_cache_v16` for model-member caches and expects `catboost_v16_fold_{fi}.npz`.
- No `gci_cache_v16` directory was found by the recursive directory-name search. Local `gci_cache_final` and `v_3_cache` artifacts exist, but have no manifest binding their contents to the executed notebook, raw-data hashes, row IDs, or folds. Two same-named view caches have different hashes.
- The local optimizer `D:\gci\gemini\train_v16_and_optimized.py` is a separate pipeline. It grid-searches blend weights against the same OOF labels whose AUC it reports, so its selected blend AUC is not an unbiased evaluation. Its bundle stores `final_auc=0.782000`, while the stored `final_oof` scores `0.79467683` against the current labels. The stored `oof_e1Aet`, `oof_cb`, and `final_oof` arrays are exact duplicates (array SHA-256 values: `{"final_oof": "2febb38d37455ee7875a8889a1cb6f4204c822510656904d0380602774649c1e", "oof_cb": "2febb38d37455ee7875a8889a1cb6f4204c822510656904d0380602774649c1e", "oof_e1Aet": "2febb38d37455ee7875a8889a1cb6f4204c822510656904d0380602774649c1e"}`); this bundle is rejected as a V16 reproduction.

## Candidate V16-formula calculation

The fixed blend formula was applied without training to the local candidate model caches, using member weights from the notebook and the source's fixed 0.9/0.1 blend. Candidate scores are recorded in `V16_CACHE_SCORE_COMPARISON.csv`. The candidate set AUCs do not reproduce the saved notebook outputs, and the final diagnostic is **not a reproduced V16 result**: the cache files lack proven row/fold identity. The CatBoost shards were assigned using the source's seed-42 split order solely to calculate a diagnostic; the shards themselves contain no fold indices and are named `catboost_product_fold_*`, not `catboost_v16_fold_*`.

## EXP85 final OOF recovery

- The frozen EXP85 artifact is a 61,500-row test submission, not an OOF vector.
- `D:\gci\gemini\dependent\exp85_notebook_run\cache\rebuilt_eight_engine_oof.npy` is a finite 171,202-by-8 upstream component matrix; it does not contain the final test-only Anchor74 plus eight-engine synthesis and has no embedded row IDs/fold sidecar.
- The available Anchor74 recovery is test-only. No Anchor74 OOF counterpart or final champion OOF/fold map was recovered.
- The source and README under `D:\gci\_gci_push_tmp\champion\EXP85_CHAMPION_IMPROVEMENT\experiments\exp02_anchor74_oof` describe a different raw-data surrogate and hybrid-cache-assisted diagnostic. It has 2 files and no run-output directory; it cannot substitute for historical EXP85 evidence.

## Competition validation boundary

The prior static code audit recorded pooled train+test covariate transforms and nested supervised encodings in V16. Those methods remain subject to organizer-rule clarification. This recovery run does not decide their permissibility or use them to train anything.

## Decision

**BLOCKED — exact EXP85 final OOF and fold assignments are not recoverable from the inspected artifacts.** Do not run a champion-comparison experiment, claim an improvement, or replace the frozen submission based on these cache diagnostics. The next valid gate is to obtain a provenance-bound final OOF vector, fold IDs, row IDs, member lineage, and Anchor74 out-of-fold analogue; otherwise define and label a new fold-safe challenger separately from historical reproduction.
