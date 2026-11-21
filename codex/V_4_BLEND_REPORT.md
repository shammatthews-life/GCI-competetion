# v_4 v17 + LightGBM rank-blend probe

## Decision

**v_4 is the best-supported diagnostic leaderboard probe made in this workspace.** It passes a paired, same-fold comparison against the v_3 standalone CatBoost-E1 model. It is not proven to beat the earlier v15 submission or public score 0.77891, because the v15 prediction/cache bundle is missing.

If you submit v_4, keep the earlier 0.77891 result as your incumbent and report back the exact public score. Do not assume this will restore that score.

## Submission file

- CSV: [v_4_v17_lgb_rank_blend.csv](./v_4_v17_lgb_rank_blend.csv)
- Upload path: `D:\gci\codex\v_4_v17_lgb_rank_blend.csv`
- Metrics: [v_4_blend_metrics.json](./v_4_blend_metrics.json)
- OOF predictions: [v_3_standalone_v17_oof.npz](./v_3_standalone_v17_oof.npz) and [v_4_lgb_seed42_oof.npz](./v_4_lgb_seed42_oof.npz)
- Reproduction: [run_v2_submission.py](./run_v2_submission.py) with `GCI_SEEDS=42`, followed by [assess_v4_v17_blend.py](./assess_v4_v17_blend.py)

## What was compared

Both branches have OOF predictions for the exact same 5 outer folds (`StratifiedKFold`, seed 42):

- **Reference:** standalone v17 CatBoost-E1, including residual features and consistent EXT/LOAN/PERSON views.
- **Added branch:** formula-aware LightGBM with leakage-safe Q4/Q3/Q4×contract target encodings.
- **Fixed blend:** 90% rank(CatBoost-E1) + 10% rank(LightGBM). This weight was declared before evaluating the blend; no grid search was used.

The LightGBM seed-42 run was re-executed so that each row's OOF score came only from the same held-out fold assignment as the CatBoost-E1 score. The earlier two-seed v_2 OOF file averaged predictions from different fold partitions and was not used for this test.

## Paired local results

| Measure | Result |
|---|---:|
| v_3 CatBoost-E1 OOF AUC | 0.770051 |
| v_4 fixed rank-blend OOF AUC | 0.770668 |
| Paired gain | **+0.000617** |
| Fold gains | +0.000638, +0.000739, +0.000710, +0.000544, +0.000495 |
| Positive folds | **5/5** |
| Paired stratified-bootstrap 95% CI | **[+0.000488, +0.000762]** |

This supports adding a small LightGBM component to this standalone CatBoost-E1 branch. It does **not** prove the same branch improves v15. The supplied v17 notebook's previous exact full-ensemble tests failed their own gate; this newly tested blend uses a different LightGBM challenger and must still be judged by the actual leaderboard.

## CSV validation

The generated file has 61,500 rows, exactly `SK_ID_CURR,TARGET`, unique IDs in sample-submission order, and finite rank scores. SHA-256:

`c272bce6a9498b3972546d5945bd02f6be18987d650ce51f1554036a59771433`

The displayed leaderboard results for previous probes were 0.77198 (v_2) and 0.77528 (v_3). v_4 is designed as a modest fixed blend improvement over v_3's standalone prediction; its public score is unknown until submitted. It is a reasonable next single probe, but no honest local analysis can promise it will reach 0.77891.
