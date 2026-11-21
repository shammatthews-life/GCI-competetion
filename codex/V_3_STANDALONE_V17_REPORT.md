# v_3 standalone v17 CatBoost-E1 probe

## Decision

This is the requested standalone v17 probe. It is **not a validated improvement over v15**, and the local results do not make it a likely route past public score 0.77891. Submit it only as a diagnostic leaderboard probe if you want to compare it with the v_2 result.

## Submission

- CSV: [v_3_standalone_v17_catboost_e1.csv](./v_3_standalone_v17_catboost_e1.csv)
- Exact upload path: `D:\gci\codex\v_3_standalone_v17_catboost_e1.csv`
- Metrics: [v_3_standalone_v17_metrics.json](./v_3_standalone_v17_metrics.json)
- OOF/test predictions: [v_3_standalone_v17_oof.npz](./v_3_standalone_v17_oof.npz)
- Runner: [run_v3_standalone_v17.py](./run_v3_standalone_v17.py)

## Run and validation

The job ran in `D:\gci\codex\.venv` using the supplied train/test/template files. It rebuilt the v17 residual features and consistent EXT/LOAN/PERSON views, trained a five-fold CatBoost-E1 model, and averaged the fold test predictions. It took about 29 minutes in this environment.

| Fold | OOF AUC |
|---:|---:|
| 1 | 0.767524 |
| 2 | 0.772691 |
| 3 | 0.774230 |
| 4 | 0.766290 |
| 5 | 0.769802 |
| **Pooled** | **0.770051** |

Submission validation passed: 61,500 rows, exactly `SK_ID_CURR,TARGET`, IDs unique and in sample-submission order, finite rank scores. SHA-256: `c915ce79344e81114ca2729c7020739a9c57f40a0d1f7285a1b2720aee5d03fb`.

## Important limitations

- This is **CatBoost-E1 by itself**, not the v15+v17 full rank blend. The exact v15 OOF/member/test cache is absent, so the accepted v15 formula cannot be reproduced or compared here.
- The supplied v17 review reports a previous standalone AUC of 0.77444, but its matched full-ensemble comparisons were only about +0.00000 for replacement and +0.00002 for averaging the CatBoost branches; both failed its promotion gate. This fresh local rebuild scored 0.770051, so it did not reproduce the earlier standalone result exactly.
- The displayed v_2 leaderboard score was 0.77198; despite v_3's higher local OOF than v_2, no result supports expecting v_3 to beat 0.77891.
- The v17 notebook's original decision cell wrote CSV files even for failed candidates. This standalone output is deliberately labelled as a probe rather than an accepted improvement.

If you submit it, keep the 0.77891 candidate as the incumbent and send back the exact score. Do not replace the incumbent unless the competition leaderboard score is higher.
