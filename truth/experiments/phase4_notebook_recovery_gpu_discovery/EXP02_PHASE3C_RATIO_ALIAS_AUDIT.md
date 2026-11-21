# EXP02 ratio overlap with Phase 3C baseline

- Audit status: **PASS** (`2026-11-04T04:38:24.981886+00:00`)
- Training rows audited: `171,202`. `TARGET` was removed before feature generation.
- Input train SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- Preserved baseline source SHA-256: `c49aab451d1500e1bd7e0d618a764efeb1815c172f74e1bc5174e38a793ff3fe`
- Phase 3C model source SHA-256: `b560a8b593fc0d7dbe770fbf548a1b3fdbb5e9490f652ac098aab92ea9c701b2`

## Mapping

| EXP02 feature | Formula | Existing Phase 3C feature | Relation | Comparable rows | Equivalent rows | Candidate missing | Baseline missing |
|---|---|---|---|---:|---:|---:|---:|
| `FIN_CREDIT_TO_INCOME` | `AMT_CREDIT / AMT_INCOME_TOTAL` | `CREDIT_TO_INCOME` | exact | 171202 | 171202 | 0 | 0 |
| `FIN_ANNUITY_TO_INCOME` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | `ANNUITY_TO_INCOME` | exact | 171202 | 171202 | 0 | 0 |
| `FIN_GOODS_TO_CREDIT` | `AMT_GOODS_PRICE / AMT_CREDIT` | `CREDIT_TO_GOODS` | reciprocal | 171202 | 171202 | 0 | 0 |
| `FIN_CREDIT_TO_ANNUITY` | `AMT_CREDIT / AMT_ANNUITY` | `CREDIT_TERM` | exact | 171202 | 171202 | 0 | 0 |
| `FIN_INCOME_PER_FAMILY_MEMBER` | `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS` | `INCOME_PER_FAMILY` | exact | 171200 | 171200 | 2 | 0 |
| `FIN_CREDIT_PER_FAMILY_MEMBER` | `AMT_CREDIT / CNT_FAM_MEMBERS` | `CREDIT_PER_FAMILY` | exact | 171200 | 171200 | 2 | 0 |
| `FIN_ANNUITY_PER_FAMILY_MEMBER` | `AMT_ANNUITY / CNT_FAM_MEMBERS` | `ANNUITY_PER_FAMILY` | exact | 171200 | 171200 | 2 | 0 |

## Findings

- The Phase 3C raw feature builder already creates all seven ratio concepts. Six EXP02 formulas match those features on comparable rows; `FIN_GOODS_TO_CREDIT` is the reciprocal of existing `CREDIT_TO_GOODS` on rows where both prices are valid.
- The alternative transforms differ at edge cases: EXP02 requires a finite positive denominator; the source baseline `safe_ratio` permits any nonzero denominator. It fills missing family size with 1 before per-family ratios, while EXP02 leaves those values missing. It fills missing goods price with credit before `CREDIT_TO_GOODS`, whereas EXP02 leaves `GOODS_PRICE / CREDIT` missing.
- There are no missing `AMT_GOODS_PRICE` rows in this training dataset, so the goods-price fallback did not alter this audit. The observed missingness difference is limited to 2 rows with missing family size.
- Consequently, the EXP02 gain over the raw-feature CatBoost experiment cannot be described as discovery of seven novel ratios for Phase 3C: those ratios were already part of Phase 3C's rowwise baseline. The seed-20261105 experiment demonstrated a useful joint engineered-feature signal relative to its raw-only input; it did not test incremental lift over Phase 3C.
- The Phase 3C fixed blend's reported `0.763957` already includes the existing ratio features. Adding a second copy would test duplicate representation/model split competition, not new financial information. The planned duplicate-addition training run was correctly stopped before execution.
- No AUC was calculated in this audit, no model was trained, and no existing model, champion, or experiment artifact was modified.

## Artifacts

- Machine-readable mapping: `EXP02_PHASE3C_RATIO_ALIAS_AUDIT.csv`
- Execution log and hashes: `EXP02_PHASE3C_RATIO_ALIAS_AUDIT_LOG.txt`
