# Independent row-wise feature builder specification

## Purpose and boundary

This specification defines a new, label-free transformation of the permitted
application predictors. It is not a reconstruction, port, or copy of any
historical feature-builder implementation. It does not claim to reproduce the
Phase 9 row-wise feature matrix. The Phase 9 builder's complete mathematical
specification is not present in the permitted raw-data and official-dictionary
materials, so exact source parity is not asserted.

Inputs are the supplied training predictors and the official
`HomeCredit_columns_description.xlsx`. `TARGET` is not accepted by either
feature function. `SK_ID_CURR` is retained outside the feature matrix solely
for row alignment and is excluded from model features. No test rows, external
tables, archived predictions, cached feature matrices, or holdout data are
used to construct features.

## Independent row-wise formulas

For each application row, the output starts with the 32 provided predictor
columns except `SK_ID_CURR`. Categorical values remain row-local strings;
missing category values become the literal `MISSING`. Numeric source values
remain numeric, with missing values left missing.

The following deterministic columns are added:

| Output | Mathematical definition |
|---|---|
| `DAYS_EMPLOYED_IS_365243` | `1[DAYS_EMPLOYED = 365243]` |
| `DAYS_EMPLOYED_CLEAN` | `NaN` for the observed sentinel value `365243`; otherwise the input value |
| `RAW_MISSING_COUNT` | Count of missing values over the provided predictor columns, excluding `SK_ID_CURR` |
| `EXT_SOURCE_MISSING_COUNT` | Count of missing values among `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` |
| `AGE_YEARS_FROM_DAYS_BIRTH` | `-DAYS_BIRTH / 365.25` |
| `CREDIT_TO_INCOME` | `AMT_CREDIT / AMT_INCOME_TOTAL` |
| `ANNUITY_TO_INCOME` | `AMT_ANNUITY / AMT_INCOME_TOTAL` |
| `CREDIT_TO_ANNUITY` | `AMT_CREDIT / AMT_ANNUITY` |
| `ANNUITY_TO_CREDIT` | `AMT_ANNUITY / AMT_CREDIT` |
| `GOODS_TO_CREDIT` | `AMT_GOODS_PRICE / AMT_CREDIT` |
| `CREDIT_TO_GOODS` | `AMT_CREDIT / AMT_GOODS_PRICE` |

Ratios are computed only when numerator and denominator are finite and the
denominator is nonzero; otherwise the result is `NaN`. No statistic is
estimated across rows, and no label enters a feature formula.

The `365243` value is treated only as an observed numeric sentinel for this
deterministic experiment; no occupational or economic interpretation is
asserted.

## Independently specified product keys

Product keys are separate string-categorical columns and use only values from
the current row:

| Key | Definition |
|---|---|
| Q3, Q4, Q5 | Integerize `round((AMT_ANNUITY / AMT_CREDIT) * 10^d)` for `d = 3, 4, 5` |
| P4, P5 | Integerize `round(((AMT_ANNUITY - 130.5) / AMT_CREDIT) * 10^d)` for `d = 4, 5` |
| AG3 | Integerize `round((AMT_ANNUITY / AMT_GOODS_PRICE) * 10^3)` |
| KQ3, KQ4 | For cash loans with `C > G > 0`, join `round((C/G - 1)/0.0066)` with Q3 or Q4 using `|` |

Here `A` is annuity, `C` is credit, and `G` is goods price. Rounding is
nearest-integer using NumPy's ties-to-even `rint` operation. Invalid ratios
use `MISSING`; ineligible KQ values use `NONCASH_OR_INVALID`. The constants
`130.5` and `0.0066` are explicitly exploratory hypotheses, not confirmed
fees, rates, terms, or organizer-approved facts.

## Provenance and limitations

The raw field names and descriptions come from the official local competition
dictionary. The arithmetic transforms above are specified here from those
fields and the user-provided product-key hypotheses. The independent
implementation is `phase10_independent_feature_builder.py`; formula tests are
in `test_phase10_independent_features.py`.

The prior Phase 9 CatBoost baseline included a row-wise feature builder whose
implementation is outside the permitted dependency boundary. This independent
specification deliberately does not infer its undocumented transforms from
historical model artifacts. Therefore the Phase 9 parity preflight can only
diagnose a changed feature space; it cannot establish exact reproduction.
