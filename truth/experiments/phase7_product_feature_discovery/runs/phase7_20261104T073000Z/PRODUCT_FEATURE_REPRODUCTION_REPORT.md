# Product feature reproduction report

## Exact source and fold-safe implementation

The Phase 3 feature source is `D:\gci\truth\experiments\phase3_hidden_pattern_discovery\run_phase3_hidden_pattern_discovery.py` (SHA-256 `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`); Phase 3B ablation source is `D:\gci\truth\experiments\phase3b_deep_validation\phase3b_ablation.py` (SHA-256 `1470a412ab42aa6015f94af01859615a96fb127b7fc124d045acf8c9a81cf8ae`). The direct implementation uses `fit_fee_and_step(raw_fit)`, `fit_product_maps(raw_fit, params)`, and `make_product_features(raw, params, maps)`. Parameters and frequency maps are fit only on each outer fit fold and then applied unchanged to its validation fold. No target is passed into these functions and test data is not used.

- Formula columns reproduced: `16` model columns (`K=5`, `Q=4`, `P=7`) from a `51`-definition Phase 3 catalog.
- Cash-loan masks, fit-only frequency tables, Q/P rounding precision, fee offset, K step, and K validity threshold are unchanged from source.
- Per-fold fee/step values, feature missingness/cardinality, and parameter-fit row counts are in `PRODUCT_FEATURE_FOLD_DIAGNOSTICS.csv`.
- Independent equation checks for Q5, P5, and K coordinate passed at max absolute error <= 1e-10 in every development fold.
- Formula definitions, aliases, and eligibility status are in `PRODUCT_FEATURE_FORMULA_REGISTER.csv`.
- Existing Phase 3 `NO_REDUNDANT` feature list has been de-duplicated only to prevent duplicate column insertion; feature definitions themselves are unchanged.

## Interpretation

The product decoder remains a numerical feature hypothesis. Its estimated offset is not an established fee; its K coordinate is not an actual financing tier; Q ratios do not establish installment count or loan duration. Features based on `AMT_ANNUITY` retain the dictionary limitation that its payment period is unspecified.
