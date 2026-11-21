# Numerical lattice hypothesis report

## Formula tested

For cash-loan rows with positive goods price and credit greater than goods price:

`coordinate = (AMT_CREDIT / AMT_GOODS_PRICE - 1) / step`

Prespecified candidate steps were `0.0066` and `0.0132`. Negative-control placebo steps were `0.0057`, `0.0097`, `0.0173`, and `0.0231`. A fit-only selected step from the original Phase 3 decoder was also evaluated on each held-out development fold. No target labels enter step fitting or residual calculations.

## Results

See `LATTICE_FOLD_STABILITY.csv` for every fold, sample size, residual mean/median/p95, and shares below absolute residual tolerances 0.01, 0.05, and 0.1; `LATTICE_STEP_COMPARISON.csv` aggregates stability. Each fitted step was estimated only from its fold's development-fit rows; evaluation used that fold's held-out rows. The confirmation holdout was not used for lattice discovery.

## Controls and interpretation

Placebo step sizes are deliberately not selected on validation labels. Compare their held-out support to the prespecified hypotheses, and report all tested values. Any fit-only grid search is descriptive and its selected step must be evaluated on held-out rows; do not quote fit support as generalization. Integer residuals demonstrate only a mathematical pattern in these variables, not a contract increment, fee, interest rate, repayment term, or causal mechanism.
