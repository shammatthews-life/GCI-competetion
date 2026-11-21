# Diagnostic interpretation of term-proxy gain

## Measured result

The strongest single feature in this probe was `snapped_paid_ratio`: pooled AUC 0.763049, a +0.000557 paired pooled change versus the refit control, with mean fold delta +0.000564 and positive deltas on 3/5 folds. The alternate rounded residual yielded pooled AUC 0.762983 and improved on 4/5 folds. The six-month-snapped residual improved on only 3/5 folds. These are exploratory on reused folds.

The candidate formulas are deterministic transforms of `AMT_CREDIT / AMT_ANNUITY`, not independent observations of loan contract terms. They must not be described as actual repayment duration, total paid, or interest.

The rounded proxy is `n = clip(round(P/A), 6, 72)`. Its paid-ratio feature is `A*n/P = n/(P/A)`, a sawtooth rounding residual close to one. The six-month proxy rounds `n` to six-month steps; its paid-ratio feature deliberately creates a coarser quantization residual. Therefore, any AUC change indicates that this piecewise transformation helps the tree model represent nonlinearities in the existing credit/annuity ratio; it does not establish a true six-month contract structure.

See `TERM_PATTERN_DIAGNOSTICS.csv` for extrema/quantiles, `TERM_PROXY_DEFAULT_RATES.csv` and `PAID_RATIO_QUANTILE_DEFAULT_RATES.csv` for exploratory target rates, `TERM_PROXY_CORRELATIONS.csv` for redundancy, and `TERM_SUBGROUP_AUC_DIAGNOSTICS.csv` for whether the OOF gain is concentrated in contract/product subgroups. These subgroup analyses reuse the same exposed labels and are hypothesis-generating only.

For the saved subgroup diagnostic, the six-month residual changes AUC by +0.001498 within revolving loans (n=16214). This is a descriptive subgroup pattern, not independent evidence; the smaller group and the multiple subgroup comparisons make it especially susceptible to chance.

The OOF deltas were independently recalculated from the saved predictions in `TERM_PATTERN_AUC_RECALCULATION.csv`. All-fold improvements are not independent confirmation because the folds were used in earlier phases and multiple formula variants were compared.
