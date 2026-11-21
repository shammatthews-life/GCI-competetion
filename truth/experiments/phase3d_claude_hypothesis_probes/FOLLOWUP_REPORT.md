# Phase 3D follow-up: contract, external-score, monotonicity, and fixed blends

## Result

All formula/model probes completed on the preserved five folds. Baseline reference is the Phase 3C LightGBM raw-plus-product OOF vector (AUC 0.762491). The fixed fold iteration counts and preprocessing are unchanged; all fold-fitted transformations use outer-training rows only. These are exploratory because the folds have previously been examined.

See `FOLLOWUP_SUMMARY.csv` for each candidate's pooled OOF AUC, paired fold delta, and number of fold wins. `FOLLOWUP_FOLD_RESULTS.csv` has per-fold values. `FIXED_BLEND_COMPARISONS.csv` and `MODEL_RANK_CORRELATIONS.csv` compare predeclared equal-weight rank/probability blends; no blend weights were tuned on these reported labels.

| Follow-up candidate | Pooled OOF AUC | Change vs. Phase 3C LGB product | Improved folds |
|---|---:|---:|---:|
| Refit control | 0.762491 | +0.000000 | — |
| `round(C/A)` term proxy | 0.762104 | -0.000387 | 1/5 |
| six-month-snapped term proxy | 0.762199 | -0.000293 | 1/5 |
| `A*round(C/A)/C` rounded residual | 0.762983 | +0.000492 | 4/5 |
| rounded `A*n-C` gap | 0.762527 | +0.000036 | 3/5 |
| `A*n/C` six-month residual | 0.763049 | +0.000557 | 3/5 |
| six-month `A*n-C` gap | 0.762451 | -0.000040 | 2/5 |
| fixed 0.20/0.35/0.45 external-score summary | 0.762025 | -0.000467 | 1/5 |
| nested cross-fitted logistic external score | 0.762418 | -0.000074 | 2/5 |
| monotone `DAYS_BIRTH` increasing | 0.761803 | -0.000688 | 1/5 |
| monotone `DAYS_BIRTH` decreasing | 0.762075 | -0.000417 | 1/5 |
| monotone external scores decreasing | 0.762368 | -0.000123 | 2/5 |

The two paid-ratio residuals are the only promising single-feature signals here. The six-month version has slightly higher pooled AUC but less consistent fold improvement; the rounded version wins 4/5 folds. Do not select between them as a final candidate on these already-examined folds.

## Interpretation of the contract hypotheses

No inferred term or rate is an observed contract quantity. `round(credit/annuity)` is only a scale proxy; interest makes the amortization duration differ from credit/annuity. The 6-month snap and 6..72 clipping are assumptions. `annuity*n/credit` is near one by construction for that heuristic, except where rounding/clipping intervenes. `annuity*n-credit` is therefore a heuristic residual. A fixed-term inverse-rate is monotone in annuity/credit, already supplied to the model, so it was not presented as an independent financial signal.

The experiment does test the rounded/clipped and snapped proxy variants separately. Their outcomes do not establish true term or rate; look for robustness in the fold deltas rather than treating a small AUC change as contract discovery.

The paid-ratio residuals are near one by construction (`0.99986` mean, `0.01765` SD for rounded; `1.00451` mean, `0.11708` SD for snapped). The rounded residual's Spearman correlation with `credit/annuity` is only `0.027`, which is consistent with a sawtooth/quantization transform that exposes fractional-ratio structure not easily represented by tree splits. This is a plausible modeling explanation, not evidence that the feature is an actual repayment schedule. The snap is an unverified six-month assumption.

## External scores and monotonic constraints

The user-proposed 0.20/0.35/0.45 combination is tested as a fixed row-wise weighted mean (weights renormalized across available scores). A second feature uses logistic-regression weights derived only from each outer training fold; training-row scores are cross-fitted internally. Baseline already includes mean/min/max/std/count/products/differences, so an added score is an alternate summary, not new source information.

Monotonic experiments (positive/negative `DAYS_BIRTH` and negative `EXT_SOURCE_1..3`) are reported separately. A positive constraint on `DAYS_BIRTH` means risk must increase as its numeric value increases (becomes less negative/younger). A constraint on three raw external scores does not constrain their engineered combinations; interpret it as a limited model restriction, not a global causal claim.

## Blend caution

The best existing fixed probability blend remains LightGBM/CatBoost at 0.763957. A post-probe 50/50 probability blend of the snapped residual model and CatBoost has AUC 0.764338 (+0.000381 vs. that earlier blend); the rounded-residual/CatBoost blend has 0.764328 (+0.000371) and improves 5/5 folds against it. Because the formula was selected after examining these results, these blend gains are still adaptive discovery, not independent confirmation. The all-five-model equal rank blend scores 0.762585; rank correlations among product models are high (0.925–0.984), limiting the value of indiscriminate blending.

Nested optimization of weights was not performed: available OOF vectors were generated across the global fold partition, so using those vectors to fit weights inside one held-out fold is not a rigorously nested outer evaluation. A proper nested rank-weight experiment requires retraining every base estimator within each outer training fold. The Claude `solution.py`, notebook, model OOF files and its seven-model definition were not found locally; its 0.7647 score cannot be reconstructed or attributed to these models.

## Files and reproducibility

The execution scripts, OOF predictions, formula audit, term distributions, fold results, subgroup diagnostics, fixed blend tables, model correlations, environment, input hashes, and output manifest are saved in this folder. No champion or earlier experiment was modified.
