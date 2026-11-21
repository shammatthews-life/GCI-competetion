# Phase 3D formula experiment report

## Status and scope

Completed a fold-safe, row-wise experiment on the supplied 171,202-row training set. No test labels, external data, pooled train/test statistics, label-derived formula values, or edits to the champion/earlier experiments were used. All results reuse the existing five fixed folds and are discovery evidence only.

## Prespecified model and comparison

- Reference: preserved LightGBM `raw_plus_product` OOF predictions, aligned by applicant ID, TARGET, and the saved five-fold assignment.
- Each candidate reuses the same Phase 3C fold-local selected boosting rounds: `{"1": 436, "2": 542, "3": 583, "4": 388, "5": 387}`.
- Product features and model categorical encodings are fitted inside each outer training fold. Added formulas are deterministic per-row operations.
- No outer validation labels were used for feature selection, early stopping, or parameter fitting.
- Candidate scores: see `BASELINE_VS_FORMULAS_RESULTS.csv` and `FOLD_LEVEL_CV_RESULTS.csv`.
- Best screened formula configuration: `formula__risk_adjusted_ltv`, pooled OOF AUC `0.762373`, paired fold delta `-0.000100`. This is not a promotion decision.

## Formula interpretation

Five nonduplicate candidates were tested: risk-adjusted LTV, normalized external-score dispersion, LTV weighted by external-score dispersion, a financial-stress interaction, and the month-to-older bureau-window count ratio. See `FORMULA_CORRECTNESS_AUDIT.csv` for guarded definitions and excluded hypotheses. In particular:

- The baseline already contains LTV, annuity/income, family-normalized income, external-score mean/standard deviation, and an income/annuity coverage transformation. Re-testing those as novel inputs would be redundant.
- The inverse external-score ratio is stabilized with a fixed 0.05 on the normalized score scale because the observed source values include near-zero values; the score direction/meaning is not further defined in the local dictionary. Treat the result only as a hypothesis.
- The dictionary calls `AMT_ANNUITY` “Loan annuity” but does not state its time period. Annualizing it, labeling it monthly, or treating income/annuity as a calibrated DSCR is unsupported by this local dictionary.
- `AMT_GOODS_PRICE - AMT_CREDIT` is not established as an actual down payment; its normalized version is exactly `1 - LTV`.
- The implied-rate inversion cannot identify a contractual rate without repayment count and payment interval. For fixed assumed term, it is only a monotonic transformation of annuity/credit.
- Bureau ratio is descriptive, not an “acceleration” measure. The local dictionary’s timing descriptions do not establish equal-length, non-overlapping rates for every window.

## V5 notebook audit

The supplied v5 notebook has 10 cells; all saved code cells have `execution_count=None`, despite containing saved stdout. The saved output says old/new/mix pooled held-out AUCs were 0.77202/0.77198/0.77213; these values cannot be independently recomputed because the referenced old cache (`gci_cache_repro`) and OOF vectors are not present locally. Its own fold table has the all-new blend below the old blend on 3 of 5 folds, with mean paired fold delta -0.00005. The 50/50 mix is positive on 4/5 folds but mean gain is only +0.00010, below the notebook’s own +0.00020 gate. These are notebook claims, not verified new estimates. The notebook itself says to keep v3_A (claimed 0.77746) when the gate fails; that v3_A score/vector was not found for independent reproduction.

## Outputs and limitations

See `SOURCE_CODE_AUDIT.md`, `FORMULA_CORRECTNESS_AUDIT.csv`, `FORMULA_DISTRIBUTIONS.csv`, `FORMULA_OOF_PREDICTIONS.csv`, `FOLD_LEVEL_CV_RESULTS.csv`, `BASELINE_VS_FORMULAS_RESULTS.csv`, `V5_NOTEBOOK_CELL_INVENTORY.csv`, `V5_NOTEBOOK_SAVED_OUTPUT.txt`, `V5_SCORE_CLAIMS.csv`, `INPUT_SHA256_MANIFEST.csv`, `OUTPUT_FILE_MANIFEST.csv`, `environment.json`, and `EXECUTION_LOG.txt`.

No candidate is promoted. The folds have been examined in earlier phases, and selecting a result on them adds researcher degrees of freedom. Confirm any frozen candidate on genuinely unseen labels if available; a new random split of the same exposed labels is not independent confirmation.
