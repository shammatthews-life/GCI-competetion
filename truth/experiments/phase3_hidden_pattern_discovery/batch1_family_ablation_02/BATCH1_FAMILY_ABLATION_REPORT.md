# Batch 1 family ablation — discovery results

Run: 2026-11-03T14:02:14.568981+00:00 to 2026-11-03T14:07:11.958387+00:00 UTC; 297.4 seconds.

The saved corrected baseline OOF was re-aligned by ID and labels. It remains the comparator (pooled AUC 0.760658450595). Each of the four feature families got its own inner-split early-stopping choice, then a refit on the complete outer training fold. Product constants and key frequencies were fit on the current fit partition only. No test covariates, TARGET-derived features, or external data were used.

## Family-level comparison

                             family  features_added  pooled_oof_auc  mean_fold_auc  std_fold_auc  mean_paired_fold_delta  std_paired_fold_delta  pooled_delta_vs_verified_baseline  folds_improved  fold_1_delta  fold_2_delta  fold_3_delta  fold_4_delta  fold_5_delta
               product_decoder_only              16        0.762826       0.762902      0.003929                0.002173               0.001473                           0.002168               4      0.003333      0.003347      0.002519      0.001893     -0.000227
            financial_formulas_only              11        0.760724       0.760799      0.003740                0.000070               0.000860                           0.000065               2      0.001079      0.000871     -0.000323     -0.000902     -0.000376
product_financial_interactions_only              11        0.760324       0.760471      0.002885               -0.000258               0.001325                          -0.000334               2      0.001638      0.000286     -0.001825     -0.000311     -0.001079
         all_phase3_batch1_features              38        0.761353       0.761438      0.003583                0.000709               0.001212                           0.000694               3      0.002753      0.000654      0.000546     -0.000179     -0.000229

`product_decoder_only` isolates Q/P rounded keys, fit-only frequency signals, fold-estimated fee candidates, and the financing lattice. `financial_formulas_only` tests domain ratios without product decoding. `product_financial_interactions_only` tests external-score/financing interactions. The combined family is included to measure complementarity.

These are development-fold results on folds already used in Phase 2, not untouched confirmation. A positive family result is only a discovery lead and is not promoted.

## Fee hypothesis

See `FEE_HYPOTHESIS_FOLD_TESTS.csv`. It compares unadjusted Q5, the fold-estimated fee P5, and v17's fixed 129.5 P5 using fit-set key duplication and validation key repetition in the fit map. A fee is not supported merely because a rounded key appears in the fit set; the adjusted key should improve held-out repeated-key coverage consistently.

## Lattice hypothesis

See `FINANCING_LATTICE_FOLD_TESTS.csv`. Fold-local searches selected the coarser-grid candidate 0.0066 in the initial run, exactly half of v17's 0.0132. Both are stable candidates but they have different integer interpretations. A strong 0.0066 score can arise because every 0.0132 step is also an even 0.0066 step; inspect held-out odd-integer support and actual error distributions before naming a primitive increment.

## Decision

No individual formula is promoted. The 0.79 objective remains unachieved/unverified. Continue only after reviewing the separate family ablations and freeze a new confirmation protocol before feature selection.
