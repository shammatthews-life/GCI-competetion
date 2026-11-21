# Experiment decision report

## Primary comparison

- Frozen primary: `raw_plus_all_product`.
- Development raw baseline pooled OOF AUC: `0.756878879`.
- Development primary product-block pooled OOF AUC: `0.759320302`.
- Paired development delta: `+0.002441424`; positive fold deltas: `5/5`.
- Paired row-bootstrap 95% interval: `[+0.001393071, +0.003443783]` (descriptive on previously exposed labels).
- One-time locked-holdout raw AUC: `0.761843784`.
- One-time locked-holdout primary AUC: `0.762763150`; delta `+0.000919365`.
- Best control development AUC: `0.758047420` (`random_fee_P`).

## Secondary results

Top individual single-feature discovery ablations:

                                  pooled_development_oof_auc  delta_vs_raw_baseline  positive_paired_fold_deltas
variant                                                                                                         
single:PROD_P5_FIT_FREQUENCY_LOG                    0.757896               0.001017                            3
single:PROD_P4                                      0.757426               0.000547                            5
single:PROD_P3                                      0.757368               0.000489                            4

Product block ablations:

                       pooled_development_oof_auc  delta_vs_raw_baseline  positive_paired_fold_deltas
variant                                                                                              
raw_plus_all_product                     0.759320               0.002441                            5
raw_plus_QP                              0.759200               0.002321                            5
raw_plus_P                               0.758639               0.001761                            5
raw_plus_no_redundant                    0.758465               0.001586                            5
raw_plus_KP                              0.758092               0.001213                            5
raw_plus_Q                               0.757593               0.000714                            4
raw_plus_K                               0.757519               0.000640                            4
raw_plus_KQ                              0.757474               0.000595                            4

Negative controls:

                 pooled_development_oof_auc  delta_vs_raw_baseline  positive_paired_fold_deltas
variant                                                                                        
random_fee_P                       0.758047               0.001169                            3
random_step_K                      0.757416               0.000538                            4
random_keys                        0.756240              -0.000639                            2
random_permuted                    0.755968              -0.000911                            1

See the corresponding CSVs for all candidates, folds, runtimes, ranking differences, and controls.

## Decision

**RESEARCH CANDIDATE ONLY — independent confirmation unavailable.** The holdout's labels were already used in previous project OOF analyses, so it is not independent confirmation. Even if the primary block improves on this one-time partition, the required untouched labeled confirmation is unavailable. No candidate is promoted, no 0.77 target is claimed, and no submission or champion change is made.

## Statistical caution

The individual/block/control matrix contains many comparisons; secondary winners are selection-biased. The paired bootstrap interval is not a substitute for a genuinely independent sample and does not adjust for feature-family search. Fold-average differences alone are not a significance test.
