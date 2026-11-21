# Product decoder component ablation

Run: 2026-11-03T14:08:57.203126+00:00 to 2026-11-03T14:13:29.866448+00:00 UTC; 272.7 seconds.

The same fixed outer folds, preserved baseline builder/model configuration, fold-only fee/step estimation, inner-only early stopping, and untouched outer predictions were used for each feature family.

                      family  feature_count  pooled_oof_auc  mean_fold_auc  std_fold_auc  mean_paired_fold_delta  std_paired_fold_delta  pooled_delta_vs_baseline  folds_improved  fold_1_delta  fold_2_delta  fold_3_delta  fold_4_delta  fold_5_delta
              k_lattice_only              5        0.761012       0.761097      0.003699                0.000368               0.000478                  0.000354               4      0.000899      0.000698      0.000372      0.000211     -0.000338
   q_rounded_ratio_keys_only              4        0.760939       0.761049      0.003169                0.000319               0.000888                  0.000281               3      0.001460      0.000865     -0.000864      0.000190     -0.000054
    p_fee_adjusted_keys_only              7        0.762184       0.762280      0.003591                0.001551               0.001191                  0.001526               4      0.002319      0.001441      0.001939      0.002507     -0.000450
all_product_decoder_features             16        0.762826       0.762902      0.003929                0.002173               0.001473                  0.002168               4      0.003333      0.003347      0.002519      0.001893     -0.000227

This identifies whether the Batch 1 product-decoder family derives signal from Q rounded annuity/credit ratios, P fee-adjusted ratios, or the K financing lattice. These remain discovery results on reused folds; no component is promoted without a frozen confirmation.
