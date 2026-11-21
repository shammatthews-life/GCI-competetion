# Fixed versus fold-estimated fee ablation

Run: 2026-11-03T14:14:13.905304+00:00 to 2026-11-03T14:17:49.247152+00:00 UTC; 215.3 seconds.

The same seven P-family features were evaluated with a fee selected on the current outer training partition, with the notebook's fixed 129.5, and with unadjusted Q keys as a control. Each variant used its own inner-only early-stopping choice, the preserved LightGBM configuration, and the same outer folds.

                    family  pooled_oof_auc  mean_fold_auc  std_fold_auc  mean_paired_fold_delta  std_paired_fold_delta  pooled_delta_vs_baseline  folds_improved  fold_1_delta  fold_2_delta  fold_3_delta  fold_4_delta  fold_5_delta
 p_keys_fold_estimated_fee        0.762184       0.762280      0.003591                0.001551               0.001191                  0.001526               4      0.002319      0.001441      0.001939      0.002507     -0.000450
p_keys_fixed_v17_fee_129_5        0.761656       0.761768      0.003426                0.001039               0.000536                  0.000997               5      0.001956      0.000897      0.000687      0.001020      0.000635
 q_keys_unadjusted_control        0.760939       0.761049      0.003169                0.000319               0.000888                  0.000281               3      0.001460      0.000865     -0.000864      0.000190     -0.000054

The 129.5 candidate is not treated as an established contract fee. This is a fixed-hypothesis comparison, not independent confirmation; see the fold estimates and prior key-repeat diagnostics.
