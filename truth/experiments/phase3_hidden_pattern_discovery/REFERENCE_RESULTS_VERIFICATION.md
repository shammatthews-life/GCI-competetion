# Historical OOF reference verification

Executed: 2026-11-03T13:59:27.328469+00:00
Method: recomputed pooled ROC-AUC and all five fold AUCs from saved predictions; aligned rows by SK_ID_CURR; checked saved TARGET against supplied train.csv TARGET; checked fold IDs.

                    variant  pooled_oof_auc  mean_fold_auc  std_fold_auc  fold_1_auc  fold_2_auc  fold_3_auc  fold_4_auc  fold_5_auc  claimed_historical_auc  absolute_difference_from_claim  reproduced
               raw_baseline        0.760658       0.760729      0.003770    0.757195    0.761577    0.766257    0.757183    0.761433                0.760658                             0.0        True
   existing_domain_formulas        0.760372       0.760452      0.003521    0.757389    0.762220    0.765294    0.756762    0.760597                0.760372                             0.0        True
     peer_relative_features        0.760693       0.760800      0.003309    0.758953    0.762216    0.764975    0.756288    0.761570                0.760693                             0.0        True
external_score_interactions        0.760058       0.760178      0.003202    0.757824    0.760804    0.764583    0.756380    0.761300                0.760058                             0.0        True
           combined_phase2b        0.759909       0.759994      0.003633    0.757609    0.760212    0.765481    0.755942    0.760727                0.759909                             0.0        True

All five references reproduced within 1e-10. This verifies saved prediction metrics and row/label alignment, not an independent rerun of historical model training.
