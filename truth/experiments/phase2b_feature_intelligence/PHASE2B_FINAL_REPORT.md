# Phase 2B — first feature-intelligence results

Execution timestamp: 2026-11-03T09:58:03Z

## Results

                    experiment  mean_fold_auc  std_fold_auc  pooled_oof_auc  delta_pooled_vs_baseline  folds_improved_vs_baseline
                  raw_baseline       0.760729      0.003770        0.760658                  0.000000                           0
         external_interactions       0.760178      0.003202        0.760058                 -0.000600                           1
               peer_deviations       0.760800      0.003309        0.760693                  0.000034                           3
combined_peer_and_interactions       0.759994      0.003633        0.759909                 -0.000750                           1

## Decision

No family is promoted as a confirmed improvement. The peer-deviation family's pooled delta is only +0.000034, while the interaction and combined families are negative. The tiny peer difference is below a practically meaningful threshold and may be fold noise; it needs a separate predeclared confirmation before any retention decision. Do not select or submit these features based on this run.

## Scope not yet executed

Clustering, categorical frequency/cross features, comprehensive per-column hypothesis generation, broad nonlinear screening, residual-driven discovery, model comparison, and ensemble optimization remain pending. This run registered 17 concrete candidates, not 300–500. No claim is made that Phase 2B is complete or that the 0.80 target is reached.
