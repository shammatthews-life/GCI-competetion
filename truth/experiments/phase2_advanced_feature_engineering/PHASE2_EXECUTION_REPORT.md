# Phase 2 execution report — corrected validation gate

- Status: completed at 2026-11-03T08:38:11Z
- Train/test dimensions: (171202, 34) / (61500, 33)
- Runtime seconds: 124.48
- Input train SHA-256: b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1
- Input test SHA-256: dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65

## Corrected comparison

                        variant  mean_fold_auc  std_fold_auc  pooled_oof_auc  delta_pooled_vs_baseline
                   raw_baseline       0.760729      0.003770        0.760658                  0.000000
baseline_plus_existing_formulas       0.760452      0.003521        0.760372                 -0.000287

The numbers above are the valid result of this experiment; prior OOF-like metrics from outer-fold early stopping are invalid for unbiased comparison. These results are local OOF measurements, not leaderboard scores.

## Formula interpretation cautions

- Annualizing annuity is a constant rescaling of an existing baseline feature, not a new domain concept.
- Family-size multiplication assumes equal allocation of the payment burden and is not retained as a preferred feature hypothesis.
- Bureau windows are retained separately; their meanings do not justify treating them as rates over equivalent time intervals.
- Relative external-score dispersion is undefined with fewer than two observed scores and can be unstable when its mean is close to zero; missingness and distribution are recorded for follow-up.
- REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE has uncertain semantics and is exploratory only.

## Next gate

Only after reviewing the corrected baseline and this predeclared formula comparison should feature-discovery screening begin. The full 500-hypothesis request is not treated as a quota and has not been started.