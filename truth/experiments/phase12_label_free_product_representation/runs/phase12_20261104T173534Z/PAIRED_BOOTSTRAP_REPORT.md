# Paired bootstrap report

Intervals are paired, stratified row bootstraps of the saved OOF predictions. They are conditional on fitted models and previously examined labels, not independent confirmation.

| Comparison | Paired delta | 95% low | 95% high | Repetitions |
|---|---:|---:|---:|---:|
| raw_numeric_vs_raw_numeric | +0.000000 | +0.000000 | +0.000000 | 1000 |
| raw_plus_keys_ohe_vs_raw_numeric | +0.015473 | +0.013742 | +0.017175 | 1000 |
| raw_plus_keys_frequency_ohe_vs_raw_numeric | +0.016559 | +0.014728 | +0.018325 | 1000 |
| raw_plus_decompositions_vs_raw_numeric | +0.010569 | +0.009271 | +0.011878 | 1000 |
| raw_plus_interactions_vs_raw_numeric | -0.000466 | -0.001253 | +0.000368 | 1000 |
| raw_plus_keys_decompositions_interactions_vs_raw_numeric | +0.015165 | +0.013355 | +0.016825 | 1000 |
| raw_plus_keys_hash_vs_raw_numeric | +0.016042 | +0.014329 | +0.017698 | 1000 |
