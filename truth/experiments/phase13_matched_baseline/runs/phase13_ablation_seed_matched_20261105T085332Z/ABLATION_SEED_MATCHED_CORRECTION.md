# Phase 13 matched-seed key ablation correction

- UTC start: 2026-11-05T08:53:32.080619+00:00
- Original Phase 13 run: `experiments\phase13_matched_baseline\runs\phase13_20261105T075946Z`
- Original OOF SHA-256: `775f11742e2be8f1bfdf75d4a34c7da7603788fb0201ec3052990cf471e6c488`; train SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Rows: 171,202; seed: 20261113; folds: 5.
- Refit each key subset with the same fold-specific LightGBM seed, D-selected configuration, and D-selected iteration count for that fold. Only the key columns vary across ablation variants.
- This rerun corrects the earlier ablation-only seed mismatch. It does not alter the primary A–F benchmark, selected configurations/rounds, controls, or secondary-seed run.
- All transformations are fitted on each outer-fit partition; all rows are predicted exactly once out of fold.

| Variant | Pooled OOF AUC | Mean fold delta vs all eight | Folds better than all eight |
|---|---:|---:|---:|
| leave_out_KQ4 | 0.765127 | +0.000159 | 3/5 |
| leave_out_P5 | 0.765116 | +0.000134 | 3/5 |
| leave_out_Q5 | 0.765019 | +0.000035 | 2/5 |
| all_eight | 0.764983 | +0.000000 | 0/5 |
| preregistered_P4_Q4_AG3 | 0.764947 | -0.000028 | 1/5 |
| leave_out_AG3 | 0.764922 | -0.000047 | 2/5 |
| leave_out_Q4 | 0.764919 | -0.000046 | 2/5 |
| leave_out_Q3 | 0.764787 | -0.000180 | 3/5 |
| leave_out_KQ3 | 0.764678 | -0.000302 | 3/5 |
| leave_out_P4 | 0.764585 | -0.000395 | 2/5 |
| preregistered_P4_P5_Q4 | 0.764372 | -0.000604 | 2/5 |
| preregistered_Q3_Q4_Q5 | 0.763559 | -0.001407 | 1/5 |
| single_Q3 | 0.762941 | -0.002032 | 0/5 |
| single_P4 | 0.762668 | -0.002319 | 0/5 |
| single_P5 | 0.761895 | -0.003087 | 0/5 |
| single_Q4 | 0.761771 | -0.003219 | 0/5 |
| single_KQ3 | 0.761454 | -0.003530 | 0/5 |
| single_AG3 | 0.760666 | -0.004309 | 0/5 |
| single_Q5 | 0.760504 | -0.004466 | 0/5 |
| single_KQ4 | 0.760418 | -0.004548 | 0/5 |

This is a methodological correction to the Phase 13 ablation sub-analysis. The original run artifacts are preserved. These reused OOF labels remain exploratory and are not independent confirmation.
