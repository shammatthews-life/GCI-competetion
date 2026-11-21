# Phase 25 - EXP02 fold-safe, train-only feature ablation

- Input train CSV SHA-256: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- Rows: 171202; positive rate: 0.08072920.
- Protocol: 5-fold stratified OOF, seed 20261113; CatBoost GPU, 700 iterations, depth 6, learning rate 0.05.
- Fixed parameters across folds; no validation-fold early stopping, feature selection, or blend fitting.
- Read train.csv only; no test file, external data, cached predictions, or submission was used or written.
- Excluded train-label-to-test linkage, UID/identity label features, test/test smoothing, pooled train/test statistics, and the unverified 129.5/0.0132/130.5/0.0066 product constants. The matched base model uses only constant-free Q3/Q4/Q5/AG3 ratio keys.
- Formula/peer features are deterministic or reference only each outer fit fold. Product-ratio target encodings use inner OOF values for outer-fit rows and mappings learned only from the outer-fit labels.
- The local compliance matrix allows nested target encoding with leakage controls, but the official README does not explicitly address it; treat that candidate as research-only until organizer clarification.
- Formula/peer features are annuity/credit, its interaction with the provided EXT_SOURCE mean, and an occupation peer percentile learned from each fit fold. No inferred loan terms, guessed rate anchors, or test-population statistics are used.
- No blend weights were fitted. Prior label exposure and multiple candidate comparisons make these exploratory results, not independent confirmation or submission approval.

| Candidate | Pooled AUC | Mean fold AUC | Pooled delta | Mean fold delta | Folds improved |
|---|---:|---:|---:|---:|---:|
| baseline | 0.765682908 | 0.765688795 | +0.000000000 | +0.000000000 | 0/5 |
| formula_peer | 0.766157572 | 0.766168920 | +0.000474664 | +0.000480124 | 3/5 |
| ratio_target_encoding | 0.765970451 | 0.766039189 | +0.000287543 | +0.000350394 | 5/5 |
| combined | 0.765822543 | 0.765876192 | +0.000139636 | +0.000187397 | 2/5 |

- Runtime: 321.9 seconds.
- GPU utilization before run: 0%.
- GPU free before run: 5648 MiB.
