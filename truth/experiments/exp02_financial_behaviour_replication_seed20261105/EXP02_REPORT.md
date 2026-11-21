# EXP02 - Financial behaviour and affordability

## Decision

**NEEDS_INDEPENDENT_SEED_REPLICATION**. The independent-seed replication is not complete, so this is not an accepted model or submission candidate. No submission was created. Official leaderboard score: **NOT EVALUATED**.

## Hypothesis and rule compliance

A predeclared financial-ratio block was compared with the same raw-feature CatBoost model. Inputs matched the course package hashes. No external data, undocumented fee/product constants, validation/test covariate fitting, target encodings, manual prediction overrides, or public-score tuning were used.

## Results

| Metric | Raw-feature baseline | Financial feature model | Delta |
|---|---:|---:|---:|
| Pooled 5-fold OOF AUC | 0.75170772 | 0.75804387 | +0.00633614 |
| Mean paired fold AUC | - | - | +0.00629682 |

| Fold | Raw AUC | Financial AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.74979512 | 0.75670954 | +0.00691442 |
| 2 | 0.75445683 | 0.76087749 | +0.00642066 |
| 3 | 0.75561479 | 0.76014279 | +0.00452800 |
| 4 | 0.75040178 | 0.75646123 | +0.00605945 |
| 5 | 0.74866441 | 0.75622599 | +0.00756158 |

Screen: positive pooled delta = True; mean paired delta >= 0.0002 = True; positive folds = 5/5. A pass still requires independent-seed replication.

## Protocol and leakage checks

Five stratified outer folds (seed 42) share inner 10% stopping splits. Iteration selection uses only the inner split inside each outer training fold; final models refit on the full outer training fold. Validation labels are not used in early stopping. Feature formulas and checks are in `FEATURE_MANIFEST.json`.

## Hardware and integrity

GPU: `NVIDIA GeForce RTX 4050 Laptop GPU`. During-training polling: 668 samples; max sampled utilization 96%; max sampled VRAM 5126 MiB of 6141 MiB. These are sampled values, not guaranteed true peaks. Total run time: 714.6 seconds.

Input, notebook source, and artifact SHA-256 values are in `SHA256_MANIFEST.json`. The notebook independently recomputed folds, OOF metrics, saved arrays, and test/sample ID alignment. It generated test prediction arrays only; no submission CSV.

## Next step

If the predeclared screen passes, run the frozen pipeline with a new outer seed as an independent replication. Otherwise reject this feature block. Do not select or edit rows based on leaderboard feedback.
