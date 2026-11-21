# EXP02 - Financial behaviour and affordability

## Decision

**NEEDS_INDEPENDENT_SEED_REPLICATION**. The independent-seed replication is not complete, so this is not an accepted model or submission candidate. No submission was created. Official leaderboard score: **NOT EVALUATED**.

## Hypothesis and rule compliance

A predeclared financial-ratio block was compared with the same raw-feature CatBoost model. Inputs matched the course package hashes. No external data, undocumented fee/product constants, validation/test covariate fitting, target encodings, manual prediction overrides, or public-score tuning were used.

## Results

| Metric | Raw-feature baseline | Financial feature model | Delta |
|---|---:|---:|---:|
| Pooled 5-fold OOF AUC | 0.75126491 | 0.75849375 | +0.00722884 |
| Mean paired fold AUC | - | - | +0.00728454 |

| Fold | Raw AUC | Financial AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.74786378 | 0.75753725 | +0.00967347 |
| 2 | 0.75098639 | 0.75853291 | +0.00754652 |
| 3 | 0.75669424 | 0.76189240 | +0.00519816 |
| 4 | 0.74767490 | 0.75423617 | +0.00656127 |
| 5 | 0.75382536 | 0.76126866 | +0.00744330 |

Screen: positive pooled delta = True; mean paired delta >= 0.0002 = True; positive folds = 5/5. A pass still requires independent-seed replication.

## Protocol and leakage checks

Five stratified outer folds (seed 42) share inner 10% stopping splits. Iteration selection uses only the inner split inside each outer training fold; final models refit on the full outer training fold. Validation labels are not used in early stopping. Feature formulas and checks are in `FEATURE_MANIFEST.json`.

## Hardware and integrity

GPU: `NVIDIA GeForce RTX 4050 Laptop GPU`. During-training polling: 736 samples; max sampled utilization 99%; max sampled VRAM 5347 MiB of 6141 MiB. These are sampled values, not guaranteed true peaks. Total run time: 792.7 seconds.

Input, notebook source, and artifact SHA-256 values are in `SHA256_MANIFEST.json`. The notebook independently recomputed folds, OOF metrics, saved arrays, and test/sample ID alignment. It generated test prediction arrays only; no submission CSV.

## Next step

If the predeclared screen passes, run the frozen pipeline with a new outer seed as an independent replication. Otherwise reject this feature block. Do not select or edit rows based on leaderboard feedback.
