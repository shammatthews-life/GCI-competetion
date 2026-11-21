# Phase 57 — row-local feature CatBoost ablation

## Research question

Does a small, predeclared set of row-local financial, timing,
external-score, inquiry, and coarse Q3 features improve a matched
one-hot CatBoost baseline without cross-row or target-derived
feature construction?

## Protocol

- 5 stratified folds; seed `20261113`.
- CatBoost `GPU`, `1000` fixed iterations; no outer-fold early stopping.
- CatBoost one-hot encodes every categorical column (checked cardinality <= `256`); no categorical target statistics are needed.
- Baseline uses raw supplied predictor columns; candidate adds only deterministic row-local formulas.
- All transforms use the current row only. Each fold model sees labels from its training rows only.
- No test/train pooling, explicit target encoding, entity matching, prediction caches, external data, or CSV blending.

## Results

| Arm | Pooled AUC | Mean fold AUC | Log loss | Fold AUCs |
|---|---:|---:|---:|---|
| Raw one-hot baseline | 0.752315081 | 0.752335369 | 0.247856594 | 0.755515, 0.757223, 0.750426, 0.748787, 0.749726 |
| Rowwise + Q3 candidate | 0.760615629 | 0.760668784 | 0.245440330 | 0.764067, 0.767216, 0.756756, 0.757096, 0.758208 |

- Candidate minus baseline pooled AUC: `+0.008300548`; positive folds `5/5`.
- Log-loss change (candidate minus baseline): `-0.002416264`.
- Conditional paired-bootstrap interval: `[+0.006743061, +0.009835467]`.

## Interpretation

The candidate improved pooled AUC and was positive in at least four folds; retain it only as an exploratory feature set.
Treat this as exploratory: the project labels and folds have prior exposure, and the bootstrap is conditional on fixed OOF predictions. The leaderboard score is unknown until a submission is evaluated; public-score feedback is not a substitute for untouched validation.

- Runtime seconds: `284.87`.
- Runner SHA-256: `29d3823e7f11ba5ba80906bc19b763f62a5665132bda406a2f7b43e2d9773940`.
