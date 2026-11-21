# Phase 58 — fold-local LightGBM and rowwise features

## Protocol

- Five stratified folds, seed `20261113`; fixed `1400` rounds.
- One-hot category dictionaries are fit on each outer training fold only; unseen validation/test categories map to all-zero indicators.
- Baseline uses raw supplied predictors; candidate adds deterministic row-local ratios, score summaries, time ratios, bureau summaries, and Q3.
- No target encoding, cross-row aggregates, test pooling, entity matching, external inputs, or CSV blending.

## Ranking results

| Model | Pooled AUC | Average precision | Log loss | Fold AUCs |
|---|---:|---:|---:|---|
| Raw one-hot baseline | 0.751397545 | 0.235770552 | 0.248063391 | 0.754435, 0.757618, 0.747891, 0.747682, 0.749414 |
| Rowwise + Q3 candidate | 0.761907220 | 0.246221682 | 0.245133296 | 0.765715, 0.769107, 0.757220, 0.759639, 0.758078 |

- Candidate minus baseline AUC: `+0.010509675`; positive folds `5/5`.
- Paired conditional bootstrap interval: `[+0.008872683, +0.012100391]`.

## User-reported leaderboard result

The user reports a leaderboard score of `0.76861` for this submission. This
score has not been independently checked against the competition platform.
It is `+0.00129` above the user-reported Phase 56 score (`0.76732`) and
`+0.00333` above Phase 57 (`0.76528`), but remains `0.01139` below the
`0.78000` goal. The Phase 58 OOF AUC (`0.761907`) is slightly below Phase
56's OOF AUC (`0.762144`), so the higher reported leaderboard result is not
explained by the local CV comparison. Do not compare the leaderboard score
directly with historical OOF scores.

## OOF confusion and threshold diagnostics

Confusion counts are `[TN, FP, FN, TP]`. Precision/recall are evaluated only on OOF rows.

| Model | Threshold rule | TN | FP | FN | TP | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | fixed_threshold_0.5 | 157103 | 278 | 13513 | 308 | 0.525597 | 0.022285 |
| baseline | outer_train_prevalence | 110986 | 46395 | 4637 | 9184 | 0.165242 | 0.664496 |
| baseline | top_10_percent_per_fold | 144821 | 12560 | 9259 | 4562 | 0.266441 | 0.330077 |
| candidate | fixed_threshold_0.5 | 157066 | 315 | 13444 | 377 | 0.544798 | 0.027277 |
| candidate | outer_train_prevalence | 111998 | 45383 | 4506 | 9315 | 0.170299 | 0.673974 |
| candidate | top_10_percent_per_fold | 145007 | 12374 | 9073 | 4748 | 0.277304 | 0.343535 |

These threshold metrics diagnose operating points and class imbalance; they do not replace ROC-AUC. The top-decile rule measures how many positives are captured among the highest-scored 10% of each fold, without selecting a threshold from validation labels.

## Limitations

All labels/folds have prior project exposure. The bootstrap is conditional on fixed OOF predictions and does not include model/research-selection uncertainty. The leaderboard score above is user-reported, not independently verified.

- Runtime seconds: `440.41`.
- Source SHA-256: `8dad43df1ea59408575ddd374e715eeeb4be03bf3c86f32cce9beacf52115342`.
