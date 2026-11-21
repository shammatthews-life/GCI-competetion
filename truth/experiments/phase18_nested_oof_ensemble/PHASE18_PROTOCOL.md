# Phase 18 — nested same-fold CatBoost / target-encoded LightGBM blend

## Question

Does a blend of Phase 15 native-category CatBoost and Phase 17 constant-free
key target-encoded LightGBM improve when the blend weight is selected only
inside each outer training partition?

## Frozen predictions and folds

- Phase 15 CatBoost and Phase 17 LightGBM OOF files must both align exactly
  to train IDs, targets, row order and Phase 13 outer folds (seed 20261113).
- The outer validation predictions are those saved component OOF predictions.
- Outer validation labels are not used in selecting blend weights.

## Nested weight selection

For each outer training partition, generate three-fold inner OOF predictions
from:

1. GPU CatBoost with the fixed Phase 15 settings (700 iterations, depth 6,
   learning rate 0.05, one-hot threshold 10, CTR complexity 1).
2. LightGBM with Phase 17's constant-free Q3/Q4/Q5/AG3 target-rate/log-count
   features. Its iteration count is selected using a 10% early-stop split
   inside each inner-fit partition; all target maps are fit only on inner
   training labels, with inner-fit rows themselves encoded OOF.

Choose the best pooled inner OOF AUC among fixed CatBoost weights
`0.00, 0.25, 0.50, 0.75, 1.00`. Apply that fold's weight to its saved
outer-validation component predictions. All weight choice is isolated from
that outer fold's labels.

## Evaluation and constraints

Report pooled outer OOF AUC, fold metrics, selected inner weights, paired
bootstrap vs each component, and artifact integrity. This nested estimate
improves on the post-hoc blend diagnostic but remains exploratory: the same
labels have been studied in prior phases. Phase 15 uses unresolved product
formula hypotheses and CatBoost CTRs; Phase 17 uses label-derived target
statistics whose rule status is unresolved. The blend is not eligible or a
champion. No test rows, external data, submissions, or champion changes.
