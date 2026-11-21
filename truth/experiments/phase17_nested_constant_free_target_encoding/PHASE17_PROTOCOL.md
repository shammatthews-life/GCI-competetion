# Phase 17 — nested target statistics on constant-free product keys

## Research question

Do smoothed, nested target rates for the four constant-free ratio keys
Q3/Q4/Q5/AG3 add predictive value over the same LightGBM pipeline with raw,
rowwise, one-hot product keys and fold-fitted key counts?

## Frozen design

- Train only: `D:\gci\inpuy\train.csv`.
- Outer folds: five stratified folds, seed `20261113`.
- Constant-free keys only: `PRODUCT_KEY_Q3`, `PRODUCT_KEY_Q4`,
  `PRODUCT_KEY_Q5`, and `PRODUCT_KEY_AG3`. Do not use the 130.5 annuity
  offset or 0.0066 financing-step keys.
- Matched model variants: same raw and rowwise features, same four product
  keys with fold-fitted frequency-supported OHE, same LightGBM settings,
  same iteration count and seed per outer fold. The treatment adds eight
  features: smoothed target rates and log counts for each key.
- Smoothing strength: alpha 30, fixed from the prior Phase 8 protocol.
- For each outer training set, target rates for training rows are inner-fold
  OOF with five stratified folds; each row's label is excluded from its map.
  Outer validation rates use maps built only from the complete outer training
  partition. The outer validation target is never used in feature fitting.
- A 10% early-stopping partition inside each outer training set selects the
  iteration count for both matched variants using the baseline feature set
  only. The final candidate and baseline are refit on the complete outer
  training set using that same selected count.

## Outcomes and limits

Primary comparison is pooled outer OOF ROC-AUC of the target-statistic model
minus its matched baseline, with paired fold AUCs and a class-stratified
bootstrap interval. This is exploratory model development because all labels
have previously been inspected.

The target statistics are label-derived and remain in a separate research
track pending explicit rule clarification. This protocol avoids formula
hypotheses with undocumented constants but does not itself establish
competition eligibility. No test rows, external data, submissions, or
champion changes are permitted.
