# Strictly inductive AUC research

This folder is a separate workspace for new, rule-conscious model experiments.
It does not modify or replace historical submissions, the V22 reference, or
previous experiments.

## Research boundary

- Read only the organizer-provided `train.csv` for this first experiment.
- Do not use external data, prior prediction files, target encoding, peer-label
  transfer, pooled train/test statistics, manually edited predictions, or
  empirically chosen prediction cutoffs.
- Fit category vocabularies and frequencies on each applicable training
  partition. Keep outer validation rows out of preprocessing, stopping,
  hyperparameter selection, and model fitting.
- Report cross-validation results as development evidence, not leaderboard
  scores. The project has substantial historical label exposure, so this is
  not an untouched generalization estimate.

## First experiment: nested positive-class weighting

`run_v1_class_weight.py` tests the AUC/imbalance hypothesis by comparing
`scale_pos_weight` values 1, 2, 4, and 8. For each outer fold, a stratified
inner split selects the weight by validation ROC-AUC; iteration count is
selected on that same inner split. Every candidate is then refit on the full
outer training fold and scored on the untouched outer fold. The script also
reports each fixed weight separately and records row-level OOF predictions.

The script uses the existing local Phase 33 feature builder and Phase 38
fold-fitted matrix/model helpers. Their source paths and hashes are recorded
in the run summary. No test CSV or historical prediction artifact is read.

## Run

```powershell
python D:\gci\truth\compliant_inductive_auc_20261114\run_v1_class_weight.py
```

By default, each run writes into a new UTC-stamped directory under `runs`.
The runner refuses to overwrite an existing output directory and fails if
XGBoost cannot train on the configured CUDA device.

## Interpretation

The fold-selected candidate estimates the full nested selection procedure;
it is not a fixed-weight model. Compare it with the fixed-weight candidates
using paired outer-fold AUCs and pooled OOF AUC. Do not select or report the
largest fixed-weight OOF number as an unbiased winner. Any promising result
requires a predeclared confirmation run and still cannot guarantee a public
or private leaderboard score.

The nested-selection OOF predictions can come from different selected
weights in different folds. Because ROC-AUC is sensitive to ranking across
all pooled rows, between-fold score-scale shifts can distort its pooled AUC.
For that adaptive candidate, mean outer-fold AUC is the primary metric; the
raw pooled value is retained only with an explicit scale-sensitivity warning.

## Second experiment: fixed rank blend

`run_v2_rank_blend.py` compares a pre-existing clean Phase 37 multi-seed
LightGBM/XGBoost OOF component with the V1 fixed-weight-2 OOF component.
It validates exact ID, target, and fold alignment, then compares a fixed
50/50 probability average with a fixed 50/50 within-fold rank average.
Blend weights are not tuned on the OOF labels. This tests whether rank-scale
normalization helps combine diverse predictions without training on any
outer-validation label. It produces OOF evidence only; it does not generate
test predictions or a submission.

## Third experiment: application-process features

`run_v3_application_process.py` adds row-wise missingness indicators and
missingness-by-age/contract interactions to the fold-safe Phase 33/38 feature
pipeline. These features use no labels or other applicants' records. It
reports the model alone and predeclared fixed 50/50 probability/rank blends
with the aligned Phase 37 OOF reference. All blend metrics are mean outer-fold
AUC; no blend weights are tuned.

## Reproducible test inference

`build_submission_v1.py` fits the fixed Phase 37-style four-model ensemble
and the V3 application-process XGBoost component on all official training
rows, then produces their fixed 50/50 probability blend for the official test
rows. It checks all three official input hashes, fits encoders/frequencies on
training rows only, chooses iterations from an inner training split, requires
CUDA for XGBoost, and validates the exact sample-submission schema and ID
order. It writes the candidate CSV and a hash/config summary under a new
output directory. No leaderboard submission is performed.

```powershell
python D:\gci\truth\compliant_inductive_auc_20261114\build_submission_v1.py `
  --output-dir D:\gci\truth\compliant_inductive_auc_20261114\runs\new_submission_run
```

## References

- Official competition source: `Hareen-Liyanage/GCI_World_Competition`,
  `README.ipynb` (data restrictions, hand-labeling, reproducibility,
  evaluation and submission requirements).
- XGBoost: Chen and Guestrin, “XGBoost: A Scalable Tree Boosting System,”
  KDD 2016.
