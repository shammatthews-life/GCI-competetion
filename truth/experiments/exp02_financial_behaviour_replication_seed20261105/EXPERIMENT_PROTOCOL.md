# EXP02 independent seed replication

## Status before execution

Predeclared replication of the existing `EXP02_FINANCIAL_BEHAVIOUR` result.
The original notebook, reports, run arrays, champion, and submissions are not
modified. This replication uses a fresh experiment directory and a fresh run
ID. The model will be trained only after this protocol is recorded.

## Hypothesis

The previously tested seven-feature financial-ratio block will improve a
matched raw-feature CatBoost model on a new set of outer folds. This tests
replication of that block-level result; it does not identify which individual
ratio contributes or establish improvement over the GCI champion.

## Frozen data and features

Use only the supplied `train.csv`, `test.csv`, and `sample_submission.csv`
with SHA-256:

- `train.csv`: `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- `test.csv`: `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfcd525f7c08de15c6d2d42c65`
- `sample_submission.csv`: `6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db`

The candidate adds only these deterministic rowwise ratios:

1. `AMT_CREDIT / AMT_INCOME_TOTAL`
2. `AMT_ANNUITY / AMT_INCOME_TOTAL`
3. `AMT_GOODS_PRICE / AMT_CREDIT`
4. `AMT_CREDIT / AMT_ANNUITY`
5. `AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`
6. `AMT_CREDIT / CNT_FAM_MEMBERS`
7. `AMT_ANNUITY / CNT_FAM_MEMBERS`

Non-finite inputs and non-positive denominators map to missing. No annualizing
constant, product/fee hypothesis, external data, pooled train/test statistic,
target encoding, feature selection, or manual prediction rule is introduced.
The amount ratios are empirical predictors; dictionary semantics do not
establish annuity frequency or equal household allocation.

## Frozen evaluation

- Baseline and candidate use identical features except for the seven additions.
- Model and CatBoost parameters are copied unchanged from EXP02.
- Five stratified outer folds, shuffled, outer seed `20261105`.
- Inner stopping seed for fold `k`: `20262105 + k`.
- Matched model seed for fold `k`: `20263105 + k`.
- Early stopping uses only a 10% stratified stopping subset inside each outer
  training fold. Both variants share the same inner split and model seed.
- Refit each model on its complete outer training fold at its selected
  iteration count. The outer validation fold is used only for scoring.
- No hyperparameter search, blend optimization, public leaderboard feedback,
  or changes after viewing the replication results.

## Predeclared decision rule

The replication supports the prior signal only if the candidate has a positive
pooled OOF AUC difference, mean paired fold improvement of at least `+0.0002`,
and improves at least 4 of 5 folds. A pass is still not a champion result:
the seven features must subsequently be tested against the champion using
identical folds and a matched model protocol. A failure is retained and
reported without tuning this replication.

No submission will be produced. All predictions, fold assignments, reports,
logs, and hashes belong only in this replication directory.

## Pre-run correction — input hash transcription

The preregistered `test.csv` hash above contains a transcription error. Before
training, direct SHA-256 checks of both the project and official-package copies,
and the existing EXP02 run manifest, agree on:

`dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`

The train and sample-submission hashes are unchanged. No input file changed;
only the recorded test hash is corrected. This correction was made before
training and does not change the model, folds, features, or decision rule.
