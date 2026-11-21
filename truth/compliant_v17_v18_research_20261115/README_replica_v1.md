# V17/V18 compliant research continuation

This is a separate, additive research workspace. It does not alter the V17,
V18, V22, previous experiment folders, or the user's submission files.

## What was tested

- `run_clean_v17v18_catboost_probe.py`: five-fold CUDA CatBoost using only the
  official training CSV, row-wise transformations, and constant-free product
  ratio categories. It excludes fixed pricing offsets, train/test pooling,
  repeat-applicant label transfer, and historical prediction files as features.
- `analyze_clean_catboost_aligned_blend.py`: checks ID, target, and fold
  alignment, then compares CatBoost to a Phase 33/34 baseline using the exact
  same five folds.
- `run_oof_regime_diagnostics.py`: checks row/label/fold alignment in selected
  saved OOF artifacts, computes fixed rank blends, and reports AUC within
  feature-defined strata. It is diagnostic work on previously examined labels,
  not an independent test. Its Phase 37 reference averages predictions from
  different fold assignments, so those blends are secondary diagnostics, not
  strict fold-aligned evidence.

## Reproduce

```powershell
python D:\gci\truth\compliant_v17_v18_research_20261115\run_clean_v17v18_catboost_probe.py
python D:\gci\truth\compliant_v17_v18_research_20261115\analyze_clean_catboost_aligned_blend.py
python D:\gci\truth\compliant_v17_v18_research_20261115\run_oof_regime_diagnostics.py
```

The CatBoost runner uses CUDA and fails rather than falling back to CPU. It
writes its OOF artifacts to `runs\clean_catboost_constant_free_v1`. The
aligned analysis writes to `runs\fold_aligned_catboost_blend_v1`; the
historical diagnostic runner writes to
`runs\historical_oof_diagnostics_clean_v1`. Existing run directories are
never overwritten. Run the commands in a fresh copy, or update each script's
output path to a new directory before rerunning in this workspace.

## Main result

The same-fold Phase 33/34 baseline reached mean-fold OOF ROC-AUC **0.765360**;
the clean CatBoost expert reached **0.765945**; and a fixed 50/50 within-fold
rank blend reached **0.767492** (+0.002132 over the aligned baseline), higher
in all five folds. An earlier Phase 37 comparison used a multi-seed OOF
reference whose seeds had different fold assignments; it is withdrawn as the
primary comparison and documented as secondary diagnostic evidence in the
report.

These are exploratory results on labels used repeatedly during development,
not an independent generalization estimate, test-set score, or leaderboard
result. At this first research pass, no test submission was created or
uploaded; a later hash-locked candidate is documented below.

See `RESEARCH_FINDINGS.md` for the rule audit, correction details, exact
fold-aligned comparisons, subgroup diagnostics, limitations, and next-step
protocol.

## Follow-up: stricter submission run and mathematical experiments

The preserved four research Markdown files remain in place. A separate v2
generator was added; it verifies the expected SHA-256 values for the official
train, test, and sample files before reading them, keeps CatBoost train/test
frames separate, and runs test-covariate and validation-label perturbation
checks. The checks passed. The generated candidate is
`ideas_v1/submissions/product_te_20pct_v2/submission_compliant_product_te20_v2.csv`.
Its regenerated mean-fold OOF AUC was **0.767756** versus **0.767450** for
its freshly regenerated reference (+**0.000305**, positive on all five
folds). The file passed schema, ID-order, uniqueness, finite-value, range,
and hash checks. It is not a leaderboard score; no upload was made.

Two further mathematical tests were run. Balancing the positive Logloss
weight to the fold-specific negative/positive count ratio (about 11.387)
gave a weaker standalone CatBoost and only a +0.000140 mean-fold fixed-blend
gain on 4/5 folds. A sampled pairwise-logistic ranker, whose training pairs
were formed only from each outer training fold, produced a weaker standalone
model but a +0.000204 90/10 blend gain on 5/5 folds. A fixed three-way
80/10/10 blend was only +0.000057 above TE20 and lost on one fold; its
conditional paired interval included zero, so it was not promoted to a new
submission. The details and limitations are in `ideas_v1/IDEA_EXPERIMENTS_v1.md`.

The exact v2 generator was run twice. Both runs passed the OOF gate:
0.767756 (+0.000305) and 0.767772 (+0.000303), each positive on all five
folds. The submissions were highly rank-correlated (Spearman 0.999990) but
not byte-identical; maximum absolute `TARGET` difference was 0.01333, with
variation isolated to the GPU CatBoost stream. Keep the first v2 candidate as
the primary artifact rather than selecting the slightly higher replay after
seeing its score. Seeds, versions, input/code hashes, and output hashes are
recorded, but exact GPU replay is not guaranteed. The competition's official
interpretation of every modeling choice cannot be certified from local tests
alone.

## Latest pairwise-AUC research update

The next rule-conscious experiment tested global ranking objectives. A GPU
CatBoost `PairLogit` model did not improve the TE20 blend. A GPU XGBoost
`rank:pairwise` model, trained only on each outer training fold, produced a
positive fixed 90/10 rank blend on all five folds. The full candidate
generator regenerated every component and confirmed a mean-fold OOF AUC of
**0.767979**, versus **0.767743** for its regenerated TE20 baseline
(**+0.000236**, 5/5 folds positive). The new candidate is
`compliant_v17_v18_research_20261115/ideas_v1/submissions/te20_xgb_rank_pairwise_v1/submission_compliant_te20_xgb_rank_pairwise_v1.csv`.
It is a local OOF result, not a leaderboard score; no upload was made, and
**0.79 has not been reached or guaranteed**. The AUC-pair decomposition,
per-fold results, references, and strict-rule caveats are appended in
`ideas_v1/IDEA_EXPERIMENTS_v1.md`.
