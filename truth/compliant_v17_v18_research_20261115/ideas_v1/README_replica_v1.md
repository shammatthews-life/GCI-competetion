# Rules-conscious improvement experiments

This folder contains separate experiments building on the corrected,
same-fold Phase 33/34 + clean CatBoost reference. New scripts do not overwrite
the earlier research or run artifacts.

## What was tested

1. `run_clean_catboost_second_seed.py` — a second GPU CatBoost run with the
   same features and hyperparameters but a fixed, independent seed.
2. `run_clean_catboost_external_geometry_v1.py` — row-wise external-score
   ratios and score-order category, using only the supplied columns.
3. `run_nested_product_te_lgbm_v1.py` — a LightGBM view with four
   constant-free product-key target rates and counts. Training-row target
   statistics are inner-OOF within each outer training fold; each validation
   map is fit only on the outer training rows.
4. `analyze_ideas_v1.py` — fixed, within-fold rank blend comparisons of the
   new OOF predictions against the prior aligned baseline.
5. `create_compliant_submission_te20_v1.py` — independently regenerates the
   four model views from official inputs, computes OOF checks, and writes a
   submission candidate only if its OOF mean-fold AUC improves and all five
   fold deltas are positive.

## Current screening result

The prior aligned blend scored mean-fold ROC-AUC **0.767492**. The best
candidate in the finite exploratory screen was an 80/20 rank blend of that
reference and the nested target-encoding LightGBM: **0.767790**, a
**+0.000298** mean-fold delta, positive on all five folds. The 10% target
encoding blend also improved on all five folds (+0.000201). A 50% addition
was worse, so more target-encoding weight is not automatically better.

The best fixed weight was selected after comparing 10%, 20%, and 50% on the
same repeatedly studied folds. This introduces model-selection optimism.
The result is a submission experiment, not an independent estimate, a
leaderboard score, or proof of a lift.

### Latest regenerated candidate and additional multi-view screen

The submission-generation script was run end-to-end on the official training
and test files. Its recomputed five-fold OOF baseline was **0.767477** and its
80/20 candidate was **0.767778** (+**0.000301**), with positive deltas on all
five folds. The change from the earlier screening estimate (**0.767790**) is a
small difference from the fresh model regeneration; use the manifest and OOF
artifacts from the generated candidate as the authoritative record of that
run. The CSV passed checks for the exact `SK_ID_CURR,TARGET` columns, 61,500
unique IDs in sample order, finite predictions, and values in `[0,1]`.

A further GPU experiment trained separate global external-score, loan, and
person feature views. Their standalone mean-fold AUCs were **0.712330**,
**0.692380**, and **0.672145**. An equal-rank combination added at 10% to the
prior reference gained **+0.000078** on average but improved only 4/5 folds;
20% and 30% additions were worse. This exploratory result was not used to
replace the generated submission candidate. Both screens reused examined
labels, so neither gain is an independent estimate or a leaderboard score.
The user must upload the candidate to learn its competition score.

See `IDEA_EXPERIMENTS_v1.md` for all tested metrics, fold deltas, and
limitations. The submission generator does not read cached prediction files
as model inputs and does not upload anything.

## Reproduction

Run the three model scripts, then the analysis script. The run directories
are fixed and scripts refuse to overwrite them; use a fresh copy or choose
new output paths before rerunning.

```powershell
python D:\gci\truth\compliant_v17_v18_research_20261115\ideas_v1\run_clean_catboost_second_seed.py
python D:\gci\truth\compliant_v17_v18_research_20261115\ideas_v1\run_clean_catboost_external_geometry_v1.py
python D:\gci\truth\compliant_v17_v18_research_20261115\ideas_v1\run_nested_product_te_lgbm_v1.py
python D:\gci\truth\compliant_v17_v18_research_20261115\ideas_v1\analyze_ideas_v1.py
```

The submission generator uses the same fixed data paths, checks the official
sample schema and ID order, and writes under `ideas_v1\submissions`:

```powershell
python D:\gci\truth\compliant_v17_v18_research_20261115\ideas_v1\create_compliant_submission_te20_v1.py
```

## Latest strict run and further AUC research

`create_compliant_submission_te20_v2.py` is the current candidate generator.
It preserves v1, asserts the expected official input hashes, keeps model
training on training partitions, and runs test-covariate and validation-label
perturbation checks. Its candidate is
`submissions/product_te_20pct_v2/submission_compliant_product_te20_v2.csv`.
The generated five-fold result was 0.767756 versus a regenerated 0.767450
reference (+0.000305, all five folds positive). The CSV and manifest were
validated; the leaderboard score is unknown until it is uploaded.
The same generator was replayed into a separate directory: replay OOF was
0.767772 versus 0.767468 reference (+0.000303, all folds positive). The two
submission files are not identical (Spearman 0.999990; maximum absolute
`TARGET` difference 0.01333); component comparison localized variation to
CatBoost GPU predictions. Keep the first v2 output as the primary candidate
rather than selecting the slightly higher replay after seeing its OOF result.
Neither OOF result is a leaderboard score.

`run_balanced_loss_catboost_v1.py` tested class-total-balanced Logloss;
`run_pairwise_auc_linear_v1.py` and `_v2.py` tested a sampled pairwise
logistic AUC surrogate; `analyze_pairwise_te_blend_v2.py` evaluated a fixed
80/10/10 blend. The pairwise blend gave +0.000204 versus the reference on
all five folds, but adding it to TE20 gave only +0.000057 with a conditional
interval including zero, so no additional candidate was promoted. Exact
details and caveats are appended in `IDEA_EXPERIMENTS_v1.md`.

### Latest pairwise-ranking candidate

`run_catboost_pairlogit_auc_v1.py` tested GPU CatBoost PairLogit and did not
improve the fixed blend. `run_xgboost_rank_pairwise_v1.py` tested a global
XGBoost `rank:pairwise` objective. Its pre-specified 90/10 rank blend scored
0.768057 versus 0.767756 TE20 (+0.000301, all five folds positive).
`create_compliant_submission_te20_xgb_rank_pairwise_v1.py` then regenerated
all streams and passed its own gate at 0.767979 versus 0.767743 (+0.000236,
5/5 positive folds). The submission is
`submissions/te20_xgb_rank_pairwise_v1/submission_compliant_te20_xgb_rank_pairwise_v1.csv`.

An exact AUC decomposition by baseline-negative score decile showed that most
of this lift came from deciles 8–10, not the low-risk bulk. This is a
post-hoc diagnostic only; it did not create manual or row-routed predictions.
The candidate has no leaderboard score yet, and local OOF evidence neither
reaches nor guarantees 0.79. See `IDEA_EXPERIMENTS_v1.md` for fold metrics,
the decomposition, citations, and compliance limitations.
