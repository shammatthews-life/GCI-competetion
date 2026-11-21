# GCI residual-ranking submission: forensic diagnosis

Date: 2026-10-18

## Leaderboard observation

- v15 rank-space consistent-view submission: user-reported public AUC 0.77891.
- Residual-ranking probe: user-reported public AUC 0.77869.
- Difference: -0.00022.

A public score alone cannot establish whether all of the difference is model degradation or partly public-split variability. The implementation audit nevertheless identifies concrete train/inference and evaluation-comparison flaws.

## Root cause 1 — wrong baseline for the claimed +0.000095

The +0.000095 OOF AUC was measured against a compact, newly built LightGBM baseline (pooled OOF AUC 0.753331), not against the v15 champion's aligned OOF predictions (historical OOF AUC 0.77410395). It therefore never demonstrated improvement over v15. The leaderboard candidate's top-level blend used the residual branch to perturb v15 despite no direct paired OOF test of that formula.

## Root cause 2 — training/inference iteration mismatch

The OOF baseline recorded in `oof_predictions.npz` came from fold models with early-stopped best iterations around 168–180 (see `fold_results.csv` and `run.log`). But `create_residual_leaderboard_probe.py` trains the full-data test-inference baseline with `n_estimators=60`. The correction model is trained using OOF baseline scores from the longer models, then receives test scores from the much shorter 60-tree model. This is a train/test feature-distribution mismatch in the correction model's `BASE_OOF_SCORE` input.

## Root cause 3 — most of the added 5% was not the correction model

The submitted formula is:

`0.95 * rank(v15) + 0.05 * rank(0.95 * rank(compact_baseline) + 0.05 * rank(correction))`

Thus the overall candidate gives the compact baseline an effective 4.75% contribution and the correction model only an effective 0.25% contribution before rank nonlinearities. This is not a clean 5% residual-correction blend. The compact baseline is substantially weaker than v15, and the correction's tiny weight cannot be assumed to repair that dilution.

## What changed in the submitted ranking

The files have matching IDs and the candidate is structurally valid. Comparing prediction ranks:

- Rank correlation between v15 and candidate: 0.9997344.
- Mean absolute percentile-rank change: 0.004999.
- Maximum absolute percentile-rank change: 0.03935.
- Top 1% overlap: 96.92%.
- Top 5% overlap: 98.02%.
- Top 10% overlap: 98.33%.

So the file made a small but material reordering of cases; the AUC loss is consistent with the added ranking signal being unhelpful/noisy on the public evaluation set. These rank statistics do not by themselves identify which individual pairwise inversions caused the loss.

## New plan: champion-conditioned, cross-fitted rank correction

Do not blend a weak standalone model into v15 and call it residual ranking. Instead:

1. Rebuild v15 OOF predictions using the exact v15 feature and multiview pipeline, with every training row scored only by models that did not train on that row.
2. Fit a small correction learner on the v15 OOF score plus a small, pre-specified set of features that could explain local ranking errors (EXT_SOURCE values/spread, credit/income burden, employment sentinel, bureau enquiry recency/bursts, and selected categorical groups). Use only cross-fitted score features for the meta-training rows.
3. On an untouched outer fold, generate v15 predictions and correction predictions using exactly the same training procedure as meta-training. Compare the correction directly to v15 on the same held-out rows.
4. Test a pre-registered fixed rank-space blend (e.g. 1%, 2%, 3%; choose on inner validation only, then freeze it before outer confirmation). Also test a gated correction that only applies where the correction is confident and disagrees with v15; gating thresholds must be chosen inside inner folds.
5. Promote only if paired AUC improves on a fresh outer split, fold gains are stable, and the confidence interval supports a gain. Then train the matched pipeline for test predictions and generate one CSV.

## Current decision

- Keep v15 (public 0.77891) as champion.
- Do not resubmit the current residual probe.
- Do not claim the residual idea failed in general; this submission did not validly test it against v15.
- The v15 notebook/source pipeline and aligned v15 OOF predictions are needed to complete the decisive champion-conditioned test. The v15 test CSV alone is not enough to train and validate a correction model without leakage or distribution mismatch.
