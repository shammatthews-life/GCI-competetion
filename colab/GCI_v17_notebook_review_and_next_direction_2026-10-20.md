# GCI v17 CatBoost-E1 notebook review — 20 October 2026

## Decision

**Use v17 as research evidence; do not submit either v17 CSV candidate.** The notebook's own champion-comparison cell says both candidates fail its predeclared acceptance rule.

## What the uploaded notebook does

`GCI_v17_catboost_E1_stack_colab(1).ipynb` trains CatBoost on the E1 stack:
- base application features and decoded product categorical keys (`Q3`, `Q4`, `Q5`, `P3`, `P4`, `P5`, `PF4`, `K_DEC`, `FEE`);
- 32 residual features (`RESID1`);
- consistent-view stacking features from EXT, LOAN and PERSON views, plus pairwise view differences;
- the same five outer folds (`StratifiedKFold`, shuffle=True, random_state=42).

The motivation is sound: the prior CatBoost branch had only received base features, although the E1/ExtraTrees LightGBM branches benefited from residual and view representations.

## Results shown in the user's screenshots / notebook output

| Result | Pooled held-out OOF AUC / delta | Interpretation |
|---|---:|---|
| Original base-feature CatBoost branch | 0.76943 | Baseline CatBoost branch |
| CatBoost-E1 standalone | 0.77444 | Strong standalone improvement, about +0.00501 vs the base CatBoost branch |
| v15 champion formula | 0.77404 | Matched OOF baseline shown by the notebook |
| C1: replace CatBoost with CatBoost-E1 | 0.77404, delta +0.00000 | Fold gains `[+0.00018, -0.00015, -0.00013, -0.00004, +0.00011]`; 2/5 positive; FAIL |
| C2: average both CatBoost branches | 0.77406, delta +0.00002 | Fold gains `[+0.00011, -0.00005, -0.00005, 0.00000, +0.00007]`; 3/5 positive; FAIL |

The two CatBoost branches have Spearman rank correlation **0.9758**. The new branch has a stronger standalone score, but its ranking is very similar to the original CatBoost branch, so the full ensemble gets almost no new ranking information.

### Important caveat

The `0.77444` standalone AUC and the `0.77404` full-ensemble AUC refer to different predictions/recipes. They must not be compared as though they were the same model. The valid promotion test is the notebook's matched comparison of C1/C2 against the v15 formula; both fail.

## Implementation warning

Cell 6 prints `submit only if PASS`, but it writes each candidate CSV regardless of the `ok` value. A CSV's existence does **not** mean the candidate passed. Fix the notebook to write a candidate only inside `if ok:` and otherwise print `REJECTED — no submission written`. Keep the champion CSV unchanged.

The earlier `ModuleNotFoundError: No module named 'catboost'` shown in the Gemini pane is an environment dependency issue, not a model-quality result. Install CatBoost in the Colab runtime before running the relevant cell, then rerun dependent cells in order.

## What this experiment teaches us

1. Giving CatBoost residual/view features helps its standalone performance substantially.
2. Standalone AUC is not enough; ensemble diversity is the bottleneck.
3. The high 0.9758 rank correlation suggests that adding more versions of this same CatBoost branch is unlikely to help unless we create a model with materially different errors.
4. Do not repeatedly adjust blend weights on the same five folds. The fold results are already weak, and repeated tuning increases selection bias.

## Recommended next direction

Prioritize **conditional error diversity**, not more CatBoost feature stacking:

1. Recover the exact v15 OOF arrays, fold indices, and all member predictions from `gci_cache_final`.
2. For each candidate branch, calculate paired v15 OOF AUC delta, fold deltas, Spearman rank correlation, and performance on positive-negative pairs that v15 misranks.
3. Screen a genuinely different model family or feature view only if it can be justified by a new hypothesis; do not repeat the rejected DAE, MLP, KNN, XGBoost, pairwise ranker, or additional CatBoost seed/depth tests listed in the project log.
4. The bureau-enquiry burst block was a prior promising quick-screen result, but its full v15 confirmation was negative (delta approximately -0.0000011; bootstrap interval crossed zero), so it is not a current lead.
5. A new feature hypothesis should be tested on identical folds and compared against the full v15 formula, with an acceptance rule declared before running.
6. Require at least +0.00015 pooled OOF gain, 4/5 positive folds, and paired-bootstrap lower bound above zero before producing a submission candidate. This is a practical project gate, not a guarantee of leaderboard gain.

## Current leaderboard context

The user-reported public champion remains `GCI_submission_v15_rankspace_consistent_view.csv`, score **0.77891** (the screenshot's displayed score). v18 scored **0.77882** and should remain rejected. No v17 public leaderboard score is established by the screenshots.

## Source basis

- Uploaded notebook: `GCI_v17_catboost_E1_stack_colab(1).ipynb`, especially cells 0, 4, 5 and 6.
- User screenshots dated 20 October 2026 showing CatBoost-E1 standalone output and champion comparison.
- Project master handoff and full chat research log, which record the v15 formula, previously rejected experiments and evaluation cautions.
