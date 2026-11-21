# v15 score audit and next research direction

**Reviewed:** 21 October 2026  
**Primary source:** `D:\gci\pre info\GCI_FINAL_v15_end_to_end_colab.ipynb`  
**User-reported public result from its output:** approximately **0.7789**  
**Metric:** ROC-AUC

## Executive conclusion

The v15 score is strong because it is a carefully validated **diverse ensemble over the same small application dataset**, not because of one unusually high-scoring standalone model. Its signal comes from:

1. discovering and encoding repeated loan-pricing products;
2. combining several differently regularized LightGBM members;
3. adding label-free residual representations and three specialized views;
4. using `extra_trees=True` to change the LightGBM members' errors;
5. adding product-native CatBoost as another ranking family; and
6. mixing complementary predictions in a fixed rank-space formula.

The next result is unlikely to come from another minor ratio, another CatBoost variant, or a standalone model. The remaining bottleneck is **incremental ranking information after the whole v15 blend**. A candidate must improve the v15 OOF predictions on the same rows; improving a weaker compact baseline does not answer that question.

## What the supplied end-to-end notebook actually does

The notebook documents the champion formula as:

`0.90 × rank(0.75 × rank(B) + 0.25 × rank(E1Aet)) + 0.10 × rank(CatBoost_product)`

where `B = (OLD + E1 + E1ET) / 3`. Each of OLD, E1, E1ET, and E1Aet is itself a fixed-weight rank blend:

| LightGBM member | Weight |
|---|---:|
| `qf_shallow` | 0.24 |
| `qf_reg` | 0.12 |
| `v1_allfam_oh` | 0.32 |
| `v2_allfam_oh` | 0.16 |
| `v2_mono` | 0.16 |

These are **multiple complementary fitted members**; the five weights are fixed in advance, not reoptimized on the leaderboard.

### Feature and model layers

- **Original application features:** cleaned employment sentinel/missingness; EXT_SOURCE geometry; financial ratios and logs; age/employment ratios; bureau-enquiry summaries; category-frequency and group-deviation features.
- **Decoded pricing signal:** `AMT_ANNUITY ≈ Q × AMT_CREDIT + F`, with `F ≈ 129.5` for many cash loans. Multiple raw/fee-corrected product keys (`Q3/Q4/Q5`, `P3/P4/P5`, `PF4`) and counts are included. A separate `K_DEC` / credit-to-goods tier is used in selected members.
- **Leakage-controlled target encoding:** the training-fold rows receive nested out-of-fold product encodings; an outer validation fold is encoded from outer-training rows only. This is much safer than computing a row's product default rate from the entire training target.
- **Residual representation:** 16 cross-fitted `HistGradientBoosting` regressions predict selected input variables from the remaining variables; both predictions and observed-minus-predicted residuals form 32 label-free features.
- **Multi-view stacking:** separate EXT, LOAN, and PERSON models produce logits and pairwise view differences. E1Aet's “consistent” variant uses the average of the inner models for validation/test, matching the type of prediction used to make training-row stack features.
- **ExtraTrees diversity:** E1ET/E1Aet switch LightGBM to `extra_trees=True`; the earlier research reports improvements for all five members in that specific pipeline.
- **Product CatBoost:** CatBoost receives native categorical versions of the source categories and decoded product keys, including `K_DEC` and `FEE`.

The result is then assembled in rank space rather than by assuming that raw model probabilities are calibrated alike.

## Why this recipe raised the score

The research archive shows a cumulative progression rather than a single “big lift”:

| Stage | Reported OOF AUC | Change/interpretation |
|---|---:|---|
| v9 reproduced ensemble | 0.773301 | OLD + E1, fixed blend |
| v13_B (OLD + E1 + E1 ExtraTrees) | 0.773768 | ExtraTrees members add useful ranking diversity |
| v14G (B + product CatBoost) | 0.773937 | Product CatBoost adds a complementary signal |
| v15 rank-space consistent-view candidate | 0.774104 | Consistent-view E1Aet adds a small stable OOF gain |

The user-reported leaderboard progression from the archived research is 0.77865 → 0.77884 → 0.77891. These are public-score reports; they are **not numerically interchangeable with OOF AUC**. The end-to-end notebook itself contains no saved executed-cell outputs, and its OOF print is a reference value, not a newly captured result in this workspace.

The particular v15 matched local gain was only about **+0.000167 OOF** over v14G, despite a leaderboard improvement of about +0.00007 as displayed. That is consistent with the main lesson: once the ensemble is strong, the remaining wins are small and need error diversity.

## What the v17 experiment tells us

The v17 notebook adds residual and consistent-view features to the CatBoost member. The historical matched test is especially informative:

- CatBoost-E1 standalone: reported 0.77444 vs old CatBoost standalone 0.76943.
- But its rank correlation with old CatBoost was **0.9758**.
- Replacing the champion's CatBoost member changed the full formula by about **+0.00000**; averaging both CatBoost versions changed it by about **+0.00002**.
- Both failed the notebook's predeclared full-ensemble gate.

Thus, “standalone AUC went up” did not translate into better v15 ranking: the E1 CatBoost branch mostly learned signal the other ensemble branches already had. This is why the earlier standalone v_3/v_4 experiments did not establish a v15 improvement. Do not treat v_4's gain over standalone v_3 as a gain over the v15 champion.

## What is lacking / the gap to the top

The 20 October leaderboard snapshot in the research archive placed 0.77891 at rank 17; rank 10 was 0.78616, a gap of about **+0.00725 public AUC**. The reported v15 full OOF lift over v9 was around +0.0008, and the v15 successor lift over v14G was +0.00017. A top-10 jump is therefore not a reasonable expectation from another small feature or a 1–5% blend perturbation.

The supplied training table has only 32 predictors. It has no applicant's historic bureau records, prior loan applications, installment repayment history, POS/cash-loan history, or credit-card history. Such longitudinal information can be highly predictive in related credit-risk tasks, but it is not in the provided files. **Do not add outside tables unless the actual GCI rules explicitly permit them.** If the competition is restricted to the supplied files, the information ceiling may be a major part of the gap.

Other constraints:

- The exact fold/member OOF cache is absent locally; v15 needs it for valid paired tests.
- The newly supplied notebook has **zero saved outputs**. Its code and recipe are valuable, but it does not include the executed member predictions needed to reproduce the matched acceptance analysis.
- Its full scratch run estimates roughly eight hours. Only `resid16.pkl` and `e1A_views.pkl` from the previous standalone v17 work exist in this workspace; the OLD/E1/E1ET/E1Aet member predictions and original CatBoost folds do not.
- Extensive reuse of five folds creates selection bias. Any late-stage candidate should have a fresh confirmation or untouched holdout in addition to the development comparison.

## Direction I recommend

### Priority 1 — establish an exact v15 baseline bundle

Run the supplied notebook in its intended Colab/runtime with its actual `gci_cache_final`; preserve the exact scored v15 CSV. Export a compact, versioned bundle containing:

- target and exact outer-fold indices;
- OOF predictions for every OLD/E1/E1ET/E1Aet member and product CatBoost fold;
- test predictions for those same members;
- final v15 OOF/test predictions;
- notebook hash, package versions, parameters, and ID-order checks.

This is the prerequisite for every honest improvement test. The local directory currently does not contain that bundle.

### Priority 2 — audit and lock test-time recipe equivalence

Before fitting new features, verify that:

- train OOF target encodings are nested and match the model's target mapping at validation/test;
- fold/test views use the same consistent A recipe for E1Aet;
- the test ensemble's fold-average/full-data prediction mix is exactly the one used in the tested recipe;
- every blend transforms/ranks the same row population as in the original submission;
- the generated CSV is compared to the user-scored file by Spearman correlation, maximum rank change, and checksum.

This is more valuable now than adding another weak model; earlier lower scores show implementation mismatch can waste a leaderboard attempt.

### Priority 3 — test only genuinely new error information

Given the experiment history, I would **not** rerun unchanged product-only Q encodings, generic ratios, bureau bursts, contract-affordability, CatBoost seed/depth, ExtraTrees seed, or v17's CatBoost-E1 replacement. Conditional product encodings tested in the recent fresh v_1 run were inconsistent/negative.

For the next feature experiment, define a single hypothesis not already represented by the application features: **whether the 32 label-free residuals and three view logits expose a subpopulation where the current ensemble is systematically misranking positives**. Build cross-fitted diagnostic strata on development folds; then, only if a stable pattern appears, test one correction model inside the exact v15 outer folds. Train the correction using only out-of-fold v15 predictions, keep its scoring/transform identical at inference, and compare the final fixed blend to v15. Avoid hand-tuning a weight on public leaderboard results.

If no stable misranked subpopulation is found, stop feature search rather than submitting noise. The next practical alternative is to check competition rules for additional allowed source data; if none is allowed, set an attainable target and preserve 0.7789.

### Promotion rule for any future candidate

Do not generate a submission candidate until the *full fixed v15 formula* comparison achieves:

- paired pooled OOF AUC gain ≥ **+0.00015**;
- at least **4/5** positive fold gains;
- paired bootstrap 95% lower bound above zero;
- the same conclusion on a fresh confirmation split or untouched validation data;
- exact submission ID/order/finite-prediction checks.

The final arbiter remains the public score. Keep the 0.7789 submission as fallback until a candidate scores higher.

## Current recommendation

**Do not submit v_2, v_3, or v_4 as a replacement for v15.** The two lowest reported standalone leaderboard scores (0.77198 and 0.77528) support that decision. The right next action is to use the new end-to-end notebook to recover exact v15 OOF/member/test predictions, then run a single predeclared conditional-error experiment against that full formula. No new CSV is created by this audit.

## Source details

- Main source: [GCI_FINAL_v15_end_to_end_colab.ipynb](../pre%20info/GCI_FINAL_v15_end_to_end_colab.ipynb)
- Historical experiment ledger: [GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md](../pre%20info/GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md)
- v17 evaluation: [GCI_v17_notebook_review_and_next_direction_2026-10-20.md](../pre%20info/GCI_v17_notebook_review_and_next_direction_2026-10-20.md)
- Recent v_1 conditional confirmation: [V_1_CONDITIONAL_CONFIRMATION.md](./V_1_CONDITIONAL_CONFIRMATION.md)
