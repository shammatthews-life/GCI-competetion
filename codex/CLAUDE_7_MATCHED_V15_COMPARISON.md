# Claude_7: matched comparison against cached v15 OOF

**Date:** 21 October 2026  
**Metric:** ROC-AUC  
**Decision:** The household and inquiry specialist models show a small positive matched OOF signal at the fixed 1% weight, but none clears the promotion threshold. Keep v15 as champion. No new submission CSV was created.

## Cache recovery and v15 baseline check

The user signed in to Google Drive in the integrated browser. The `gci_cache_final` folder was downloaded as [gci_cache_final-20261021T075423Z-1-001.zip](./gci_cache_final-20261021T075423Z-1-001.zip) and extracted to [v15_cache/](./v15_cache/). It contains 49 files, including all OLD/E1/E1ET/E1Aet member OOF/test prediction arrays, both view caches, and five product-CatBoost fold caches.

I reconstructed the final OOF from the cached component predictions using the formula in [GCI_FINAL_v15_end_to_end_colab.ipynb](../pre%20info/GCI_FINAL_v15_end_to_end_colab.ipynb):

`v15 = 0.90 × rank(0.75 × rank((OLD + E1 + E1ET) / 3) + 0.25 × E1Aet) + 0.10 × rank(CatBoost_product)`

The reconstructed OOF AUC is **0.77404144**, matching the notebook's printed reference **0.77404**. This validates that the correct cached member arrays and five-fold assignments were loaded. OOF AUC is not directly comparable to the user-reported public leaderboard AUC of 0.77891.

## Matched test setup

- Candidate specialist predictions were regenerated on the same full 171,202 training rows and the exact `StratifiedKFold(5, shuffle=True, random_state=42)` assignments used by the v15 cache.
- Each candidate row prediction is held out from its fold's model; product encodings are nested/cross-fitted within the training partition.
- The candidate model is the compact product-aware LightGBM used in the prior feature screen, with household and bureau-inquiry-profile feature blocks—not a retrained full v15 member. This makes the comparison a component screen, not a complete v15 architecture replacement.
- Before comparing to v15, the rank blend weight was fixed at **1%**. The same weight is used for household, inquiry, and their equal-rank blend.
- Paired bootstrap intervals are conditional on these OOF predictions and do not account for prior feature screening or repeated research.

## Results: fixed 1% blend into v15

| Candidate | Standalone OOF AUC | Spearman with v15 | Blended OOF AUC | Delta vs v15 | Positive folds | Paired 95% CI | Promotion gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Household specialist | 0.753786 | 0.894464 | 0.774065 | +0.000023 | 5/5 | [+0.000004, +0.000043] | **Fail** |
| Inquiry-profile specialist | 0.753173 | 0.895043 | 0.774059 | +0.000017 | 5/5 | [−0.000001, +0.000037] | **Fail** |
| Equal-rank household + inquiry specialist | 0.754160 | 0.897709 | 0.774062 | +0.000021 | 5/5 | [+0.000001, +0.000040] | **Fail** |

The household-only blend's interval is above zero, and the joint blend's lower endpoint is barely above zero, but their point gains are much smaller than the project gate of **+0.00015 OOF AUC**. The inquiry-only interval includes zero. Positive results in every fold are encouraging but do not turn these tiny changes into a robust promotion.

### Post-hoc weight sensitivity (diagnostic only)

After the fixed 1% comparison, I inspected weights from 0.5% to 50%. The largest observed deltas in that grid were at 5%: household **+0.000071**, inquiry **+0.000038**, and joint **+0.000057**. All had 5/5 positive folds, but every gain remained below the +0.00015 gate. These weights were selected after inspecting the same OOF results, so the sensitivity grid is **not an unbiased validation** and must not be used to justify a leaderboard submission.

## Conclusion and next direction

The experiment confirms that the new household and inquiry rankings contain a little ordering signal not already captured by v15. However, their much lower standalone AUC and high correlation with v15 mean that the usable incremental gain is tiny. The earlier OLD/NEW diversity lesson is directionally present, but these particular compact specialists are too weak to move the champion materially.

**Do not submit these branches or a 5% sensitivity blend.** No new file was added to `output/`. Keep the existing 0.77891 v15 result as the incumbent.

The complete comparison is reproducible with [run_claude_7_matched_v15_specialists.py](./run_claude_7_matched_v15_specialists.py). Its machine-readable results are [claude_7_matched_v15_specialist_results.csv](./claude_7_matched_v15_specialist_results.csv), [claude_7_matched_v15_specialist_folds.csv](./claude_7_matched_v15_specialist_folds.csv), and [claude_7_matched_v15_specialist_metadata.json](./claude_7_matched_v15_specialist_metadata.json). Aligned OOF arrays are in [claude_7_matched_v15_specialist_oof.npz](./claude_7_matched_v15_specialist_oof.npz). The exploratory weight curve is [claude_7_blend_weight_sensitivity.csv](./claude_7_blend_weight_sensitivity.csv).
