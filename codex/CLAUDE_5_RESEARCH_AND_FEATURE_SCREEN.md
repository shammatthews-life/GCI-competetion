# Claude_5 research review and feature screen

**Date:** 21 October 2026  
**Goal:** Find feature ideas that add information beyond the existing product-aware LightGBM pipeline, without weakening or replacing the v15 fallback.  
**Decision:** Keep v15. Two feature blocks merit a matched v15 test; none of the five has passed that test, so no leaderboard CSV was created.

## What the literature suggests for this dataset

This was a targeted literature review, not a claim that papers can predict the result on this competition's split. The most relevant lessons are:

1. **Domain-derived features can matter when they encode real data-generating structure.** Qiu et al. specifically study feature engineering on the Kaggle Home Credit Default Risk dataset and report that expert-knowledge-generated features paired with LightGBM were useful. This supports testing loan/account mechanisms, not adding arbitrary polynomial features.
2. **Credit-scoring model rankings depend on the dataset and evaluation protocol.** The large credit-scoring benchmark by Lessmann et al. and the ensemble comparison by Li and Chen motivate matched out-of-fold comparisons. A higher standalone AUC is not sufficient evidence for adding a branch to v15.
3. **Boosted trees remain a strong default for medium-sized tabular data.** Grinsztajn et al. benchmarked tree-based and neural approaches across 45 datasets. That evidence, plus this project's own failed MLP/autoencoder probes, argues against spending the next round on another scratch neural model.
4. **Leakage control is essential for categorical risk encodings.** CatBoost's ordered categorical treatment and the existing nested-encoding work both support cross-fitting any new target-derived feature. The new experiment therefore nests encodings inside each outer training fold.
5. **Feature selection should be empirical and task-specific.** The feature-selection literature supports screening coherent feature blocks and rejecting them when they fail fresh validation, rather than retaining a large kitchen-sink feature set.

### References

- Z. Qiu, Y. Li, P. Ni, and G. Li (2019), [“Credit Risk Scoring Analysis Based on Machine Learning Models”](https://doi.org/10.1109/ICISCE48695.2019.00052). Its abstract explicitly describes feature engineering and LightGBM evaluation on Kaggle Home Credit Default Risk.
- S. Lessmann, B. Baesens, H.-V. Seow, and L. C. Thomas (2015), [“Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research”](https://doi.org/10.1016/j.ejor.2015.05.030).
- Y. Li and W. Chen (2020), [“A Comparative Performance Assessment of Ensemble Learning for Credit Scoring”](https://doi.org/10.3390/math8101756). The paper's abstract reports that stacking outperformed individual learners in its tested datasets, while metric leadership varied by model.
- L. Grinsztajn, E. Oyallon, and G. Varoquaux (2022), [“Why do tree-based models still outperform deep learning on tabular data?”](https://doi.org/10.48550/arXiv.2207.08815).
- M. Schmitt (2022), [“Deep Learning vs. Gradient Boosting: Benchmarking state-of-the-art machine learning algorithms for credit scoring”](https://doi.org/10.48550/arXiv.2205.10535).
- J. Jemai and A. Zarrad (2023), [“Feature Selection Engineering for Credit Risk Assessment in Retail Banking”](https://doi.org/10.3390/info14030200).
- L. Prokhorenkova et al. (2018), [“CatBoost: unbiased boosting with categorical features”](https://doi.org/10.48550/arXiv.1706.09516).

The review used bibliographic records and abstracts for the papers where full-text access was unavailable. The exact dataset's prior experiments and the v15 notebook remain more directly applicable than any paper's generic AUC result.

## Existing v15 coverage and prior rejects

The v15 recipe already includes decoded annuity/credit product keys, nested product target encodings, loan-to-income and annuity-to-income ratios, external-score summary/interactions, age/employment ratios, bureau totals/recent-count ratios, category counts and group-mean deviations, learned residual features, EXT/LOAN/PERSON stack views, ExtraTrees LightGBM members, and product-aware CatBoost.

The archive also records previous failures for generic KNN risk, unsupervised clusters/anomaly scores, deep autoencoder/MLP features, repeated CatBoost variants, bureau-burst features in the full v15 branch, product-by-applicant encodings, and several EXT/affordability interactions. Those were not repeated as if they were new ideas. The present inquiry-profile block is a narrower transformation of the four inquiry fields available here; it is still treated cautiously because the related full-v15 bureau-burst test was neutral.

This supplied data does **not** contain social-circle default counts or the historical bureau/prior-loan/card/POS tables used in the broader Home Credit Kaggle dataset. Those cannot be engineered from the current files.

## Experiments run

Reproduction code: [run_claude_5_feature_hypothesis_screen.py](./run_claude_5_feature_hypothesis_screen.py).  
Validation artifacts: [claude_5_feature_hypothesis_metrics.csv](./claude_5_feature_hypothesis_metrics.csv), [claude_5_feature_hypothesis_fold_metrics.csv](./claude_5_feature_hypothesis_fold_metrics.csv), [claude_5_feature_hypothesis_oof.npz](./claude_5_feature_hypothesis_oof.npz), and [claude_5_feature_hypothesis_metadata.json](./claude_5_feature_hypothesis_metadata.json).

- Full 171,202 training rows; same compact product-aware LightGBM scaffold for reference and candidate.
- Two fresh `StratifiedKFold(3)` partitions (seeds 20261051 and 20261052), 450 boosting iterations.
- Product, categorical-group, categorical-pair and risk-band target encodings were nested within the outer training partition and inner cross-fits. No validation-fold labels entered their features.
- Each row received an OOF prediction for all variants. Paired bootstrap intervals below are conditional on each set of OOF predictions; they do not account for testing five ideas.
- This compact reference is not the v15 ensemble. Its pooled AUC varies around 0.752 across folds and must not be compared numerically to v15's approximately 0.774 OOF result or 0.77891 public score.

| Feature idea | Seed 20261051 delta (95% paired CI) | Fold wins | Seed 20261052 delta (95% paired CI) | Fold wins | Decision |
|---|---:|---:|---:|---:|---|
| Household adult-equivalent affordability | +0.001081 [+0.000445, +0.001776] | 3/3 | +0.000348 [−0.000384, +0.001010] | 2/3 | Best provisional feature lead; fresh-split interval crosses zero |
| Bureau inquiry profile / recency shares | +0.000816 [+0.000223, +0.001577] | 3/3 | +0.000382 [−0.000366, +0.001224] | 2/3 | Provisional only; related bureau feature work was neutral against v15 |
| Organization/occupation/etc. smoothed target encodings | +0.000550 [−0.000230, +0.001186] | 2/3 | −0.000587 [−0.001259, +0.000064] | 0/3 | Reject; inconsistent and negative on fresh split |
| Organization×occupation and income×education target encodings | +0.000803 [+0.000207, +0.001611] | 2/3 | −0.000410 [−0.001189, +0.000217] | 1/3 | Reject; does not reproduce |
| Coarse continuous-risk-band target encodings | −0.000256 [−0.001079, +0.000506] | 1/3 | −0.000385 [−0.001174, +0.000369] | 1/3 | Reject |

The two provisional leads were consistently positive in pooled AUC on both seeds, but their second-seed bootstrap intervals cross zero and one fold regressed on each. This is enough to prioritize them for one carefully matched v15 test, not enough to claim an improvement.

## What the OLD/NEW experiments teach us

The archive records two related but distinct pieces of evidence:

- Historically, the compact OLD member scored 0.755212 and NEW scored 0.755694; their equal rank blend scored 0.756750, +0.001056 over the better single member. This is not a v15-matched comparison, and the surviving archive does not preserve enough of the compact OLD/NEW runner to attribute that difference to one specific feature.
- In the v15 lineage, OLD uses the original features and nested product encodings; E1 adds 16 label-free learned-expectation residual representations and EXT/LOAN/PERSON view scores; E1ET changes the LightGBM split sampling with `extra_trees=True`; E1Aet fixes the validation/test view-generation mismatch by averaging inner view models. Product-native CatBoost supplies another model family. The measured lifts are small but cumulative.

The defensible interpretation is that **new views and member changes help when they add complementary ordering errors**, not simply because they increase feature count or standalone AUC. The strongest v15-era change was the ExtraTrees replacement: all five E1 member replacements improved in that particular pipeline, whereas ExtraTrees on a later compact family performed worse. Likewise, v17 CatBoost-E1 improved standalone AUC but had 0.9758 rank correlation with the existing CatBoost and almost no full-ensemble gain. The *context and error diversity* matter.

### Follow-up: OLD/NEW-inspired diversity test

To check whether that ensemble lesson appears in the new feature screen, I formed a fixed equal-weight rank blend of the compact reference, household block, and inquiry-profile block. Ranks were computed **within each held-out fold**, so the comparison does not exploit score-distribution shifts between fold models. No blend weight was tuned.

Reproduce with [run_claude_5_diversity_diagnostic.py](./run_claude_5_diversity_diagnostic.py); outputs are [claude_5_diversity_fold_diagnostics.csv](./claude_5_diversity_fold_diagnostics.csv) and [claude_5_diversity_diagnostic.json](./claude_5_diversity_diagnostic.json).

| Seed | Fold | Compact baseline AUC | Equal-rank blend AUC | Delta |
|---:|---:|---:|---:|---:|
| 20261051 | 1 | 0.748918 | 0.750072 | +0.001154 |
| 20261051 | 2 | 0.753525 | 0.755510 | +0.001985 |
| 20261051 | 3 | 0.753927 | 0.755625 | +0.001698 |
| 20261052 | 1 | 0.751017 | 0.752674 | +0.001656 |
| 20261052 | 2 | 0.753704 | 0.754643 | +0.000938 |
| 20261052 | 3 | 0.753718 | 0.754862 | +0.001144 |

The initial diagnostic was positive on 6/6 folds (mean fold delta +0.001429), and each fold's paired stratified-bootstrap interval was above zero. Household and inquiry member ranks were themselves highly correlated (mean Spearman about 0.985), so the uplift is not evidence of dramatic diversity. The small gain may reflect a combination of complementary ordering and averaging away model-specific noise.

### Fresh third-seed confirmation

I then froze equal one-third rank weights for the compact reference, household member, and bureau inquiry-profile member and ran a new split (seed 20261053). No weights or features were tuned on this split. The held-out fold comparison was:

| Fold | Compact baseline AUC | Household AUC | Inquiry AUC | Fixed equal-rank blend AUC | Delta vs baseline | Paired bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.752866 | 0.753444 | 0.753308 | 0.754170 | +0.001303 | [+0.000586, +0.002018] |
| 2 | 0.752199 | 0.751303 | 0.752046 | 0.752848 | +0.000650 | [−0.000069, +0.001380] |
| 3 | 0.748509 | 0.748689 | 0.748979 | 0.749679 | +0.001170 | [+0.000426, +0.001888] |

The fixed blend improved all 3 held-out folds on this untouched seed, mean fold delta +0.001041. Combined with the exploratory screen, its direction is positive in all 9 folds across three seeds. The third-seed fold-2 interval slightly crosses zero, and the model ranks remain highly correlated. This is evidence that the blend is useful against this **compact baseline**, but not proof of full-ensemble lift.

Reproduction code: [run_claude_6_fresh_diversity_confirmation.py](./run_claude_6_fresh_diversity_confirmation.py). Results: [claude_6_fresh_diversity_confirmation.csv](./claude_6_fresh_diversity_confirmation.csv), [claude_6_fresh_diversity_confirmation.json](./claude_6_fresh_diversity_confirmation.json), and [claude_6_fresh_diversity_confirmation_oof.npz](./claude_6_fresh_diversity_confirmation_oof.npz).

**This is still not a v15 improvement.** Do not submit this compact blend alone: it is far below v15's reported OOF level. The next actual promotion test is to add the two specialist member rankings separately and jointly to the exact v15 OOF rank with a small, predeclared weight and matched folds. The full v15 cache is still absent, so no honest matched claim or safe submission CSV can be made from this screen.

## Next decision

1. Preserve v15 unchanged.
2. Before a v15 comparison, provide the complete aligned champion cache: exact v15 OOF predictions, fold indices, component/member OOF predictions, and matching test predictions. The output submission CSV alone cannot support a leakage-safe matched comparison.
3. Test the household feature block first and the inquiry profile second, separately, against the frozen v15 formula. The fixed equal-rank blend is now confirmed only against the compact baseline on a fresh split; test it against v15 only as a locked candidate. Keep folds and member settings fixed and report paired per-fold deltas, rank correlation, and paired uncertainty.
4. Promote neither unless it passes the existing gate on matched v15 predictions and then survives a fresh confirmation. Do not tune a blend weight on the same OOF labels and present it as unbiased.

There is no submission file from this run. The best available result is a research shortlist, not a replacement for the user's 0.77891 public-score v15.
