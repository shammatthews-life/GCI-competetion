# GCI World 2026 - Credit default competition: final package (v15 champion)

**Submitted file:** `GCI_submission_v15_rankspace_consistent_view.csv` (public ROC-AUC 0.77891). **Reproduction notebook:** `GCI_FINAL_v15_end_to_end_colab.ipynb`.
Metric: ROC-AUC. 61,500 rows, columns `SK_ID_CURR,TARGET`, IDs in `sample_submission.csv` order.

## 1. How to reproduce
1. Open `GCI_FINAL_v15_end_to_end_colab.ipynb` in Google Colab. Put `train.csv`, `test.csv`, `sample_submission.csv` in `/content` or anywhere in Drive; the notebook validates them (34 columns, TARGET present, 171,202 rows) and skips look-alike files.
2. Run the cells in order. From scratch it needs roughly 8 hours on a Colab CPU runtime; every member, fold and feature cache is saved to Drive (`gci_cache_final`), so a disconnect only costs the piece that was running, and with the caches the whole notebook finishes in minutes.
3. The last cell writes `GCI_submission_v15_reproduced.csv` and, if you upload the submitted CSV, prints the Spearman correlation and the maximum difference. Expect a correlation extremely close to 1 (tiny differences can come from multi-threaded LightGBM / library versions). **[Competitor: run this once and record the printed values here before submitting.]**

## 2. Data statement
Only the three supplied files are used. No external data, no test labels, no pseudo-labelling, no pretrained weights, no hand-edited predictions. Test rows are used only for label-free features (value counts, group statistics, the product decode, the learned-expectation models); the target comes only from `train.csv`. An exact-duplicate audit found 0 of 61,500 test rows identical to a training row (excluding ID and target).

## 3. Pipeline
1. **Features:** 67 core and 130 "all-families" features (raw columns, DAYS_EMPLOYED sentinel flag, missing flags, financial ratios, age/employment ratios, bureau totals, EXT_SOURCE aggregates, count encodings, group differences); two members also use a decoded credit/goods tier.
2. **Label-free product decode:** for cash loans annuity = Q0 x credit + F (F about 129.5, a monthly fee on about 75% of cash loans); a row is fee-bearing when (annuity - 129.5)/credit lands on a more frequently shared 5-decimal value than annuity/credit. This turns 5,335 distinct q values into 1,047 products.
3. **Nested target encodings** of q, the product (3/4/5 decimals) and product x fee: smoothed default rates and counts; training rows use inner 5-fold out-of-fold encodings (seed 7); validation/test rows use fold-training rows only.
4. **Four LightGBM member sets** (five members each, fixed weights 0.24 qf_shallow / 0.12 qf_reg / 0.32 v1_allfam_oh / 0.16 v2_allfam_oh / 0.16 v2_mono; rounds fixed in advance; each member = 50% five fold models + 50% full-data models over several seeds):
   - **OLD:** features 1-3 only.
   - **E1:** OLD + 16 learned-expectation residual models (label-free regressors fitted on train+test with 5-fold out-of-fold predictions; the EXT-score residuals carry the gain) + multi-view stacking features (EXT / loan / person views, out-of-fold).
   - **E1ET:** E1 with LightGBM `extra_trees=True`.
   - **E1Aet:** E1ET with *consistent* view features (validation/test view = average of the inner models).
5. **CatBoost branch:** 5-fold CatBoost (depth 7, 900 iterations) with the decoded product keys as native categoricals; test prediction = fold average.
6. **Final formula (fixed in advance):** `champion = 0.90 x rank( 0.75 x rank(B) + 0.25 x E1Aet ) + 0.10 x rank(CatBoost)`, where `B = (OLD + E1 + E1ET) / 3`.

## 4. Validation and decision history
- Held-out (5-fold out-of-fold) AUC of the champion formula: **0.77404** (reproduced independently as 0.77404144 by a second assistant). Component blends: OLD 0.77202, E1 0.77258, E1ET 0.77357, E1Aet blend about 0.7737.
- Public: 0.77321 (start) -> 0.77715 (product keys) -> 0.77746 (blend) -> 0.77836 (residual features) -> 0.77865 (multi-view) -> 0.77884 (extra_trees + CatBoost) -> **0.77891 (consistent views)**. Differences below about 0.0003 on the public board are within noise.
- Gains came from four mechanisms: the product decode, the EXT residuals, extra_trees regularisation of noisy stacked features, and consistent train/test view features. A learning-curve check shows each doubling of training rows adds only about +0.003 AUC.
- **Rejected under pre-set rules** (each needed a mean fold gain of at least +0.0002 on at least 4 of 5 folds): decoded credit/goods tier, product-relative features, region/organization/occupation encodings, more or standardised residual targets, product-aware expectations, applicant-quality-adjusted product encodings, digit/round-number features, pairwise ranker, XGBoost, DART, GOSS, path_smooth / bagging / per-node feature sampling, class weighting, sampled pairwise loss, neural networks and denoising autoencoder, sklearn ExtraTrees forest, nearest-neighbour target-rate features (including the exact 500-neighbour feature from the original Home Credit winner), regional-category CatBoost, household-affordability and employment-consistency features, subgroup specialists (drivers, high-credit, young applicants), spline scorecard, importance-weighted training, importance of ID order (none), re-optimised blend weights and blending-function changes (rank vs probability vs logit).
- Train and test are not perfectly identically distributed (adversarial AUC 0.518, mostly EXT scores and age); importance-weighted training did not help.

## 5. AI-assistance disclosure  *(competitor: confirm every statement and compare it with the current instructions in Omnicampus before submitting)*
**AI-assistance disclosure:** Anthropic Claude was used for exploratory analysis, product-decoding research, residual-feature, multi-view and consistent-view feature engineering, notebook and validation-script development, and documentation. OpenAI assistants (ChatGPT and a Codex coding agent) were used for further experiments, the CatBoost branch, the extra_trees test, the rank-space v15 candidate and independent re-validation. The reproduction notebook was AI-generated. The competitor supplied the competition data, executed and checked the Colab runs, evaluated the results, selected the submissions and made the final decisions. Some exploratory scripts were run by the assistants on scikit-learn stand-in models; all reported LightGBM/CatBoost competition results come from the competitor's runs. To the best of the assistants' knowledge no code was copied from other competitors' public solutions; methods borrowed from published sources are cited below. **[Competitor: confirm.]**

## 6. Sources
- Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
- Prokhorenkova et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS.
- Geurts, Ernst, Wehenkel (2006). Extremely randomized trees. Machine Learning 63. (idea behind LightGBM's `extra_trees`)
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
- Micci-Barreca (2001). A preprocessing scheme for high-cardinality categorical attributes. SIGKDD Explorations 3(1). (smoothed target encoding)
- Wolpert (1992). Stacked generalization. Neural Networks 5(2). (multi-view stacking)
- Libraries: pandas, NumPy, SciPy. Competition data and brief supplied by the GCI World 2026 organizers.

## 7. Known limits
- The end-to-end notebook was smoke-tested on a small subset with stand-ins; the full run must be performed and compared with the submitted file by the competitor (Section 1, step 3).
- The final ranking uses your last submitted file: make sure the last upload is a valid CSV (a notebook uploaded as a submission scores -999).
