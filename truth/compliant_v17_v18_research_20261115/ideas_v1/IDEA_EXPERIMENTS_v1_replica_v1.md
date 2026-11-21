# Improvement experiments v1 — rules-conscious OOF evidence

**Scope:** Continue from the corrected Phase 33/34 + constant-free CatBoost
OOF comparison. The prior same-fold reference has mean-fold AUC **0.767492**.
All experiments use the supplied training CSV and the same five
`StratifiedKFold` partitions (seed `20261113`). No external data, test
features, test labels, peer-label transfer, or per-row prediction rules are
used.

## Experiments and results

| Experiment | Standalone mean-fold OOF AUC | Best fixed blend with prior reference | Mean-fold blend AUC | Delta | Positive folds |
|---|---:|---|---:|---:|---:|
| Second-seed CatBoost | 0.765735 | 20% second-seed rank | 0.767619 | +0.000127 | 3/5 |
| External-score geometry CatBoost | 0.765788 | 20% geometry rank | 0.767580 | +0.000088 | 3/5 |
| Nested product-key TE LightGBM | 0.764848 | 20% TE rank | **0.767790** | **+0.000298** | **5/5** |
| Equal 10% total addition of all three new views | — | 90% prior + 3.33% each new view | 0.767631 | +0.000139 | 5/5 |

The standalone models were weaker than the reference. Their value, if any,
comes from complementary ranking information. The nested LightGBM had the
lowest Spearman rank correlation to the prior blend (**0.9744**, versus
0.9863 for the second seed and 0.9878 for external geometry), consistent with
greater ranking diversity. Diversity alone does not establish a score gain.

### Best exploratory blend by fold

| Fold | Prior reference | 80/20 prior + nested TE | Delta |
|---:|---:|---:|---:|
| 1 | 0.772128 | 0.772376 | +0.000248 |
| 2 | 0.774324 | 0.774856 | +0.000532 |
| 3 | 0.762217 | 0.762461 | +0.000244 |
| 4 | 0.764957 | 0.765063 | +0.000106 |
| 5 | 0.763834 | 0.764193 | +0.000359 |

These are mean-fold OOF comparisons, not leaderboard scores. The same labels
have been examined across prior research and this fixed-weight screen. The
best 20% weight was selected from a finite 10%/20%/50% sweep on these folds;
therefore the observed **+0.000298** is subject to selection optimism and
should be tested on the competition leaderboard before further weighting
decisions.

## Feature and validation details

### Second-seed CatBoost

- Reuses the existing row-wise feature builder and CatBoost settings:
  700 iterations, depth 6, learning rate 0.05, GPU.
- Uses the same outer fold assignments with a fixed seed offset of 50,000.
- No extra features or data sources.
- A 20% rank contribution slightly improved average fold AUC, but only on
  3/5 folds; it is not included in the submission candidate.

### External-score geometry

- Adds row-wise ratios for each pair among `EXT_SOURCE_1..3` and a category
  encoding their observed high-to-low ordering.
- Values come only from the provided training records; missing or zero
  denominators produce missing values.
- A 20% contribution gave only +0.000088 on 3/5 folds; it is not promoted.

### Nested product-key target statistics

- Uses the constant-free `AMT_ANNUITY / AMT_CREDIT` keys at 3/4/5 decimals
  and `AMT_ANNUITY / AMT_GOODS_PRICE` at 3 decimals.
- Adds smoothed target rates (alpha 30) and log support counts.
- Training-row encodings are inner five-fold OOF within each outer training
  partition. Each map's fit rows are disjoint from its encoded rows.
- Outer validation statistics use only that outer fold's training rows.
- Early-stopping rounds are selected using an inner split of the outer
  training partition; outer-validation labels do not affect fitting.
- It excludes the archived fee-offset and financing-step keys, direct
  matched-label propagation, train/test pooled statistics, and manual
  prediction edits.

The user-provided rules checklist explicitly accepts ordinary target
encoding when kept fold-local, but organizer interpretation is not formally
certified by this experiment. This is category-level supervised learning,
not copying labels between matched applicants.

## Submission candidate generation

`create_compliant_submission_te20_v1.py` retrains all four model streams
(Phase 33 LightGBM, Phase 34 XGBoost, clean CatBoost, nested TE LightGBM) on
the official training data and makes test predictions through those models.
It reconstructs OOF predictions, checks the proposed blend against the
reference, and writes no CSV unless the recomputed candidate improves mean
fold AUC and each fold is positive. The 20% weight is fixed and recorded.

The script reads `sample_submission.csv` only to verify the required two
columns and test-ID order. It does not submit or upload the CSV. Test AUC is
unknown until the user evaluates the file on the competition site.

## Artifacts

- Model code: the four `run_*.py` and `create_compliant_submission_te20_v1.py`
  scripts in this folder.
- OOF screen: `../runs/ideas_v1_blend_screen_v2/`.
- Model OOF runs: `../runs/clean_catboost_second_seed_v1/`,
  `../runs/clean_external_geometry_catboost_v1/`, and
  `../runs/nested_product_te_lgbm_v1/`.
- Submission files and manifest will be written to
  `submissions/product_te_20pct_v1/` if the independent OOF gate passes.

## Decision

The nested target-encoding stream is the only one of these three new
components worth carrying into a leaderboard trial. The local gain is small
and selected on repeatedly viewed labels; it is not evidence that the final
private leaderboard will improve. If the generated submission does not
improve externally, do not respond by searching more blend weights on these
same folds. The next controlled research direction is a fully nested,
fold-local multi-view stack, evaluated with a new outer split and a matched
baseline.

## Execution addendum — regenerated submission and multi-view experiment

The submission generator was executed from its script, retraining the
components rather than consuming cached OOF files as model inputs. The
five-fold OOF gate passed:

| Fold | Regenerated prior blend | Candidate with 20% nested TE | Delta |
|---:|---:|---:|---:|
| 1 | 0.772128 | 0.772376 | +0.000248 |
| 2 | 0.774324 | 0.774856 | +0.000532 |
| 3 | 0.762217 | 0.762461 | +0.000244 |
| 4 | 0.764881 | 0.765004 | +0.000122 |
| 5 | 0.763834 | 0.764193 | +0.000359 |
| **Mean** | **0.767477** | **0.767778** | **+0.000301** |

This regenerated baseline is 0.000015 lower than the earlier aligned-screen
baseline (0.767492). Treat the regenerated run's manifest and fold metrics as
authoritative for its CSV; the small discrepancy means the earlier screen
should not be presented as an exact reproduction of the regenerated run.
The candidate was emitted because its mean-fold gain was positive and all
five fold gains were positive. These folds were repeatedly examined during
development; this is a selection-biased development check, not an unbiased
generalization estimate or evidence of a leaderboard increase.

The generated file is
`submissions/product_te_20pct_v1/submission_compliant_product_te20_v1.csv`.
Independent checks confirmed exactly 61,500 rows, exactly the required
`SK_ID_CURR,TARGET` schema, unique IDs matching `sample_submission.csv` order,
and finite predictions within `[0,1]`. `MANIFEST.json` records the inputs,
code hashes, package versions, fixed blend, and rule controls. No upload was
performed; the competition score remains unknown.

### GPU multi-view probe

`run_compliant_multiview_catboost_v1.py` trained global CatBoost models on
three fixed feature sets, using only training rows and the same five
stratified folds. Standalone mean-fold AUCs were 0.712330 (external-score
geometry), 0.692380 (loan), and 0.672145 (person). A fixed equal-rank
combination of these views added at 10% to the prior reference reached
0.767570 (**+0.000078**, positive on 4/5 folds). Adding 20% reached 0.767361
and adding 30% reached 0.766847; both were below the prior reference
(0.767492). This screen therefore did not justify replacing the TE candidate
or adding the views to it.

The probe did not read `test.csv`, perform row-specific routing, pool
train/test statistics, or change individual predictions. Although these
feature views are rules-conscious and use only supplied columns, their blend
screen also reuses examined labels and is exploratory. The generated
submission remains the 80/20 prior-plus-TE candidate; leaderboard testing is
the next meaningful check.

## Mathematical follow-up and stricter reproduction run

The four Markdown research files were preserved. New code was added as
separate versioned scripts; prior runners and artifacts were not overwritten.

### Hash-locked TE20 candidate

`create_compliant_submission_te20_v2.py` checks SHA-256 values for the
official train, test, and sample files before loading them. CatBoost train
and test frames are kept separate. The model pipeline uses only official
training labels; test rows are used for inference, and the sample is used
only for schema and ID-order validation. Outer-training target statistics
are inner-OOF; validation and test maps use labels from the corresponding
outer-training fold only. No `partner_enc`, `PY_*`, matched-applicant label
transfer, external applicant data, manual overrides, or row-specific routing
is present.

The v2 test-perturbation check changed/reordered test applicant covariates
and verified unchanged sampled training and validation matrices. The
validation-label perturbation check flipped only outer-validation labels and
verified unchanged target-encoding features. Both passed. The script's
expected-data-hash assertions, output schema, ID order, uniqueness,
finite-value/range checks, OOF metrics, and output hashes all passed.

| Fold | Fresh reference | TE20 candidate | Delta |
|---:|---:|---:|---:|
| 1 | 0.772128 | 0.772376 | +0.000248 |
| 2 | 0.774324 | 0.774856 | +0.000532 |
| 3 | 0.762217 | 0.762461 | +0.000244 |
| 4 | 0.764749 | 0.764892 | +0.000143 |
| 5 | 0.763834 | 0.764194 | +0.000360 |
| **Mean** | **0.767450** | **0.767756** | **+0.000305** |

Candidate:
`submissions/product_te_20pct_v2/submission_compliant_product_te20_v2.csv`
(61,500 rows; exactly `SK_ID_CURR,TARGET`; IDs match sample order; values
finite and within `[0,1]`). No competition upload or leaderboard score is
known.

### AUC pair-count interpretation

AUC is the fraction of positive-negative pairs ordered correctly, with ties
counted as one-half. The train data contain 13,821 positives and 157,381
negatives, or 2,175,162,801 possible pairs. A 0.000305 AUC lift means roughly
0.0305 percentage points of the evaluated pair mass. Multiplying that
fraction by all full-data pairs would be an illustrative extrapolation only:
the measured OOF AUC compares rows within validation folds, and model
selection and fold dependence remain.

### Class-total-weighted Logloss

`run_balanced_loss_catboost_v1.py` used the training-fold ratio
`n_negative/n_positive` (about 11.387) as the positive class weight. This
balances total class contribution to Logloss; it is not equivalent to AUC
optimization and can alter learned splits/ranks.

- Standalone mean-fold AUC: **0.763813**.
- Fixed 90% baseline + 10% weighted-model rank blend: **0.767632**.
- Mean-fold lift: **+0.000140**, positive on 4/5 folds.

The fixed blend is weak and inconsistent; do not promote this model.

### Sampled pairwise-logistic ranker

`run_pairwise_auc_linear_v2.py` sampled 100,000 positive-negative pairs
uniformly from each outer training fold and trained on both directions of
each sparse feature difference. Uniform pair sampling targets an empirical
pairwise logistic surrogate that weights each positive-negative pair
equally in expectation. It does not directly optimize finite-sample ROC-AUC.
Imputation, scaling, and one-hot encoders are fit only on the outer training
rows. The code used no test rows, external data, or cross-row test
statistics. Increasing the SGD iteration limit from 30 to 100 resolved the
convergence warning but changed mean blend AUC only from 0.767689 to
0.767696.

- Standalone mean-fold AUC at 100 iterations: **0.757320**.
- Fixed 90/10 baseline/ranker blend: **0.767696**.
- Mean-fold lift: **+0.000204**, positive on 5/5 folds.

The low standalone score but small positive blend effect suggests limited
complementary ranking information, not a better standalone model.

### Fixed three-stream screen and promotion decision

`analyze_pairwise_te_blend_v2.py` tested one fixed formula without a weight
grid: 80% prior rank + 10% nested-TE rank + 10% pairwise rank, within each
fold. It scored **0.767835** versus **0.767477** for the freshly generated
reference (+**0.000358**, positive on 5/5 folds), only **+0.000057** over
TE20. It was worse than TE20 on fold 2. A paired DeLong-style interval
conditional on these fixed OOF scores was approximately
**[-0.000097,+0.000210]**, including zero. The interval ignores model
selection and cross-fold training dependence; it is not a formal
generalization interval. No pairwise-enhanced submission was generated.

For the TE20-vs-reference comparison, the analogous conditional interval
was approximately **[+0.000118,+0.000493]**. This does not correct for the
20% weight having been selected on repeatedly examined labels, so it is not
an independent significance claim.

### Reproducibility and rules boundary

The exact v2 generator was run twice into separate output directories. Both
OOF comparisons passed the five-fold gate:

| Run | Reference mean fold AUC | Candidate mean fold AUC | Gain | Positive folds |
|---|---:|---:|---:|---:|
| Primary v2 | 0.7674503449 | 0.7677555762 | +0.0003052312 | 5/5 |
| Same-script replay | 0.7674683940 | 0.7677716589 | +0.0003032649 | 5/5 |

The submissions were highly rank-correlated (**Spearman 0.999990**) but
not identical: maximum absolute `TARGET` difference **0.01333**, mean
absolute difference **0.000860**. Phase 33, Phase 34, and nested-TE test
prediction streams matched exactly; the difference was isolated to the
CatBoost GPU stream. This is evidence of GPU execution variability with
fixed seeds and versions, not a byte-reproducible run. The manifests record
input, source, and output hashes and execution versions.

Retain the first v2 output as the primary candidate; do not choose the
slightly higher-scoring replay after examining its OOF result. These are
development-fold metrics, not leaderboard scores or independent estimates.
The v2 candidate and its manifest remain the primary artifact.

The experiments follow the supplied restrictions: official data only,
model-generated predictions, fold-local target statistics, no label
propagation, no manual test edits, and no test-set statistics used to fit
training features. The user-supplied checklist accepts fold-local target
encoding, but no local analysis can guarantee organizer-certified
interpretation of every rule. The current candidate is around **0.7678 OOF**;
**0.79 has not been reached or guaranteed**. Public leaderboard testing is
the next external check, not another weight sweep on these same folds.

## Direct pairwise ranking research — 2026-11-15

### Mathematical motivation

For binary targets, AUC is the average positive-over-negative ordering
indicator across all \(N_+N_-\) pairs, with ties receiving one-half credit.
It is invariant to strictly monotone transformations, so the important
quantity is each positive-negative score margin, not probability
calibration. Pairwise logistic loss replaces the discontinuous pair
indicator with a smooth penalty:

\[
L_{\mathrm{pair}}=\frac{1}{|S|}\sum_{(i,j)\in S}
\log\!\left(1+\exp(-(s_i-s_j))\right),\quad y_i=1,\ y_j=0.
\]

With all pairs and a globally comparable score this aligns more closely
with AUC than ordinary pointwise Logloss, but it is only a surrogate. The
CatBoost experiment capped its generated pair set at one million pairs per
outer training fold; XGBoost used its `mean` pair-construction strategy with
eight sampled pairs per sample (about 1.1 million comparisons per fold).
Those samples approximate, rather than enumerate, the roughly billions of
possible training-fold pairs.

### Fold-isolated experiments

Both experiments used the fixed five-fold stratified split, GPU training,
and official `train.csv` only. Each validation fold's labels were excluded
from fitting and used only to calculate its AUC. Each pairwise model used one
global ranking group inside its training fold, so generated comparisons were
positive-versus-negative applicants from that fold. The baseline OOF file
was used only for alignment and the pre-specified 90/10 rank blend; it was
not a model feature.

| Model | Standalone mean-fold AUC | 90/10 blend vs TE20 | Gain | Positive folds | Decision |
|---|---:|---:|---:|---:|---|
| CatBoost `PairLogit:max_pairs=1000000`, 500 iterations | 0.758634 | 0.767719 vs 0.767756 | -0.000037 | 3/5 | Do not promote |
| XGBoost `rank:pairwise`, mean strategy, 8 pairs/sample, 1,000 trees | 0.760821 | 0.768057 vs 0.767756 | +0.000301 | 5/5 | Rebuild as candidate |

The CatBoost result shows that selecting a loss that resembles AUC does not
guarantee a lift: the tree's learned ordering and pair sample still matter.
The XGBoost result is more useful as a complementary stream than as a
standalone model.

### End-to-end candidate regeneration

The separate generator
`create_compliant_submission_te20_xgb_rank_pairwise_v1.py` retrained the
existing Phase 33, Phase 34, clean CatBoost, nested product-TE LightGBM, and
new XGBoost ranking components; it did not read cached prediction files as
model inputs. It uses a fixed final blend:

\[
s_{\mathrm{new}}=0.90\,\operatorname{rank}(s_{\mathrm{TE20}})
+0.10\,\operatorname{rank}(s_{\mathrm{XGBPairwise}}).
\]

For the test set, the pairwise model's within-test ranks are averaged across
the five fold models, then combined with the TE20 score. The per-row output
is generated by those models and this fixed formula; there is no manual
override, label propagation, or row-specific routing.

| Fold | Regenerated TE20 baseline | Pairwise candidate | Delta |
|---:|---:|---:|---:|
| 1 | 0.772376 | 0.772523 | +0.000147 |
| 2 | 0.774683 | 0.774888 | +0.000206 |
| 3 | 0.762460 | 0.762495 | +0.000035 |
| 4 | 0.765004 | 0.765493 | +0.000489 |
| 5 | 0.764193 | 0.764498 | +0.000305 |
| **Mean** | **0.767743** | **0.767979** | **+0.000236** |

The generator's fold seed schedule differs from the earlier isolated
pairwise screen, and it regenerates the base models; therefore its fresh
OOF comparison is the authoritative one for the candidate. It passed the
predeclared all-five-fold improvement gate. The file
`submissions/te20_xgb_rank_pairwise_v1/submission_compliant_te20_xgb_rank_pairwise_v1.csv`
has 61,500 rows, exactly `SK_ID_CURR,TARGET`, unique IDs in sample order,
finite values in `[0,1]`, and a manifest-verified SHA-256 of
`7cf21df3ba752d09ad487593fca8905c834f0e1d7334c97b6c953952d37a3023`.
No leaderboard upload was performed.

### Exact negative-decile AUC decomposition

To investigate whether the lift came from the low-risk bulk or risky region,
each validation fold's negatives were sorted by the TE20 baseline score and
split into ten near-equal deciles. For decile \(k\), every validation
positive was compared with every negative in that bin. Since each
positive-negative pair belongs to exactly one negative bin, the decomposition
is exact:

\[
\Delta\operatorname{AUC}
=\sum_{k=1}^{10}\frac{N^-_k}{N^-}\Delta\operatorname{AUC}_k.
\]

The mean-fold weighted contributions for deciles 1 (lowest baseline risk)
through 10 (highest) were:

| Baseline-negative decile | Weighted AUC contribution |
|---:|---:|
| 1 | +0.000001 |
| 2 | +0.000001 |
| 3 | -0.000002 |
| 4 | +0.000044 |
| 5 | +0.000017 |
| 6 | -0.000014 |
| 7 | +0.000009 |
| 8 | +0.000034 |
| 9 | +0.000084 |
| 10 | +0.000063 |
| **Sum** | **+0.000236** |

The gain is concentrated in negative deciles 8–10, especially decile 9.
Although AUC gives each positive-negative pair equal weight, this model did
not improve the low-risk bulk materially. This decomposition is descriptive
OOF diagnosis, not a rule used to alter or route test predictions.

The separate CatBoost PairLogit branch was excluded from the final
generator. The XGBoost candidate's score remains an OOF result, not a
leaderboard score; it is around **0.768**, so **0.79 has not been reached or
guaranteed**. These same folds have been examined in earlier research, so
do not tune further weights on them. A next defensible experiment should
nest blend selection inside each outer training partition.

### Rule and source notes

- Official train labels only; no external datasets, matched-applicant label
  propagation, or manually assigned test scores.
- Pair training and key-frequency feature fitting are restricted to each
  outer training fold; validation labels are metric-only.
- `test.csv` is used only for inference; the sample file is used only for
  output schema/ID-order checks. All test scores come from trained models.
- Fixed blend weights; no per-row model routing. Hash-locked inputs,
  versions, source, and outputs are in the candidate `MANIFEST.json`.
- These controls follow the user-provided competition checklist, but do not
  constitute an organizer ruling. Exact replay of GPU CatBoost output is
  not guaranteed.

References:

- [CatBoost ranking objectives, `PairLogit`, and `max_pairs`](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
- [CatBoost AUC definition](https://catboost.ai/docs/en/concepts/loss-functions-classification)
- [XGBoost learning-to-rank objectives and pair sampling](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [Cawley & Talbot (2010), model-selection overfitting and evaluation bias](https://jmlr.org/papers/v11/cawley10a.html)
- [GCI World Competition repository](https://github.com/Hareen-Liyanage/GCI_World_Competition)
