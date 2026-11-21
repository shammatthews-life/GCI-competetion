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

## Follow-up research — fold-local group ranks and histogram resolution (2026-11-16)

### Decision

Neither new approach reaches the predeclared **+0.005 mean-fold OOF AUC**
promotion threshold. Do not submit a CSV from either experiment. The requested
rule-compliant improvement target is still unmet; these experiments do not
support an expected **+0.002 leaderboard** gain. There is no mathematical
conversion from a local OOF delta to a leaderboard delta: the public score is
computed on a different subset, and its relationship to these repeatedly
examined folds is unknown.

The comparison baseline is the regenerated TE20 OOF candidate, on the same
five outer folds. The previous OOF file is a benchmark only; neither new model
uses its predictions as input features.

| Experiment | Standalone expert mean-fold AUC | Fixed blend mean-fold AUC | Delta vs TE20 | Positive folds | Decision |
|---|---:|---:|---:|---:|---|
| Fold-local group-conditional rank expert; fixed 90/10 rank blend | 0.762655 | 0.767949 | +0.000194 | 5/5 | Reject for submission; below threshold |
| XGBoost `max_bin=512`; fixed 90/10 rank blend | 0.764265 | 0.767710 | -0.000046 | 2/5 | Reject |

Baseline mean-fold AUC was **0.767756**. The first expert was weaker by
0.005100 as a standalone model; its small positive blend effect comes from
complementarity, not from beating the baseline itself. The second model
underperformed the baseline both standalone and in the fixed blend.

### AUC mathematics and why the deltas are small

For \(P\) positives and \(N\) negatives, empirical AUC is the average
positive-negative ordering indicator (ties count one-half):

\[
\operatorname{AUC}(s)=\frac{1}{PN}\sum_{i:y_i=1}\sum_{j:y_j=0}
\left[\mathbf{1}(s_i>s_j)+\tfrac12\mathbf{1}(s_i=s_j)\right].
\]

A strictly increasing transform of one score vector cannot change its AUC.
A blend can change AUC only when it changes the relative order of positive-
negative pairs. A weak model can therefore add a small gain if it repairs a
few baseline pairwise errors, but its noisy rank disagreements can just as
easily create new errors. Model probability calibration is not itself the
source of a lift.

For diagnosis, each fold's negatives were sorted by the TE20 baseline and
partitioned into ten equal-count deciles. For decile \(k\), every validation
positive was compared with every negative in that bin. The weighted
contributions sum exactly to the fold AUC delta:

\[
\Delta\operatorname{AUC}
=\sum_{k=1}^{10}\frac{N_k^-}{N^-}
\left(\operatorname{AUC}_{k,\mathrm{candidate}}
-\operatorname{AUC}_{k,\mathrm{baseline}}\right).
\]

Mean-fold contributions for the group-rank blend were approximately
**[-0.000002, -0.000009, +0.000013, +0.000005, +0.000012, +0.000008,
+0.000021, +0.000091, +0.000037, +0.000017]** from lowest- to highest-risk
baseline negative decile; their sum is +0.000194 (rounding). Nearly all of
the net gain came from deciles 8–10, especially decile 8. The low-risk
negative bulk did not provide a meaningful gain. This is diagnostic only;
no group or decile was used to route or edit predictions.

For `max_bin=512`, the corresponding contributions were approximately
**[-0.000006, -0.000002, -0.000006, -0.000003, +0.000014, -0.000003,
-0.000012, -0.000022, -0.000028, +0.000023]**, summing to -0.000046.
There is no consistent pairwise region of improvement. The fine-bin model's
extra resolution did not translate into better out-of-fold ordering; its
rank stream mostly adds noise relative to TE20 at the fixed 10% weight.

### Experiment 1: conditional empirical ranks within supplied groups

For each supplied applicant, seven continuous values (three external scores,
credit/income, annuity/credit, birth days, and cleaned employment days) were
mapped to their midrank empirical CDF within six supplied categories. Each
CDF used only its fit partition; groups with fewer than 30 fit rows fell back
to that partition's global CDF. Missing values stayed missing. The expert
also retained the standard Phase 33 raw features, so it could learn from
absolute values and relative position together.

Training-row group-rank features were cross-fitted within each outer training
fold. Outer-validation group ranks used only the outer-fit rows. The new
model's labels, encoders, vocabularies, and frequency matrices never included
the scored fold. A fixed 10% within-fold rank blend with the TE20 baseline
was specified before running; no weight sweep was performed.

Why it did not reach the target: the expert's standalone mean AUC (0.762655)
was lower than TE20, showing that these relative ranks do not replace the
global loan-risk ordering. The small blend gain indicates some complementary
pair corrections, concentrated in upper-risk negative deciles, but the
signal is too small to justify a submission. A plausible contributor to the
weak standalone result is redundant feature competition: the underlying
LightGBM already has raw measurements and category indicators from which it
can learn many of these interactions. The experiment establishes the
performance result, not a unique causal explanation for it.

### Experiment 2: higher GPU histogram resolution

XGBoost's `max_bin` was fixed at 512; other Phase 34 model settings and the
fold-local early-stopping procedure were kept fixed. This followed a prior
single-inner-split signal for finer bins, but the full five-fold result did
not reproduce that signal. Mean-fold AUC was 0.764265 for the expert and
0.767710 for the fixed 90/10 blend, **-0.000046** versus TE20, with only two
positive folds. The prior one-split gain was therefore not sufficient
evidence of general improvement. Finer bins increase threshold resolution,
but do not guarantee lower generalization error; the added split choices
can fit fold-specific noise, and the original regularization/round schedule
may not be optimal for the changed histogram resolution.

### What the wider version history tells us

The results across the earlier versioned work are not one comparable
leaderboard. Fold seeds, validation protocols, features, and model
provenance differ; a score from one version cannot be subtracted from a
score from another and called a controlled gain. In the aligned experiments
available here, the pattern is:

- **Useful economic ratios / interactions:** can help when they expose a
  nonlinear repayment boundary that a tree would otherwise need many splits
  to approximate. In the aligned Phase 33 feature screen, financial ratios
  alone scored 0.759404 versus 0.758592 for the base view (+0.000812), while
  constant-free product keys/frequencies scored 0.763309 (+0.004717 vs base).
  Adding the broader `all_safe` feature set scored 0.763277, essentially flat
  versus keys/frequencies (-0.000031): more safe features did not improve the
  model. Larger lifts in earlier screens used different baselines and must be
  retested under the exact final validation protocol before being treated as
  durable gains.
- **Nested constant-free product target encoding:** provides a small,
  repeatable complementary gain in the current aligned runs (about +0.00030
  mean-fold AUC), but it does not by itself bridge a large leaderboard gap.
- **Class reweighting:** balancing logloss changes split gradients, not
  directly the pairwise AUC objective; the prior fixed blend gained only
  +0.000140 on 4/5 folds.
- **Pairwise rankers:** the tested ranker was weaker alone (about 0.7608)
  and helped the TE20 blend by only about +0.00024 in the regenerated run.
  A ranking-shaped loss is a surrogate and does not guarantee better
  held-out pair ordering.
- **More views or features are not automatically better:** the prior three
  multi-view CatBoost streams improved by only +0.000078 at 10% total weight,
  while 20% and 30% additions lost. Redundant columns can dilute tree split
  selection; separate specialists can lose sample size and cross-group
  comparisons.
- **Nested stacking can be more promising than manually chosen routing,**
  but it must regenerate all level-one predictions inside each outer
  training fold. The historical V28 report lists 0.771679 under a different
  multi-stream, nested design and split. It is not a same-fold comparison to
  the current 0.767756 baseline and should not be claimed as a fresh gain
  until its complete code and fold provenance are independently reproduced.
  Its EDA-screened financing constant is also outside this pass's conservative
  feature boundary until its derivation and organizer interpretation are
  established.

These observations fit the pairwise AUC definition: a feature or model helps
only when it corrects enough positive-negative orderings to outweigh new
pair inversions. Calibration changes alone cannot do that. The fixed blend
screen also avoids fitting weights on this evaluation set, but the folds
have been examined in earlier development, so even these OOF deltas are
exploratory rather than independent estimates.

### Reproduction artifacts and compliance boundary

- Group-rank code:
  `ideas_v2/group_conditional_rank_v1.py`
- Group-rank OOF and metrics:
  `ideas_v2/runs/group_conditional_rank_v1/OOF_PREDICTIONS.csv` and
  `ideas_v2/runs/group_conditional_rank_v1/FOLD_METRICS.csv`
- Histogram-resolution code:
  `ideas_v2/xgboost_maxbin512_rank_v1.py`
- Histogram-resolution OOF and metrics:
  `ideas_v2/runs/xgboost_maxbin512_rank_v1/OOF_PREDICTIONS.csv` and
  `ideas_v2/runs/xgboost_maxbin512_rank_v1/FOLD_METRICS.csv`
- Each run's `SUMMARY.json` records exact fold results, configuration,
  software versions, runtime, SHA-256 hashes, and controls.

The group-rank model used only official `train.csv`; the high-bin model also
used only official `train.csv` and CUDA XGBoost 3.4.1. Neither run read
`test.csv`, used external data, transferred labels between applicants,
pooled test covariates, manually changed predictions, or generated a
submission. The saved TE20 OOF baseline was read only for row/fold alignment
and metric comparison, never as a feature. Neither experiment passed the
predeclared +0.005 gate, so no test submission CSV was created.

## Follow-up research — raw-feature TabPeriodic-SwiGLU (2026-11-16)

### Hypothesis and controlled design

The previous two tests showed that a weak rank stream can at best contribute
tiny complementary gains and that finer tree bins do not guarantee improved
pair ordering. A genuinely different GPU model family was therefore tested:
periodic Fourier embeddings plus gated residual layers. This choice was
motivated by the historical nested V28 report, where two neural streams had
nonzero meta-weights. That historical result was only a lead: its model
inputs included residuals and supervised view features, and its folds and
feature pipeline differ from the current TE20 benchmark.

This scratch implementation intentionally used a conservative feature set:
raw numeric columns, guarded row-wise financial ratios, external-score
summaries/missingness, and training-fitted one-hot categorical fields. It
excluded target encoding, cached component predictions, EDA-selected
financing offsets, external data, train/test pooling, and applicant-label
transfer. Numeric imputers, clipping bounds, scales, and categorical
vocabularies were fit on each inner/outer fit partition. Epoch count was
selected only on an inner holdout of each outer training fold. A fixed
80/20 within-fold rank blend with TE20 was specified; there was no weight
sweep.

### Result: reject

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Raw-feature TabPeriodic-SwiGLU expert | 0.747903 | -0.019853 | 0/5 |
| Fixed 80/20 TE20/neural rank blend | 0.766660 | -0.001096 | 0/5 |

The neural model reduced AUC in every outer fold. Its inner stopping AUCs
were only 0.7383–0.7484, so this is not a case where calibration alone
concealed a strong ranker. The fixed blend also lost in all folds; no test
submission was generated and the +0.005 promotion gate was not approached.

### Failure analysis and next-search filter

This failure differs from the earlier pairwise and group-rank failures. The
new architecture did not recover enough signal from a raw-only application
representation. The historical neural lead used a richer input containing
fold-local covariate residuals, nested product target statistics, and
cross-fitted model-view predictions. Removing those inputs made the model
rules-conservative but much weaker. This suggests the model family alone is
not the improvement; representation and cross-fitting are likely decisive.
The result does **not** justify increasing the neural blend weight or
searching more epochs on these same outer folds.

The next candidate should add only fold-local, reproducible learned
representation (not test-pooled statistics or transferred labels). In
particular, unsupervised covariate residuals can be built by fitting
regressors on each inner-fit partition, generating residual features for its
held-out training rows, and applying outer-fit regressors to outer validation
rows. This tests the missing representation with no `TARGET` use in residual
construction. Any target statistics used in a later stack must likewise be
nested inside the relevant training partition.

### Artifacts and controls

- New code: `ideas_v2/tabperiodic_swiglu_v1.py`
- OOF predictions: `ideas_v2/runs/tabperiodic_swiglu_v1/OOF_PREDICTIONS.csv`
- Fold metrics and selected epochs:
  `ideas_v2/runs/tabperiodic_swiglu_v1/FOLD_METRICS.csv`
- Hashes, package versions, GPU, and fold-level results:
  `ideas_v2/runs/tabperiodic_swiglu_v1/SUMMARY.json`

The run used official `train.csv` only and required CUDA
(`NVIDIA GeForce RTX 4050 Laptop GPU`; PyTorch `2.14.1+cu130`). The baseline
OOF file was used solely for row/fold alignment and score comparison; it was
not a model feature. Validation labels were used only for inner epoch
selection and outer AUC reporting. No test predictions or submission were
created.

## Follow-up research — fold-local covariate residual expert (2026-11-16)

### Hypothesis and controlled design

The raw-only neural model failed, while historical nested V28 used richer
learned views. This test isolated one such view: deviations of selected
applicant covariates from expectations learned from other supplied
covariates. For each of `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`,
`AMT_CREDIT`, `AMT_INCOME_TOTAL`, and `DAYS_EMPLOYED`, an XGBoost regressor
predicted that field from the other numeric and one-hot encoded training
fields. The response field itself was masked from its predictor matrix.
Amounts were modeled on the log scale; the employment sentinel was treated
as missing.

Inside each outer training fold, three-way cross-fitting generated
out-of-fold predicted-value and residual columns for every training row.
Each outer-validation row received predictions from regressors trained only
on the corresponding outer-fit partition; residuals used only its supplied
covariate values. A LightGBM classifier combined these features with the
fold-safe `keys_frequency` feature view. The single predeclared comparison
was a 90/10 within-outer-fold rank blend with the TE20 OOF benchmark.

### Result: reject for promotion

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Covariate-residual expert | 0.763772 | -0.003984 | — |
| Fixed 90/10 TE20/residual rank blend | 0.768141 | +0.000385 | 5/5 |

The blend improved each outer fold, but the gain was **0.000385**, far below
the fixed **+0.005** promotion threshold. The standalone residual expert
lost to the benchmark in every fold. No submission CSV was generated.

The pairwise decomposition divides each fold's negatives into ten bins by
baseline score, then computes each bin's weighted contribution to the change
in positive-versus-negative AUC:

`delta_AUC = sum_j (n_neg,j / n_neg) * delta_AUC_j`.

The mean contribution from negative deciles 1–3 was approximately
`-0.000005`; deciles 7–10 contributed `+0.000297` of the `+0.000385`
total. This says the small gain came mainly from correcting rankings among
higher-scored baseline negatives, not from a broad correction of the
low-risk bulk. The residual view contains some complementary pairwise
signal, but its weaker standalone ordering limits its useful blend share.
Do not respond by blindly increasing its fixed weight; the next distinct
test should evaluate nested model-view stacking, which was present in the
historical V28 approach but is absent from the raw-only neural candidate.

### Artifacts and rule controls

- New code: `ideas_v2/covariate_residual_expert_v1.py`
- OOF predictions: `ideas_v2/runs/covariate_residual_expert_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/covariate_residual_expert_v1/FOLD_METRICS.csv`
- Versions, fold metrics, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/covariate_residual_expert_v1/SUMMARY.json`
- Code SHA-256:
  `9d859b30baf369a45458288e6252316277e2859ee01fc46bc0c9d2efd316a463`
- OOF SHA-256:
  `4e31bb89637a6d2f09122ab42acc6a27b915e42113fcffe8496260b1ecc3f4c4`
- Runtime: 375.9 seconds; XGBoost 3.4.1 CUDA residual regressors,
  LightGBM 4.7.0 classifier.

Only official `train.csv` was read as model data; `test.csv` was not read.
Residual regressors did not use `TARGET`; the supervised classifier used
only outer-fit labels, with inner cross-fitting for its training residual
features. The prior OOF file served only as an aligned benchmark and was
never a feature. No external data, applicant-label transfer, manual
prediction edits, or submission file was used.

## Follow-up research — cross-fitted model-view stack (2026-11-16)

### Hypothesis and controlled design

The residual experiment produced a small, consistent blend gain, and the
historical V28 architecture used cross-fitted model-view logits. This test
implemented that principle without copying V28's EDA-selected pricing
offsets or label-transfer methods. Four existing rule-first views (`base`,
guarded `financial`, `peer`, and `keys_frequency`) were fit as inner-fold
LightGBM models. For each outer training row, its four view scores were
generated only by models that excluded that row. Outer-validation view
scores were averaged across models trained only on outer-fit rows. A
fold-safe `all_safe` LightGBM model used those scores alongside its ordinary
features. The fixed comparison remained a 90/10 within-fold rank blend with
TE20.

### Result: reject for promotion

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Cross-fitted model-view expert | 0.761273 | -0.006483 | — |
| Fixed 90/10 TE20/model-view rank blend | 0.767919 | +0.000163 | 5/5 |

This is the fourth added candidate below the +0.005 gate. The weak
all-safe view family did not become a strong standalone model merely by
stacking its scores. Its small complementarity is repeatable across the
five folds, but materially smaller than the residual-view blend
(+0.000385). The fixed 10% share was not optimized after observing the
results.

The same negative-decile pair decomposition attributes about `+0.000153`
of the net `+0.000163` to baseline-negative deciles 8–10; deciles 1–3
contributed about `-0.000037`. Thus the stack mostly adjusts a narrow
higher-risk part of the negative ranking and slightly harms the lowest
negative-score bins. This does not support a claim that the view scores
improve the full ranking by the required amount. Next experiments should
prefer a stronger supervised representation rather than further combining
these already-weak rule-first experts.

### Artifacts and rule controls

- New code: `ideas_v2/crossfit_model_views_v1.py`
- OOF predictions: `ideas_v2/runs/crossfit_model_views_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/crossfit_model_views_v1/FOLD_METRICS.csv`
- Versions, fold metrics, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/crossfit_model_views_v1/SUMMARY.json`
- Code SHA-256:
  `719d2ba7b774eedb6ba6aff90820f8176be9cd24f30e55862721dd8756b55fec`
- OOF SHA-256:
  `4ad7bfc5e9e04457279b33c927a3a5007df9ec99ab17965b6f9a957c68163afb`
- Runtime: 889.1 seconds; LightGBM 4.7.0.

Only official `train.csv` was read as model data; `test.csv` was not read.
Training view scores were inner out-of-fold; outer validation view scores
were generated from outer-fit rows only. Labels were used only to fit each
partition's supervised models and to report outer AUC. The benchmark was
used only for alignment and comparison. There was no external data, label
transfer, manual prediction editing, or submission generation.

## Follow-up research — fold-local categorical target encoding (2026-11-16)

### Hypothesis and controlled design

The existing TE20 candidate target-encodes four rounded loan-ratio keys.
This test asked whether additional supervised encoding of supplied
categorical fields and four predeclared interactions could add information.
It encoded all eight categorical columns and the fixed pairs
`organization×occupation`, `income type×education`, `gender×education`,
and `region rating×housing`. Each key received a smoothed mean,

`TE(k) = (sum(TARGET_k) + 30 * prior) / (count(k) + 30)`,

plus `log1p(count)`. The TE and count for every outer-training row were
constructed from only the other two inner folds; outer-validation maps used
only outer-fit labels. The existing `all_safe` feature matrix was retained,
and the only comparison was a predeclared 90/10 within-fold rank blend with
TE20.

### Result: reject for promotion

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Categorical-TE expert | 0.762852 | -0.004903 | — |
| Fixed 90/10 TE20/category-TE rank blend | 0.767955 | +0.000199 | 5/5 |

The improvement is consistent but too small: `+0.000199` versus the
unchanged +0.005 gate. The category-TE expert is weaker alone on every fold.
The paired negative-decile breakdown is diffuse compared with the
covariate-residual expert: the positive contribution is spread mostly over
baseline-negative deciles 5–10 rather than being a single large change in
one bin. The maximum mean per-decile contribution is about `+0.000038`.
The categorical summaries therefore add limited pairwise complementarity,
not the stronger global ordering needed to approach the goal.

### Artifacts and rule controls

- New code: `ideas_v2/fold_local_category_te_v1.py`
- OOF predictions: `ideas_v2/runs/fold_local_category_te_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/fold_local_category_te_v1/FOLD_METRICS.csv`
- Versions, fold metrics, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/fold_local_category_te_v1/SUMMARY.json`
- Code SHA-256:
  `03b8fbb547fecd3a88b6cb9a12e5b1144f327edcf25b876b2cb4b8c63e0cb48a`
- OOF SHA-256:
  `70d9a69f735d88841f1fa2d40df88e2ce73291f24a844a68694561bde64a8332`
- Runtime: 158.3 seconds; LightGBM 4.7.0.

Only official `train.csv` was read as model data; `test.csv` was not read.
Training encodings were inner out-of-fold; outer validation encodings used
only outer-fit labels. No applicant ID or matches were used as encoding
keys. The TE20 file was used solely for alignment and score comparison.
No external data, label transfer, manual edits, or submission was generated.

## Follow-up research — engineered CatBoost GPU expert (2026-11-16)

### Hypothesis and controlled design

The fold-local target-encoding expert was weak alone, but it gave a small
positive fixed-blend gain. A separate model family may capture categorical
and feature interactions differently, so the existing constant-free feature
builder was evaluated with CatBoost's native ordered categorical processing
and GPU training. The feature set uses supplied row fields, guarded ratios,
external-score summaries, and rounded financial keys. The model was fixed at
700 iterations, depth 6, learning rate 0.05, and `max_ctr_complexity=1`.
The fixed comparison remained a 90/10 within-fold rank blend with TE20.

### Result: reject

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Engineered CatBoost GPU expert | 0.765945 | -0.001810 | — |
| Fixed 90/10 TE20/CatBoost rank blend | 0.767752 | -0.000004 | 2/5 |

CatBoost's standalone score was closer to TE20 than the raw-only neural,
but the fixed blend was effectively unchanged overall and inconsistent
across folds. This shows that a different learner alone does not guarantee
complementary pair corrections. Do not promote or retune against these same
folds based on this result.

### Artifacts and rule controls

- New code: `ideas_v2/catboost_engineered_expert_v1.py`
- OOF predictions: `ideas_v2/runs/catboost_engineered_expert_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/catboost_engineered_expert_v1/FOLD_METRICS.csv`
- Versions, fold metrics, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/catboost_engineered_expert_v1/SUMMARY.json`
- Code SHA-256:
  `5df935b55b2827f08d0ec47383f35cf5151ce279c183bdf03a3a4f7592739bdf`
- OOF SHA-256:
  `9eb70dce41e1886892fcd7fa4c1fabb6ed694859c00f4981c44d60e658fc92f5`
- Runtime: 200.5 seconds; CatBoost 1.2.10 using CUDA.

Only official `train.csv` was read as model data; `test.csv` was not read.
CatBoost training used only outer-fit labels, and evaluation was on the
excluded fold. The benchmark was used only for alignment and comparison.
No external data, applicant-label transfer, manual edits, test predictions,
or submission was used.

## Follow-up research — fixed blend of four complementary experts (2026-11-16)

### Hypothesis and controlled design

The group-conditional rank, covariate-residual, cross-fitted model-view, and
fold-local categorical-TE experts each showed small complementary effects
with TE20. This test combined their already-generated, fold-aligned OOF
scores with equal rank shares, without fitting another model or searching
weights:

`0.90 rank(TE20) + 0.025 * sum(rank(each of four experts))`.

All ranks were calculated within the same outer validation fold. The source
files were checked for identical row IDs, labels, fold assignments, and
baseline predictions before combining.

### Result: reject for promotion

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Fixed four-expert 90/10 rank blend | 0.768025 | +0.000269 | 5/5 |

This fixed combination is the best of these new trials so far, but remains
far below the required **+0.005**. The highest fold gain was +0.000440;
the gain is consistently positive but modest. The negative-decile
decomposition places most of the lift in baseline-negative deciles 6–10;
deciles 8–9 alone contribute about +0.000130 on average. The lower-risk
bulk is largely unchanged. Combining complementary weak learners helps
slightly, but the four streams share enough of the same global ranking that
their errors do not yield a large additional correction.

Because this is a post-hoc combination of experts selected after repeatedly
examining the same fixed OOF folds, the result is exploratory rather than
an independent generalization estimate. It is not a submission candidate.

### Artifacts and rule controls

- New code: `ideas_v2/complementary_expert_blend_v1.py`
- OOF predictions: `ideas_v2/runs/complementary_expert_blend_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/complementary_expert_blend_v1/FOLD_METRICS.csv`
- Negative-decile decomposition:
  `ideas_v2/runs/complementary_expert_blend_v1/NEGATIVE_DECILE_DECOMPOSITION.csv`
- Full formula, fold metrics, input/output hashes, and controls:
  `ideas_v2/runs/complementary_expert_blend_v1/SUMMARY.json`
- Code SHA-256:
  `74477a45967e057d25022a5497f3e25185be30901cd76677c65f08cd8926005e`
- OOF SHA-256:
  `ec417666c7f98f1e7e5db193f791cf3a063ad5443d5060d522fdadd68040ffd1`
- Runtime: 3.2 seconds.

This post-processing step used official `train.csv` only for ID/label checks
and OOF metric calculations, plus the four existing fold-isolated OOF
prediction files. It read no test rows, used no external data, performed no
row-specific edits or label transfer, and generated no submission.

## Follow-up research — five-expert equal-share blend (2026-11-16)

### Hypothesis and controlled design

The four-expert mixture gained +0.000269. A previously evaluated
fold-isolated XGBoost `rank:pairwise` stream had also produced positive
pairwise blending results, so it was added as a fifth equally weighted
stream. The fixed formula keeps the total non-baseline contribution at
10%:

`0.90 rank(TE20) + 0.02 * sum(rank(each of five experts))`.

The five experts are group-conditional rank, covariate residual,
cross-fitted model views, fold-local categorical TE, and XGBoost pairwise.
No weights were tuned.

### Result: reject for promotion

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Fixed five-expert 90/10 rank blend | 0.768048 | +0.000293 | 5/5 |

This improves only +0.000024 over the four-expert blend, still far short of
+0.005. The incremental pairwise stream adds a tiny consistent contribution
under its 2% share, but does not materially change the conclusion.
All expert combinations so far mainly improve the upper half of the
baseline-negative ranking; they do not achieve a large correction across
the low-risk bulk. The cumulative evidence is that the current weak experts
are complementary but share too much ranking signal, and simply adding
more of them is unlikely to reach the promotion threshold.

The result is post-hoc on repeatedly used folds, not an independent
generalization estimate or a submission candidate.

### Artifacts and rule controls

- New code: `ideas_v2/five_expert_blend_v1.py`
- OOF predictions: `ideas_v2/runs/five_expert_blend_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/five_expert_blend_v1/FOLD_METRICS.csv`
- Formula, fold metrics, input/output hashes, and controls:
  `ideas_v2/runs/five_expert_blend_v1/SUMMARY.json`
- Code SHA-256:
  `1d08be93b7381fb6130ec2360fb66f1cefc0a3a57458a631e3db3fed7c239f44`
- OOF SHA-256:
  `6e66bb0f6bacf425a7c6ff6637cb4dfb81b256a604b574f7c789d07c019ec2f6`
- Runtime: 3.1 seconds.

The combination used official training data only for alignment and OOF
metrics, and existing outer-fold OOF predictions from the five models.
It read no test rows, external data, or matching labels; it made no
row-specific edits and generated no submission.

## Follow-up research — fixed six-expert blend (2026-11-16)

### Hypothesis and controlled design

The five-expert blend gained +0.000293. A sixth, cross-fitted hard-example
expert was added with the same equal-share rule, reducing each expert's
contribution from 2% to 1.667% while retaining 90% TE20:

`0.90 rank(TE20) + (0.10 / 6) * sum(rank(each of six experts))`.

All component OOF files were checked for identical IDs, labels, folds, and
baseline predictions. No weights were searched.

### Result: reject; adding the sixth expert slightly reduced lift

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Fixed five-expert blend | 0.768048 | +0.000293 | 5/5 |
| Fixed six-expert blend | 0.768034 | +0.000278 | 5/5 |

The hard-example stream did not add useful ranking information to the
mixture. The gain stayed positive across folds but slipped by about
0.000015 versus the five-expert blend. The best new result in this round
remains the covariate-residual expert's fixed 90/10 blend at **+0.000385**;
none of the new tests met the +0.005 gate. This trend—many weak streams
yielding small, unstable incremental gains—warns against extending the
post-hoc blend search on these same folds.

### Artifacts and rule controls

- New code: `ideas_v2/six_expert_blend_v1.py`
- OOF predictions: `ideas_v2/runs/six_expert_blend_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/six_expert_blend_v1/FOLD_METRICS.csv`
- Formula, input/output hashes, fold results, and compliance checks:
  `ideas_v2/runs/six_expert_blend_v1/SUMMARY.json`
- Code SHA-256:
  `832f2ab645646b2e3a9f9bec7ed491d77724f863e7c9cbdac27bae29312fbc64`
- OOF SHA-256:
  `05053e2826a4b5b55f7de1ec34fe6baaa6fb762695f68ffc327ece9dde28893d`
- Runtime: 2.2 seconds.

The six OOF prediction files are fold-isolated model outputs; official
`train.csv` was used only for alignment checks and OOF metrics. No test
data, external data, label transfer, manual prediction editing, or
submission generation occurred.

### Promotion-gate status

The requested +0.005 mean-fold OOF improvement has **not** been achieved.
The best new fixed blend in this experiment sequence is +0.000385 over
TE20, which is not sufficient to justify a submission. These repeated
comparisons use the same fixed outer folds and have accumulated model
selection exposure; they are exploratory and should not be treated as an
independent estimate of future leaderboard lift. An honest next claim
requires a genuinely new, predeclared hypothesis with fold-safe training,
or a fresh validation design—not further tuning of these weights on the
same OOF predictions.

## Follow-up research — higher-order CatBoost CTRs (2026-11-16)

### Hypothesis and controlled design

The constant-free CatBoost expert used single-category CTR combinations
(`max_ctr_complexity=1`). This follow-up tested whether more complex
categorical combinations could discover useful demographic and employment
interactions. The fixed configuration increased the depth to 7, iterations
to 850, and `max_ctr_complexity` to 2, using the same supplied-field
feature builder, five outer folds, and fixed 90/10 rank blend with TE20.
This was one predeclared candidate, not an iteration/weight grid.

### Result: reject

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| CatBoost depth-7 / CTR2 expert | 0.766240 | -0.001516 | — |
| Fixed 90/10 TE20/CTR2 rank blend | 0.767840 | +0.000084 | 5/5 |

Increasing native categorical interaction complexity did not materially
strengthen the CatBoost stream. The blend's gain was positive in every fold
but only +0.000084 overall, below the +0.005 gate and slightly weaker than
the 700-iteration, CTR1 trial. The limiting factor is not merely too few
CatBoost category combinations; the expert remains less accurate than TE20
and only slightly complementary.

### Artifacts and rule controls

- New code: `ideas_v2/catboost_ordered_ctr2_expert_v1.py`
- OOF predictions: `ideas_v2/runs/catboost_ordered_ctr2_expert_v1/OOF_PREDICTIONS.csv`
- Fold metrics: `ideas_v2/runs/catboost_ordered_ctr2_expert_v1/FOLD_METRICS.csv`
- Versions, fold metrics, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/catboost_ordered_ctr2_expert_v1/SUMMARY.json`
- Code SHA-256:
  `5ee5810f526478fc2f6ff90a044c341dbe7a6b1d647155f12321289217c9169b`
- OOF SHA-256:
  `9dfef9e49b2d7fef44c8d6237522a6448f80d5bf6ac485a63528fb077299100c`
- Runtime: 604.7 seconds; CatBoost 1.2.10 using CUDA.

Only official `train.csv` was read as model data; `test.csv` was not read.
CatBoost fit only outer-training labels and scored the excluded fold. The
baseline OOF file was used solely for alignment and metric comparison. No
external data, label transfer, manual edits, or submission was used.

## Follow-up research — cross-fitted hard-example weighting (2026-11-16)

### Hypothesis and controlled design

The previous tests suggest weak experts gain only by correcting some
pairwise errors. This test emphasized difficult training examples while
keeping each weight out-of-sample with respect to the row being weighted.
Within each outer-fit partition, three-fold LightGBM predictions were
generated for every training row by models that excluded that row. The
weight was fixed as

`w_i = 1 + 3 * abs(y_i - p_i)`, with `1 <= w_i <= 4`.

A weighted `all_safe` LightGBM was then trained on the outer-fit partition;
the outer validation fold had no role in weight creation, fit, or tuning.
The benchmark comparison was a fixed 90/10 within-fold rank blend with
TE20.

### Result: reject

| Model | Mean-fold OOF AUC | Delta vs TE20 | Positive blend folds |
|---|---:|---:|---:|
| TE20 baseline | 0.767756 | — | — |
| Cross-fitted hard-example expert | 0.758923 | -0.008833 | — |
| Fixed 90/10 TE20/hard-example rank blend | 0.767892 | +0.000137 | 3/5 |

The expert underperformed strongly and its blend was negative in folds 1–2.
The weights averaged about 3.52 for positives and 1.21 for negatives, so
the formula substantially increased the positive-class contribution in
addition to emphasizing hard rows. That changed the learned global rank
without producing a competitive expert. This indicates that naive
error-weighted Logloss is not a suitable AUC correction here; do not
repeat it with arbitrary strength changes on these same folds.

### Artifacts and rule controls

- New code: `ideas_v2/crossfit_hard_example_weights_v1.py`
- OOF predictions: `ideas_v2/runs/crossfit_hard_example_weights_v1/OOF_PREDICTIONS.csv`
- Fold metrics and weight diagnostics:
  `ideas_v2/runs/crossfit_hard_example_weights_v1/FOLD_METRICS.csv`
- Full configuration, versions, runtime, rule checks, and SHA-256 values:
  `ideas_v2/runs/crossfit_hard_example_weights_v1/SUMMARY.json`
- Code SHA-256:
  `5596929954e85e3a35886f82a8b624f130aeca13e599535c500968d3095e94a7`
- OOF SHA-256:
  `59da9acaf9b455c80093698fe87a2d7b54d66d71009379aa6f91d6cf26796325`
- Runtime: 329.6 seconds; LightGBM 4.7.0.

Only official `train.csv` was read as model data; `test.csv` was not read.
Training weights used only inner-OOF predictions and outer-training labels;
the scored fold was excluded. The TE20 file was used only for alignment and
comparison. No external data, label transfer, manual edits, or submission
was generated.

## Current stopping point — +0.005 gate remains unmet

The best fixed candidate from this research sequence remains the
covariate-residual blend: TE20 mean-fold AUC **0.767756** to blend mean-fold
AUC **0.768141**, a gain of **+0.000385** on 5/5 folds. The fixed four-,
five-, and six-expert combinations scored gains of **+0.000269**,
**+0.000293**, and **+0.000278**, respectively. The hard-example-weighted
expert and both CatBoost expert trials did not provide a stronger route.
No experiment passed the required **+0.005** gate; no submission was
generated.

These are repeated examinations of the same five outer folds, so choosing
new transformations based on the observed decile errors or continuing to
search blend weights would further bias the OOF estimate. The next
defensible step is to pre-register a genuinely new modeling hypothesis and
test it on a fresh validation design, or stop claiming OOF lift until such
validation exists. Do not treat the small OOF gains as guaranteed
leaderboard improvement.
