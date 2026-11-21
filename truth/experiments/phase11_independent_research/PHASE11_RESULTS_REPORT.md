# Phase 11 Results: Independent Product-Key Research

## Executive result

The independent five-fold GPU CatBoost comparison supports the product-key
signal, but does not validate a new competition champion. On 171,202 permitted
training rows, adding all eight mathematical keys raised pooled OOF ROC-AUC from
**0.759456 to 0.767399** (**+0.007944**). All five fold deltas were positive
(+0.005861 to +0.009826). A paired, stratified row bootstrap gave a conditional
95% interval of **+0.006485 to +0.009387**.

The historical Phase 9 CatBoost score of 0.768240 was not independently
reproduced; the Phase 11 score is lower and uses a newly specified feature
builder. These results must not be treated as a like-for-like comparison.
Repeated CV folds reuse applicants and labels, and the bootstrap is conditional
on the fitted folds; neither provides independent confirmation.

The broader train-only key scan found no compelling second hidden-key family.
Target-rate blends helped a raw-feature model, but added essentially nothing
once the eight-key CatBoost OOF model was already present. A nested logistic
blend reached 0.767769, only +0.000370 over the primary eight-key OOF score;
the paired bootstrap interval included zero, and only three of five fold deltas
were positive. No result established a robust score above 0.77.

## 1. Independent GPU CatBoost comparison

| Model | OOF ROC-AUC | Paired change |
|---|---:|---:|
| Independent raw-feature baseline | 0.759456 | — |
| Same features + eight product keys | **0.767399** | **+0.007944** |

Both use the same fixed stratified five-fold assignments, 700 iterations,
depth 6, learning rate 0.05, and two model seeds averaged within each outer
fold. The outer validation rows were excluded from every model that predicted
them. Per-fold paired changes were +0.006218, +0.008729, +0.009182, +0.009826,
and +0.005861. Prediction Spearman correlation between the raw and key models
was 0.9533.

Seed averaging gave a small additional lift versus the mean score of the
individual replicas: about +0.00054 for the raw model and +0.00056 for the
eight-key model. This is a within-fold seed comparison, not a new independent
CV confirmation.

Authoritative outputs: [`RUN_MANIFEST.json`](runs/phase11_gpu_cv_20261105/RUN_MANIFEST.json),
[`fold_metrics.csv`](runs/phase11_gpu_cv_20261105/fold_metrics.csv), and
[`oof_predictions.csv`](runs/phase11_gpu_cv_20261105/oof_predictions.csv).

## 2. Individual and leave-one-out key ablations

An 18-variant, five-fold GPU CatBoost grid used the same folds and 350
iterations per model. The all-key variant scored 0.765319 versus 0.756336 for
its same-grid raw baseline (+0.008983); this is a separate, lower-iteration
experiment and should not be compared directly to the 700-iteration primary
result.

Among single keys, P4 was strongest (0.764562), followed by Q4 (0.763993) and
P5 (0.763715). AG3 alone was nearly neutral (0.756944). Removing Q3 produced
the best leave-one-out score (0.766046), +0.000727 versus all eight keys and
better on four of five folds. Other leave-one-out deltas were small and mixed.
Because all variants reuse the same exposed folds, this is a screening result,
not evidence to drop or promote a key.

Full results and train-fold key support are recorded in
[`RUN_MANIFEST.json`](runs/phase11_ablation_cv_20261105/RUN_MANIFEST.json),
[`fold_metrics.csv`](runs/phase11_ablation_cv_20261105/fold_metrics.csv), and
[`oof_predictions.csv`](runs/phase11_ablation_cv_20261105/oof_predictions.csv).

## 3. Train-only hidden-key scan and controls

The candidate search scored **1,695** distinct candidates: 14 numeric
single-value keys, pairwise sums/differences/ratios at predeclared resolutions,
smooth-bin controls, low-cardinality pairs and triples, selected numeric-bin
crosses, and the eight explicit product keys. Encodings used smoothing 30 and
were fitted only on each OOF fit partition.

The best fixed 10% blend against the raw-model OOF predictions was P4 at
0.762633 (+0.003178); P5 and Q4 followed at +0.003074 and +0.002735. The best
numeric pair was credit/annuity at four significant digits (+0.002660).
Low-cardinality crosses and smooth-bin controls had maximum lifts of only
+0.000125 and +0.000013, respectively. All reported top-candidate results are
adaptive to these same OOF labels and are exploratory.

Forty shuffled-key controls preserved each key's category-frequency
distribution while breaking its row association. Their 10% target-rate blend
deltas were centered near zero. The corrected label-free frequency diagnostic
had standalone AUCs from 0.471 to 0.508, so category frequency alone did not
reproduce the product-key signal. Raw frequency scores are not directly blended
with model probabilities because they are not probability-calibrated.

The scan and controls are in
[`SCAN_SUMMARY.json`](runs/phase11_candidate_scan_20261105/SCAN_SUMMARY.json),
[`candidate_scan.csv`](runs/phase11_candidate_scan_20261105/candidate_scan.csv),
[`frequency_preserving_controls.csv`](runs/phase11_candidate_scan_20261105/frequency_preserving_controls.csv),
and [`label_free_frequency_recomputed.csv`](runs/phase11_candidate_scan_20261105/label_free_frequency_recomputed.csv).

## 4. Target-rate encodings versus the eight-key model

Fourteen predeclared product/amount-ratio target-rate features were also
compared with the existing eight-key model's OOF predictions, not just the raw
baseline. The best result was credit/annuity at five significant digits with a
5% fixed blend: **+0.000078**. The best 10% blend was negative
(-0.000044), and the best 20% blend was -0.000910. Forty frequency-preserving
nulls for four keys remained near zero.

This is not a credible incremental score gain: the best change is tiny, was
chosen among multiple candidates, and has no independent confirmation. The
feature is label-derived, and organizer eligibility remains unresolved. Do not
promote target encoding or the blend for submission.

See [`RUN_MANIFEST.json`](runs/phase11_te_vs_product_20261105/RUN_MANIFEST.json),
[`product_baseline_target_rates.csv`](runs/phase11_te_vs_product_20261105/product_baseline_target_rates.csv),
and [`frequency_preserving_nulls_vs_product.csv`](runs/phase11_te_vs_product_20261105/frequency_preserving_nulls_vs_product.csv).

## 5. Label-free one-hot LightGBM and nested stacking

LightGBM used fold-fitted one-hot encoding of raw categories and product keys;
unknown validation categories were ignored, and no label-derived feature
encoding was used. The eight keys raised LightGBM OOF AUC from 0.759687 to
0.764494 (+0.004807), with a positive change on all five folds. This supports
label-free usefulness of the constructed keys, although this CPU LightGBM
model remained below the GPU CatBoost key model.

The nested experiment selected CatBoost depth/learning-rate settings using
three inner folds inside each of five outer folds. The default depth-6,
0.05-learning-rate setting was selected in all five outer folds. Its 400-tree
OOF AUC was 0.766316. The inner-OOF logistic stack of that CatBoost model and
the one-hot LightGBM key model scored 0.767769. Against the primary
700-iteration eight-key CatBoost model, this is +0.000370; paired fold changes
were negative in two folds, and the conditional paired-bootstrap 95% interval
was **[-0.000233, +0.001012]**. Treat the apparent lift as inconclusive.

The nested OOF predictions and inner model choices are in
[`RUN_MANIFEST.json`](runs/phase11_nested_ensemble_20261105/RUN_MANIFEST.json),
[`fold_metrics.csv`](runs/phase11_nested_ensemble_20261105/fold_metrics.csv),
[`nested_oof_predictions.csv`](runs/phase11_nested_ensemble_20261105/nested_oof_predictions.csv),
and [`inner_model_selection.csv`](runs/phase11_nested_ensemble_20261105/inner_model_selection.csv).

## 6. CatBoost statistics and competition eligibility

The fitted CatBoost configuration used `one_hot_max_size=10` and
`max_ctr_complexity=1`. All eight product keys had more than 90 observed
categories; the fitted model's effective `simple_ctr` configuration included
`Borders` with target-border settings as well as `FeatureFreq`. CatBoost is
therefore using internal target-dependent categorical statistics for these
high-cardinality keys. The key formulas themselves are label-free, but the
CatBoost representation is not label-free.

The one-hot LightGBM representation avoids that internal categorical target
statistic mechanism. Neither modeling choice resolves the competition rules:
the separate local rules audit still marks target-derived categorical
processing as unresolved. Obtain organizer guidance before any submission use.

## 7. Integrity, reproducibility, and decision

- Source data was only `D:\gci\inpuy\train.csv` (SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`).
- No test data, locked holdout, external data, or submission was opened or used.
- The independent Phase 10 builder was used; no EXP85 helper or Phase 9
  cached feature/prediction artifact was imported.
- CatBoost runs executed on the NVIDIA GeForce RTX 4050 Laptop GPU. LightGBM
  ran on CPU because the installed build did not provide the required GPU
  learner.
- Six Phase 11 unit tests and Python compilation checks passed. Small smoke
  runs verified GPU and nested-pipeline plumbing only; their scores are not
  evidence.
- The candidate search and ablation grid reuse exposed labels and are
  exploratory. The paired bootstrap intervals quantify row-sampling
  uncertainty conditional on these folds, not dataset/selection uncertainty.

**Decision:** retain the eight-key CatBoost result as a research candidate, not
an official champion. The clean independent OOF result supports predictive
value, but does not surpass the historical Phase 9 report, establish a
competition-eligible CatBoost categorical procedure, or validate a score
above 0.77. Continue only after eligibility is clarified or through the
label-free one-hot branch with a preregistered, organizer-approved evaluation.
