# V17/V18 ideas under the supplied competition rules

**Status:** Exploratory research complete for this pass; no submission was
generated or uploaded.  
**Data boundary:** Official `train.csv` for the new model; previously saved,
row-aligned training OOF artifacts only for diagnostics. No external data,
test-set covariates, test labels, or historical submission CSVs were used by
the new model.  
**Validation boundary:** Five-fold stratified OOF using seed `20261113`.
These labels and several component pipelines were studied before, so this is
not independent confirmation.

## Executive decision

The reported `0.76989` for `submission_candidate_v1.csv` is user-reported
leaderboard evidence. It is a different result from this work's local OOF
metrics and cannot be treated as directly comparable. Likewise, the earlier
V22 `0.78239` score is a historical user-reported leaderboard result, not a
locally reproduced clean-model estimate. This research did **not** establish
a rule-compliant route to `0.79`.

The most useful result from this pass is a clean V17-style product-category
expert and a fixed rank blend, evaluated against a **strictly fold-aligned**
Phase 33/34 baseline:

| Model | Pooled OOF AUC | Mean of 5 fold AUCs |
|---|---:|---:|
| Same-fold Phase 33/34 LightGBM + XGBoost blend | 0.765335 | 0.765360 |
| New clean CatBoost expert | 0.765877 | 0.765945 |
| Fixed 50/50 within-fold rank blend | **0.767488** | **0.767492** |

The clean expert by itself improved mean-fold AUC by 0.000585. Its fixed
equal-rank blend improved mean-fold AUC by **0.002132** (21.3 AUC points per
10,000) over the aligned reference and was higher in **all five folds**:

| Fold | Same-fold Phase 33/34 | Clean CatBoost | Fixed rank blend | Blend delta vs aligned baseline |
|---:|---:|---:|---:|---:|
| 1 | 0.770442 | 0.770154 | 0.772128 | +0.001686 |
| 2 | 0.770916 | 0.774104 | 0.774324 | +0.003407 |
| 3 | 0.760209 | 0.760526 | 0.762217 | +0.002009 |
| 4 | 0.762586 | 0.763676 | 0.764957 | +0.002371 |
| 5 | 0.762648 | 0.761264 | 0.763834 | +0.001186 |

This is promising complementarity, not proof of a leaderboard lift. The
equal blend was fixed rather than searched on the validation labels, but the
model family and feature hypotheses were chosen after extensive prior work on
these same labels. The run is not an untouched confirmation.

### Evaluation correction

An initial diagnostic compared the new model to Phase 37's reference, which
averages predictions from three different fold assignments. That aggregate
is valid as a rowwise multi-seed OOF diagnostic, but it is **not a strictly
outer-fold-aligned control**: for some rows, those component models may have
trained on labels of other rows in the fold being scored. I do not use that
comparison for the main lift claim.

The corrected comparison uses Phase 33 and Phase 34 OOF predictions from the
same five folds as the new CatBoost model. Each evaluated fold is excluded
from every model used to score that fold. The exact alignment and blend
calculation are reproduced by `analyze_clean_catboost_aligned_blend.py`.
The clean runner's original `SUMMARY.json` preserves the earlier Phase 37
comparison as run provenance; its cross-model blend metrics are secondary,
not the primary result. Use `runs/fold_aligned_catboost_blend_v1/SUMMARY.json`
for the corrected comparison. The earlier Phase 37 rank-blend results remain
in the diagnostic output but are not promotion evidence.

## What V17/V18 got right—and what cannot be carried forward

### Useful ideas retained

1. **Represent loan pricing explicitly.** `AMT_ANNUITY / AMT_CREDIT`,
   `AMT_ANNUITY / AMT_GOODS_PRICE`, and `AMT_GOODS_PRICE / AMT_CREDIT` expose
   product structure more directly than the raw amounts alone. The new
   CatBoost run used only rounded, constant-free forms of these ratios.
2. **Use a product specialist alongside a general model.** CatBoost's native
   categorical handling on product-key categories produced useful ranking
   differences from the same-fold Phase 33/34 general-model blend. The
   combined result was stronger than either standalone stream.
3. **Keep the full population for the general learner.** AUC depends on
   comparisons across the entire positive/negative population. Hard-splitting
   training rows into small regime-specific populations can starve the model
   and destabilize cross-regime scores. Add regime information as features or
   combine fold-trained experts instead of manually routing selected test rows.
4. **Use nested cross-fitting for learned features.** Any supervised category
   statistics or level-one view predictions used as training features must
   exclude that row's label. Outer validation labels must not affect those
   features, stopping rounds, model weights, or the model itself.
5. **Diagnose ranking by feature-defined groups.** Conditional AUC and
   pairwise rank comparisons are useful for discovering complementarity.
   They are not a license to hard-code group-specific test adjustments.

### Excluded or unresolved historical mechanisms

| Historical mechanism | Decision | Reason |
|---|---|---|
| `partner_enc` / `PY_*` | **Exclude** | Copies observed training labels through applicant-matching keys to other rows. This is the label-transfer method the user asked to remove; it is not ordinary category target encoding. |
| Manual test-row weight changes or hand-set predictions | **Exclude** | Conflicts with the no-hand-labeling boundary. |
| Train-plus-test pooled feature statistics | **Avoid** | The supplied README does not clearly authorize transductive fitting. Use fit-fold-only statistics and apply them to held-out rows. |
| Fixed offsets such as `129.5` or financing-step `0.0132` | **Hold** | Their rule-compliant derivation from only the permitted data has not been established in the reviewed materials. Constant-free ratios were used instead. |
| Residual models trained on pooled validation/test covariates | **Exclude** | Violates the strict fold-isolation policy even when the residual target is not `TARGET`. |
| Cached predictions as a final model input | **Do not use for reproduction** | The submitted pipeline should regenerate predictions from official inputs and the documented model code. Cached OOF files here were used only to compare research hypotheses. |

The audited archived V17/V18 source contained both `partner_enc` label
transfer and train/test covariate pooling. Therefore, the old V17/V18/V22
output cannot be treated as a compliant reference implementation merely
because the final blending code applies a mathematical rank average. The
blend inherits the provenance and eligibility of its component predictions.

## The "confusion matrix" request and AUC

A confusion matrix requires a classification threshold. ROC-AUC does not:
it measures positive-negative pair ordering across all thresholds. At the
competition's roughly 8% positive rate, a default threshold of `0.5` would be
arbitrary and could make a useful ranking model look as if it predicts almost
no positives. It would not explain or improve AUC.

This analysis therefore reports **within-group AUC** for covariate-defined
groups, and compares those results to the full-fold rank blend. Group AUC
answers whether a model orders positives above negatives *inside that group*;
it does not measure all cross-group positive-negative pairs. A group can have
high within-group AUC while still shifting group scores badly relative to
other groups.

### Where the clean CatBoost stream helps

Values below are mean AUC across the five outer folds, calculated within each
stratum. They are diagnostics, not separately held-out subgroup tests.

| Supplied-data stratum | Rows | Positives | Same-fold Phase 33/34 | Clean CatBoost | Fixed rank blend |
|---|---:|---:|---:|---:|---:|
| Cash loans | 154,988 | 12,921 | 0.764770 | 0.765545 | **0.767021** |
| Revolving loans | 16,214 | 900 | 0.757198 | 0.756172 | **0.758369** |
| 1 external score observed | 38,611 | 3,498 | 0.745548 | 0.747996 | **0.748703** |
| 2 external scores observed | 96,310 | 7,742 | 0.768944 | 0.768682 | **0.770632** |
| 3 external scores observed | 36,164 | 2,570 | 0.774510 | 0.775657 | **0.776923** |
| Employment sentinel absent | 140,304 | 12,145 | 0.764952 | 0.766148 | **0.767386** |
| Employment sentinel present | 30,898 | 1,676 | 0.741895 | 0.738572 | **0.742580** |

The product specialist alone is not uniformly better: it is weaker on the
revolving-loan and employment-sentinel strata. Yet the fixed blend is higher
than the aligned baseline in every listed stratum. This supports a **global blend of
complementary predictors**, not selecting V17/V18 for individual rows using
EDA-derived conditions. The subgroup with zero observed external scores has
only 89 rows and 11 positives; its subgroup AUC is too unstable to guide a
model decision.

## What the historical OOF evidence says

These are previously run experiments, independently checked here for row,
ID, target, and fold alignment where applicable:

| Evidence | Mean-fold AUC | Interpretation |
|---|---:|---|
| Phase 33/34 same-fold LightGBM/XGBoost fixed blend | 0.765360 | Strictly fold-aligned control used for the new blend comparison. |
| Phase 37 fixed multi-seed XGBoost/LightGBM reference | 0.765982 | Useful rowwise multi-seed OOF reference; not the aligned control for fold-wise blend assessment. |
| Phase 17 constant-free product-key nested target encoding | 0.765496 | Small standalone value; fold-local training-row encodings and constant-free keys. |
| Phase 37 + Phase 17 50/50 within-fold rank diagnostic | 0.767323 | Secondary cross-seed/post-hoc diagnostic; not strict same-fold blend evidence or nested weight evidence. |
| Phase 18 nested CatBoost/TE blend | 0.768722 | Highest relevant prior nested blend; **not eligible as a strict candidate** because its CatBoost side uses product constants/features whose provenance remains unresolved. |
| Phase 44 feature-routed contract-expert mixture | 0.765231 | Below Phase 37 alone. |
| Phase 37 + Phase 44 contract mixture, 25% expert rank blend | 0.766173 | Secondary cross-seed/post-hoc diagnostic; do not treat as a strict nested improvement. |
| Phase 52 V7-style label-free feature model | 0.758662 | Lower standalone AUC; its blends with Phase 37 are secondary cross-seed diagnostics. |

Phase 18's fold-local weights were selected using only inner OOF predictions
inside each outer training fold, and its blend exceeded each of its two
components in all five outer folds. It is a valuable **architecture lead**:
diverse product/category learners can complement each other. It does not
validate its unresolved features or establish that a compliant replacement
will retain the same score. The new constant-free CatBoost probe above is a first rules-conscious test of
that product-specialist idea.

## Model and experiment details

### New clean CatBoost expert

- **Input:** only `D:\gci\inpuy\train.csv`.
- **Rows/labels:** 171,202 rows; 13,821 positives.
- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=20261113)`.
- **Features:** 77 total, including the supplied columns, deterministic
  external-score and financial row-wise features, and five constant-free
  rounded ratio categories (`annuity/credit` at 3/4/5 decimals,
  `annuity/goods` at 3 decimals, and `goods/credit` at 3 decimals).
- **Model:** CatBoost GPU, 700 iterations, depth 6, learning rate 0.05,
  L2 5, one-hot maximum 10, maximum CTR complexity 1; distinct fixed fold
  seeds.
- **Fit boundaries:** Each fold's model sees only that fold's training rows.
  Categories and CatBoost's learned statistics are generated within the
  model fit; no validation or test rows are included.
- **Explicit exclusions:** no `TARGET`-based matching or peer-label feature,
  no empirical fee/step constant, no train/test concatenation, no OOF cache
  as model input, and no per-row prediction override.
- **Runtime:** 90.6 seconds on the detected NVIDIA RTX 4050 Laptop GPU.
- **Versions:** Python 3.12.10, pandas 3.1.0, NumPy 2.5.4, CatBoost 1.2.10.

The user-supplied checklist treats ordinary target encoding as acceptable when
strictly fold-local and lists CatBoost as an allowed model family. The local
README does not separately adjudicate CatBoost's internal categorical target
statistics. This implementation keeps them inside each model's training
fold, but that narrow interpretation remains a competition-rule question for
the organizers if the user wants a formal ruling.

### Artifacts

- Runner: `run_clean_v17v18_catboost_probe.py`
- Fold-aligned comparison: `analyze_clean_catboost_aligned_blend.py`
- Diagnostic runner: `run_oof_regime_diagnostics.py`
- New CatBoost OOF: `runs/clean_catboost_constant_free_v1/OOF_PREDICTIONS.csv`
- New fold metrics: `runs/clean_catboost_constant_free_v1/FOLD_RESULTS.csv`
- New run manifest and metrics: `runs/clean_catboost_constant_free_v1/SUMMARY.json`
- Cross-model/subgroup summary:
  `runs/historical_oof_diagnostics_clean_v1/OOF_SUMMARY.json`
- Subgroup fold AUCs:
  `runs/historical_oof_diagnostics_clean_v1/REGIME_FOLD_AUC.csv`
- Corrected aligned-blend summary:
  `runs/fold_aligned_catboost_blend_v1/SUMMARY.json`
- Corrected aligned-blend fold metrics:
  `runs/fold_aligned_catboost_blend_v1/FOLD_METRICS.csv`
- Corrected aligned-blend subgroup metrics:
  `runs/fold_aligned_catboost_blend_v1/REGIME_FOLD_AUC.csv`
- Official training-data SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`
- New OOF CSV SHA-256:
  `ba58b1f6832b2b62eb466a8437ac668c46df6baa1511934eb82ad145ae6ed0c3`

The model was not run on `test.csv`; this work generated no submission CSV and
did not upload anything to the leaderboard.

## Best next step

The evidence favors a **global, fold-isolated multi-view blend**, not manual
or hard subgroup routing:

1. Keep a general same-fold baseline and the new clean CatBoost product-key
   expert as distinct components. Treat the multi-seed Phase 37 comparisons
   as secondary unless its component OOF predictions are regenerated on
   aligned folds.
2. Rebuild the complementary LightGBM component with constant-free
   product-key target statistics. For model-training rows, make those
   statistics inner-OOF; for outer validation/test rows, fit them on the
   outer training partition only. Do not copy labels between inferred
   applicant matches.
3. In each outer fold, generate inner OOF predictions for **all** components,
   select any blend weight only from those inner predictions, then evaluate
   on the untouched outer fold. Do not tune a weight against global saved OOF
   vectors and call that nested validation.
4. Use fixed, documented blend rules and record both pooled AUC and the mean
   of fold AUCs, per-fold values, input/code hashes, package versions, and
   submission schema checks.
5. Compare model complementarity with subgroup AUC and rank-pair diagnostics;
   do not use a thresholded confusion matrix as the AUC objective.

The first next model experiment should be a fully nested clean
CatBoost + constant-free LightGBM product-encoding blend. Its expected result
is unknown; the available evidence gives no defensible guarantee of reaching
`0.79`. If that blend does not improve consistently, stop escalating
feature/weight searches on the same labels and report the limitation rather
than overfit the public leaderboard.

## Sources reviewed

- `D:\gci\ORIGIN_MASTER_VERIFICATION_LEDGER.md`
- `D:\gci\past.md` and `D:\gci\truth\past.md`
- `D:\gci\truth\reports\DEEP_ACTUARIAL_RESEARCH_AND_STAGE_OPTIMIZATION.md`
- `D:\gci\truth\reports\GCI_WORLD_2026_LEADERBOARD_SUBMISSIONS_AND_MATHEMATICAL_LEDGER.md`
- `D:\gci\truth\reports\GCI_WORLD_2026_MATHEMATICAL_FORMULAS_AND_ARCHITECTURAL_TREATISE.md`
- `D:\gci\truth\reports\NEW_MATERIALS_FORENSIC_AUDIT.md`
- `D:\gci\truth\reports\RESEARCH_10_REGIMES_REPORT.md`
- `D:\gci\truth\reports\COMPETITION_MATERIALS_REVIEW.md`
- User-supplied audit snapshot of the archived V17/V18 blend notebook.
- Phase 17/18, Phase 37, Phase 44/45, and Phase 52 source and saved OOF
  artifacts, cited in the tables above.

This is a focused, evidence-ranked review of the named reports, audit
snapshot, and relevant experiment implementations/results—not a claim that
every line in every historical file was read.

## Follow-up audit and AUC-objective experiments

This addendum preserves the earlier findings and the historical V17/V18/V22
artifacts. It does not rehabilitate their archived `partner_enc` or pooled
preprocessing. Those outputs remain out of the compliant-model path.

### Strict v2 submission run

`ideas_v1/create_compliant_submission_te20_v2.py` is a new version; v1 and
its artifacts were not overwritten. Before loading data, v2 checks the
recorded SHA-256 values for the official train, test, and sample files. The
active candidate path uses only those three files: training labels fit
models and fold-local category statistics, test rows are inference inputs,
and the sample is used only for schema and ID-order checks. It does not read
old V17/V18 submissions or prediction caches as model features.

The selected pipeline has no residual-regression feature stage. Phase 33
encoders, category vocabularies, frequency counts, and target maps are fit
on the relevant outer-training partition. Product target statistics for
outer-training model rows are inner-OOF; outer validation/test maps use only
outer-training labels. The v2 test-set perturbation check changed/reordered
test covariates and verified that sampled training and validation matrices
were unchanged. The validation-label perturbation check changed only
outer-validation labels and verified that fold-local target-map features
were unchanged. Both passed.

| Fold | Regenerated reference | TE20 candidate | Delta |
|---:|---:|---:|---:|
| 1 | 0.772128 | 0.772376 | +0.000248 |
| 2 | 0.774324 | 0.774856 | +0.000532 |
| 3 | 0.762217 | 0.762461 | +0.000244 |
| 4 | 0.764749 | 0.764892 | +0.000143 |
| 5 | 0.763834 | 0.764194 | +0.000360 |
| **Mean** | **0.767450** | **0.767756** | **+0.000305** |

The generated CSV has exactly 61,500 rows and `SK_ID_CURR,TARGET` columns,
unique IDs in sample order, finite predictions in `[0,1]`, and hashes
recorded in its manifest. Its path is
`ideas_v1/submissions/product_te_20pct_v2/submission_compliant_product_te20_v2.csv`.
No leaderboard upload occurred.

The measured +0.000305 is a development-fold estimate, not an independent
confirmation: the same labels have informed extensive feature and weight
choices. A paired DeLong-style calculation conditional on these fixed OOF
scores gives an approximate interval of +0.000118 to +0.000493, but it does
not account for model selection, training-fold dependence, or GPU
reproducibility and must not be reported as a formal confidence interval.
The 20% TE weight itself was chosen from an earlier screen on these labels.

### Direct pairwise ranking and class-weight experiment

For binary AUC,
\[
\operatorname{AUC}=
\frac{1}{N_+N_-}\sum_{i:y_i=1}\sum_{j:y_j=0}
\left[\mathbf{1}(s_i>s_j)+\tfrac12\mathbf{1}(s_i=s_j)\right].
\]
The training file has 13,821 positives and 157,381 negatives, or
2,175,162,801 positive-negative pairs. A +0.000305 AUC change corresponds
to roughly 0.000305 of the evaluated pair mass; multiplying by the full-data
pair count would be only an illustrative extrapolation, not an observed
number of corrected pairs, because OOF AUC is evaluated within folds.

`run_balanced_loss_catboost_v1.py` set positive Logloss weight to
`n_negative/n_positive` within each fit fold (about 11.387), which balances
the total class weight but does not directly optimize pair ordering. Its
standalone mean-fold AUC was 0.763813. A fixed 90/10 rank blend scored
0.767632 (**+0.000140**, positive on 4/5 folds), so this experiment was not
promoted.

`run_pairwise_auc_linear_v2.py` samples 100,000 positive-negative pairs
uniformly with replacement from each outer training partition and trains a
linear model on their feature differences using logistic pairwise loss. This
is a surrogate for maximizing AUC: uniform sampling makes each pair
equally likely in the empirical pairwise objective, but finite sampling and
logistic loss mean it is not exact AUC optimization. All imputation,
scaling, and one-hot vocabulary are fit only on the outer training rows.
Its standalone mean-fold AUC was 0.757320; a fixed 90/10 rank blend scored
0.767696 (**+0.000204**, positive on 5/5 folds). Increasing optimizer
iterations from 30 to 100 did not materially change the result.

A fixed 80/10/10 prior/TE/pairwise screen reached mean-fold 0.767835,
**+0.000057** over TE20, but lost to TE20 on fold 2. A paired conditional
interval for this difference was approximately [-0.000097, +0.000210].
Because the incremental gain is small, uncertain, and selected on reused
labels, no pairwise-enhanced submission was generated.

### Reproducibility and strict-rule boundary

The hash-locked v2 script and perturbation tests strengthen provenance and
fold isolation; they do not prove an organizer ruling. The exact v2 script
was run twice from the same source hash into separate output directories.
Both OOF gates passed: 0.767756 (+0.000305) and 0.767772 (+0.000303), each
positive on all folds. The CSVs were rank-correlated at 0.999990 but not
byte-identical (maximum absolute `TARGET` difference 0.01333). Phase 33,
Phase 34, and nested-TE test predictions matched; the difference was isolated
to CatBoost GPU predictions. Keep the first v2 run as primary rather than
choosing a higher replay after inspection. Exact output reproduction is
therefore not established; the manifests record seeds, versions, input/code
hashes, and output hashes. This is consistent with the competition note that
exact reproducibility is not always guaranteed, but it remains an important
limitation.

No external data, applicant matching labels, manual test overrides, or
row-specific routing were used in these experiments. Fold-local category
target statistics are retained because the user-provided checklist accepts
them; the competition materials do not formally adjudicate every
implementation detail. Therefore the work follows the supplied conservative
controls but cannot honestly guarantee organizer-certified “100%” compliance
without organizer confirmation. It has not reached or guaranteed leaderboard
0.79.

## Pairwise AUC objective and error-mass analysis

For binary labels, ROC-AUC is the correctly ordered mass over every
positive-negative pair:

\[
\operatorname{AUC}(s)=\frac{1}{N_+N_-}
\sum_{i:y_i=1}\sum_{j:y_j=0}
\left(\mathbf{1}[s_i>s_j]+\frac{1}{2}\mathbf{1}[s_i=s_j]\right).
\]

This makes the pairwise logistic loss a defensible surrogate:
\(\log(1+\exp(-(s_i-s_j)))\) penalizes a positive-negative margin error
smoothly. It is not the same finite-sample objective as AUC, and a sampled
pair set and model capacity can still underperform a classifier trained with
Logloss.

Two GPU experiments used the shared five folds and the current TE20 OOF
candidate as a fixed comparator:

| Model/objective | Standalone mean-fold AUC | Fixed 90/10 blend mean-fold AUC | Gain vs TE20 | Positive folds |
|---|---:|---:|---:|---:|
| CatBoost `PairLogit`, 1,000,000 capped pairs, 500 iterations | 0.758634 | 0.767719 | -0.000037 | 3/5 |
| XGBoost `rank:pairwise`, mean pair sampling, 8 pairs/sample, 1,000 trees | 0.760821 | 0.768057 | +0.000301 | 5/5 |

The CatBoost pairwise branch was not promoted. The XGBoost result justified a
separate end-to-end candidate generator, which retrained all components and
passed its own strict five-fold gate: TE20 baseline **0.767743**, candidate
**0.767979**, gain **+0.000236**, positive in **5/5** folds. Its fold gains
were +0.000147, +0.000206, +0.000035, +0.000489, and +0.000305. The small
difference from the initial OOF screen is expected because the full generator
regenerates all base models and uses its own fixed fold-seed schedule; the
candidate manifest is authoritative. No leaderboard submission was made.

### AUC mass by baseline-negative decile

To quantify *where* the gain came from, negatives were split into ten
near-equal bins by the TE20 baseline rank within each validation fold. Every
positive was compared with all negatives in each bin. This gives an exact
decomposition for each fold:

\[
\Delta\operatorname{AUC}
=\sum_{k=1}^{10}\frac{N^-_k}{N^-}
\left(\operatorname{AUC}_{k,\mathrm{candidate}}
-\operatorname{AUC}_{k,\mathrm{baseline}}\right).
\]

For the final generator's OOF predictions, the mean-fold contributions for
negative deciles 1 through 10 were respectively
**+0.000001, +0.000001, -0.000002, +0.000044, +0.000017, -0.000014,
+0.000009, +0.000034, +0.000084, +0.000063**; they sum to **+0.000236**.
Thus, this experiment's gain came mostly from baseline high-risk negative
deciles 8–10 (especially decile 9), not from the low-risk bulk. The deciles
are diagnostic bins, not hand-coded prediction rules; all rows are still
ranked by the trained models.

### Compliance, reproducibility, and decision

The new candidate is
`ideas_v1/submissions/te20_xgb_rank_pairwise_v1/submission_compliant_te20_xgb_rank_pairwise_v1.csv`.
Its hash is
`7cf21df3ba752d09ad487593fca8905c834f0e1d7334c97b6c953952d37a3023`.
The generator checked official input hashes, sample ID order, uniqueness,
finite `[0,1]` values, and all five OOF deltas. It uses only official
train/test files; labels used for rank training are confined to the current
outer training fold. Fold-local Phase 33 feature matrices are fit on those
training rows. The sample file is used only for output-schema and ID checks;
there is no matched-applicant label transfer, external data, row-specific
routing, or manual test correction. This is evidence of following the
provided checklist, not an organizer ruling. Fixed seeds and hashes are
recorded; exact replay of the GPU CatBoost components is not guaranteed.

Do not continue sweeping pairwise weights on these same folds: both the
candidate construction and many historical experiments have already
examined their labels. The next defensible work is a nested evaluation of a
pre-specified blend/feature hypothesis, with every weight and preprocessing
decision made inside the outer training partition. This OOF result is about
**0.768**, not 0.79, and does not predict the leaderboard score.

References consulted:

- [CatBoost ranking objectives (`PairLogit`, `max_pairs`)](https://catboost.ai/docs/en/concepts/loss-functions-ranking)
- [CatBoost AUC pair definition](https://catboost.ai/docs/en/concepts/loss-functions-classification)
- [XGBoost learning-to-rank objectives and pair construction](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
- [Cawley & Talbot, selection bias in model evaluation](https://jmlr.org/papers/v11/cawley10a.html)
- [GCI World Competition repository](https://github.com/Hareen-Liyanage/GCI_World_Competition)
