# Phase 15 research map — historical reconciliation and next experiments

**Status:** Initial forensic map completed before Phase 15 model training.
**Scope:** Existing supplied-train experiments and their saved OOF artifacts.
No new model is promoted by this map. All previously used labels remain
exposed; none of the comparisons below is independent confirmation.

## 1. Why the recent scores look lower

These are not a controlled score time series. Feature matrices, algorithms,
categorical handling, fold seeds, and aggregation differ.

| Result | Reported OOF AUC | Model/representation | Key comparability facts |
|---|---:|---|---|
| Phase 9 | 0.768240 | CatBoost GPU, native high-cardinality categories, eight keys | Mean of three seed-specific pooled AUCs: 0.767955 (seed 42), 0.768000 (seed 7), 0.768764 (seed 2026). Source builder uses an archived Phase 2 raw-rowwise helper; independent Phase 10 builder does not claim exact matrix parity. |
| Phase 11 | 0.767399 | Independent CatBoost GPU, two replicas, eight keys | Five folds seed 2026, 700 iterations, depth 6; `one_hot_max_size=10`, so high-cardinality categories use internal CTR statistics. Feature builder is Phase 10 independent implementation. |
| Phase 13 E | 0.766038 | CPU LightGBM, complete raw+rowwise+eight keys, frequency-supported OHE | Five folds seed 20261113; OHE and fit-fold category support, not CatBoost native CTRs. It is the strongest matched Phase 13 representation, not a reconstruction of Phase 9/11. |
| Phase 14 augmented LightGBM | 0.765719 | CPU LightGBM, Phase 13 E representation plus deterministic numeric block | Same Phase 13 folds. The new block was not beneficial at this model/configuration; 2/5 folds beat E and the conditional paired interval included zero. |
| Phase 14 CatBoost | 0.759529 | CPU CatBoost, explicit one-hot handling for all categories | `one_hot_max_size=10,000`, above the largest fit-fold cardinality (4,564), intentionally prevented CTRs. This is materially different from Phase 9/11 native categorical models and explains why its score is not a fair reproduction of them. |
| Phase 2 | 0.771103 | 50:50 LightGBM/CatBoost blend with decoded-fee features and nested target statistics | Five folds seed 42. Uses constants 129.5/0.0132 and label-dependent target encodings; rules eligibility/provenance unresolved. Research-only, not a competition candidate. |

### Main regression diagnosis

1. **Model-family / categorical representation changed.** The strong Phase 9
   and Phase 11 models are GPU CatBoost with native categorical handling and
   internal target statistics. Phase 13's model is CPU LightGBM with one-hot
   categories. Phase 14's CatBoost was explicitly configured to avoid CTRs by
   forcing one-hot for all features. Those are different learners, not
   like-for-like reproductions.
2. **The rowwise feature matrix changed.** Phase 11 documents a separate
   independent builder, and Phase 10 explicitly blocks exact Phase 9 parity
   because the allowed official data dictionary does not define the complete
   historical builder. Phase 13's complete baseline is independently built
   and tested but is not claimed to recover every historical transform.
3. **Phase 14's hand-designed numeric expansion did not add useful ranking
   information for its LGBM configuration.** It lost 0.000318 AUC vs E,
   with mixed folds; the constant-free subset and equal blend were lower.
   This is evidence against that block/configuration, not all feature
   engineering.
4. **Phase 14's equal blend was a poor candidate.** The augmented CatBoost
   member was about 0.0065 below E and its prediction correlation with
   augmented LightGBM was about 0.955. The 50:50 blend lost 0.001152 vs E.
5. **The higher Phase 2 estimate is dominated by a distinct, unresolved
   supervised feature pathway.** Its raw-only matched blend was 0.761362;
   the Phase 2 feature/model blend was 0.771103 (+0.009741). The Phase 2
   protocol nests target encoding, but its label-derived encodings and
   fixed fee/product constants remain unresolved under competition rules.
6. **The data scope is narrow.** The supplied training file has 171,202 rows,
   32 predictors and a 8.07% positive rate. It does not contain related
   application-history, bureau, installment, POS/cash, or credit-card tables.
   Those missing table families cannot be imported because external data is
   prohibited. The ceiling may therefore be constrained by the provided
   feature scope; no score target is guaranteed.

## 2. Saved OOF inventory and compatibility

All listed prediction vectors were checked against 171,202 training rows.
Phase 9 IDs, Phase 11 row index/target, and Phase 13/14 row index/ID/target
align exactly to `D:\gci\inpuy\train.csv`. Phase 2 arrays have the verified
training-row shape and were checked by its existing artifact verifier. They
were generated with different fold assignments/seeds; row alignment permits
paired diagnostics but does **not** make their folds or training procedures
identical.

| OOF family | Available | Same row/target order | Fold protocol | Direct use |
|---|---|---|---|---|
| Phase 9, 45 prediction columns | Yes, CSV | Exact IDs | Three different 5-fold seeds: 42, 7, 2026 | Score/error diagnostics; seed averaging is an exploratory OOF ensemble |
| Phase 11, 6 main predictions | Yes, CSV | Exact row index and target | 5 folds, seed 2026, 2 replicas per variant | Score/error diagnostics; independently built feature path |
| Phase 13, 35 predictions | Yes, CSV | Exact row index, ID, target | 5 folds, seed 20261113 | Primary matched baseline and feature comparison |
| Phase 14, 7 predictions | Yes, CSV | Exact row index, ID, target | Same Phase 13 folds | Paired diagnostics vs Phase 13 E |
| Phase 2, four NumPy arrays | Yes, `.npy` | Verified 171,202-row arrays | 5 folds, seed 42 | Research-only diagnostics; target encodings/constants unresolved |
| Phase 8/11 exploratory target-rate scans | Yes, CSV | ID-aligned in their runs | Nested/cross-fitted, repeated exposed folds | Exploratory only; excluded from Track A promotion |

### OOF aggregation diagnostic (not a new validated score)

- Phase 9 per-seed CatBoost-8-key AUCs: `0.767955`, `0.768000`,
  `0.768764`; their mean is the reported `0.768240`.
- Averaging the three Phase 9 seed-specific OOF vectors row-by-row gives
  `0.769839` pooled AUC. Every component prediction for each row is out of
  fold, but these folds/seeds and labels have been repeatedly inspected.
- Phase 2 OOF blend: `0.771103`.
- Phase 2/Phase 9 averaged-vector correlation: Spearman `0.977745`.
- Descriptive fixed blend sweep: weights on Phase 9 `0.0` to `1.0` in 0.1
  steps gave a best observed Phase 9 weight `0.4` / Phase 2 weight `0.6`,
  AUC `0.771833` (vs Phase 2 `0.771103`). This weight was selected after
  examining these same labels and is **not** an unbiased ensemble estimate,
  eligible feature, champion, or confirmation.
- The diagnostic indicates modest OOF complementarity, not a path to claim
  `0.785`. A meta-model must be cross-fitted within outer training folds;
  fitting blend weights directly to the saved OOF labels would overfit.

## 3. Feature and source reconstruction

### High-score pipelines recoverable for diagnostics

- **Phase 9:** Full result report, 45-column OOF file, source-provenance CSV,
  model metrics and independent AUC audit are present. Its runner is also
  present, but exact standalone reconstruction remains blocked under the
  project clean-room rule because it imports the archived baseline feature
  helper. The artifact records CatBoost GPU with depth 6, learning rate 0.05,
  `max_ctr_complexity=1`; high-cardinality key CTR processing is part of the
  method. Use its saved predictions for research diagnostics, not to assert
  exact clean-room reproduction.
- **Phase 11:** Runner, independent Phase 10 builder, complete OOF and
  manifest are present. Reproducible independently as a research model;
  the two-replica run uses CatBoost GPU, depth 6, learning rate 0.05,
  700 iterations, `one_hot_max_size=10`, and native target-statistical
  categorical handling. Competition eligibility of those CTRs is unresolved.
- **Phase 13/14:** Feature builder/tests, runners, fold predictions, metrics
  and manifests are complete and independently verified. Phase 14's numeric
  block does not improve the label-free E reference; retain it as rejected
  evidence, not an automatic feature registry addition.
- **Phase 2:** Runner, feature source snapshot, OOF arrays, run manifest and
  verifier are present. The 0.771103 result is reproducible as a research
  score but has both unresolved constants and supervised encodings; retain
  as Track B only.

### Proposed immediate tests

| Rank | Hypothesis | Track | Cheap screen | Full-run gate |
|---:|---|---|---|---|
| 1 | Native-category CatBoost performance recovers the missing lift; Phase 14's forced one-hot setting caused model-family regression | B (CTR eligibility unresolved) | GPU benchmark one primary fold using the Phase 13/14 rowwise blocks and 8 keys; verify device, runtime, fit-partition CTR scope | Run matched 5-fold OOF only if useful lift/runtime is demonstrated |
| 2 | The Phase 14 numeric block interacts differently with native CatBoost than LightGBM | B with numeric formulas label-free; model CTRs unresolved | Same-fold CatBoost base vs base+Phase 14 features on one fold, identical seed and stopping split | Full paired folds only if positive and computationally feasible |
| 3 | XGBoost can be practical when using native categorical support instead of 12k sparse one-hot columns | A candidate | Benchmark XGBoost `hist` with fit-fold category dtype on a representative 6k–20k sample; measure memory/runtime | Scale only if smoke runtime/memory is acceptable |
| 4 | Fold-fitted numeric-bin × raw-category one-hot crosses or carefully bounded category crosses add interaction signal beyond E | A | Predeclare small, domain-agnostic crosses (organization × credit/age/external-score bins); one fold with identical E reference | Full OOF only if useful and fold-stable |
| 5 | Time-life ratios among registration, ID publication, phone change, employment and age add signal absent from Phase 14 | A | Implement guarded rowwise ratios; test one fold / all five via efficient LGBM | Keep only if incremental and stable |
| 6 | Existing OOF model complementarity yields an honest blend gain | B for Phase 2/CTR predictions; A for label-free bases | Use only as descriptive error analysis initially; don't report tuned OOF blend as validation | Any final stack requires nested base-model and meta-model cross-fitting |

## 4. Compute and validation plan

- GPU available: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB; currently
  idle. LightGBM build is CPU-only. CatBoost can use GPU; verify live GPU
  utilization and memory for each run.
- XGBoost with the Phase 13 12k-column sparse OHE representation was stopped
  twice because representative fits were disproportionately slow. Do not
  repeat that encoding; benchmark native categorical support first.
- Phase 14 completed in 36 minutes at roughly 1.0 GiB peak working set. Its
  source and results are preserved.
- Primary fold seed: Phase 13 `20261113` for directly matched label-free
  experiments; seeds `42`, `2026`, etc. remain historical robustness only.
- All transforms, category vocabularies, frequency counts, feature selection,
  model selection and early stopping must fit within outer-training/inner-fit
  data. Never use outer validation labels to tune the outer prediction.
- Every full candidate must save aligned OOF predictions, per-fold AUC,
  paired differences, uncertainty, feature/runtime/memory details, negative
  controls where applicable, eligibility status, and a verified manifest.
- Existing folds are already exposed. Additional cross-validation is
  controlled development, not independent confirmation. No test data,
  external data, submission generation, or champion mutation is authorized.

## 5. Decision before Phase 15 training

The most plausible recoverable score gap is not a missing handful of simple
ratios; it is the difference between native categorical boosting and sparse
one-hot models, plus supervised product statistics/constants in the research
track. The Phase 14 added numeric block did not help its LGBM setting. Phase 15
will first benchmark native-category GPU CatBoost on the verified independent
feature builder, then benchmark native-category XGBoost and one bounded set
of novel rowwise time-life features. The two abandoned sparse-XGBoost runs
remain excluded. The first new full OOF run is gated on representative
runtime/memory and one-fold performance.

The score target `0.785` is aspirational. No evidence currently supports
promising it; the available best reproducible research estimate is
approximately `0.7711` from an unresolved Track B Phase 2 method.

## 6. Post-map execution update

The map above was written before training. The subsequent research cycle
completed the ranked screens and changed the working diagnosis:

1. **Native-category CatBoost did recover performance.** On matched
   Phase 13 folds, GPU CatBoost with native CTRs and the independent feature
   builder reached `0.767511` for one seed. Averaging three OOF seeds reached
   `0.768825`; the OOF artifact and seed alignment were independently
   verified. This is the best complete Phase 15 model result, but remains
   Track B because CTR eligibility and some key constants are unresolved.
2. **The additional numeric block did not establish its own lift.** Against
   the matched CatBoost base it added `+0.000227` AUC; the paired 95%
   bootstrap interval included zero. Seven time-lifecycle features raised
   that comparison to `+0.000507` vs base; their isolated contribution was
   `+0.000280`, with a 300-resample interval
   `[-0.000253, +0.000817]`.
3. **The targeted organization × external-risk-bin interaction was not
   useful at OOF level.** Its pooled delta was `+0.000045`, positive on
   three of five folds, with paired interval `[-0.000490, +0.000536]`.
4. **More complex CTR combinations were not justified.** CatBoost
   `max_ctr_complexity=2` lost `0.000331` in the matched fold and was about
   2.7× slower. Native-categorical XGBoost scored `0.737658` on a
   30,000-row screen without keys and `0.723070` with four keys; do not
   spend a full OOF run on that configuration.
5. **The ID was not a hidden time feature.** `SK_ID_CURR` is the row index
   (`0..171201`); its ID-only AUC is `0.502897` and fold results are mixed.
   Keep it for alignment only.
6. **Same-label blends remain exploratory.** A fixed grid reached
   `0.771659` for Phase 15/Phase 2 OOF vectors, and Phase 2/Phase 9 reached
   `0.771833`; weights were selected after label exposure, and Phase 2 is
   unresolved. These are not validation or champion scores.

The result reconciles the regression: Phase 13/14 relied on CPU LightGBM
one-hot categories or deliberately CTR-free CatBoost; the strongest earlier
models relied on GPU CatBoost's native category statistics. Phase 15
recovered that modeling regime rather than finding a second product-key
signal. It did not reach `0.78000` or `0.78500`; further work on the same
32-predictor table has diminishing returns. See
[`PHASE15_EXECUTION_REPORT.md`](./PHASE15_EXECUTION_REPORT.md) for all
fold metrics, run details, eligibility boundaries and verification results.
