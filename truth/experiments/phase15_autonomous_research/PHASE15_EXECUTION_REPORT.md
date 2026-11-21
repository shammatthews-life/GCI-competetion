# Phase 15 — forensic recovery and adaptive model research

**Status:** Research cycle completed; the 0.78000 and aspirational 0.78500
targets were not reached. No champion or submission was changed.
**Classification:** Controlled development and exploratory OOF evidence on
previously exposed labels. No result is independent confirmation.

## Executive result

The apparent Phase 13/14 score regression was not a like-for-like model
regression. Phase 9 and Phase 11 used GPU CatBoost with native categorical
target statistics; Phase 13 used CPU LightGBM with fold-fitted one-hot
features; Phase 14's CatBoost was deliberately forced to one-hot all
categories, preventing the CTR path. A matched-fold GPU CatBoost preflight
recovered most of the gap.

The best complete new OOF result was a three-seed average of the Phase 15
native-category CatBoost feature variant: **0.768825 pooled OOF ROC-AUC**.
Its component OOF AUCs were `0.767511`, `0.767946`, and `0.767081`. This
slightly exceeds Phase 9's reported *mean of seed AUCs* (`0.768240`) but is
below the Phase 9 rowwise average of its three saved OOF vectors (`0.769839`).
The Phase 15 model uses all eight product keys and CatBoost native CTRs; both
the fee/step constants and CTR eligibility remain unresolved. It is
**research-only**, not a competition candidate or champion.

Other tested directions did not create stable lift:

- The seven time-lifecycle features added `+0.000280` pooled OOF AUC over the
  Phase 14 numeric block in the matched single-seed run; the 300-resample
  paired interval was `[-0.000253, +0.000817]`.
- Organization × external-score-quartile crossing added only `+0.000045`
  pooled AUC over the matched time-feature model (`3/5` folds positive;
  paired interval `[-0.000490, +0.000536]`).
- CatBoost `max_ctr_complexity=2` lost `0.000331` on the matched preflight
  fold and took about 2.7× as long to fit.
- Native-categorical XGBoost on a 30,000-row screen reached `0.737658`
  without keys and `0.723070` with four constant-free keys; no full OOF run
  was justified.
- Applicant ID is exactly the zero-based row index and has no useful
  monotonic signal: ID-only AUC `0.502897`; fold AUCs ranged
  `0.494498`–`0.507718`.

The largest measured same-label post-hoc blend diagnostic was `0.771833`
(Phase 2 OOF blend + Phase 9 three-seed average with 60%/40% weights). A
Phase 15 / Phase 2 fixed-grid blend reached `0.771659`. These blends were
computed after examining the same labels, include unresolved Phase 2
target-derived features/constants, and are not nested validation, promotion
evidence, or an eligible score. Neither success target is close enough to
claim.

## What caused the score difference

The historical numbers describe distinct pipelines:

| Pipeline | OOF AUC | Relevant settings and status |
|---|---:|---|
| Phase 9 CatBoost | `0.768240` reported | Mean of seed AUCs `0.767955`, `0.768000`, `0.768764`; GPU, depth 6, LR 0.05, 700 iterations, native high-cardinality categories and eight keys. |
| Phase 9 saved-prediction average | `0.769839` recomputed | Rowwise mean of the three saved, aligned OOF vectors. This aggregation was not the reported Phase 9 headline and is still same-label exploratory evidence. |
| Phase 11 CatBoost | `0.767399` | Independently built feature pipeline; GPU native categoricals, `one_hot_max_size=10`, max CTR complexity 1, two replicas. |
| Phase 13 E | `0.766038` | CPU LightGBM, complete baseline plus eight keys, frequency-supported OHE. |
| Phase 14 CatBoost | `0.759529` | CPU CatBoost, one-hot threshold 10,000 (above every observed fit-fold cardinality), intentionally no CTRs. |
| Phase 14 augmented LightGBM | `0.765719` | Same Phase 13 folds; extra numeric features were not beneficial for that model/configuration. |
| Phase 15 CatBoost, one seed, numeric + time | `0.767511` | Independent rowwise builder; GPU native CTRs, 700 iterations, depth 6, LR 0.05. |
| Phase 15 CatBoost, three-seed OOF average | `0.768825` | Each row is averaged from three predictions whose respective models excluded that row. Research-only; not independent confirmation. |
| Phase 2 supervised-feature blend | `0.771103` | Nested product-key target encodings and constants 129.5/0.0132; competition provenance unresolved. |

Thus, comparing Phase 14's forced-one-hot CatBoost or Phase 13's LightGBM
directly with the Phase 9/11 native-category CatBoost conflates a change in
categorical learning, model family/device, feature-builder version, fold
aggregation, and seeds. The one-fold Phase 15 preflight showed native
CatBoost scoring `0.772628` vs Phase 14's one-hot CatBoost `0.763587` on
that fold; this is consistent with the categorical/model-regime explanation,
but is not an isolated causal ablation because their training settings and
pipelines differ.

Phase 14's formula block itself was not the explanation for the weaker score:
on the matched Phase 15 CatBoost run, the block changed pooled AUC from
`0.767004` to `0.767231`, and its addition with time features gave
`0.767511`. The respective paired bootstrap intervals vs the base model
included zero. The primary recovery came from native-category GPU CatBoost,
not a dramatic new deterministic ratio.

## Phase 15 research map (delivered before training)

The initial map is preserved as
[`PHASE15_RESEARCH_MAP.md`](./PHASE15_RESEARCH_MAP.md). It inventoried
historical OOF files and their fold/model differences before any Phase 15
model fit. Its central findings were:

1. Phase 9 is available for diagnostics but exact clean-room feature parity
   is not established; its runner imports an archived rowwise helper.
2. Phase 11 is independently reproducible and is the closest native-category
   reference.
3. Phase 13/14 OOFs align by row/ID/target, but have different learners,
   fold seeds, and preprocessing.
4. Phase 2's stronger `0.771103` score contains nested supervised
   target-derived features and unresolved constants.
5. The supplied table contains 171,202 rows and 32 predictors; there are no
   related bureau, previous-application, installment, POS/cash, or
   credit-card tables in the permitted input. External datasets are
   prohibited.

## Experiment results

### A. GPU native-category CatBoost preflight and matched OOF

The one-fold benchmark used Phase 13 fold 1 (seed `20261113`, 136,961 fit
rows, 34,241 validation rows). The three feature variants all used the eight
keys; the numeric variant added the pre-existing Phase 14 block, and the
third added seven new deterministic time-lifecycle features.

| One-fold variant | Fold AUC | Delta vs Phase 13 E fold |
|---|---:|---:|
| Native CatBoost, base all-eight-key features | `0.771323` | `+0.000465` |
| Plus Phase 14 numeric block | `0.772578` | `+0.001720` |
| Plus Phase 14 numeric and time-lifecycle features | `0.772628` | `+0.001770` |
| Phase 13 E reference | `0.770858` | — |
| Phase 14 forced-one-hot CatBoost reference | `0.763587` | — |

These are screening results, not full-model scores. The complete controlled
run froze the Phase 13 five folds, used the same CatBoost model seed for each
feature variant within each fold, and fixed the Phase 11 model settings:
GPU, 700 iterations, depth 6, learning rate 0.05, `one_hot_max_size=10`,
`max_ctr_complexity=1`, L2 5, random strength 1, border count 128. No
outer-validation labels were used for early stopping or model selection.

| Variant | Pooled OOF AUC | Delta vs native base | Paired 95% bootstrap interval vs base | Positive folds vs base |
|---|---:|---:|---:|---:|
| Native CatBoost, base all eight | `0.767004` | `0.000000` | `[0.000000, 0.000000]` | — |
| Plus Phase 14 numeric block | `0.767231` | `+0.000227` | `[-0.000505, +0.001031]` | 4/5 |
| Plus Phase 14 numeric and time-lifecycle block | `0.767511` | `+0.000507` | `[-0.000276, +0.001362]` | 4/5 |

The isolated time-feature contribution over the Phase 14 numeric block was
`+0.000280`; fold deltas were `+0.000366`, `+0.000030`, `-0.000716`,
`+0.001394`, `+0.000284` (4/5 positive). A separate 300-resample paired
bootstrap interval was `[-0.000253, +0.000817]`; no stable incremental gain
is established.

### B. Secondary-seed robustness and OOF seed average

The time-augmented native CatBoost model was run on two secondary fold seeds
using the same feature definitions and fixed model settings. Every row was
out of fold for all three seeds.

| Seed/aggregation | Pooled OOF AUC | Mean fold AUC |
|---|---:|---:|
| 20261113 primary | `0.767511` | `0.767525` |
| 20261114 secondary | `0.767946` | `0.767956` |
| 20261115 secondary | `0.767081` | `0.767166` |
| Rowwise average of all three OOF vectors | **`0.768825`** | Not a single fold assignment |

The three-seed result is a reproducible model-development candidate on
previously exposed rows, not independent confirmation. It is not directly
comparable to a test ensemble trained on 100% of the rows: each development
component was trained on its outer 80% partition.

### C. Other model and feature directions

| Direction | Result | Decision |
|---|---|---|
| XGBoost native categorical, 30,000-row screen (24,000 fit / 6,000 validation) | No keys `0.737658` (best iteration 157); four constant-free keys `0.723070` (best iteration 23) | Rejected from full OOF; substantially weak on a representative sample. This does not claim XGBoost is universally weak. |
| CatBoost CTR complexity 2, one matched fold | `0.772544` vs same-fold CTR complexity 1 `0.772875`; delta `-0.000331`; fit `56.8s` vs about `20.7s` | Rejected from full OOF; slower and lower on the screen. |
| Organization × external-score quartile, cuts fit on each outer training fold | OOF `0.767642` vs matched base `0.767596`; delta `+0.000045`; 3/5 folds positive; 400-resample interval `[-0.000490, +0.000536]` | No useful gain; not promoted. |
| Time-lifecycle features | Seven ratios/age-at-event features from supplied date offsets | Deterministic and label-free formulas; effect is small and interval includes zero. |
| Applicant ID ordering | ID equals zero-based source row index; ID-only AUC `0.502897`; fold AUCs `0.507718`, `0.501050`, `0.494498`, `0.505169`, `0.506125` | Excluded. No stable target signal; retaining ID only for alignment is safer. |

### D. Fixed OOF blend diagnostics (post-hoc; not validation)

All values below were calculated after examining the same training labels.
Weights were not nested or chosen independently, and they must not be used
to claim generalization or eligibility.

| OOF combination | Observed pooled AUC |
|---|---:|
| Phase 15 three-seed average alone | `0.768825` |
| 75% Phase 15 + 25% Phase 13 E | `0.769682` |
| 50% Phase 15 + 50% Phase 13 E | `0.769533` |
| 75% Phase 15 + 25% Phase 9 three-seed mean | `0.769508` |
| 50% Phase 15 + 50% Phase 9 three-seed mean | `0.769897` |
| 25% Phase 15 + 75% Phase 9 three-seed mean | `0.770007` |
| 25% Phase 15 + 75% Phase 2 research blend | `0.771659` |
| Phase 2 60% + Phase 9 three-seed mean 40% | `0.771833` |
| Equal Phase 15 / Phase 9 / Phase 2 | `0.771459` |

Phase 15 correlates `0.990869` (Spearman) with the Phase 9 seed-averaged OOF
predictions and `0.974369` with Phase 2. These pairs offer little additional
error complementarity. The highest value in this table uses Phase 2's
unresolved supervised features/constants and is not a valid champion score.

### E. Saved-OOF diagnostic reproduction

`generate_phase15_diagnostics.py` independently loads the saved Phase 9,
Phase 13 and Phase 15 OOF vectors, Phase 2 research vector, and supplied train
IDs/targets. It checks row/ID/target alignment before calculation and does not
fit models or use test data. It writes the 18-row fixed-blend grid,
pairwise prediction correlations, fold-level ID-order checks, and a
300-resample paired interval for the time-feature ablation. The fixed-weight
grid is descriptive only: all underlying labels were previously exposed, and
the weights were not selected inside nested folds. Input hashes and the
method warning are saved in `PHASE15_POSTHOC_DIAGNOSTICS.json`.

## Data, governance, and limitations

- Training rows: 171,202; positives: 13,821; target rate: 8.0729%;
  predictors: 32. Training SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Fixed fold seed for the main comparisons: `20261113`; 5 stratified folds
  of 34,241 / 34,241 / 34,240 / 34,240 / 34,240 rows.
- Only `train.csv`, saved Phase 13 OOF references, and historical saved OOF
  vectors used for diagnostics were read. No test rows, sample submission,
  external data, or test-to-test statistics were used. No submission file
  was created.
- All eight keys include the unresolved 130.5 and 0.0066 hypotheses.
  CatBoost native CTRs use fold-training labels internally. The exact rule
  status of those models/features remains unresolved, so all Phase 15
  CatBoost runs are Track B/research-only.
- The XGBoost no-key sample screen uses only supplied predictors and
  deterministic rowwise transformations; it is still only a screen and
  does not establish a competition-ready model.
- No score is independent confirmation: all labels have been exposed by
  prior phases and repeated research.
- No champion or submission has been created or changed.

## Reproducibility and independent verification

The independent verifier recalculated all saved Phase 15 AUCs and checked
all run manifests, IDs, targets, row alignment and fold coverage:

- 95 manifest-registry entries: 92 hash checks passed, two execution-log
  hashes required supplemental corrections, and one early failed attempt
  had no manifest because it failed before training.
- All full OOF files contain 171,202 aligned rows and finite predictions.
  The multiseed average is finite and each component is OOF for every row.
- The independent AUC audit recalculated all OOF and sample-screen scores
  exactly to the saved CSV precision.
- The additional post-hoc diagnostics validated the historical prediction
  row/ID/target order before calculating any score. Their source OOF hashes,
  fixed blend records, Spearman correlations, ID fold values, and bootstrap
  settings/results are persisted in the diagnostic outputs.
- Six Phase 15 unit tests passed; the scripts were compiled before runs.
- Peak process working set during the main CatBoost OOF run was about
  902 MiB. GPU CatBoost completed without memory errors; each primary
  candidate fit took about 9–21 seconds per fold. Full five-fold, three
  variant runtime was about 12 minutes, including paired bootstrap
  computation. GPU is an RTX 4050 Laptop with 6,141 MiB.

The original run manifests were preserved. In the successful one-fold
preflight and full OOF run, the final completion message was written after
the manifest captured `EXECUTION_LOG.txt`. The declared log hash is
therefore stale; the actual final hash and original declared hash are both
recorded in `PHASE15_MANIFEST_CORRECTIONS.csv`. All other entries in those
manifests, and all entries in the other completed manifests, passed.

The first preflight run (`phase15_native_catboost_preflight_20261105T123915Z`)
failed during feature-block assembly with `ValueError: too many values to
unpack`. It did not fit a model and is preserved as incomplete; the corrected
preflight is `phase15_native_catboost_preflight_20261105T124003Z`.

## Deliverables and run folders

- Initial map, created before training:
  [`PHASE15_RESEARCH_MAP.md`](./PHASE15_RESEARCH_MAP.md).
- Feature builders/tests/runners:
  `phase15_features.py`, `phase15_interactions.py`,
  `generate_phase15_diagnostics.py`,
  `build_phase15_final_manifest.py`,
  `test_phase15_features.py`, `run_phase15_catboost_preflight.py`,
  `run_phase15_catboost_oof.py`, `run_phase15_catboost_multiseed.py`,
  `run_phase15_xgboost_category_preflight.py`,
  `run_phase15_ctr2_preflight.py`,
  `run_phase15_org_ext_cross_preflight.py`,
  `run_phase15_org_ext_cross_oof.py`, and
  `verify_phase15_artifacts.py`.
- Complete OOF run:
  `runs/phase15_native_catboost_oof_20261105T124238Z`.
- Secondary seeds:
  `runs/phase15_native_catboost_multiseed_20261105T125826Z`.
- XGBoost screen:
  `runs/phase15_xgboost_category_preflight_20261105T125653Z`.
- CTR-complexity preflight:
  `runs/phase15_ctr2_preflight_20261105T130340Z`.
- Organization interaction preflight and OOF:
  `runs/phase15_org_ext_cross_preflight_20261105T130627Z` and
  `runs/phase15_org_ext_cross_oof_20261105T130802Z`.
- Independent AUC registry/corrections:
  `PHASE15_ARTIFACT_REGISTRY.csv`,
  `PHASE15_INDEPENDENT_AUC_VERIFICATION.csv`,
  `PHASE15_MANIFEST_CORRECTIONS.csv`, and
  `PHASE15_VERIFICATION_SUMMARY.json`.
- Final source/document/input hash index:
  `PHASE15_FINAL_SHA256_MANIFEST.csv`.
- Saved-OOF diagnostics:
  `PHASE15_POSTHOC_DIAGNOSTICS.json`,
  `PHASE15_FIXED_BLEND_DIAGNOSTICS.csv`,
  `PHASE15_OOF_CORRELATIONS.csv`,
  `PHASE15_ID_ORDER_DIAGNOSTICS.csv`, and
  `PHASE15_TIME_FEATURE_PAIRED_BOOTSTRAP.json`.

## Decision and next research boundary

Phase 15 recovered a useful native-category CatBoost research model and
explained why the forced-one-hot Phase 14 CatBoost score was low. It did not
reach 0.78000 or 0.78500. Simple new ratios, a single category cross, deeper
CTR combinations, XGBoost native categoricals, and applicant ID did not
produce a stable path to the remaining gap.

The best current research model to carry forward is the Phase 15
three-seed-average native-category CatBoost (`0.768825`) **only in the
separate unresolved Track B**. It is not a final champion, not organizer
certified, and not independent confirmation. For the conservative
label-free track, retain Phase 13 E as the best matched model
(`0.766038`); Phase 14's constant-free LightGBM result was `0.764053`.
Further score claims require either a genuinely untouched labeled
evaluation or substantial, permitted additional data sources. Neither is
available in this workspace.
