# Phase 16 — nested CatBoost parameter selection

## Result

**Nested-tuned pooled OOF ROC-AUC: `0.767652809`.** The matched Phase 15
single-seed baseline on the same five outer folds scored `0.767510928`.
The observed difference is `+0.000141881`, with 4/5 positive fold deltas.
The 1,000-resample paired interval is `[-0.000442134, +0.000659163]`, so
this does not establish a reliable increase. The Phase 15 three-seed OOF
average (`0.768824708`) remains higher; Phase 2's `0.771102653` remains a
separate research-only result with unresolved constants/target-derived
features.

The parameter search did **not** close the gap to `0.78000` or `0.78500`.
It does indicate that choosing a different parameter set by fold is
insufficient: the five outer folds selected four different configurations,
and no one configuration won consistently.

## Frozen design

- Train input: `D:\gci\inpuy\train.csv`, 171,202 rows and 13,821 positives.
- Outer folds: five stratified folds, seed `20261113`, exactly matching
  Phase 13 and the Phase 15 baseline.
- Within each outer training partition: three stratified inner folds.
  Candidate selection used pooled inner OOF AUC; outer-validation labels were
  not used for selection, stopping, or fit.
- All eight Phase 13 product keys, Phase 14 deterministic numeric features,
  and seven Phase 15 date-offset features were held fixed.
- GPU CatBoost: Logloss, AUC reporting metric, `one_hot_max_size=10`,
  `max_ctr_complexity=1`, border count 128; no early stopping.
- Four parameter sets were preregistered in
  [`PHASE16_PROTOCOL.md`](./PHASE16_PROTOCOL.md). The historical model was
  included as a candidate control.

| Outer fold | Inner-selected configuration | Outer-fold AUC | Phase 15 AUC | Delta |
|---:|---|---:|---:|---:|
| 1 | depth 5, LR 0.05, 850 iterations | 0.773427082 | 0.772874781 | +0.000552301 |
| 2 | depth 6, LR 0.035, 1,000 iterations, L2 8 | 0.775154537 | 0.774919672 | +0.000234864 |
| 3 | historical depth 6, LR 0.05, 700 iterations | 0.760985652 | 0.761288394 | -0.000302743 |
| 4 | depth 6, LR 0.035, 1,000 iterations, L2 8 | 0.765225817 | 0.765141478 | +0.000084339 |
| 5 | depth 7, LR 0.04, 800 iterations, L2 8 | 0.763926674 | 0.763401437 | +0.000525238 |

Full candidate-level inner scores, selected configurations, per-fold metrics,
and row-aligned predictions are saved in the run folder
`runs/phase16_nested_catboost_20261105T132953Z`.

## Interpretation and constraints

The primary model-tuning hypothesis was not supported as a meaningful
performance improvement: the paired interval contains zero and the pooled
gain is tiny. The selected configurations vary by outer fold, which argues
against freezing one of these alternatives as a clear winner.

The supplied package contains only the application-level train/test files
and 32 predictors in `train.csv`; no bureau, prior-application, installment,
POS/cash, or credit-card history tables are present in the input directory.
Those are substantial missing information sources, but importing them or
other external data is not permitted by the local competition materials.
Further sophistication on this narrow table cannot be promised to reach
`0.785`.

All eight product keys include the unresolved 130.5 and 0.0066 hypotheses;
native CatBoost CTR eligibility is also unresolved. Phase 16 is therefore
Track B research only, not a competition candidate, independent confirmation,
or champion. No test rows, external data, submissions, or champion files
were used or changed.

## Validation and artifacts

`verify_phase16_run.py` independently checked the full 171,202-row ID/target
alignment, frozen fold assignment, finite OOF predictions, all five AUCs,
inner-choice consistency, train/Phase 13 hashes, and all 13 run-manifest
entries. The audit passed. Runtime was 1,427 seconds (about 23.8 minutes).

Run outputs and the reproducible source:

- [`run_phase16_nested_tuning.py`](./run_phase16_nested_tuning.py)
- [`verify_phase16_run.py`](./verify_phase16_run.py)
- [`PHASE16_PROTOCOL.md`](./PHASE16_PROTOCOL.md)
- `runs/phase16_nested_catboost_20261105T132953Z/SUMMARY.json`
- `runs/phase16_nested_catboost_20261105T132953Z/INNER_CONFIG_SELECTION.csv`
- `runs/phase16_nested_catboost_20261105T132953Z/FOLD_METRICS.csv`
- `runs/phase16_nested_catboost_20261105T132953Z/OOF_PREDICTIONS.csv`
- `runs/phase16_nested_catboost_20261105T132953Z/SHA256_MANIFEST.csv`

## Decision

Do not promote the Phase 16 output. Model parameter tuning alone produced no
material lift. A next bounded experiment should test whether nested,
smoothed target statistics for constant-free Q3/Q4/Q5/AG3 keys add value
over an identical complete raw-feature baseline; this must remain
research-only until the local rule ambiguity on target encoding is resolved.
