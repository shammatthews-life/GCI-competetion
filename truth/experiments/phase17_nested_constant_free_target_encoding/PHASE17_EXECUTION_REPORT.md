# Phase 17 — nested target statistics on constant-free keys

## Result

The matched LightGBM baseline with four constant-free key OHE/frequency
features scored **0.764330449 pooled OOF AUC**. Adding four fold-local
smoothed target rates and four log-count features reached **0.765290753**:
delta **+0.000960304**, positive on 3/5 folds. The 1,000-resample paired
interval is `[-0.000065046, +0.002013920]` and includes zero. The lift is
uncertain and does not exceed Phase 13 E (`0.766037689`) or the Phase 15
three-seed CatBoost OOF average (`0.768824708`).

| Fold | Baseline AUC | Nested target-stat AUC | Delta | Baseline-selected rounds |
|---:|---:|---:|---:|---:|
| 1 | 0.768388337 | 0.770874946 | +0.002486609 | 383 |
| 2 | 0.770014491 | 0.772564096 | +0.002549605 | 383 |
| 3 | 0.759420324 | 0.758560079 | -0.000860245 | 593 |
| 4 | 0.761900977 | 0.763739019 | +0.001838042 | 354 |
| 5 | 0.762810866 | 0.761739815 | -0.001071050 | 863 |

This checks the core product-key hypothesis on Q3/Q4/Q5/AG3 without the
unverified 130.5 fee-offset or 0.0066 financing-step keys. It shows some
incremental signal in target rates, but the result is heterogeneous and
uncertain.

## Frozen design and implementation

- Train only: `D:\gci\inpuy\train.csv`.
- Same five Phase 13 outer folds, stratified seed `20261113`.
- Features: complete supplied raw and rowwise features, Q3/Q4/Q5/AG3 keys
  with fold-fitted one-hot and frequency support.
- Treatment adds smoothed target rates (alpha 30) and log counts for each of
  the four constant-free keys.
- Training-row target encodings are inner five-fold OOF within each outer
  training partition. Each row's target is excluded from its own map.
  Outer-validation maps use the outer-training rows only.
- LightGBM configuration and early-stopping rounds are selected on a
  baseline-only 10% split inside each outer training set; both matched outer
  variants use the same selected rounds and model seed.
- No outer validation labels, test rows, external data, submissions, or
  champion files were used or changed.

Target encoding is label-derived; its competition-rule status remains
unresolved under the local materials. This is research-only and does not
certify eligibility.

## Same-fold blend diagnostic

A separate post-hoc grid combines Phase 15 CatBoost, Phase 16 nested-tuned
CatBoost, and Phase 17 target-encoded LightGBM predictions. All were checked
to have the same outer-fold IDs. The equal three-model mixture scored
`0.769061967`; the best two-member rows were around `0.7687`. These weights
were explored after seeing the same labels, are not nested, and are not
validation evidence, promotion evidence, or an eligible score. Spearman
correlations were 0.990237 (Phase 15/16), 0.954683 (Phase 15/17), and
0.956839 (Phase 16/17). The lower correlation indicates some complementarity
but does not justify choosing a blend from this same-label sweep.

The next experiment is a properly nested two-model blend: choose weights
inside each outer training partition using inner OOF predictions, then score
only that outer validation partition. It remains exploratory due extensive
prior label exposure.

## Integrity and artifacts

The independent verifier passed: all 171,202 row IDs, targets and folds
aligned; pooled/fold AUCs and the paired bootstrap were recomputed; all nine
run-manifest entries passed. Runtime was 300 seconds.

See [`PHASE17_PROTOCOL.md`](./PHASE17_PROTOCOL.md),
[`run_phase17_nested_te.py`](./run_phase17_nested_te.py),
[`verify_phase17_run.py`](./verify_phase17_run.py), and
[`diagnose_matched_oof_blends.py`](./diagnose_matched_oof_blends.py).
Run artifacts are under
`runs/phase17_nested_te_20261105T140620Z`.
Post-hoc blend outputs are
[`PHASE17_MATCHED_OOF_BLEND_DIAGNOSTICS.csv`](./PHASE17_MATCHED_OOF_BLEND_DIAGNOSTICS.csv)
and
[`PHASE17_MATCHED_OOF_CORRELATIONS.csv`](./PHASE17_MATCHED_OOF_CORRELATIONS.csv).

## Decision

Do not promote the standalone target-encoded model or its post-hoc blends.
Nested target rates on constant-free keys have a possible small increment
over their matched four-key LightGBM baseline, but not a demonstrated
increment over the strongest existing model and not a route to `0.785`.
