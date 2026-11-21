# Phase 18 — nested CatBoost / target-encoded LightGBM blend

**Status:** Completed and independently verified; research-only. The nested
blend improved on both of its single-seed components, but did not exceed the
previous Phase 15 three-seed OOF average. No champion or submission was
changed.

## Result

The fold-local blend scored **0.768668866 pooled OOF ROC-AUC** on 171,202
training rows. It improved by `+0.001157938` over the matched Phase 15
single-seed CatBoost OOF vector (`0.767510928`) and by `+0.003378113` over
the Phase 17 nested target-encoded LightGBM vector (`0.765290753`).

| Outer fold | CatBoost weight | LightGBM weight | CatBoost AUC | LightGBM AUC | Nested blend AUC | Delta vs CatBoost |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.50 | 0.50 | 0.772874781 | 0.770874946 | 0.774078041 | +0.001203260 |
| 2 | 0.75 | 0.25 | 0.774919672 | 0.772564096 | 0.775868913 | +0.000949241 |
| 3 | 0.50 | 0.50 | 0.761288394 | 0.758560079 | 0.762384059 | +0.001095665 |
| 4 | 0.75 | 0.25 | 0.765141478 | 0.763739019 | 0.766281304 | +0.001139826 |
| 5 | 0.75 | 0.25 | 0.763401437 | 0.761739815 | 0.764998069 | +0.001596633 |

Paired, class-stratified 1,000-resample 95% bootstrap intervals:

- Versus Phase 15 single-seed CatBoost: `[+0.000690214, +0.001589754]`.
- Versus Phase 17 target-encoded LightGBM: `[+0.002590654, +0.004231289]`.
- The blend beat the single-seed CatBoost component on all five outer folds.

The best previously reported Phase 15 three-seed OOF average is `0.768824708`,
which is `0.000155842` higher than this Phase 18 blend. Phase 18 therefore
does not set a new best score and remains well below the aspirational `0.785`
target. Its comparison with the three-seed result is descriptive, not a fresh
independent evaluation.

## Frozen design and implementation

- Train only; five stratified outer folds, seed `20261113`.
- For each outer training partition, create three-fold inner OOF predictions
  for both components. LightGBM stopping rounds are chosen using an internal
  split contained in each inner-fit partition.
- Select the CatBoost weight from `{0.00, 0.25, 0.50, 0.75, 1.00}` by pooled
  inner OOF AUC; apply that weight to the frozen, aligned Phase 15 and 17
  OOF predictions on the outer validation fold.
- The selected CatBoost weights were `0.50, 0.75, 0.50, 0.75, 0.75`.
- Phase 15 is GPU CatBoost with native categories, all eight product keys,
  and unresolved formula/CTR issues. Phase 17 is LightGBM with nested
  label-derived target rates and counts for the constant-free Q3/Q4/Q5/AG3
  keys.
- No test rows, external data, submissions, or champion files were used or
  changed.

## Interpretation and limitations

The complementarity appears real within this frozen, nested selection
procedure: all five folds favored an interior blend, and every outer fold
improved on either component. Nevertheless, the score is not a clean
independent confirmation. These labels and component pipelines were examined
and adapted across prior phases; nested weight selection limits direct
outer-fold leakage for this blend choice, but it does not erase the history
of research selection.

Competition eligibility is unresolved. The CatBoost component relies on
native categorical CTRs and product keys including historically hypothesized
constants; the LightGBM component uses label-derived target statistics.
Do not treat this result as an eligible submission score or promote it to a
champion. The result gives no basis to claim that `0.785` is reachable using
the currently available train-only inputs.

## Integrity and artifacts

The independent verifier passed all 171,202 row, ID, target, and fold
alignments; recomputed pooled and fold AUCs; verified all five fold-specific
weights against the full 25-row inner selection grid; independently
recomputed both paired bootstrap intervals; and checked all 16 run-manifest
entries. Training data SHA-256:
`b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
Runtime was `723.35` seconds.

Run outputs are under
`runs/phase18_nested_blend_20261105T142048Z/`, including:

- `OOF_PREDICTIONS.csv` — aligned outer OOF component and blend predictions.
- `FOLD_METRICS.csv` — fold metrics, selected weights, and inner-fit details.
- `INNER_BLEND_SELECTION.csv` — each weight's pooled inner OOF AUC per fold.
- `SUMMARY.json`, `EXECUTION_LOG.txt`, and `SHA256_MANIFEST.csv`.

Code and protocol:
[`PHASE18_PROTOCOL.md`](./PHASE18_PROTOCOL.md),
[`run_phase18_nested_blend.py`](./run_phase18_nested_blend.py), and
[`verify_phase18_run.py`](./verify_phase18_run.py).

**Decision:** retain as an exploratory ensemble result, not a champion. Stop
tuning this blend on the same OOF labels. Any next model claim needs a
predeclared untouched evaluation set or genuinely independent permitted data;
the currently available local materials do not provide either.
