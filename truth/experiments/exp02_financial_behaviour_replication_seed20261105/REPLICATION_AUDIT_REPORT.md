# EXP02 independent-seed replication audit

- Verification: **PASS**
- Verified UTC: `2026-11-04T04:33:10.702663+00:00`
- Run ID: `financial_ratios_seed20261105`
- Evidence source: independently recomputed from saved OOF arrays, folds, inputs, and SHA-256 manifests.

## Reproduced OOF results

| Metric | Raw-feature baseline | Financial feature block | Delta |
|---|---:|---:|---:|
| Pooled OOF ROC-AUC | 0.751707724 | 0.758043865 | +0.006336142 |
| Mean paired fold delta | - | - | +0.006296822 |

Fold-delta sample SD: `0.001138140`. Positive folds: `5/5`.

| Fold | Raw AUC | Feature AUC | Paired delta |
|---:|---:|---:|---:|
| 1 | 0.749795121 | 0.756709537 | +0.006914416 |
| 2 | 0.754456830 | 0.760877486 | +0.006420656 |
| 3 | 0.755614786 | 0.760142786 | +0.004528000 |
| 4 | 0.750401777 | 0.756461232 | +0.006059454 |
| 5 | 0.748664409 | 0.756225990 | +0.007561581 |

## Protocol and integrity

- The predeclared screen passed: positive pooled delta, mean fold delta at least `+0.0002`, and at least 4/5 positive folds.
- Outer seed `20261105`; inner stopping seeds `20262106` through `20262110`; model seeds `20263106` through `20263110`.
- Input, notebook-source, result, prediction, fold, and execution-log hashes were checked. Five stratified folds were independently regenerated and matched the saved assignments.
- Training/test dimensions: `171,202 x 34` and `61,500 x 33`; sample/test ID order matched. OOF and test predictions were finite and within `[0,1]`.
- CatBoost GPU execution completed. The execution log includes four Pandas `inplace` deprecation warnings and CatBoost GPU AUC metric-period notices; neither stopped training.

## Inherited metadata correction

The generated `EXP02_REPORT.md`, `RESULTS.json`, and `run_manifest.json` retain static text from seed 42. In particular, they report seed 42, old inner/model seed rules, and `independent_seed_replication_complete=false`. These fields are stale, not the executed configuration. The executed notebook source sets the preregistered seed constants and passes the fold-specific constants into the splitters and CatBoost models. Original generated artifacts are preserved unchanged; this independently verified audit is the corrected status record.

## Interpretation and limitations

- The seven-feature block reproduced a consistent improvement over its matched raw-feature baseline across this independent seed: 5/5 folds improved.
- This supports a repeatable predictive signal in the joint ratio block, not a causal explanation, individual-feature attribution, or improvement over the champion.
- The next valid step is a predeclared matched champion ablation on identical folds. Do not claim champion lift yet.
- The available dictionary does not establish `AMT_ANNUITY` frequency; per-family-member ratios assume equal allocation. These semantic caveats remain.
- No public leaderboard score was evaluated; no submission was created.

## Artifacts

- Original execution evidence: `EXECUTION_LOG.txt`
- Executed notebook: `EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb`
- Independent verifier: `verify_exp02_seed_replication.py`
- Machine-readable audit: `REPLICATION_AUDIT_RESULTS.json`
- Original generated files were not edited by this verifier.
