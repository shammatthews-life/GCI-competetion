# Scratch Research Master Log — Phase 58

Phase 58 is a new, standalone LightGBM experiment. Its only runtime inputs are
the official `D:\gci\inpuy\train.csv` and `D:\gci\inpuy\test.csv`. It does not
read Phase 56/57 predictions, V43 CSVs, caches, or other feature files.

## What the non-improvement diagnostics say

The user reports leaderboard scores of `0.76732` for Phase 56, `0.76528`
for Phase 57, and `0.76861` for Phase 58. These are platform results reported
by the user; local files cannot independently verify them. Phase 58 is
`+0.00129` above Phase 56 and `+0.00333` above Phase 57 on the reported
leaderboard scores, but remains `0.01139` short of `0.78000`.

The matched local OOF evidence points in the same direction:

| Model | OOF AUC | Average precision |
|---|---:|---:|
| Phase 56 Q3 LightGBM | `0.762144` | `0.246025` |
| Phase 57 rowwise CatBoost | `0.760616` | `0.244279` |

Phase 57's submission therefore had weaker OOF ranking and minority-class
precision-recall area than Phase 56 Q3. The lower leaderboard score is
consistent with the weaker ranking model; it is not evidence that a
classification threshold caused the AUC drop.

Phase 58's reported leaderboard score is higher than both earlier reported
scores, despite its OOF AUC (`0.761907`) being `0.000237` below Phase 56
(`0.762144`). This local-CV/leaderboard discrepancy is unresolved; the scores
must remain distinct metrics. Phase 58 has not shown a reliable CV gain over
Phase 56, and its reported `0.76861` still misses the user's `0.78` goal.
Historical Phase 15 `0.768825` is an OOF score, not a leaderboard score, so
it is not directly comparable to Phase 58's reported leaderboard result.

### OOF confusion / operating-point comparison

Positive prevalence is `8.073%`. Confusion counts use `[TN, FP, FN, TP]`:

| Model | Threshold rule | Confusion matrix | Precision | Recall |
|---|---|---|---:|---:|
| Phase 57 candidate | fixed `0.5` | `[157070, 311, 13474, 347]` | `0.5274` | `0.0251` |
| Phase 56 Q3 | fixed `0.5` | `[157062, 319, 13462, 359]` | `0.5295` | `0.0260` |
| Phase 57 candidate | train prevalence `0.080729` | `[110488, 46893, 4375, 9446]` | `0.1677` | `0.6835` |
| Phase 56 Q3 | train prevalence `0.080729` | `[111345, 46036, 4426, 9395]` | `0.1695` | `0.6798` |
| Phase 57 candidate | top 10% of each fold's scores | `[144963, 12418, 9117, 4704]` | `0.2747` | `0.3404` |
| Phase 56 Q3 | top 10% of each fold's scores | `[144968, 12413, 9112, 4709]` | `0.2750` | `0.3407` |

At `0.5`, recall is low because only about `0.4%` of rows cross that threshold;
this is not a suitable default operating point for this rare outcome. At the
training-prevalence threshold, precision remains low and about one-third of
rows are flagged. The threshold-free top-decile comparison shows Phase 57
captured five fewer positives than Phase 56. This is a small but directionally
consistent ranking gap. Precision/recall are diagnostic; the competition
metric is ROC-AUC.

## Why V43's reported score is not a clean recipe

The V43 source is
`D:\gci\v22_champion_audit_certified_v43_grand_apex_soVEREIGN_master.py`;
its SHA-256 is
`715801a7843979f8c1619d20f9cc86ad5864b4d37fb0688e80697abe30fbc869`.
The locally saved V43 CSV hash is
`b1406094e95665d92d963f1e9ec91d48e391a82e05c30e49c36e8f8a926d3ce0`.

The existing ledger records `0.77749` for V12 and lists V43 as "LB Ready" with
`0.77270` OOF. The user's later V43 leaderboard report of `0.77749` may be
correct, but cannot be verified from local code or a prediction CSV alone.

The inspected V43 code:

- pools train/test covariates before group/person feature construction;
- constructs custom target encodings and multiview OOF signals before a
  separate outer validation procedure, allowing outer-validation labels to
  influence those features;
- optimizes routing weights against the complete OOF label vector and reports
  performance on that same vector.

These explain why V43's OOF score is not a fair, leakage-controlled benchmark.
They do not prove why its public score was high. Model diversity and more
informative features may genuinely help, but the V43 public score alone cannot
identify the cause. The Phase 58 implementation does not reuse V43 code or
predictions.

## Phase 58 research question and protocol

The Phase 57 result shows its rowwise features help a weak matched one-hot
CatBoost baseline, but the absolute score stays low. Phase 58 tests whether a
single stronger LightGBM model can extract more ranking signal from the same
rowwise feature block without transductive statistics or target encoding.

- **New runner:**
  `experiments/phase58_fold_local_lightgbm/phase58_fold_local_lightgbm.py`
- **Arms:** raw supplied predictors versus the same predictors plus
  deterministic row-local financial, time, external-score, bureau, missingness,
  and Q3 features.
- **Model:** LightGBM CPU, fixed 1,400 rounds; same configuration and folds for
  both arms; no outer-fold early stopping.
- **Categorical processing:** `OneHotEncoder` vocabularies fit within each
  outer training fold. Unseen validation/test categories map to all-zero
  indicator columns.
- **Validation:** five stratified folds, seed `20261113`. Every fold's
  validation labels are excluded from model fitting and preprocessing.
- **Diagnostics in the runner:** per-arm OOF ROC-AUC, average precision,
  log loss, paired bootstrap, confusion counts, precision, recall, specificity,
  and F1 at fixed `0.5`, the outer-training prevalence, and top 10% score
  capacity. Threshold metrics do not tune the ROC-AUC model.
- **Excluded:** train/test pooling, custom target encoding, entity/group
  matching, external data, previous predictions, leaderboard-dependent edits,
  and CSV rank blending.

Local competition materials prohibit external data and hand-labeling but do
not specify every validation or threshold convention. Phase 58 uses labels to
fit the supervised models, but creates no custom target-derived predictors;
its engineered features are deterministic functions of each row. All labels
and folds have prior exposure, so this remains exploratory rather than
untouched confirmation.

### Pre-run checks

- Python syntax check: passed.
- Synthetic feature, fold-local one-hot, and confusion-metric self-tests:
  passed.
- Output directory did not exist.
- Runner SHA-256:
  `8dad43df1ea59408575ddd374e715eeeb4be03bf3c86f32cce9beacf52115342`.
- Official train SHA-256:
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Official test SHA-256:
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- Full run completed in `440.41` seconds.
- Independent verification recalculated AUC, average precision, log loss,
  confusion matrices, input alignment, submission range, and all manifest
  hashes; all matched.

## Phase 58 results

### Paired ranking results

| Model | OOF AUC | Average precision | Log loss |
|---|---:|---:|---:|
| Phase 58 raw one-hot LightGBM | `0.751398` | `0.235771` | `0.248063` |
| Phase 58 rowwise + Q3 LightGBM | `0.761907` | `0.246222` | `0.245133` |
| Phase 56 Q3 LightGBM | `0.762144` | `0.246025` | — |
| Phase 57 rowwise CatBoost | `0.760616` | `0.244279` | — |

Within Phase 58's matched comparison, rowwise features improved AUC by
`+0.010510` and average precision by `+0.010451`; all five folds improved.
The paired bootstrap interval was `[+0.008873, +0.012100]`, conditional on
these fixed OOF predictions. This confirms useful signal relative to Phase
58's weaker raw one-hot baseline, not a new champion. The candidate is still
`0.000237` below Phase 56 in OOF AUC and `0.006918` below the historical
Phase 15 result (`0.768825`). Its average precision is only `0.000197` above
Phase 56, so the two measures do not indicate a meaningful overall advance.

### Requested error/threshold diagnostics

For Phase 58's candidate, at threshold `0.5`, confusion counts
`[TN, FP, FN, TP]` are `[157066, 315, 13444, 377]`; precision is `0.5448`,
recall `0.0273`. At the top 10% of each fold's ranking, precision is `0.2773`
and recall `0.3435`. With positive prevalence `8.073%`, the default `0.5`
threshold flags only `0.404%` of rows. This explains the low fixed-threshold
recall, not ROC-AUC. Precision/recall and the confusion matrix depend on the
operating threshold; changing the threshold cannot improve a leaderboard
ROC-AUC.

### Decision and artifacts

Do not treat Phase 58 as a validated champion based on these results. The
user-reported leaderboard score is `0.76861`, but the local OOF AUC does not
beat Phase 56, and this CV is not an untouched confirmation after repeated
work on the same labels. The score remains below `0.78`; its cause relative
to the OOF comparison is unknown.

- Runner:
  `experiments/phase58_fold_local_lightgbm/phase58_fold_local_lightgbm.py`
- Output directory:
  `experiments/phase58_fold_local_lightgbm/runs/phase58_fold_local_lightgbm_20261117/`
- Submission:
  `experiments/phase58_fold_local_lightgbm/runs/phase58_fold_local_lightgbm_20261117/submission.csv`
- OOF predictions:
  `experiments/phase58_fold_local_lightgbm/runs/phase58_fold_local_lightgbm_20261117/PHASE58_OOF.csv`
- Report and summary:
  `experiments/phase58_fold_local_lightgbm/runs/phase58_fold_local_lightgbm_20261117/PHASE58_REPORT.md`,
  `experiments/phase58_fold_local_lightgbm/runs/phase58_fold_local_lightgbm_20261117/PHASE58_SUMMARY.json`
- Manifest: all five listed artifact/source hashes and byte counts verified.
- OOF IDs/labels and submission IDs exactly match the supplied input files;
  submission has 61,500 unique IDs and finite probabilities in `[0,1]`.
