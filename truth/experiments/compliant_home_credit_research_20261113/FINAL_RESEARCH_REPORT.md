# Strict-Compliance Home Credit Research Report

## Executive decision

The strongest completed, strictly inductive candidate in this research run is
the five-fold GPU XGBoost model with nested target encoding. It reached
**0.762910 mean fold ROC-AUC** (0.762835 pooled OOF AUC) on the fixed
five-fold development split. Against the same XGBoost model and folds without
target encoding, its mean fold AUC improved by **0.002322**, with gains in
four of five folds.

This is a measurable local improvement, not a verified 0.77 score and not a
guarantee of public- or private-leaderboard performance. Historical V22 and
V28 scores are not independently reproducible from saved row-level OOF
predictions. The V22 source fails the project's strict policy; V28 is much
safer but still has unresolved reproducibility and evaluation-exposure
limitations.

A candidate submission was generated from the best paired-OOF method. It
uses only organizer files, fold-local target encodings, and an average of
five outer-fold GPU models. Its schema, row count, IDs/order, prediction
range, input hashes, source hashes, and OOF metrics passed the artifact
auditor. The submission's true test AUC is unknowable without test labels.

## Scope, rule check, and source audit

### Official rule source

The rules were checked against
`Hareen-Liyanage/GCI_World_Competition`, commit
`8de63b276524ca7b067e35a6e8b14f5be64a11d8`, `README.ipynb` cells 2-4.
Published requirements cover permitted data, no hand-labeling, seeds and
reproducibility, citation/disclosure, ROC-AUC and submission schema,
leaderboard behavior, and code reproduction for recognition. The README
does not explicitly resolve train/test covariate pooling or target encoding.
This project therefore adopts the stricter user-approved boundary: no
train/test pooling, no matched-person label transfer, and supervised
encodings only when nested inside the active training folds.

| Requirement or policy | V22 scratch | V28 source | New v06 candidate |
|---|---|---|---|
| Only the organizer's train/test data; sample for schema | Fails strict policy: pooled train/test feature construction and optional historical prediction read | Source reads the three organizer files; no external prediction/cache read found | Pass: exact hashes required for train, test, and sample |
| No manual test decisions or per-row overrides | Partner/person label-transfer path is present | No partner-label propagation or manual test overrides found | Pass: all test scores are model predictions |
| No label transfer/matched-person outcome copying | Fails: `partner_enc` uses matched training labels | No such path found | Pass: no matching or label transfer |
| Train-only fit / fold isolation | Fails strict boundary: pooled features and residual transforms precede outer folds; TE cache key can collide | Much stronger nesting, but score exposure and artifact parity remain unresolved | Pass for candidate: row-wise features; each outer fold's training TE is inner-OOF; validation/test TE uses only that outer training partition |
| Reproducible, scored validation evidence | Printed 0.769889 without saved row-level OOF predictions | Printed 0.771679 without saved row-level OOF predictions; notebook/script defaults differ | Pass for v06 run: row-aligned OOF, fixed seed/folds, source/input/OOF hashes, independently recomputed fold AUCs |
| Submission schema and reproducibility | Historical claim not independently reproduced | Candidate artifact requires resolving script/notebook mismatch | Pass: two required columns, all 61,500 IDs in test order, unique IDs, finite `[0,1]` scores |

These are source and local artifact findings, not an organizer ruling. In
particular, the README's silence on unsupervised train/test covariate pooling
is why the new candidate avoids it rather than claiming it is expressly
forbidden.

### V22 and V28 source findings

The complete source findings are recorded in
[`V22_V28_SOURCE_AUDIT.md`](./V22_V28_SOURCE_AUDIT.md).

- **V22:** The scratch pipeline concatenates train and test for feature
  construction, uses matched-person `partner_enc` label propagation, fits
  some residual features outside the active outer fold, uses incomplete TE
  memoization keys, pools OOF outputs for its ensemble, and can read a
  historical prediction file. Its printed 0.769889 cannot be rescored because
  no row-level OOF vector was saved. It should remain an archived experiment,
  not the strict-compliance model.
- **V28:** The source has substantially better fold-local preprocessing,
  nested encoding/specialist generation, and inner-OOF meta fitting. However,
  its reported 0.771679 also lacks row-level OOF predictions; the notebook and
  script have differing CLI defaults; GPU smoke-test exceptions can lead to
  CPU fallback; and evaluation rows overlap previous development work. Thus
  the source is promising but the historical score is not independently
  certified.

## Data and evaluation setup

- Admitted data: only organizer `train.csv`, `test.csv`, and
  `sample_submission.csv`; their SHA-256 values are enforced in every
  source-hashed candidate.
- Train: **171,202 rows**, **13,821 positives**, prevalence **8.0729%**.
- Test/sample: **61,500 rows**.
- Evaluation: seed `314159`, the same five shuffled stratified folds for each
  paired comparison. Each row has exactly one prediction from a model that
  did not train on its target.
- Reported metrics: per-fold AUC, pooled OOF AUC, and mean/std fold AUC are
  kept distinct. Paired fold deltas use identical validation rows.
- Exposure caveat: this dataset has a long history of model development and
  previously inspected folds. A new random seed would not establish an
  untouched holdout. These scores are controlled development comparisons,
  not independent final generalization estimates.
- Hardware/software recorded by runs: NVIDIA RTX 4050 Laptop GPU, Python
  3.12.10, XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0,
  scikit-learn 1.9.1, PyTorch 2.14.1+cu130, CUDA runtime 13.0.
  CatBoost/XGBoost training used GPU; LightGBM was CPU. GPU kernels may not
  reproduce bit-for-bit across software/driver versions.

## Experiments and results

All rows below use the same five folds unless explicitly marked. “Pooled” is
one ROC-AUC over concatenated OOF rows; “mean fold” is the arithmetic mean of
the five fold AUCs. Paired deltas, rather than cross-run pooled differences,
are the main comparison.

| Candidate | Model | Pooled OOF AUC | Mean fold AUC (SD) | Paired result / disposition |
|---|---|---:|---:|---|
| v01 clean raw | CatBoost GPU | 0.752433 | 0.752486 (0.005023) | Clean, raw-feature baseline |
| v01 clean raw | LightGBM CPU | 0.748892 | 0.748909 (0.004022) | Clean, raw-feature baseline |
| v01 clean raw | XGBoost GPU | 0.753335 | 0.753368 (0.005208) | Clean, raw-feature baseline |
| v02 row-wise features | CatBoost GPU | 0.758701 | 0.758761 (0.004822) | +0.006275 mean paired AUC; improved 5/5 folds |
| v02 row-wise features | LightGBM CPU | 0.756889 | 0.756926 (0.003773) | +0.008017 mean paired AUC; improved 5/5 folds |
| v02 row-wise features | XGBoost GPU | 0.760556 | 0.760588 (0.005436) | +0.007220 mean paired AUC; improved 5/5 folds |
| v05 row-wise + group stats | XGBoost GPU | 0.759654 | 0.759688 (0.004987) | **−0.000901** paired mean vs v02; improved 1/5 folds; reject |
| v06 row-wise + nested TE | XGBoost GPU | 0.762835 | **0.762910 (0.003679)** | **+0.002322** paired mean vs v02; improved 4/5 folds; current candidate |
| v04 pairwise objective | XGBoost Ranker GPU | 0.500028 | 0.500000 (0.000000) | No useful ranking; reject |

### 1. Raw-feature baseline

The baseline trained CatBoost, LightGBM, and XGBoost on supplied row-wise
columns, with category transformations fit on each training fold only. This
gives a reproducible reference point; it is not intended to match historical
pipelines.

### 2. Row-wise feature engineering

The feature candidate adds deterministic transformations of each record:
credit/income/annuity/goods ratios and logs; per-family amounts; age,
employment and relative date features; external-score mean/range/dispersion,
products and pairwise differences; missingness counts; and credit-bureau
request summaries. No other applicant or test-row statistics are needed.

This is the strongest simple improvement: every model improved on all five
paired folds. The result supports spending effort on the provided columns'
financial and external-score geometry before adding more complex
cross-person structures.

### 3. Fold-local, label-free group aggregates

The v05 candidate adds group counts and group-mean residuals/ratios for
income, credit, annuity, and external scores within organization,
occupation, education, region-rating, and income-type categories. Training
rows receive inner-fold cross-fitted statistics; outer validation/test
statistics are fit only on the active outer training rows. No target is used
to form the aggregates.

It did not help: paired XGBoost mean AUC fell 0.000901, with only one fold
improving. The added complexity is not justified, so these features were not
included in the current candidate.

The first v05 attempt stopped because the guard incorrectly rejected
ordinary missing external-score values. The corrected implementation keeps
model-supported NaNs, converts infinities to NaNs, and rejects actual
infinities. The corrected run and tests passed; the failed attempt is retained
in its separate empty run directory for traceability.

### 4. Nested target encoding

The v06 key set includes individual categorical keys, selected categorical
crosses, and rounded financial-ratio bins. For each outer fold, each training
row's encoding is calculated from an inner-training subset that excludes
that row; smoothing is 30 and priors are calculated within the corresponding
inner fit. Validation/test encodings are fit using only the outer training
partition. Unknown query keys use the training prior and zero count.

On XGBoost GPU, nested TE improved paired mean AUC by 0.002322 (four of five
folds). It also reduced fold-score standard deviation relative to the v02
XGBoost run. This is the best tested candidate, but it remains a
post-selection estimate on a development history with prior fold exposure.

A multi-engine TE run began but exited with process status `-1` after entering
the second outer fold, before saving OOF metrics. No result is claimed from
that run. The focused v06 XGBoost-only experiment then completed successfully.

### 5. Pairwise ranking objective

The rationale was that AUC is a ranking metric, so a pairwise ranking loss
might align training more closely with evaluation. In the tested
XGBoostRanker configuration the competition-data OOF output had only five
distinct scores and AUC 0.5 in every fold. A small synthetic smoke test
confirmed the library/API can learn a separable ranking problem, so this is a
failure of the tested data/configuration, not evidence that pairwise ranking
is universally ineffective. Do not use this candidate without a fresh,
predeclared implementation/tuning test.

## Current candidate submission

The separate builder fits one XGBoost model per outer fold and uses that same
fit to predict both its held-out validation rows and test rows. This avoids
duplicating each GPU fit; one export attempt through the older runner
terminated before completion. Each fold's test encodings are based
only on that fold's training rows; test predictions are averaged across the
five models.

- Builder: [`build_v06_submission.py`](./build_v06_submission.py)
- Configuration:
  [`v06_nested_te_xgboost_submission.json`](./configs/v06_nested_te_xgboost_submission.json)
- Candidate CSV:
  [`submission_xgboost_gpu.csv`](./runs/v06_submission_predict_once/submission_xgboost_gpu.csv)
- OOF rows and metrics:
  [`oof_predictions.csv`](./runs/v06_submission_predict_once/oof_predictions.csv),
  [`metrics.json`](./runs/v06_submission_predict_once/metrics.json)
- Candidate CSV SHA-256:
  `B7BCD3895CE46888BF3CD88B6BAD53DAC5DC807F0576DD52393119DD8A04AB04`
- Reproduction command:

  ```powershell
  python D:\gci\truth\experiments\compliant_home_credit_research_20261113\build_v06_submission.py `
    --output-dir D:\gci\truth\experiments\compliant_home_credit_research_20261113\runs\v06_submission_predict_once
  ```

The output directory must not already exist. Reproduction requires the same
three organizer input files and installed dependencies. The candidate CSV
should not be described as having 0.7629 test AUC: **0.7629 is its cross-
validated mean fold AUC only**.

## Verification

- Seven unit tests pass, including deterministic row-wise feature handling,
  training-only category vocabulary, unknown-category alignment, target-key
  independence, exact inner-fold target-encoding values/priors, unknown-key
  fallback, and fold-local group feature alignment/finite-value handling.
- The final artifact auditor independently checks admitted input hashes,
  source/config hashes, OOF ID order and uniqueness, finite predictions,
  per-fold and pooled AUC reproduction, OOF file hashes, submission schema,
  all 61,500 test IDs in order, and prediction bounds. It passed for v04,
  v05, v06 and the candidate submission.
- The auditor fails closed for v01/v02 because those earlier runs do not
  contain source-code hashes. Their OOF AUCs can be recomputed from saved
  vectors and input hashes, but their exact source-at-run-time cannot be
  cryptographically confirmed. Treat the paired v01-to-v02 feature result as
  strong diagnostic evidence, not fully source-attested artifact provenance.
- No historical report, notebook, pipeline, submission, or verification
  ledger was edited. This report and all new candidates/runs are in the new
  research workspace.

## Conclusion and next safe steps

Promote v06 only as the current **research candidate**, not as a guaranteed
0.77 solution. Keep v05 group statistics and v04 pairwise ranker out of the
candidate. Do not restore V22's label-transfer/pooling methods to chase its
printed OOF value. A further improvement should be predeclared, compared on
the same folds, and then checked on a genuinely locked evaluation set if one
can be established as historically untouched. Avoid repeatedly selecting
features or blend weights on these same five folds.

The existing V01/V02 source-manifest gap should be closed before using those
runs as formal audit artifacts. The current v06 submission is reproducible
from the saved builder/config and verified inputs, but no public/private
leaderboard outcome is inferred here.
