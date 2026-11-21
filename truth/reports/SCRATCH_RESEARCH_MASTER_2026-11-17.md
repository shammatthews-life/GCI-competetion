# Scratch Research Master Log

This is a new, append-only record for the independently written scratch
experiments created in response to the request on 2026-11-17. It does not
replace or modify the existing GCI verification ledger. Each code version gets
its own source path and run directory; previous versions and outputs must be
preserved.

## Competition constraints applied

- Read only the organizer-provided `train.csv` and `test.csv` in executable
  experiments; no previous submissions, OOF caches, feature stores, or external
  data may be read by a scratch runner.
- The test file may only be used to build final model inputs, generate
  predictions, and verify submission IDs. Do not use pooled train/test
  statistics, test labels, test-to-test score adjustments, or leaderboard
  feedback.
- All feature transformations learned across rows must be fit only on each
  model's training partition. Prefer row-local deterministic transforms.
- Validation rows must not participate in feature construction, early
  stopping, tuning, feature selection, or model selection for their own score.
- Keep row-level OOF IDs, targets, fold IDs, and predictions for independent
  metric recalculation.
- Competition rules do not provide a definitive ruling on every target
  statistic or inferred-identity technique. This conservative track avoids
  custom target encodings, label propagation between rows, and inferred
  applicant matching.
- AUC is a ranking metric. A monotonic score transformation cannot create
  discrimination; rank averaging or calibration is not a substitute for
  better out-of-fold ordering.

## Evidence carried forward

### Friend's 0.7788 material (2026-11-17)

- `D:\gci\friend code\help 7788\help.txt` and `help2.txt` are byte-identical.
  Their 100,014-byte script-like text configures a ten-fold/four-seed
  multi-model feature-fusion lab with hidden-product and UID features. It is
  source text, not an executed report.
- `accepted_prune_hidden_uid.json` lists 131 dropped features and sets
  `hidden`, `uid`, and `uidte` to true; it contains no metrics or validation
  evidence.
- The screenshot labels a result `0.76849`, while the user reports a
  leaderboard score `0.7788`. The artifacts do not tie that leaderboard claim
  to the supplied submission CSV. The screenshot describes five folds and a
  seven-model stack; the text config differs (ten folds and four seeds).
- `sub_hill_climb_modelos.csv` contains 61,500 IDs matching test order and
  rank-like scores. It cannot independently establish a leaderboard score.
- The pasted multi-layer stack has two important evidence concerns: base-model
  early stopping on the fold subsequently scored, and meta-validation built
  from OOF vectors whose base models may have trained on the meta-validation
  labels. Its “nested” hill-climb estimate is therefore not a clean nested
  evaluation as written.
- Useful hypotheses, not certified gains: deterministic financial ratio
  representations and genuinely diverse learners can help; repeated-applicant
  label transfer may be predictive but is legally unresolved and is excluded
  from this scratch track.

### Existing project evidence (context only; not inputs to scratch runners)

- The existing Phase 52 clean-room XGBoost ablation found that the
  raw-plus-financial-ratio/log family scored `0.759889` OOF versus `0.752413`
  for its raw one-hot baseline, a `+0.007476` matched delta. The full
  label-free feature bundle scored `0.758640`; these are exploratory and use a
  different learner/protocol from Phase 53.
- Phase 50's exploratory three-seed synthesis scored `0.767070`; it used
  prior OOF artifacts and is not a source for the new standalone code.
- Phase 51's fold-local twin/fingerprint label-statistic candidate had only
  about 2% validation rows matched to training keys and a modest score lift;
  that label-derived family remains rule-ambiguous and is excluded here.
- Historical Phase 15 CatBoost OOF (`0.768825` three-seed mean) and the user's
  reported leaderboard scores are context, not directly matched Phase 53
  baselines and not evidence that the new run will achieve `0.78`.

## Version register

| Version | Python source | Research question | Code/data scope | Run location | Result | Decision |
|---|---|---|---|---|---|---|
| 53.1 | `experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py` | Do row-local financial ratios, log scales, and external-score geometry improve a fixed CatBoost model over its raw-feature control? | Standalone; official train/test only; paired five-fold control; no cross-row or target-derived features | `experiments/phase53_scratch_financial_geometry/runs/phase53_scratch_financial_geometry_20261117/` | Engineered OOF AUC `0.759685`; `+0.007380` vs matched raw control | Retain as exploratory feature-family evidence; no 0.78 claim or champion promotion |

## Phase 53.1 — preregistration and audit checklist

- **Source:** `experiments/phase53_scratch_financial_geometry/phase53_scratch_financial_geometry.py`.
- **Output directory:** `experiments/phase53_scratch_financial_geometry/runs/phase53_scratch_financial_geometry_20261117/`.
- **Model:** single CatBoost family, CPU, 700 fixed iterations, depth 6,
  learning rate 0.05, four threads; five seeded stratified folds.
- **Paired comparison:** the same fold assignments, seed per fold, and
  hyperparameters are used for raw predictors and an expanded row-local
  financial-geometry feature set. Validation labels are not used for stopping.
- **Feature hypotheses:** scale-free credit/income, annuity/income,
  credit/annuity, goods/credit, down-payment and family-normalized ratios;
  log-scaled amounts; annualized age/employment; row-wise external-score
  moments and pairwise geometry; missingness and bureau-activity summaries.
  Rounded ratios are numeric columns, not target-encoded categories.
- **Excluded friend mechanisms:** UID/fingerprint matching, target transfer,
  hidden-product target encoding, feature-driven submission edits,
  multi-layer hill-climb, train/test pooled statistics, and CSV rank blends.
- **Categorical handling:** original low-cardinality categories only; one-hot
  threshold 255. The run aborts if a training category exceeds this cap, to
  avoid implicit CatBoost CTR/target-statistic handling.
- **Submission:** arithmetic mean of the five fold-model probabilities from
  this one experiment; original test ID order; no rank transformation.
- **Required audit:** syntax/self-test, output schema and ID alignment,
  finite/range checks, OOF coverage, pooled and fold AUC, paired fold deltas,
  conditional paired-bootstrap interval, input/source hashes, and output
  manifest. A failed gate must be recorded here; never silently repair by
  reading a previous CSV.
- **Pre-run checks:** Python compilation and the script's row-invariance,
  feature-exclusion, finite-value, and ratio self-tests passed. The runner was
  started on CPU against the two official input paths; no active Python
  training process was found immediately before launch.
- **Input SHA-256:** train
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`;
  test
  `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`.
- **Pre-run source SHA-256:**
  `10b98d608ca7a10a1d239443a4d219e3b8fba51024db4c9c44750ff60a81db2e`.

## Phase 53.1 — completed results (2026-11-17)

### Results

- **Raw-feature control:** pooled OOF AUC `0.752305736`; mean fold AUC
  `0.752328334`; log loss `0.247768430`.
- **Financial-geometry candidate:** pooled OOF AUC `0.759685265`; mean fold
  AUC `0.759725317`; log loss `0.245708820`.
- **Matched delta:** `+0.007379529` pooled AUC and `-0.002059610` log loss;
  all five fold AUC differences were positive:
  `+0.007402793`, `+0.008838135`, `+0.005144664`, `+0.006558474`,
  `+0.009040851`.
- **Paired OOF bootstrap:** conditional 95% interval
  `[+0.006189928, +0.008573897]` from 300 stratified row resamples; it does not
  account for model-training variability, previous experimentation on these
  labels, or leaderboard sampling.
- **Interpretation:** the broad deterministic financial/score-geometry bundle
  improved its matched raw-feature control, in the same general direction as
  Phase 52's XGBoost ratio-family result. This does not isolate which of the
  53 added features caused the gain. The folds have prior project exposure;
  the result is exploratory and not independent confirmation.
- **Target comparison:** the candidate OOF result is not a verified 0.78, and
  it does not supersede the historical Phase 15 three-seed OOF reference.
  Fold/model differences make that historical score contextual rather than a
  matched baseline. The friend-reported `0.7788` remains unverified against the
  supplied friend artifacts.
- **Decision:** retain the feature-family hypothesis for carefully controlled
  follow-up, but do not promote this run as a champion or claim it predicts a
  leaderboard score. No inference about 0.78 follows from the paired delta.

### Validation and artifacts

- Completed in `632.26` seconds on CPU using CatBoost `1.2.10`, five folds,
  fixed 700 iterations, depth 6, learning rate `0.05`, and four threads.
- `python -m py_compile` and the script self-test passed before training.
- Independent post-run checks recalculated both pooled AUCs and log losses
  directly from saved OOF rows, verified all 171,202 row IDs/targets/folds,
  checked 61,500 submission IDs against test order, checked prediction range,
  and confirmed all recorded source/output hashes.
- Submission:
  `experiments/phase53_scratch_financial_geometry/runs/phase53_scratch_financial_geometry_20261117/submission.csv`
  SHA-256
  `01a9c82271c99c87f4180174cb1db73d38590a8b22b94b52db5abded21c962ca`.
- OOF rows:
  `experiments/phase53_scratch_financial_geometry/runs/phase53_scratch_financial_geometry_20261117/PHASE53_OOF_PREDICTIONS.csv`
  SHA-256
  `4ff706c2ebbb81ef9227afd7ef11ee5a999d45114aa4612605bc21e75dac4175`.
- Detailed run report:
  `experiments/phase53_scratch_financial_geometry/runs/phase53_scratch_financial_geometry_20261117/PHASE53_REPORT.md`.
- Summary and full SHA-256 manifest are in that same run directory as
  `PHASE53_SUMMARY.json` and `PHASE53_SHA256_MANIFEST.json`.
- Final source SHA-256:
  `10b98d608ca7a10a1d239443a4d219e3b8fba51024db4c9c44750ff60a81db2e`.

## Cumulative research notes

Add dated findings after each later run. Record negative results and
code/output locations as well as improvements. Do not promote an OOF winner
without independent evaluation, and do not call a public leaderboard score a
model validation metric.
