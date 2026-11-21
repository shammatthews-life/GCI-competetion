# V22 and V28 source audit

## Scope and evidence standard

This is a read-only source audit of:

- `D:\gci\truth\reports\scratch_v_1.py` (V22 scratch candidate)
- `D:\gci\solo_champion_v28_certified.py`
- `D:\gci\solo_champion_v28_certified.ipynb`
- Existing `D:\gci\truth\audit_check_compliance.py` and local rule audit
  `D:\gci\truth\reports\final_competition_audit\OFFICIAL_RULES_AUDIT.md`

The friend-code notebook and its submission are explicitly out of scope.
Source inspection can identify potential behavior but cannot certify the
Omnicampus organizer's interpretation of ambiguous rules.

## Findings by candidate

### V22 scratch pipeline — not acceptable under the confirmed strict policy

**Critical policy failures**

- `scratch_v_1.py:84`: concatenates train and test covariates in
  `build_all_features` before deriving features. Categories, one-hot
  vocabularies, frequency/group/person links and other transforms therefore
  depend on the evaluation covariates. This violates this project's
  conservative inductive boundary (the README itself does not explicitly
  resolve unsupervised pooling).
- `scratch_v_1.py:199-220, 251-263`: creates matched-person groups over the
  pooled data and derives partner/person features.
- `scratch_v_1.py:477-497, 517, 615-617`: `partner_enc` uses matched training
  `TARGET` values in features for training, validation and test records. The
  confirmed user policy expressly excludes partner/peer label transfer. This
  is disallowed for the candidate irrespective of any leaderboard gain.

**Validation/reproducibility defects**

- `scratch_v_1.py:338-385`: expectation-residual regressors are fit across
  the pooled train/test feature frame before outer CV. They are label-free
  with respect to `TARGET`, but they do not respect the outer fold boundary
  and include test covariates. Their fold OOF evidence is therefore not a
  clean inductive estimate.
- `scratch_v_1.py:388-463`: view specialists are generated globally from
  cross-validation splits and labels. Their outputs are consumed as model
  inputs in another CV layer. This layering needs fold-by-fold lineage proof;
  the overall script does not provide the two-level isolation required by
  the strict policy.
- `scratch_v_1.py:503-529`: TE memoization keys use array lengths and first
  index, not the complete fit/query index sets. Distinct partitions can
  collide and reuse encodings. The key is not a valid fold-identity key.
- `scratch_v_1.py:535`: one-hot vocabulary is constructed from concatenated
  train and test rows.
- `scratch_v_1.py:636-640, 695-704`: fixed ensemble assembly is applied to
  pooled OOF vectors. The code does not show that the ensemble choices were
  selected in a separate/nested evaluation; the resulting AUC is vulnerable
  to model-selection optimism.
- `scratch_v_1.py:719-723`: optional post-run read of a historical prediction
  CSV. It is not shown to enter predictions, but prevents a strict
  raw-input-only/reproducible execution claim.
- The run printed OOF AUC `0.769889` but saved no row-level OOF vector, so
  the value cannot be independently rescored. The script's historical public
  score claim is not verified by local code execution.

**Disposition:** Retain only as an archived research artifact. Do not repair
or use it as the final pipeline; its architecture is not a clean baseline for
the strict policy.

### V28 — materially safer design, not yet a verified final solution

**Positive source evidence**

- `solo_champion_v28_certified.py:669-671` reads explicit `train.csv`,
  `test.csv`, and `sample_submission.csv`; it has no external data/cache
  reads or partner-label propagation.
- `:109-195` performs row-wise features with category maps fit on the
  active training partition.
- `:197-280` fits covariate residual models on the active fitting partition,
  with cross-fitted predictions for fitting rows.
- `:283-375` builds aligned inner-OOF specialist views; validation labels are
  not supplied to the specialist models.
- `:378-430` builds Bayesian target encodings with inner-fold vocabularies
  and priors for fit rows; validation/test transforms use only the permitted
  fitting partition.
- `:696-741` nests preprocessing and model predictions inside outer folds.
  `:743-750` fits meta weights on inner OOF predictions in the outer-training
  partition before evaluating the outer validation fold.
- Model training uses fixed iteration/epoch counts without outer-fold
  early-stopping labels. Submission IDs/order/range are checked near
  `:795-807`.
- The existing semantic AST checker reported no V28 standalone-script
  nesting/prior flags.

**Unresolved or failed evidence**

- The notebook output reports OOF ROC-AUC `0.771679`, but row-level OOF
  predictions are not saved. The score therefore cannot be independently
  recomputed from the delivered notebook/script.
- The existing audit runner exits nonzero because the script and notebook
  differ in CLI path defaults. The model pipeline is normalized as matching
  by that checker, but the two execution artifacts are not byte/source
  equivalent.
- Broad exception handlers around GPU smoke probes (`:683-692`) downgrade
  to CPU. The run log prints actual device selection, so this is not silent in
  the saved run, but it can hide the reason acceleration failed. The new
  runner instead must fail explicitly when a configured GPU learner is not
  available.
- The saved run used seed `SEED_EVAL=2027`. Historical audit records show
  substantial overlap between this evaluation split and development/benchmark
  rows. A nested run isolates its own outer folds, but repeated architecture
  development on overlapping rows can still bias the selected score.
- Constants/keys including `128.5`, `129.5`, and `0.0066` are deterministic
  transforms of supplied columns, not direct prediction overrides. Their
  empirical discovery and subsequent model selection must be counted as
  training-label/data reuse during evaluation interpretation.
- V28 averages outer-fold model predictions for test; this is ordinary
  ensembling, but must be reproduced with its saved source and fixed
  configuration to substantiate the delivered CSV.
- The printed public score is not present in the artifact evidence and cannot
  be inferred from local OOF AUC.

**Disposition:** Use V28 to identify candidate techniques and failure modes;
do not accept its OOF number as a verified baseline. Reimplement only
justified ideas inside the new train-only runner and save row-level outputs.

## Historical evaluation exposure

The local audit helper `truth/audit_check_compliance.py` reports that each
`SEED_EVAL=2027` held-out fold shares about 79.9-80.1% of its rows with the
first training partition from the seed-42 development split, and about
26.3-27.0% with the recorded `bench_v30` validation partition. Overlap alone
does not prove an outer-fold implementation leaked those labels. It does show
that the architecture-development history and V28 evaluation do not constitute
independent datasets; the V28 score must be treated as a post-selection
estimate.

Many prior experiments exist under `truth/experiments`. Because their score
and split history is incomplete, no new random seed can be called guaranteed
untouched solely because its integer differs from a known seed. The new
research will freeze the next evaluation protocol before candidate
comparisons, retain per-row predictions, and state this remaining historical
selection limitation.

## Source/evidence check results

- Current input hashes recorded in the local V28 output:
  - train: `B8DD5A61771D8EB2821972C1B7D3F2F747CC66471F4D34153036D2D9C458ECF1`
  - test: `DFB7AEB96E5AF3D8616ACAABA6B23CEC8680845CFD525F7C08DE15C6D2D42C65`
  - sample submission: `6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB`
- V28 submission artifact: 61,500 rows; columns `SK_ID_CURR,TARGET`; hash
  `ED0D062664A47676E78586016C076F479E0E0E59A136135D1B21CC7AFCDC4874`.
  This verifies file integrity/shape only, not public score.
- Environment detected for new runs: Python 3.12.10, RTX 4050 Laptop GPU,
  CatBoost GPU smoke passed, XGBoost CUDA smoke passed. LightGBM will be
  explicitly logged as CPU unless a GPU learner is verified.
