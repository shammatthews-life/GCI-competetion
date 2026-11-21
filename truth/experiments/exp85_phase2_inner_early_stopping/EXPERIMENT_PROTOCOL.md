# EXP85 Phase 2 with inner early stopping

## Provenance and scope

The handoff named `EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb`, but that notebook
was not found in the audited workspace roots. No notebook was reconstructed.
This experiment uses the byte-verified baseline, fee-feature, and nested
target-encoding Python sources under
`research_archive/exp85_phase2_source_snapshot/`; those source copies remain
unchanged.

The currently supplied CSV headers contain 32 predictor columns, plus
`SK_ID_CURR`; the training file additionally contains `TARGET`. This is the
verified schema in `D:\gci\inpuy`, not 36 predictors. The feature builder
removes the ID and never passes the target to rowwise feature construction.

No Kaggle or other external dataset, feature table, prediction cache, or model
checkpoint is read. Derived fields are calculated from the supplied
application columns. The fee amount (`129.5`) and product rate (`0.0132`) are
fixed hypotheses present in the archived source; this experiment does not
look them up in an external source. Their historical provenance is not
established by the available files and should be checked with the organizer
if the rule wording restricts externally sourced domain constants.

## Validation protocol

- Five stratified outer folds, seed 42, preserve the available Phase 2 split
  design.
- For each outer fold, a deterministic stratified 10% holdout is taken only
  from the outer training partition to select boosting iterations.
- Inner-stopping features are rebuilt with frequency counts referenced only
  to inner-fit rows. Target-encoding training features are inner OOF; stopping
  rows are encoded using inner-fit labels only.
- The selected iteration count is used to initialize a fresh model refit on
  the complete outer training partition. Outer validation is not passed to
  either model's early-stopping API.
- Final outer-fit fee frequencies use only outer-fit covariates. Final
  outer-validation and test target encodings use only outer-fit labels.
- LightGBM's one-hot category vocabulary is learned separately from each
  model-fit partition; unseen apply categories map to all-zero dummy columns.
- The raw-only and Phase 2 feature variants use the same outer folds, inner
  stopping indices, model settings, and fixed 50:50 probability blend.
- No blend weights are optimized on validation labels. No submission CSV is
  created.

This removes the direct outer-validation early-stopping dependency visible in
the preserved model functions and nests target encodings again for the
iteration-selection split. It does not by itself certify official rule
compliance or predict an official leaderboard score.

## Rule boundary and current status

The supplied course package has now been reviewed. Its
`D:\gci\rules and info\04. Competition (due Nov 20)\Competition\README.ipynb`,
markdown cell 4, prohibits external data at every analysis stage and
hand-labeling of all or part of test predictions; requires reproducible,
automatically generated model predictions and random seeds; and permits
referencing code when the original source is cited. The competition tutorial
deck recommends constructing features from existing features (slide 22).
See `D:\gci\truth\reports\COMPETITION_MATERIALS_REVIEW.md` for the detailed
review and conservative decisions.

This run did not read an outside dataset, prediction cache, or model
checkpoint. However, the archived feature builder hard-codes an annuity fee
of `129.5` and a product rate of `0.0132`; available evidence does not
establish that these constants were derived only from the provided CSVs. The
run also imports helper code from the archived Python snapshot rather than
being a standalone notebook. Accordingly, the completed run and its OOF
metrics are **research-only and not cleared as a competition submission
candidate**. Do not submit or promote it until constant provenance is
established and the experiment is packaged as one standalone notebook.

The new financial experiment avoids those constants and uses only arithmetic
features from the supplied columns. It will fit preprocessing/model state
inside each training fold, apply the fitted model to held-out/test rows, and
generate predictions only through the model. Pooled test-covariate statistics
and manual prediction overrides are excluded.

## Execution

Example:

```powershell
python -B D:\gci\truth\experiments\exp85_phase2_inner_early_stopping\run_experiment.py `
  --data-dir D:\gci\inpuy `
  --run-dir D:\gci\truth\experiments\exp85_phase2_inner_early_stopping\runs\phase2_inner_stop_seed42
```

The runner refuses to start unless `nvidia-smi` reports an RTX 4050, requires
an empty run directory, verifies raw schemas and IDs, writes OOF/test arrays
and hashes, and intentionally does not write a submission.

To independently verify the completed run without retraining:

```powershell
python -B D:\gci\truth\experiments\exp85_phase2_inner_early_stopping\verify_run.py `
  --data-dir D:\gci\inpuy `
  --run-dir D:\gci\truth\experiments\exp85_phase2_inner_early_stopping\runs\phase2_inner_stop_seed42
```

The verifier recomputes the fold assignments, input and source hashes, AUCs,
fixed blends, array validity, and test/sample ID order. It writes
`VERIFIED_METRICS.json`.
