# Phase 9 historical dependency graph and Phase 10 independence

## Result

The Phase 10 runtime has **zero EXP85 source, import, cache, model-output, or
feature-matrix dependency**. It imports only the independent Phase 10 builder
and standard/installed Python libraries, reads the permitted `train.csv` and
official dictionary, and reads the saved Phase 9 OOF / fold summaries only as
the historical parity reference. Neither the Phase 9 runner nor its legacy
feature helper is imported or executed.

## Phase 9 dependency trace

The Phase 9 runner declares and loads the frozen `run_independent_baseline.py`
helper directly. It also loads Phase 3, Phase 3B, and Phase 7. Those local
modules declare or transitively load the same helper: Phase 3 uses it for the
row-wise base features; Phase 3B loads it for its feature comparisons; Phase 7
loads it alongside Phase 3 and Phase 3B for product feature construction.
Phase 9's CatBoost matrix is the helper's row-wise matrix plus Phase 7's
product block and the separately constructed product keys.

Phase 9 provenance also includes the Phase 8 runner/audit and saved Phase 8
OOF files. The Phase 9 runner reads the Phase 8 OOF predictions for a
post-hoc subgroup diagnostic; that path is not a Phase 10 model input. Phase
8 itself also imports the frozen helper, so the Phase 8 diagnostic lineage is
not independent of that helper either.

The explicit Phase 9 dependency path
`D:\gci\truth\research_archive\exp85_phase2_source_snapshot\project\source\run_independent_baseline.py`
is recorded in `DEPENDENCY_PROVENANCE.csv` as **prohibited and not accessed by
Phase 10**. Its code was not opened, hashed, copied, or imported for this
zero-dependency run.

## Parity boundary

The permitted official dictionary defines the supplied fields, but does not
give the complete mathematical specification of the historical row-wise
builder. Reconstructing that feature matrix from old source or cached outputs
would violate the zero-dependency requirement. The independent builder is
therefore a newly specified transformation, not a purported Phase 9 clone.
The one-fold GPU comparison is diagnostic only; exact Phase 9 reproduction is
**BLOCKED**. No full Phase 10 ablation/grid was launched.

## Prior pre-correction action

Before this governance correction arrived, a one-fold parity probe was run
using the then-current draft, which imported the prohibited frozen helper.
That probe was not saved as a Phase 10 artifact and is excluded from all
results in this run. No full Phase 10 run was launched. The corrected runner
and results below have no such dependency.

## Validation boundary

The corrected runtime uses the training labels only for the supervised
CatBoost fit and AUC evaluation. The Phase 9 prediction file is read solely as
a historical comparator. No Phase 7 holdout, `test.csv`, sample submission,
external data, submission generation, or champion modification was used.
