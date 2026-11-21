# Current TRUTH source audit

## Scope

Reviewed the current root README and compliance matrix; competition-materials
review; prior Python code-audit reports; the EXP85 Phase 2 experiment,
protocol, results, runner, verifier, and preserved source snapshot; and the
actual EXP02 notebook, protocol, results, feature manifest, fold metrics, and
hash manifest. This report is based on current files and independent checks,
not transcript claims alone.

## Current active experiment: EXP02

`experiments/exp02_financial_behaviour/EXP02_FINANCIAL_BEHAVIOUR.ipynb` is the
only notebook in that experiment directory. It has seven executed code cells
and no execution errors. The notebook source hash recomputes to the value in
its final manifest. The test ID array is still read from the original
`test_raw` DataFrame; the code uses distinct names for generated prediction
arrays. The suspected `test_raw` overwrite is **not present** in the saved
notebook.

The notebook:

- Verifies hashes of the provided train, test, and sample-submission CSVs.
- Uses the provided application columns only and deterministically adds seven
  guarded financial ratios.
- Excludes ID and target from predictors; does not import TRUTH/sibling
  experiment Python code or read historical prediction caches.
- Uses matched five-fold stratified splits and inner stopping splits; refits
  on each full outer training fold.
- Writes research prediction arrays, not a submission CSV.

The saved result reports baseline pooled OOF AUC 0.75126491, financial
feature pooled OOF AUC 0.75849375, delta +0.00722884, and 5/5 positive folds.
The independent check previously recorded and the current source/hash checks
support artifact integrity. This is one seed only; the decision remains
`NEEDS_INDEPENDENT_SEED_REPLICATION`, not an accepted model or leaderboard
claim.

**EXP02 source finding: PASS against explicit local data/hand-label rules.**
No unresolved method identified in its current implementation requires
transductive test-statistic fitting or selected-row manual prediction changes.
This is not organizer approval or an official score.

## Preserved EXP85 Phase 2 research

The old Phase 2 pipeline and its archived source snapshot remain research
evidence, not a competition candidate. It contains fee/product constants
`129.5` and `0.0132` whose derivation from only the supplied files has not
been established. Its source includes a fold-safe target-encoding approach,
but provenance of the constants and interpretation of transductive statistics
remain open. The corrected runner improved validation isolation, but did not
resolve those provenance questions and does not replace the required
standalone notebook. Do not copy its unresolved constants into a submission.

## Other TRUTH audit evidence

The prior source inventory found old code paths with optional caches,
transductive calculations, model artifacts, or incomplete extraction. Their
existence in `research_archive/` does not mean they are part of EXP02 or
cleared for use. Preserve them as read-only audit material. No champion source
or file was modified during this audit.

## Decision

EXP02 is a relatively clean, source-traceable research experiment and passes
the explicit local data-source and automatic-prediction controls. The **TRUTH
repository as a whole is not certified**: legacy hypotheses remain
unresolved, the official materials do not settle transductive practices or
all pretrained/cache provenance, and organizer clarification is still
needed. See [`FINAL_AUDIT_DECISION.md`](./FINAL_AUDIT_DECISION.md).
