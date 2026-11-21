# TRUTH

TRUTH is an isolated workspace for researching, auditing, and developing a
competition-compliant solution using preserved GCI competition research.

## Isolation rules

- This repository is independent of the preserved champion repository and has
  no Git remote configured until a dedicated TRUTH GitHub repository exists.
- Preserved source directories outside `D:\gci\truth` are read-only audit
  inputs. Do not edit them or stage their files here.
- Do not copy unavailable source files or prediction artifacts into this
  project, and do not infer missing implementation details.
- Keep audit reports and generated artifacts under their dedicated TRUTH
  directories. Do not include local datasets, credentials, or environment
  files in version control.
- A static code audit is not proof of an official leaderboard result or of
  organizer approval for an ambiguous method.

## Current development candidate

The available EXP85 Phase 2 Python sources are preserved byte-for-byte under
`research_archive/exp85_phase2_source_snapshot/`. The named
`EXP85_PHASE2_COMPLETE_SUBMISSION.ipynb` was not present in the audited roots
and has not been reconstructed. A separate corrected experiment using inner
early stopping and fold-local feature construction is under
`experiments/exp85_phase2_inner_early_stopping/`.

The corrected Phase 2 research run completed five folds and reached
0.77110265 pooled OOF AUC versus 0.76136207 for its matched raw-only baseline
(+0.00974058; all five paired fold deltas were positive). It is **not cleared
for competition use**: the archived fee/product constants have unverified
provenance, and the experiment does not yet have a standalone primary
notebook. No submission was created and no official score was evaluated. See
`reports/COMPETITION_MATERIALS_REVIEW.md` for the supplied rules, tutorial,
data dictionary, and course-deck review, and
`experiments/exp85_phase2_inner_early_stopping/RESULTS.md` for the research
metrics and verification evidence.

## EXP02 financial-feature experiment

`experiments/exp02_financial_behaviour/EXP02_FINANCIAL_BEHAVIOUR.ipynb` is
the experiment's single standalone primary notebook. It uses only the supplied
application columns and tests seven predeclared rowwise financial ratios
against a matched CatBoost raw-feature baseline with five leakage-separated
folds and inner early stopping. The first run reached 0.75849375 pooled OOF
AUC versus 0.75126491 for its baseline (+0.00722884), improving all five
folds. The independent artifact verifier passed, and RTX 4050 training and
telemetry were recorded.

The result is **provisional pending independent-seed replication**. It is not
an accepted submission candidate or proof of an official score or organizer
approval. No submission was created. See
`experiments/exp02_financial_behaviour/EXP02_REPORT.md` for the complete
metrics, validation controls, and integrity results.
