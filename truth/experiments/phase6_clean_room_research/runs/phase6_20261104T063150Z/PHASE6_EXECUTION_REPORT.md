# Phase 6 execution report

## Status

**Clean-room foundation and training-only discovery completed.** Historical Phase 2/3 source/evidence was audited; a fresh raw-only baseline was run. No large model sweeps, submissions, or champion changes.

## Inputs and validation

- Training rows/columns: `171,202 x 34`; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Test rows: `61,500`; SHA-256 `dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. Test predictors were not read; header/ID checks only.
- Saved seed-42 fold alignment: `True`; new fold file SHA-256 `a8455332203da5234e3eb180cc1852b95b5dccdc627ceb7f6d22e8d0f5cd5cfb`.
- Fresh raw-only OOF AUC: **0.760658451**, compared with historical corrected Phase 2 `0.760658`.
- Fold metrics and ID/TARGET/fold-aligned OOF predictions are preserved.

## Completed deliverables

The timestamped folder contains the requested eligibility register, clean-room rules, blocker register, Phase 2/3 code audit, experiment eligibility and validation-failure registers, raw baseline report, hidden-pattern reports and CSVs, experiment master plan, validation specification, manifest, execution log, and next-step recommendation.

## Important findings

- Confirmed leakage remains in the older domain-formula probe (outer validation used for stopping and full-label feature selection) and earlier global dummy-column preprocessing. Those older metrics are rejected as unbiased OOF.
- Corrected Phase 2 and later Phase 3 fold-local runners are suitable for **exploratory research** under the clean-room restrictions; selection over repeatedly viewed folds prevents independent confirmation.
- Existing controlled results do not establish new incremental lift for generic peer deviations, external-score interactions, missingness, or clustering. Product-structure signal is the strongest candidate family but needs a frozen incremental ablation.
- EXP02's seven finance ratios overlap Phase 3C features; its `+0.006336` compares to its own raw-only baseline, not to Phase 3C/champion.
- Pooled train/test statistics, target encoding, and test-to-test coupling remain excluded pending rule clarification.
- No actual loan term, rate, fee, or household allocation is inferred from numeric regularity.

## Remaining blockers

The local official materials are not an organizer ruling on unresolved methods, the research folds/labels are exposed, and no unseen labeled confirmation cohort is available. Model promotion and competition eligibility for any pipeline relying on those methods remain blocked.

## Execution

Run ID `20261104T063150Z`; UTC start `2026-11-04T06:31:53.233525+00:00`; runtime `105.7s`. The detailed environment, hashes, and no-training/submission assertions are in `EXECUTION_LOG.txt`.
