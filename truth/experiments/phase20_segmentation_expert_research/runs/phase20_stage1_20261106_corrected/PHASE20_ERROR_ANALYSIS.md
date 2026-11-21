# Phase 20 error and result analysis

## Scope and limitations

The 180-entry registry is complete: 130 real GPU runs, 20 explicitly GPU-blocked clustering definitions, and 30 GPU placebo controls. All real runs use the same stratified 60% discovery partition and five matched folds. Reported small AUC deltas are exploratory: the project has repeatedly examined these labels, the holdback has prior label exposure, and the 150-way search creates selection bias. No champion or submission was changed.

## Findings

- Largest discovery mixture delta: `F01_C06` at `+0.001203792`; five-fold positive-split count `5/5`.
- Highest pooled discovery mixture AUC: `F01_C08` at `0.755216907` versus matched global `0.754288165`.
- `65` of 130 completed candidates had positive pooled deltas; median delta was `-0.000015532`.
- Contract-type routing (F01) was the most consistent family: all 10 configurations had positive pooled deltas; mean `+0.000872324`, median `+0.000922136`. External-score completeness (F03) and goods/credit ratio (F10) were also positive in all 10 configurations, but effect sizes remained small.
- Employment-duration (F05) and annuity/income burden (F08) lost in all 10 configurations. Learned-gate F15 lost for both tested configurations: deltas `-0.003043125` and `-0.002915731`.
- Feature/configuration comparison: C08 (raw + fold-local constant-free key frequencies) had the strongest global reference pooled AUC (`0.754288165`) on this reduced discovery sample. This does not establish that the route improves over the full-data Phase 15 result.
- Placebo 95th-percentile delta was `-0.000074085`. 31 discovery candidates passed this empirical cutoff plus the predeclared >=4/5 positive-fold rule. Because the cutoff came from only 30 same-label controls and the many candidates share labels, qualification is a ranking filter, not a significance test or independent confirmation.
- The once-used 40% holdback's highest delta among the 10 selected candidates was `F07_C08` at `+0.001527318`; its mixture AUC was `0.753684499`. The heldout labels are not independent of past project phases, so this is not promotion evidence.

## Failure and execution record

An earlier production-path pilot exposed an exact-route output-shape bug before any candidate result was checkpointed. It was fixed, unit-tested, and a complete five-fold GPU family smoke test passed before the corrected grid started. The corrected grid then checkpointed 177/180 rows before stopping on an ID-parity placebo parser mismatch. The parser was fixed and regression-tested; the runner resumed from its durable checkpoint and completed all 30 placebos without repeating saved candidate experiments. The C10 global OOF baseline was recomputed during resume. The corrected-grid wall time was approximately 5,065 seconds, excluding preflight and pilot runs; `PHASE20_RUN_SUMMARY.json` separates total wall time from the 226-second resume process.

No row-level prediction cache was saved. Therefore this analysis does not report per-row residuals or score correlations; pooled and fold AUC arithmetic was independently checked against the durable result CSV. The 20 clustering configurations were not run because no CUDA clustering library is installed; there was no CPU fallback. GPU utilization telemetry samples the successful resume/finalization process only and is explicitly not presented as full-grid telemetry.

## Decision

Do not promote a candidate. The strongest next direction, if further research is justified, is a small, predeclared contract-type specialist comparison with a genuinely new evaluation source; do not expand the same-label search or tune on the already-used holdback. The 0.78 target was not reached, and the reduced-sample scores are not comparable to historical full-data OOF scores.
