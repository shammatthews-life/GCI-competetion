# Phase 19 — unified OOF recovery and error analysis

All listed predictions were aligned to the same 171,202 training IDs and targets before scoring. Scores reuse historically exposed labels and are research diagnostics, not independent confirmation.

## Main findings

- Phase 9 reported mean seed AUC: `0.768239947`; recomputed rowwise mean-prediction AUC: `0.769839127`.
- Phase 15 mean seed AUC: `0.767512684`; recomputed/saved three-seed mean-prediction AUC: `0.768824708`.
- These seed-averaged vectors have no single native fold assignment. Their fold rows are explicitly marked as descriptive common partitions, not native validation folds.
- Phase 9, Phase 11, and Phase 13–18 OOF sources use different CV fold definitions; only identical row IDs/targets justify pooled score and prediction-correlation comparisons. Fold deltas are interpreted only on matched splits.
- Highest score in this unified inventory: `phase9_native8_seed_average` at `0.769839127`. This is a retrospective same-label comparison, not a promotion decision.
- Highest provisional Track A candidate: `phase13_C_raw_rowwise` at `0.759619220`; this is not organizer certification.
- Lowest prediction Spearman correlation with the Phase 15 three-seed vector: `phase19_mlp_constant_free` at `0.859163`. Low correlation alone does not establish useful ensemble lift.

## Recovery interpretation

Phase 9 and Phase 15 are not controlled replications: their feature builders differ, and their multi-seed ensembles average predictions generated under different CV partitions. Phase 15's time-feature ablation was only about +0.000280 in its matched single-seed comparison, with a paired interval containing zero; Phase 16's nested parameter selection added only +0.000142 with an interval containing zero. Available evidence therefore does not isolate a causal score loss to one feature or parameter. The apparent Phase 9/15 ensemble difference is compatible with feature-builder and seed/fold variation plus averaging nonlinearity.

The main structural ceiling is likely limited input information: local inventory contains only the supplied application-level table, not bureau, previous-application, installment, POS/cash, or credit-card histories. Those data are not locally available and local competition guidance prohibits external data. More model tuning on these 32 predictors cannot be promised to add 0.01 AUC.

## Fold and subgroup diagnostics

See `PHASE19_FOLD_METRICS.csv` for native and reference-fold definitions. `PHASE19_ERROR_SUBGROUPS.csv` contains post-hoc subgroup AUCs for Phase 15 and the Phase 19 MLP if its run is present. These subgroup estimates are descriptive and not fresh confirmation.

## Decision

No tiny OOF lift alone promotes a champion. The safest raw-only complete baseline is Phase 13 C; the Phase 19 MLP is the one new label-free neural-family candidate, evaluated with fold-local preprocessing. The best research OOF score remains a retrospective ensemble result, not a competition-ready score. The current evidence does not establish a path to 0.780–0.785.
