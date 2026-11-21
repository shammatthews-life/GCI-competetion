# V16 reproduction protocol and gate

## Current status

The notebook source is recovered, but an exact replay is not currently possible. The executed notebook and source-only copy have matching code-cell source hashes, while the executed copy lacks a hashed input/cache manifest. Its configured cache files are not present in the audited local paths.

The EXP85 comparison is separately blocked because its exact final 85/15 Anchor74+suite OOF vector and fold map are absent. Do not substitute the Phase 2/3C baseline or a single upstream component.

## Required inputs before training

1. Recover `resid16.pkl`, `e1_views.pkl`, `e1A_views.pkl`, `old_*.npz`, `e1_*.npz`, `e1et_*.npz`, `e1Aet_*.npz`, and `catboost_v16_fold_*.npz`, with hashes and source/data/schema/fold provenance.
2. Recover the exact final EXP85 OOF vector and row-to-fold assignments, including a fold-safe Anchor74 equivalent for the test-only transductive anchor.
3. Pin train/test/sample hashes, dependency versions, notebook source hash, seeds, fold assignments, feature schemas, and cache hashes before running.
4. Confirm organizer permission for pooled train/test covariate transforms, target encodings, and test-to-test entity coupling.
5. Define common folds and compare candidate and baseline on the same held-out rows. Keep model parameters, preprocessing and transformations fold-local where required; do not tune on the final comparison labels.

## Static-audit command

```powershell
python -B "D:\gci\truth\experiments\phase5_champion_reproduction\audit_phase5_static.py"
```

No model-training command is provided until the missing comparator and provenance inputs are restored.
