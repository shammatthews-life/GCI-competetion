# Phase 5 recovery report

**Status: BLOCKED — champion comparison not validly runnable.**

## Recovery result

The workspace contains several plausible OOF/cache artifacts, but none is a provenance-bound OOF for the final EXP85 champion or the executed V16 notebook. The exact final EXP85 OOF vector and fold map, plus the Anchor74 OOF counterpart, remain missing.

## Quantified cache findings

- Saved V16 notebook claims: old 0.77412; e1 0.77512; e1et 0.77608; e1Aet 0.77618; fixed final 0.77649. These are saved-output claims, not independently validated.
- Local candidate cache diagnostics: fixed formula AUC 0.77404329; delta from saved final claim -0.00244671. Because cache row/fold identity is unverified, this is **not** a reproduction or valid improvement estimate.
- Separate `gemini/cache/v_16_blend_bundle.npz`: recomputed AUC 0.79467683, stored `final_auc` 0.782000; `oof_e1Aet`, `oof_cb`, and `final_oof` are byte-for-byte identical. This is an internally inconsistent, unverified artifact, not a champion result.

## Why the task remains blocked

1. The frozen EXP85 submission has only test rows; the final OOF vector and fold map are absent.
2. The eight-engine matrix is upstream of the final Anchor74-plus-suite synthesis.
3. V16-compatible local caches have no row IDs, fold sidecars, or manifest tying them to the notebook/data. Their candidate AUCs fail to match saved V16 outputs.
4. The local high-AUC bundle's stored score disagrees with its predictions, and its three OOF arrays are identical.
5. The available Anchor74 runner is explicitly a different grouped-fold surrogate and has no execution outputs.

No model training, validation-fold comparison, submission creation, or champion replacement was performed. No AUC improvement is claimed.

## Deliverables

- `RECOVERED_ARTIFACT_INVENTORY.csv`
- `V16_CACHE_SCORE_COMPARISON.csv`
- `OOF_READINESS_MATRIX.csv`
- `SOURCE_CODE_AUDIT.md`
- `ENVIRONMENT.txt`
- `recover_phase5_evidence.py`
- `EXECUTION_LOG.txt`
- `TERMINAL_OUTPUT.txt` (captured by the invoking PowerShell command)
