from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci")
TRUTH = ROOT / "truth"
RUNS = TRUTH / "experiments" / "phase5_champion_reproduction" / "runs"
TRAIN = ROOT / "inpuy" / "train.csv"
TEST = ROOT / "inpuy" / "test.csv"
V15_CACHE = ROOT / "codex" / "v15_cache" / "gci_cache_final"
V3_CACHE = ROOT / "codex" / "v_3_cache"
GEMINI_CACHE = ROOT / "gemini" / "cache"
V16_NOTEBOOK = ROOT / "pre info" / "oct22" / "GCI_v16_everything_merged_o_78222 score.ipynb"
V16_SOURCE_NOTEBOOK = Path(r"D:\exp colab\GCI_v16_everything_merged.ipynb")
V16_EXTRACT = ROOT / "gemini" / "v16_extracted.txt"
V16_OPT_SOURCE = ROOT / "gemini" / "train_v16_and_optimized.py"
V16_BUNDLE = GEMINI_CACHE / "v_16_blend_bundle.npz"
EIGHT_ENGINE_OOF = (
    ROOT / "gemini" / "dependent" / "exp85_notebook_run" / "cache"
    / "rebuilt_eight_engine_oof.npy"
)
EXP85_SUBMISSION = ROOT / "champion" / "submission_exp85_solo_champion.csv"
EXP85_REFERENCE = (
    ROOT / "champion" / "00_HISTORICAL_REFERENCE"
    / "EXP85_HISTORICAL_CHAMPION.csv"
)
EXP02_SOURCE = (
    ROOT / "_gci_push_tmp" / "champion" / "EXP85_CHAMPION_IMPROVEMENT"
    / "experiments" / "exp02_anchor74_oof"
)
N_TRAIN = 171_202
N_TEST = 61_500
MEMBER_WEIGHTS = {
    "qf_shallow": 0.24,
    "qf_reg": 0.12,
    "v1_allfam_oh": 0.32,
    "v2_allfam_oh": 0.16,
    "v2_mono": 0.16,
}
SAVED_V16_AUCS = {
    "old": 0.77412,
    "e1": 0.77512,
    "e1et": 0.77608,
    "e1Aet": 0.77618,
    "fixed_champion": 0.77649,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Rank input must contain at least two finite values")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_row(
    label: str,
    path: Path,
    provenance: str,
    note: str,
    shape_or_keys: str = "",
) -> dict[str, object]:
    exists = path.is_file()
    return {
        "artifact": label,
        "path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else "",
        "sha256": sha256(path) if exists else "",
        "shape_or_keys": shape_or_keys,
        "provenance_status": provenance,
        "finding": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"recovery_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    start = datetime.now(timezone.utc)
    command = f'python -B "{Path(__file__).resolve()}" --run-id {run_id}'
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    if train.shape != (N_TRAIN, 34) or test.shape != (N_TEST, 33):
        raise ValueError(f"Unexpected data dimensions: train={train.shape}, test={test.shape}")
    if "TARGET" not in train or "TARGET" in test:
        raise ValueError("TARGET must exist only in the training data")
    y = train["TARGET"].to_numpy()
    if not np.isfinite(y).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training TARGET is not a finite binary vector")

    cache_rows: list[dict[str, object]] = []
    for path in sorted(V15_CACHE.glob("*.npz")):
        with np.load(path) as arrays:
            summaries = [
                f"{key}:{arrays[key].shape}:{arrays[key].dtype}"
                for key in arrays.files
            ]
        cache_rows.append(
            artifact_row(
                f"V15-cache/{path.name}",
                path,
                "Unbound candidate cache",
                "No embedded data hash, row IDs, or fold-index sidecar.",
                "; ".join(summaries),
            )
        )

    important_files = [
        ("train.csv", TRAIN, "Verified local input", "Target present; 171,202 rows."),
        ("test.csv", TEST, "Verified local input", "Target absent; 61,500 rows."),
        (
            "executed V16 notebook",
            V16_NOTEBOOK,
            "Saved notebook; inputs/caches not bound by manifest",
            "Executed source and saved outputs were statically audited earlier.",
        ),
        (
            "source-only V16 notebook",
            V16_SOURCE_NOTEBOOK,
            "Source-only copy",
            "Canonical code-cell hash recorded in the existing static audit.",
        ),
        (
            "V16 extracted source/output",
            V16_EXTRACT,
            "Saved-output evidence only",
            "Contains claimed V16 AUCs and configured gci_cache_v16 path.",
        ),
        (
            "local V16 optimizer source",
            V16_OPT_SOURCE,
            "Different local pipeline; not the executed notebook",
            "Optimizes blend weights on the same OOF labels it scores.",
        ),
        (
            "local V16 blend bundle",
            V16_BUNDLE,
            "Unverified bundle; internal inconsistency",
            "No row IDs/fold map; its prediction arrays are exact duplicates.",
        ),
        (
            "eight-engine OOF matrix",
            EIGHT_ENGINE_OOF,
            "Upstream component matrix only",
            "Eight streams; not the final Anchor74-plus-suite champion OOF.",
        ),
        (
            "frozen EXP85 submission",
            EXP85_SUBMISSION,
            "Test submission only",
            "A submission is not an OOF prediction vector.",
        ),
        (
            "EXP85 historical reference",
            EXP85_REFERENCE,
            "Historical test submission only",
            "Cannot supply the missing final OOF or fold map.",
        ),
        (
            "V3 residual feature cache",
            V3_CACHE / "resid16.pkl",
            "Candidate feature cache",
            "Correct row count, but no source/data hash binding.",
            "DataFrame:232702x32",
        ),
        (
            "V3 view cache",
            V3_CACHE / "e1A_views.pkl",
            "Candidate feature cache",
            "Five folds and matching row counts; not bound to V16 execution.",
        ),
        (
            "V15 view cache",
            V15_CACHE / "e1_views.pkl",
            "Candidate feature cache",
            "Duplicate-name cache differs from V3 view cache.",
        ),
        (
            "V15 A-view cache",
            V15_CACHE / "e1A_views.pkl",
            "Candidate feature cache",
            "Duplicate-name cache differs from V3 A-view cache.",
        ),
        (
            "EXP02 OOF surrogate README",
            EXP02_SOURCE / "README.md",
            "Unexecuted surrogate protocol",
            "README explicitly says it is not historical Anchor74 reproduction.",
        ),
        (
            "EXP02 OOF surrogate source",
            EXP02_SOURCE / "run_anchor74_oof.py",
            "Source only; no run output present",
            "Uses different grouped folds and hybrid-cache-assisted diagnosis.",
        ),
    ]
    inventory = list(cache_rows)
    for label, path, status, note, *shape in important_files:
        inventory.append(
            artifact_row(
                label,
                path,
                status,
                note,
                shape[0] if shape else "",
            )
        )

    save_csv(run_dir / "RECOVERED_ARTIFACT_INVENTORY.csv", inventory)

    # Recompute only cached prediction diagnostics. No estimator is fitted here.
    set_oof: dict[str, np.ndarray] = {}
    comparison: list[dict[str, object]] = []
    for tag in ("old", "e1", "e1et", "e1Aet"):
        members = []
        for name, weight in MEMBER_WEIGHTS.items():
            path = V15_CACHE / f"{tag}_{name}.npz"
            with np.load(path) as arrays:
                pred = arrays["oof"]
            if pred.shape != (N_TRAIN,) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid cached OOF member: {path}")
            members.append(weight * rank01(pred))
        set_oof[tag] = np.sum(members, axis=0)
        measured = roc_auc_score(y, set_oof[tag])
        comparison.append(
            {
                "comparison": tag,
                "saved_notebook_claim_auc": SAVED_V16_AUCS[tag],
                "candidate_cache_auc": f"{measured:.8f}",
                "candidate_minus_claim": f"{measured - SAVED_V16_AUCS[tag]:+.8f}",
                "status": "NOT REPRODUCED; candidate cache only, lineage/row alignment unverified",
            }
        )

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    cb = np.zeros(len(y), dtype=np.float64)
    cb_status = "NOT REPRODUCED; product-fold shard mapping is not embedded"
    for fold_no, (_, valid_idx) in enumerate(folds, start=1):
        shard = V15_CACHE / f"catboost_product_fold_{fold_no}.npz"
        with np.load(shard) as arrays:
            fold_pred = arrays["oof"]
        if fold_pred.shape != (len(valid_idx),) or not np.isfinite(fold_pred).all():
            raise ValueError(f"Invalid CatBoost fold shard: {shard}")
        cb[valid_idx] = fold_pred
    cb_auc = roc_auc_score(y, cb)
    base_rank = (
        rank01(set_oof["old"])
        + rank01(set_oof["e1"])
        + rank01(set_oof["e1et"])
    ) / 3.0
    candidate_final = (
        0.90 * rank01(0.75 * rank01(base_rank) + 0.25 * rank01(set_oof["e1Aet"]))
        + 0.10 * rank01(cb)
    )
    candidate_final_auc = roc_auc_score(y, candidate_final)
    comparison.extend(
        [
            {
                "comparison": "catboost_product_fold_shards",
                "saved_notebook_claim_auc": "",
                "candidate_cache_auc": f"{cb_auc:.8f}",
                "candidate_minus_claim": "",
                "status": cb_status
                + "; seed-42 split order was assumed from source, not proven by shard metadata",
            },
            {
                "comparison": "fixed V16 formula using candidate caches",
                "saved_notebook_claim_auc": SAVED_V16_AUCS["fixed_champion"],
                "candidate_cache_auc": f"{candidate_final_auc:.8f}",
                "candidate_minus_claim": (
                    f"{candidate_final_auc - SAVED_V16_AUCS['fixed_champion']:+.8f}"
                ),
                "status": "NOT REPRODUCED; candidate cache identities and row alignment unverified",
            },
        ]
    )

    with np.load(V16_BUNDLE) as bundle:
        bundle_arrays = {
            key: bundle[key].copy()
            for key in ("oof_e1Aet", "oof_cb", "final_oof")
        }
        bundle_auc_field = float(bundle["final_auc"])
    duplicate_arrays = (
        np.array_equal(bundle_arrays["oof_e1Aet"], bundle_arrays["oof_cb"])
        and np.array_equal(bundle_arrays["oof_e1Aet"], bundle_arrays["final_oof"])
    )
    bundle_auc = roc_auc_score(y, bundle_arrays["final_oof"])
    comparison.append(
        {
            "comparison": "separate gemini/cache/v_16_blend_bundle.npz",
            "saved_notebook_claim_auc": f"stored final_auc field={bundle_auc_field:.6f}",
            "candidate_cache_auc": f"{bundle_auc:.8f}",
            "candidate_minus_claim": f"{bundle_auc - bundle_auc_field:+.8f}",
            "status": (
                "REJECTED AS V16 REPRODUCTION; arrays have no provenance and are exact duplicates"
                if duplicate_arrays
                else "UNVERIFIED; array lineage and row alignment are absent"
            ),
        }
    )
    save_csv(run_dir / "V16_CACHE_SCORE_COMPARISON.csv", comparison)

    readiness = [
        {
            "artifact": "EXP85 frozen submission",
            "OOF_coverage": "none; test predictions only",
            "fold_map": "not applicable/not present",
            "row_identity": "test IDs only",
            "pipeline_identity": "test synthesis source available; final OOF absent",
            "comparator_ready": "NO",
            "blocker": "Cannot compare validation AUC from a test submission.",
        },
        {
            "artifact": "rebuilt_eight_engine_oof.npy",
            "OOF_coverage": "171202x8 finite component matrix",
            "fold_map": "not embedded",
            "row_identity": "array shape only; no IDs",
            "pipeline_identity": "upstream suite; excludes final Anchor74 composition",
            "comparator_ready": "NO",
            "blocker": "Not the final 85/15 champion OOF.",
        },
        {
            "artifact": "V15-compatible model caches",
            "OOF_coverage": "full-length vectors present",
            "fold_map": "not embedded",
            "row_identity": "no IDs or data hash",
            "pipeline_identity": "filenames and scores do not bind them to V16 run",
            "comparator_ready": "NO",
            "blocker": "Candidate blend misses saved V16 outputs and is not provenance-bound.",
        },
        {
            "artifact": "catboost_product_fold_*.npz",
            "OOF_coverage": "five validation-sized shards",
            "fold_map": "not embedded; seed-42 ordering only inferred",
            "row_identity": "no IDs",
            "pipeline_identity": "product shard filenames differ from expected catboost_v16_fold_*",
            "comparator_ready": "NO",
            "blocker": "Fold assignment and V16 identity cannot be verified.",
        },
        {
            "artifact": "gemini/cache/v_16_blend_bundle.npz",
            "OOF_coverage": "full-length arrays",
            "fold_map": "not embedded",
            "row_identity": "no IDs",
            "pipeline_identity": "different optimizer source; OOF arrays are identical and stored AUC disagrees",
            "comparator_ready": "NO",
            "blocker": "Internally inconsistent and unbound to executed V16 notebook.",
        },
        {
            "artifact": "Anchor74 recovered artifact",
            "OOF_coverage": "test-only recovered artifact",
            "fold_map": "not applicable",
            "row_identity": "test rows only",
            "pipeline_identity": "does not provide an Anchor74 out-of-fold counterpart",
            "comparator_ready": "NO",
            "blocker": "Final champion OOF cannot be reconstructed from available files.",
        },
        {
            "artifact": "EXP02 raw Anchor74 OOF surrogate",
            "OOF_coverage": "no execution outputs found",
            "fold_map": "source proposes grouped seed 20261101 folds",
            "row_identity": "not executed",
            "pipeline_identity": "different model; README labels it surrogate",
            "comparator_ready": "NO",
            "blocker": "Would be a new diagnostic, not exact historical champion evidence.",
        },
    ]
    save_csv(run_dir / "OOF_READINESS_MATRIX.csv", readiness)

    anchor_file_count = sum(
        1 for path in EXP02_SOURCE.glob("*") if path.is_file()
    ) if EXP02_SOURCE.is_dir() else 0
    bundle_array_hashes = {
        key: hashlib.sha256(value.tobytes()).hexdigest()
        for key, value in bundle_arrays.items()
    }
    source_audit = f"""# Phase 5 evidence-recovery source audit

Audit time: {start.isoformat()}

## Data and scope

- Train: `{TRAIN}`; shape {train.shape}; SHA-256 `{sha256(TRAIN)}`.
- Test: `{TEST}`; shape {test.shape}; SHA-256 `{sha256(TEST)}`.
- `TARGET` is present in train and absent from test. This audit used train labels only to score already-saved arrays. No test labels were read.
- No model was fitted, no feature selection was performed, and no submission or notebook was modified.

## V16 sources and configured caches

- Executed notebook: `{V16_NOTEBOOK}`; saved outputs report fixed-blend AUC 0.77649.
- Source-only counterpart: `{V16_SOURCE_NOTEBOOK}`.
- The notebook extraction configures `gci_cache_v16` for model-member caches and expects `catboost_v16_fold_{{fi}}.npz`.
- No `gci_cache_v16` directory was found by the recursive directory-name search. Local `gci_cache_final` and `v_3_cache` artifacts exist, but have no manifest binding their contents to the executed notebook, raw-data hashes, row IDs, or folds. Two same-named view caches have different hashes.
- The local optimizer `{V16_OPT_SOURCE}` is a separate pipeline. It grid-searches blend weights against the same OOF labels whose AUC it reports, so its selected blend AUC is not an unbiased evaluation. Its bundle stores `final_auc={bundle_auc_field:.6f}`, while the stored `final_oof` scores `{bundle_auc:.8f}` against the current labels. The stored `oof_e1Aet`, `oof_cb`, and `final_oof` arrays are exact duplicates (array SHA-256 values: `{json.dumps(bundle_array_hashes, sort_keys=True)}`); this bundle is rejected as a V16 reproduction.

## Candidate V16-formula calculation

The fixed blend formula was applied without training to the local candidate model caches, using member weights from the notebook and the source's fixed 0.9/0.1 blend. Candidate scores are recorded in `V16_CACHE_SCORE_COMPARISON.csv`. The candidate set AUCs do not reproduce the saved notebook outputs, and the final diagnostic is **not a reproduced V16 result**: the cache files lack proven row/fold identity. The CatBoost shards were assigned using the source's seed-42 split order solely to calculate a diagnostic; the shards themselves contain no fold indices and are named `catboost_product_fold_*`, not `catboost_v16_fold_*`.

## EXP85 final OOF recovery

- The frozen EXP85 artifact is a 61,500-row test submission, not an OOF vector.
- `{EIGHT_ENGINE_OOF}` is a finite 171,202-by-8 upstream component matrix; it does not contain the final test-only Anchor74 plus eight-engine synthesis and has no embedded row IDs/fold sidecar.
- The available Anchor74 recovery is test-only. No Anchor74 OOF counterpart or final champion OOF/fold map was recovered.
- The source and README under `{EXP02_SOURCE}` describe a different raw-data surrogate and hybrid-cache-assisted diagnostic. It has {anchor_file_count} files and no run-output directory; it cannot substitute for historical EXP85 evidence.

## Competition validation boundary

The prior static code audit recorded pooled train+test covariate transforms and nested supervised encodings in V16. Those methods remain subject to organizer-rule clarification. This recovery run does not decide their permissibility or use them to train anything.

## Decision

**BLOCKED — exact EXP85 final OOF and fold assignments are not recoverable from the inspected artifacts.** Do not run a champion-comparison experiment, claim an improvement, or replace the frozen submission based on these cache diagnostics. The next valid gate is to obtain a provenance-bound final OOF vector, fold IDs, row IDs, member lineage, and Anchor74 out-of-fold analogue; otherwise define and label a new fold-safe challenger separately from historical reproduction.
"""
    (run_dir / "SOURCE_CODE_AUDIT.md").write_text(source_audit, encoding="utf-8")

    final_report = f"""# Phase 5 recovery report

**Status: BLOCKED — champion comparison not validly runnable.**

## Recovery result

The workspace contains several plausible OOF/cache artifacts, but none is a provenance-bound OOF for the final EXP85 champion or the executed V16 notebook. The exact final EXP85 OOF vector and fold map, plus the Anchor74 OOF counterpart, remain missing.

## Quantified cache findings

- Saved V16 notebook claims: old {SAVED_V16_AUCS['old']:.5f}; e1 {SAVED_V16_AUCS['e1']:.5f}; e1et {SAVED_V16_AUCS['e1et']:.5f}; e1Aet {SAVED_V16_AUCS['e1Aet']:.5f}; fixed final {SAVED_V16_AUCS['fixed_champion']:.5f}. These are saved-output claims, not independently validated.
- Local candidate cache diagnostics: fixed formula AUC {candidate_final_auc:.8f}; delta from saved final claim {candidate_final_auc - SAVED_V16_AUCS['fixed_champion']:+.8f}. Because cache row/fold identity is unverified, this is **not** a reproduction or valid improvement estimate.
- Separate `gemini/cache/v_16_blend_bundle.npz`: recomputed AUC {bundle_auc:.8f}, stored `final_auc` {bundle_auc_field:.6f}; `oof_e1Aet`, `oof_cb`, and `final_oof` are byte-for-byte identical. This is an internally inconsistent, unverified artifact, not a champion result.

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
"""
    (run_dir / "FINAL_RECOVERY_REPORT.md").write_text(final_report, encoding="utf-8")

    package_lines = []
    for package in ("numpy", "pandas", "scipy", "scikit-learn"):
        package_lines.append(f"{package}=={importlib.metadata.version(package)}")
    environment = (
        f"Python: {sys.version.split()[0]}\n"
        f"Platform: {sys.platform}\n"
        f"UTC audit time: {start.isoformat()}\n"
        "No dependency installation or model training performed.\n\n"
        "Installed package versions:\n"
        + "\n".join(package_lines)
        + "\n"
    )
    (run_dir / "ENVIRONMENT.txt").write_text(environment, encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), run_dir / "recover_phase5_evidence.py")

    output_files = sorted(path for path in run_dir.iterdir() if path.is_file())
    output_files.extend(
        [run_dir / "EXECUTION_LOG.txt", run_dir / "TERMINAL_OUTPUT.txt"]
    )
    log_lines = [
        f"STATUS: COMPLETED",
        f"UTC start: {start.isoformat()}",
        f"UTC end: {datetime.now(timezone.utc).isoformat()}",
        f"Command: {command}",
        f"Train dimensions: {train.shape[0]} x {train.shape[1]}",
        f"Test dimensions: {test.shape[0]} x {test.shape[1]}",
        f"Train SHA-256: {sha256(TRAIN)}",
        f"Test SHA-256: {sha256(TEST)}",
        "Model training: NOT RUN (OOF comparator gate remains blocked).",
        f"Candidate fixed-formula diagnostic AUC: {candidate_final_auc:.8f} (not a reproduced result)",
        f"Separate bundle recomputed AUC: {bundle_auc:.8f}; stored final_auc={bundle_auc_field:.6f}",
        f"Exact duplicate OOF arrays in separate bundle: {duplicate_arrays}",
        "Errors: none.",
        "Warnings: none.",
        "Outputs:",
    ]
    log_lines.extend(str(path) for path in output_files)
    (run_dir / "EXECUTION_LOG.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(log_lines))
    print(f"OUTPUT_DIRECTORY: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
