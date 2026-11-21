from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci")
TRUTH = ROOT / "truth"
PHASE = TRUTH / "experiments" / "phase5_champion_reproduction"
RUNS = PHASE / "runs"
TRAIN = ROOT / "inpuy" / "train.csv"
TEST = ROOT / "inpuy" / "test.csv"
V15_CACHE = ROOT / "codex" / "v15_cache" / "gci_cache_final"
V3_CACHE = ROOT / "codex" / "v_3_cache"
V15_ARCHIVE = ROOT / "codex" / "gci_cache_final-20261021T075423Z-1-001.zip"
V15_NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
V16_NOTEBOOK = ROOT / "pre info" / "oct22" / "GCI_v16_everything_merged_o_78222 score.ipynb"
V16_SOURCE_NOTEBOOK = Path(r"D:\exp colab\GCI_v16_everything_merged.ipynb")
V16_TEXT = ROOT / "gemini" / "v16_extracted.txt"
V16_OPT_SOURCE = ROOT / "gemini" / "train_v16_and_optimized.py"
V16_SUITE_SOURCE = ROOT / "gemini" / "train_full_v16_suite.py"
V16_FUSION_SOURCE = ROOT / "gemini" / "exp11_v16_grand_fusion.py"
SCAFFOLD_SOURCES = [
    ROOT / "gemini" / "scratch" / "generate_exp85_solo_notebook.py",
    ROOT / "gemini" / "scratch" / "build_clean_exp85_solo_notebook.py",
]
V16_BUNDLE = ROOT / "gemini" / "cache" / "v_16_blend_bundle.npz"
V6_BUNDLE = ROOT / "gemini" / "cache" / "v_6_blend_bundle.npz"
V8_BUNDLE = ROOT / "gemini" / "cache" / "v_8_blend_bundle.npz"
EXP85_OOF = (
    ROOT / "gemini" / "dependent" / "exp85_notebook_run" / "cache"
    / "rebuilt_eight_engine_oof.npy"
)
EXP85_SUBMISSION = ROOT / "champion" / "submission_exp85_solo_champion.csv"
EXP85_REFERENCE = (
    ROOT / "champion" / "00_HISTORICAL_REFERENCE"
    / "EXP85_HISTORICAL_CHAMPION.csv"
)
ANCHOR_SOURCE = (
    ROOT / "champion" / "02_EXP85_REBUILT" / "source"
    / "generate_anchor74_from_scratch.py"
)
ANCHOR_RUNNER = (
    ROOT / "_gci_push_tmp" / "champion" / "EXP85_CHAMPION_IMPROVEMENT"
    / "experiments" / "exp02_anchor74_oof" / "run_anchor74_oof.py"
)
ANCHOR_README = ANCHOR_RUNNER.with_name("README.md")
PREVIOUS_RUN = RUNS / "recovery_20261104T053622Z"
N_TRAIN = 171_202
N_TEST = 61_500
MEMBER_WEIGHTS = {
    "qf_shallow": 0.24,
    "qf_reg": 0.12,
    "v1_allfam_oh": 0.32,
    "v2_allfam_oh": 0.16,
    "v2_mono": 0.16,
}
SAVED_V16 = {
    "old": 0.77412,
    "e1": 0.77512,
    "e1et": 0.77608,
    "e1Aet": 0.77618,
    "fixed_final": 0.77649,
}
EXPECTED_INPUT_HASHES = {
    "train": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    "test": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Rank input must be nonempty and finite")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def npz_info(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    summary = "; ".join(
        f"{key}={value.shape}/{value.dtype}" for key, value in arrays.items()
    )
    return summary, arrays


def file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "created_utc": datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def notebook_source_hash(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    return sha256_bytes(code.encode("utf-8"))


def archive_members() -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(V15_ARCHIVE) as archive:
        return {
            Path(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }


def check_v16_cache_dir() -> bool:
    for current, dirs, _ in os.walk(ROOT):
        dirs[:] = [
            name for name in dirs
            if name not in {".git", ".venv", "node_modules", "__pycache__"}
        ]
        if Path(current).name.lower() == "gci_cache_v16":
            return True
    return False


def inventory_caches(
    zip_members: dict[str, zipfile.ZipInfo],
    input_hashes: dict[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base, family in ((V15_CACHE, "gci_cache_final candidate"), (V3_CACHE, "v_3_cache candidate")):
        for path in sorted(base.iterdir()):
            if not path.is_file():
                continue
            meta = file_metadata(path)
            archive_info = zip_members.get(path.name) if family.startswith("gci_cache") else None
            archived_hash = ""
            archive_match = "not in archive"
            archive_time = ""
            if archive_info is not None:
                with zipfile.ZipFile(V15_ARCHIVE) as archive:
                    archived_hash = sha256_bytes(archive.read(archive_info))
                archive_match = str(archived_hash == meta["sha256"])
                archive_time = datetime(*archive_info.date_time).isoformat()
            shapes = ""
            prediction_status = "Not a prediction NPZ"
            fold = ""
            if path.suffix == ".npz":
                shapes, arrays = npz_info(path)
                prediction_status = (
                    "OOF/test keys claimed; no row IDs or fold sidecar"
                    if "oof" in arrays or any(k.startswith("oof_") for k in arrays)
                    else "Not identified as OOF"
                )
                for key in ("oof", "final_oof", "oof_e1Aet"):
                    if key in arrays and arrays[key].ndim == 1 and len(arrays[key]) == N_TRAIN:
                        prediction_status += "; length matches current training rows only"
                        break
                if "_fold_" in path.stem:
                    fold = path.stem.rsplit("_fold_", 1)[-1]
            rows.append(
                {
                    "cache_family": family,
                    "filename": path.name,
                    "absolute_path": str(path),
                    "bytes": meta["bytes"],
                    "sha256": meta["sha256"],
                    "filesystem_modified_utc": meta["modified_utc"],
                    "archive_member_timestamp": archive_time,
                    "archive_sha256_match": archive_match,
                    "archive_member_sha256": archived_hash,
                    "keys_shapes_dtypes": shapes,
                    "fold_filename_hint": fold,
                    "prediction_semantics": prediction_status,
                    "train_sha256_embedded": "",
                    "test_sha256_embedded": "",
                    "row_ids_embedded": "no",
                    "fold_ids_embedded": "no",
                    "fold_seed_embedded": "no",
                    "fold_seed_candidate": (
                        "StratifiedKFold(5, shuffle=True, random_state=42) from notebook source; unverified"
                        if family.startswith("gci_cache") and path.suffix == ".npz"
                        else "not established"
                    ),
                    "feature_schema_embedded": "no",
                    "model_configuration_embedded": "no",
                    "source_code_hash_embedded": "no",
                    "provenance_assessment": (
                        "Byte-identical to Oct-2026-21 V15 cache archive; producer run/input binding absent"
                        if archive_match == "True"
                        else "Unbound local candidate; no provenance sidecar"
                    ),
                    "train_hash_matches_current": "not verifiable from array",
                    "test_hash_matches_current": "not verifiable from array",
                    "source_hash_candidate": (
                        notebook_source_hash(V15_NOTEBOOK)
                        if archive_match == "True"
                        else ""
                    ),
                }
            )
    return rows


def calculate_candidate_scores(y: np.ndarray) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    sets: dict[str, np.ndarray] = {}
    rows = []
    for tag in ("old", "e1", "e1et", "e1Aet"):
        parts = []
        for name, weight in MEMBER_WEIGHTS.items():
            path = V15_CACHE / f"{tag}_{name}.npz"
            _, arrays = npz_info(path)
            values = arrays["oof"]
            if values.shape != (N_TRAIN,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid candidate cached OOF member: {path}")
            parts.append(weight * rank01(values))
        sets[tag] = np.sum(parts, axis=0)
        measured = float(roc_auc_score(y, sets[tag]))
        rows.append(
            {
                "component": tag,
                "historical_v16_saved_output_auc": f"{SAVED_V16[tag]:.8f}",
                "candidate_v15_archive_auc": f"{measured:.8f}",
                "difference": f"{measured - SAVED_V16[tag]:+.8f}",
                "interpretation": "V15 archive diagnostic; not V16 reproduction",
            }
        )

    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y
        )
    )
    cb = np.zeros(len(y), dtype=np.float64)
    cb_fold_scores = []
    for fold_number, (_, valid_idx) in enumerate(folds, start=1):
        _, arrays = npz_info(V15_CACHE / f"catboost_product_fold_{fold_number}.npz")
        values = arrays["oof"]
        if values.shape != (len(valid_idx),):
            raise ValueError(f"Fold shard length mismatch for fold {fold_number}")
        cb[valid_idx] = values
        cb_fold_scores.append(float(roc_auc_score(y[valid_idx], values)))

    b = (rank01(sets["old"]) + rank01(sets["e1"]) + rank01(sets["e1et"])) / 3.0
    final = (
        0.90 * rank01(0.75 * rank01(b) + 0.25 * rank01(sets["e1Aet"]))
        + 0.10 * rank01(cb)
    )
    final_auc = float(roc_auc_score(y, final))
    rows.extend(
        [
            {
                "component": "catboost_product_fold_shards",
                "historical_v16_saved_output_auc": "",
                "candidate_v15_archive_auc": f"{roc_auc_score(y, cb):.8f}",
                "difference": "",
                "interpretation": (
                    "Assumed seed-42 split order; shard IDs absent; fold AUCs="
                    + json.dumps([round(score, 8) for score in cb_fold_scores])
                ),
            },
            {
                "component": "fixed_v16_formula_on_candidate_cache",
                "historical_v16_saved_output_auc": f"{SAVED_V16['fixed_final']:.8f}",
                "candidate_v15_archive_auc": f"{final_auc:.8f}",
                "difference": f"{final_auc - SAVED_V16['fixed_final']:+.8f}",
                "interpretation": (
                    "Matches the V15 reference band; candidate inputs are V15 archive, "
                    "not configured gci_cache_v16"
                ),
            },
        ]
    )
    return rows, {
        **{f"{name}_oof": value for name, value in sets.items()},
        "catboost_oof": cb,
        "candidate_fixed_blend": final,
    }


def investigate_array_identity(y: np.ndarray) -> list[dict[str, object]]:
    files = {
        "v16_bundle": (V16_BUNDLE, ("oof_e1Aet", "oof_cb", "final_oof", "test_e1Aet", "test_cb", "final_test_rank")),
        "v6_bundle": (V6_BUNDLE, ("final_oof", "final_test_rank")),
        "v8_bundle": (V8_BUNDLE, ("final_oof", "final_test_rank")),
    }
    vectors: dict[tuple[str, str], np.ndarray] = {}
    stored_scores: dict[str, float] = {}
    file_hashes: dict[str, str] = {}
    for bundle_name, (path, keys) in files.items():
        _, arrays = npz_info(path)
        file_hashes[bundle_name] = sha256_file(path)
        if "final_auc" in arrays:
            stored_scores[bundle_name] = float(arrays["final_auc"])
        for key in keys:
            if key in arrays:
                vectors[(bundle_name, key)] = arrays[key]

    representatives: dict[str, tuple[str, str]] = {}
    rows = []
    for (bundle_name, key), values in vectors.items():
        raw_hash = sha256_bytes(np.asarray(values).tobytes())
        group_id = next(
            (
                group for group, representative in representatives.items()
                if np.array_equal(values, vectors[representative])
            ),
            "",
        )
        if not group_id:
            group_id = f"identity_group_{len(representatives) + 1}"
            representatives[group_id] = (bundle_name, key)
        is_train_vector = values.ndim == 1 and len(values) == len(y)
        auc = float(roc_auc_score(y, values)) if is_train_vector else ""
        rows.append(
            {
                "bundle": bundle_name,
                "array_key": key,
                "bundle_sha256": file_hashes[bundle_name],
                "array_sha256": raw_hash,
                "shape": str(values.shape),
                "dtype": str(values.dtype),
                "identity_group": group_id,
                "matches_v16_oof_e1Aet": str(
                    np.array_equal(values, vectors[("v16_bundle", "oof_e1Aet")])
                ),
                "auc_against_current_train_target": (
                    f"{auc:.10f}" if is_train_vector else ""
                ),
                "stored_bundle_auc": (
                    f"{stored_scores[bundle_name]:.8f}"
                    if bundle_name in stored_scores else ""
                ),
                "row_id_alignment": "not verifiable; no applicant IDs embedded",
                "fold_map": "absent from array/bundle",
                "producer_code_finding": (
                    "Scaffold generator writes shared v6_oof into V16 fields; source computes "
                    "v6_oof by predicting training rows after fitting on all training labels"
                    if bundle_name == "v16_bundle" and key in {"oof_e1Aet", "oof_cb", "final_oof"}
                    else "Compared to V16 bundle; producer lineage not embedded"
                ),
                "oof_validity": (
                    "INVALID as OOF under matched scaffold generation path"
                    if is_train_vector and group_id == "identity_group_1"
                    else "Not an OOF vector" if not is_train_vector else "Unverified"
                ),
            }
        )

    # Pairwise correlations are appended as explicit records using the same schema.
    for i, (left_name, left) in enumerate(vectors.items()):
        if left.ndim != 1:
            continue
        for right_name, right in list(vectors.items())[i + 1:]:
            if right.ndim == 1 and left.shape == right.shape:
                rows.append(
                    {
                        "bundle": f"{left_name[0]}::{left_name[1]} vs {right_name[0]}::{right_name[1]}",
                        "array_key": "pairwise_identity",
                        "bundle_sha256": "",
                        "array_sha256": "",
                        "shape": str(left.shape),
                        "dtype": f"{left.dtype}/{right.dtype}",
                        "identity_group": "",
                        "matches_v16_oof_e1Aet": "",
                        "auc_against_current_train_target": "",
                        "stored_bundle_auc": "",
                        "row_id_alignment": "unknown",
                        "fold_map": "unknown",
                        "producer_code_finding": "",
                        "oof_validity": (
                            "exactly identical" if np.array_equal(left, right)
                            else f"pearson={np.corrcoef(left, right)[0, 1]:.12f}"
                        ),
                    }
                )
    return rows


def add_manifest_row(rows: list[dict[str, object]], label: str, path: Path, role: str) -> None:
    if not path.is_file():
        rows.append(
            {
                "label": label,
                "path": str(path),
                "role": role,
                "exists": "no",
                "bytes": "",
                "sha256": "",
                "modified_utc": "",
            }
        )
        return
    meta = file_metadata(path)
    rows.append(
        {
            "label": label,
            "path": str(path),
            "role": role,
            "exists": "yes",
            "bytes": meta["bytes"],
            "sha256": meta["sha256"],
            "modified_utc": meta["modified_utc"],
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / f"phase5_task3_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    command = (
        f'python -B "{Path(__file__).resolve()}" --run-id {run_id}'
    )

    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    if train.shape != (N_TRAIN, 34) or test.shape != (N_TEST, 33):
        raise ValueError(f"Unexpected data dimensions: {train.shape} / {test.shape}")
    if "TARGET" not in train or "TARGET" in test:
        raise ValueError("Expected TARGET only in training data")
    y = train["TARGET"].to_numpy()
    input_hashes = {"train": sha256_file(TRAIN), "test": sha256_file(TEST)}
    for name, expected in EXPECTED_INPUT_HASHES.items():
        if input_hashes[name] != expected:
            raise ValueError(f"{name} input SHA-256 does not match audited data")

    members = archive_members()
    cache_rows = inventory_caches(members, input_hashes)
    write_csv(run_dir / "V16_CACHE_PROVENANCE_MATRIX.csv", cache_rows)

    candidate_rows, candidate_arrays = calculate_candidate_scores(y)
    write_csv(run_dir / "V16_SCORE_DISCREPANCY_REPORT.csv", candidate_rows)
    identity_rows = investigate_array_identity(y)
    write_csv(run_dir / "OOF_ARRAY_IDENTITY_AUDIT.csv", identity_rows)

    v16_cache_exists = check_v16_cache_dir()
    bundle_meta = file_metadata(V16_BUNDLE)
    v15_archive_meta = file_metadata(V15_ARCHIVE)
    v15_code_hash = notebook_source_hash(V15_NOTEBOOK)
    v16_code_hash = notebook_source_hash(V16_NOTEBOOK)
    scaffold_meta = [file_metadata(path) for path in SCAFFOLD_SOURCES]
    with zipfile.ZipFile(V15_ARCHIVE) as archive:
        archive_entries = archive.infolist()
        archive_end = max(datetime(*item.date_time) for item in archive_entries)
        archive_start = min(datetime(*item.date_time) for item in archive_entries)

    cache_trace = f"""# V16 cache generation trace

Investigation run: `{run_id}`; UTC start `{started.isoformat()}`.

## What was recovered

- The V16 executed notebook is `{V16_NOTEBOOK}` (file SHA-256 `{sha256_file(V16_NOTEBOOK)}`; canonical code-cell SHA-256 `{v16_code_hash}`). Its setup explicitly writes model caches to Colab Drive `gci_cache_v16`; the saved notebook output identifies `/content/drive/MyDrive/gci_cache_v16`.
- The local `gci_cache_final` archive is `{V15_ARCHIVE}` (SHA-256 `{v15_archive_meta['sha256']}`, {v15_archive_meta['bytes']} bytes; archive entry times {archive_start.isoformat()} through {archive_end.isoformat()}). The contemporaneous research note identifies it as the V15 Google Drive export. It has {len(archive_entries)} members.
- Every archive member that exists in the extracted `D:\\gci\\codex\\v15_cache\\gci_cache_final` directory is inventoried in `V16_CACHE_PROVENANCE_MATRIX.csv`; the matrix records bytes, SHA-256, timestamps, member hash match, NPZ keys/shapes/dtypes, and absent row/fold/source/data sidecars. Candidate V15 NPZ files listed in the archive are byte-identical to their archived members. This verifies extraction integrity, **not** the original model training run.
- Candidate model arrays are in the V15 `gci_cache_final` archive, not in a located `gci_cache_v16` directory. A recursive local search found configured V16 cache directory present = `{v16_cache_exists}`. The V16 notebook says it creates its own cache folder and only intentionally reuses selected label-free V15 residual/view helpers. Thus the recovered V15 model arrays are the wrong generation for reproducing the V16 model sets.
- The V16 source implements 5-fold `StratifiedKFold(shuffle=True, random_state=42)`, fixed LightGBM rounds, five fixed member weights, and CatBoost folds. Its feature/label encoding code is present, but the missing V16 cache manifests do not record actual fold indices, input hashes, model versions, or feature schemas.

## Source candidates and provenance confidence

| Candidate | Evidence | Confidence |
|---|---|---|
| `{V15_NOTEBOOK}` | Source hash `{v15_code_hash}`; documented as the formula/cache lineage for V15; cache ZIP dated Oct 21 and exact extracted member hashes. | V15 lineage plausible; exact run/input binding absent. |
| `{V16_NOTEBOOK}` | Source hash `{v16_code_hash}`; executed outputs claim V16; configured cache root is `gci_cache_v16`. | Historical V16 source and saved output identified; original model caches absent. |
| `{V16_OPT_SOURCE}` | Hash `{sha256_file(V16_OPT_SOURCE)}`; a separate local optimized pipeline that fits different member code and searches blend weights on the same OOF labels. | Not the executed V16 notebook pipeline; cannot explain its reported fixed-weight metric without run evidence. |
| `{SCAFFOLD_SOURCES[0]}` | Hash `{sha256_file(SCAFFOLD_SOURCES[0])}`; contains code writing V6/V8/V16 bundles from a shared `v6_oof`. | Strong match to anomalous bundle structure; execution provenance not stored. |
| `{SCAFFOLD_SOURCES[1]}` | Hash `{sha256_file(SCAFFOLD_SOURCES[1])}`; contains the same shared-array placeholder and fixed AUC fields. | Same scaffold pattern; current file's filesystem time is later than bundle mtime. |

Filesystem timestamps are retained as supporting clues only; copying, extraction, and clock settings can change or preserve timestamps. The V16 cache provenance is **not fully recovered**.

## Why the candidate score is low

The `0.77404329` candidate fixed blend uses V15 archive members and the fixed V16 blend formula; it is in the V15 reference band (`0.77404`). The historical V16 output says it used new V16 set predictions, saved to `gci_cache_v16`, and reports `0.77649`. The candidate cache component scores do not match those outputs. The principal explanation is now identified: the previous comparison substituted V15 prediction caches for missing V16 model caches. Therefore `0.77404329` is not evidence that V16 scored lower; it is a V15-cache diagnostic and does not reproduce V16.

The archive and prediction arrays do not embed row IDs or fold IDs. Source code implies row-order use and seed-42 folds, but neither can be verified from these arrays. The CatBoost reconstruction additionally infers shard-to-fold assignment from shard order and expected validation lengths. See the discrepancy and provenance CSVs for exact metrics.
"""
    (run_dir / "V16_CACHE_GENERATION_TRACE.md").write_text(cache_trace, encoding="utf-8")

    discrepancy_report = f"""# V16 score discrepancy report

The saved V16 notebook outputs claim fixed-formula AUC `{SAVED_V16['fixed_final']:.5f}` and component blend AUCs old `{SAVED_V16['old']:.5f}`, E1 `{SAVED_V16['e1']:.5f}`, E1ET `{SAVED_V16['e1et']:.5f}`, E1AET `{SAVED_V16['e1Aet']:.5f}`. They remain saved-output claims, not independently reproduced results.

The recovered candidate model caches yield old `{roc_auc_score(y, candidate_arrays['old_oof']):.8f}`, E1 `{roc_auc_score(y, candidate_arrays['e1_oof']):.8f}`, E1ET `{roc_auc_score(y, candidate_arrays['e1et_oof']):.8f}`, and E1AET `{roc_auc_score(y, candidate_arrays['e1Aet_oof']):.8f}`; the reconstructed formula yields `{roc_auc_score(y, candidate_arrays['candidate_fixed_blend']):.8f}`. It differs from the saved final claim by `{roc_auc_score(y, candidate_arrays['candidate_fixed_blend']) - SAVED_V16['fixed_final']:+.8f}`.

## Explanation

The candidate predictions are verified as byte-identical members of the downloaded V15 cache archive. They are not verified V16 outputs. The V16 notebook configures a distinct `gci_cache_v16`; that directory was not found. The candidate final result is close to the known V15 reference because it reuses V15 sets. The lower score is explained primarily by **cache-generation mismatch**, not a demonstrated V16 regression.

Remaining uncertainty: we do not have the actual `gci_cache_v16` folder or a manifest that joins its arrays to the exact input hashes, row order, fold map, seeds, packages, and feature schema. Array length and AUC are not proof of alignment or OOF validity. `V16_SCORE_DISCREPANCY_REPORT.csv` contains the component-by-component values.
"""
    (run_dir / "V16_SCORE_DISCREPANCY_REPORT.md").write_text(discrepancy_report, encoding="utf-8")

    bundle = np.load(V16_BUNDLE, allow_pickle=False)
    v16_a = bundle["oof_e1Aet"]
    v16_b = bundle["oof_cb"]
    v16_final = bundle["final_oof"]
    stored_auc = float(bundle["final_auc"])
    calculated_auc = float(roc_auc_score(y, v16_final))
    scaffold_report = f"""# Anomalous bundle investigation

Target file: `{V16_BUNDLE}`.

## Confirmed measurements

- Bundle SHA-256 `{bundle_meta['sha256']}`; size {bundle_meta['bytes']} bytes; filesystem modified `{bundle_meta['modified_utc']}`.
- Keys: `{', '.join(bundle.files)}`.
- `oof_e1Aet`, `oof_cb`, and `final_oof` are element-for-element identical. Shared array SHA-256: `{sha256_bytes(v16_final.tobytes())}`.
- All three score `{calculated_auc:.10f}` against the current train target in current CSV row order; the stored scalar is `{stored_auc:.6f}`. The scalar is not computed from the stored arrays.
- V16 `final_oof` is byte-for-byte identical to `final_oof` in both sibling `v_6_blend_bundle.npz` and `v_8_blend_bundle.npz`. Their corresponding full-test prediction arrays are also identical. The V6 and V8 bundles store fixed scalars `.776` and `.7765`; the V16 bundle stores `.782`.
- None of these bundle arrays includes applicant IDs or fold assignments.

## Producer code explains the duplicate and invalid OOF

Both `{SCAFFOLD_SOURCES[0]}` (SHA-256 `{sha256_file(SCAFFOLD_SOURCES[0])}`) and `{SCAFFOLD_SOURCES[1]}` (SHA-256 `{sha256_file(SCAFFOLD_SOURCES[1])}`) contain the same precursor code:

1. Fit one LightGBM model on `X_train_df` with `y_train` for 150 boosting rounds.
2. Set `v6_oof = bst_v6.predict(X_train_df)`, predicting the same rows used to fit that model. This is **in-sample**, not out-of-fold.
3. Save this single `v6_oof` vector as `v_6.final_oof`, `v_8.final_oof`, and all three V16 OOF keys (`oof_e1Aet`, `oof_cb`, `final_oof`); save the same test vector for several different model fields.
4. Hard-code `final_auc=0.7760`, `0.7765`, and `0.7820` rather than calculate those fields from the predictions.

This code-level reuse pattern exactly matches the recovered sibling bundle array hashes and stored scalar pattern. The source code therefore explains the anomaly: the `{calculated_auc:.6f}` score is an apparent in-sample train score from a full-fit precursor model, not an OOF estimate; `{stored_auc:.3f}` is a hard-coded placeholder. The generated scaffold is the strongly supported producer/overwriter. No process log ties a particular script invocation to the file, so exact execution chronology remains unproven; the artifact itself is nevertheless invalid as OOF evidence.

The separate `{V16_OPT_SOURCE}` has a genuine-looking OOF-training loop in its source but optimizes ensemble weights on the same OOF labels used to report the selected AUC. It is not the executed V16 notebook source, and its intended outputs do not match the observed exact V6/V8 duplicate-array lineage. Do not conflate this separate optimized script with the scaffold that explains the bundle.

See `OOF_ARRAY_IDENTITY_AUDIT.csv` for each vector's hash, AUC, identity group, and pairwise equality.
"""
    (run_dir / "ANOMALOUS_BUNDLE_INVESTIGATION.md").write_text(scaffold_report, encoding="utf-8")

    exp85_oof_meta = file_metadata(EXP85_OOF)
    exp85_report = f"""# EXP85 champion OOF recovery update

## Status

**EXP85 champion OOF recovery: NOT RECOVERED.** The exact final Anchor74 + eight-engine champion OOF vector, row-to-fold assignments, and matching Anchor74 OOF are still unavailable.

## Evidence inspected

- Frozen submission `{EXP85_SUBMISSION}` and historical reference `{EXP85_REFERENCE}` are test-only artifacts, not OOF.
- `{EXP85_OOF}` is a finite `(171202, 8)` upstream component matrix (SHA-256 `{exp85_oof_meta['sha256']}`); it is not the final macro-blended prediction and has no row IDs or fold sidecar.
- `{ROOT / 'champion' / '01_EXP85_RECOVERY' / 'source' / 'reproduce_exact_synthesis.py'}` applies the final 85:15 Anchor74/suite synthesis to the test Anchor74 and component OOFs. It does not create an equivalent Anchor74 OOF; substituting the test Anchor74 for training OOF is invalid.
- The EXP02 runner is explicitly a raw-data surrogate and disclaims historical Anchor74 reproduction. It groups by applicant entities and uses `StratifiedGroupKFold`; it does not train a new model in this audit.

## Can the surrogate produce valid fold-safe OOF?

The raw five-family surrogate path assigns every train row to an outer validation fold, uses fixed iterations without early stopping, and recomputes graph label calibration inside each fold from outer-fit labels only. The source asserts fold coverage, fit/validation disjointness, group isolation, and finite predictions. These properties make the **raw base-family surrogate OOF** structurally fold-safe with respect to its outer labels, subject to static code review and reproducible execution.

However, its feature builder computes label-free ranks, group/frequency statistics, ridge expectation features, and entity structure over concatenated train+test covariates. The local materials do not resolve permission for pooled/transductive fitting. Its 85:15 blend also combines the grouped surrogate OOF with historical suite OOF caches whose exact fold sidecars are unavailable; that hybrid blend is **not demonstrated fold-safe** and is not an EXP85 reproduction. The cached suite components can include validation-training overlap relative to the different group folds.

The two-stream meta OOF code cross-fits logistic regression, but that cannot repair unknown fold provenance or possible upstream overlap in its cached suite input.
"""
    (run_dir / "EXP85_CHAMPION_RECOVERY_UPDATE.md").write_text(exp85_report, encoding="utf-8")

    integrity = f"""# Validation integrity audit

## Confirmed issues

1. **Anomalous V16 bundle is not OOF evidence.** The three OOF fields are exact copies of the V6 full-fit prediction vector. The matching scaffold source fits on all training labels and predicts the same training rows; it also hard-codes AUC scalars.
2. **Local optimized V16 source tunes and reports on the same OOF labels.** `train_v16_and_optimized.py` searches blend weight over its full OOF vector and selects the maximum AUC on those same labels. Its winning blend AUC is selection-biased and is not an untouched estimate.
3. **V16 caches used in the prior 0.77404329 diagnostic are V15 archive files.** They do not authenticate row IDs/folds, and are not the executed notebook's `gci_cache_v16` arrays.
4. **No exact final EXP85 OOF/fold map is present.** The test submission and eight-engine component matrix are not substitutes.

## Source-level V16 validation checks

- The executed V16 notebook source uses five-fold stratified seed 42, fixed boosting rounds, and no validation-driven early stopping.
- Its label-derived `enc(a,b)` / `train_enc(a)` functions fit encodings on only the model-fit rows and inner-cross-fit training encodings, respectively; the inspected source does not use the outer validation labels in those encodings.
- Multi-view validation predictions are generated by models trained only on outer-training partitions (with inner folds for training-row view features). The residual targets are feature targets, not `TARGET`; they are fitted on pooled unlabeled train+test covariates.
- This source-level structure is compatible with fold-local supervised validation, but no V16 OOF arrays/fold IDs/cache manifest were recovered, so execution fidelity cannot be verified.

## Competition-rule status based on supplied local rules

- No external dataset was identified in the inspected V16 sources; the notebook loads the supplied competition train/test/sample files. **Data-source rule: source appears compatible; actual cache lineage is not fully proven.**
- Feature engineering from supplied columns is encouraged/permitted.
- The local official audit says fitting unlabeled pooled train+test statistics is **unresolved**. V16 uses combined covariates for category codes, group/frequency/product/entity statistics and residual transforms. Do not certify submission compliance without written organizer clarification.
- Target-encoding permissibility is also **unresolved in the rule text**; the inspected V16 implementation is nested within the outer training fold, which addresses validation leakage but not an organizer eligibility ruling.
- The local audit flags test-to-test feature-driven score coupling as ambiguous/high-risk under the hand-labeling rule. V16/EXP85 relational partner methods must not be submitted without clarification.
- Thus, reconstructing historical V16 source may be technically possible, but using the full historical V16/EXP85 method in the competition is **not presently confirmed compliant**.

## Confirmed versus suspected

Confirmed: cache-generation mismatch; anomalous in-sample vector reuse pattern; hard-coded scalar scores in scaffold source; same-OOF blend-weight search in the separate optimizer; missing EXP85 champion OOF; absent row/fold IDs in relevant bundles.

Not established: exact execution that wrote/overwrote the anomalous file; whether every historical V16 cached model used only the current CSV hashes; exact array row alignment; whether the organizer permits pooled train/test transforms, target encodings, or test-to-test relational coupling.

No model was trained, no test label was accessed, and no submission or champion was modified.
"""
    (run_dir / "VALIDATION_INTEGRITY_AUDIT.md").write_text(integrity, encoding="utf-8")

    readiness_rows = [
        {
            "pipeline": "V16 executed notebook",
            "source": "Recovered; code hash available",
            "member_oof": "Saved outputs only; original gci_cache_v16 missing",
            "fold_ids": "source seed42 implied; no persisted map",
            "row_ids": "absent from arrays",
            "competition_rule_status": "pooled covariate/TE/test coupling unresolved",
            "common_fold_ready": "NO",
            "reason": "Original V16 predictions and provenance unavailable.",
        },
        {
            "pipeline": "V15 candidate cache archive",
            "source": "V15 notebook identified; archive confirms extracted bytes",
            "member_oof": "arrays present; candidate final diagnostic 0.77404329",
            "fold_ids": "inferred from source only",
            "row_ids": "absent",
            "competition_rule_status": "some shared transforms require same unresolved review",
            "common_fold_ready": "NO",
            "reason": "Not V16; input row alignment and fold sidecars not verified.",
        },
        {
            "pipeline": "anomalous V16 bundle",
            "source": "scaffold pattern identified; exact run invocation unproven",
            "member_oof": "three exact copies; apparent AUC 0.79467683 is in-sample lineage",
            "fold_ids": "none",
            "row_ids": "none",
            "competition_rule_status": "not eligible as OOF evidence",
            "common_fold_ready": "NO",
            "reason": "Shared vector produced by full-fit predict-on-train scaffold; scalar hard-coded.",
        },
        {
            "pipeline": "EXP85 final champion",
            "source": "test synthesis source recovered",
            "member_oof": "final OOF/Anchor74 OOF absent",
            "fold_ids": "absent",
            "row_ids": "test IDs only for submission",
            "competition_rule_status": "test coupling/transductive method unresolved",
            "common_fold_ready": "NO",
            "reason": "Cannot substitute the final test submission or upstream components.",
        },
        {
            "pipeline": "EXP02 Anchor74 raw surrogate",
            "source": "source and protocol recovered; no run outputs",
            "member_oof": "not generated in this task",
            "fold_ids": "grouped seed20261101 specified by source",
            "row_ids": "source validates raw row order",
            "competition_rule_status": "pooled unlabeled features unresolved; surrogate not historical",
            "common_fold_ready": "NO",
            "reason": "Not executed; hybrid suite blend lacks matching OOF fold provenance.",
        },
    ]
    write_csv(run_dir / "COMMON_FOLD_READINESS_UPDATE.csv", readiness_rows)

    recommendation = """# Next-phase recommendation

## Gate 1: Recover V16 artifacts without training

Request/export the exact `gci_cache_v16` folder from the Drive used by the executed notebook, including all LightGBM member NPZs, CatBoost fold NPZs, feature/encoding caches, and any notebook output files. Preserve archive metadata. For each file, bind SHA-256 to current train/test/sample hashes, `SK_ID_CURR` order, validation fold IDs, model/source hash, package versions, feature schema, and configuration. Do not reuse the V15 `gci_cache_final` NPZs as V16 model caches.

If that Drive cache cannot be recovered, classify historical V16 cache provenance as unrecoverable. A new V16 source rerun would be a reproduction attempt, not recovery of original predictions; do not start it until the rule gate below is resolved.

## Gate 2: Competition-rule clarification before any competition use

Obtain organizer guidance in writing on:

1. Unlabeled statistics fitted across train and test covariates.
2. Target encoding and cross-fitted supervised representations.
3. Applicant-to-applicant / test-to-test relational or score-coupling features.

The local official materials do not settle these methods. Until clarified, preserve the current files for forensic research but do not submit a pipeline relying on unresolved methods.

## Gate 3: EXP85

Do not use the 0.79467683 bundle. Recover the exact final Anchor74 OOF and its row/fold map, each suite member OOF and fold metadata, and the final blending code. The current Anchor74 surrogate is a distinct grouped-fold diagnostic and its hybrid suite blend is not an EXP85 comparator.

## Gate 4: Common-fold benchmark

Only after artifact provenance and rule scope pass, compare verified EXP85, reproduced V16, and Phase 2/3 models on a predeclared common validation protocol. Separate model selection from the final comparison labels. Keep failed runs and all previous scores append-only.

No feature expansion, parameter search, new training, or champion promotion is recommended before these gates.
"""
    (run_dir / "NEXT_PHASE_RECOMMENDATION.md").write_text(recommendation, encoding="utf-8")

    manifest_rows: list[dict[str, object]] = []
    source_paths = [
        TRAIN, TEST, V15_ARCHIVE, V15_NOTEBOOK, V16_NOTEBOOK, V16_SOURCE_NOTEBOOK,
        V16_TEXT, V16_OPT_SOURCE, V16_SUITE_SOURCE, V16_FUSION_SOURCE,
        *SCAFFOLD_SOURCES, V16_BUNDLE, V6_BUNDLE, V8_BUNDLE, EXP85_OOF,
        EXP85_SUBMISSION, EXP85_REFERENCE, ANCHOR_SOURCE, ANCHOR_RUNNER,
        ANCHOR_README,
    ]
    for path in source_paths:
        add_manifest_row(manifest_rows, path.name, path, "input/source/evidence")
    for path in sorted(V15_CACHE.iterdir()):
        if path.is_file():
            add_manifest_row(manifest_rows, f"v15-cache/{path.name}", path, "candidate cache")
    for path in sorted(V3_CACHE.iterdir()):
        if path.is_file():
            add_manifest_row(manifest_rows, f"v3-cache/{path.name}", path, "candidate cache")
    write_csv(run_dir / "SHA256_MANIFEST.csv", manifest_rows)

    script_copy = run_dir / "investigate_phase5_task3.py"
    script_copy.write_bytes(Path(__file__).read_bytes())
    output_names = [
        "V16_CACHE_GENERATION_TRACE.md",
        "V16_CACHE_PROVENANCE_MATRIX.csv",
        "V16_SCORE_DISCREPANCY_REPORT.md",
        "V16_SCORE_DISCREPANCY_REPORT.csv",
        "ANOMALOUS_BUNDLE_INVESTIGATION.md",
        "OOF_ARRAY_IDENTITY_AUDIT.csv",
        "EXP85_CHAMPION_RECOVERY_UPDATE.md",
        "VALIDATION_INTEGRITY_AUDIT.md",
        "COMMON_FOLD_READINESS_UPDATE.csv",
        "SHA256_MANIFEST.csv",
        "NEXT_PHASE_RECOMMENDATION.md",
        "investigate_phase5_task3.py",
    ]
    ended = datetime.now(timezone.utc)
    log = [
        "STATUS: COMPLETED",
        f"UTC start: {started.isoformat()}",
        f"UTC end: {ended.isoformat()}",
        f"Command: {command}",
        f"Train shape: {train.shape}; SHA-256: {input_hashes['train']}",
        f"Test shape: {test.shape}; SHA-256: {input_hashes['test']}",
        f"V15 archive: {V15_ARCHIVE}; members={len(members)}; SHA-256={v15_archive_meta['sha256']}",
        f"V16 cache folder located: {v16_cache_exists}",
        f"Candidate fixed blend diagnostic: {roc_auc_score(y, candidate_arrays['candidate_fixed_blend']):.8f} (V15 archive only; not V16 reproduction)",
        f"Anomalous bundle recomputed AUC: {calculated_auc:.10f}; stored scalar: {stored_auc:.6f}",
        "Training: NOT RUN.",
        "Test TARGET: NOT read.",
        "Submission/champion changes: none.",
        "Errors: none.",
        "Warnings: see reports; unresolved data/cache provenance and competition-rule scope remain.",
        "Outputs:",
    ]
    log.extend(str(run_dir / name) for name in output_names)
    (run_dir / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"OUTPUT_DIRECTORY: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
