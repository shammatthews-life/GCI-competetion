from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
PHASE13_PATH = (
    Path(r"D:\gci\truth\experiments\phase13_matched_baseline")
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)
RUN_NAMES = (
    "phase15_native_catboost_preflight_20261105T123915Z",
    "phase15_native_catboost_preflight_20261105T124003Z",
    "phase15_native_catboost_oof_20261105T124238Z",
    "phase15_native_catboost_multiseed_20261105T125826Z",
    "phase15_xgboost_category_preflight_20261105T125653Z",
    "phase15_ctr2_preflight_20261105T130340Z",
    "phase15_org_ext_cross_preflight_20261105T130627Z",
    "phase15_org_ext_cross_oof_20261105T130802Z",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_manifest(run_dir: Path, registry: list[dict], corrections: list[dict]) -> None:
    manifest_path = run_dir / "SHA256_MANIFEST.csv"
    if not manifest_path.is_file():
        registry.append(
            {
                "run": run_dir.name,
                "kind": "status",
                "file": "SHA256_MANIFEST.csv",
                "bytes": 0,
                "declared_sha256": "",
                "actual_sha256": "",
                "status": "incomplete_run_no_manifest",
            }
        )
        return
    entries = pd.read_csv(manifest_path)
    for row in entries.itertuples(index=False):
        path = run_dir / row.file if row.kind == "output" else Path(row.file)
        if not path.is_file():
            status, actual = "missing", ""
        else:
            actual = sha256_file(path)
            status = "pass" if actual == row.sha256 else "hash_mismatch"
        registry.append(
            {
                "run": run_dir.name,
                "kind": row.kind,
                "file": row.file,
                "bytes": row.bytes,
                "declared_sha256": row.sha256,
                "actual_sha256": actual,
                "status": status,
            }
        )
        if status != "pass":
            corrections.append(
                {
                    "run": run_dir.name,
                    "file": row.file,
                    "declared_sha256": row.sha256,
                    "actual_sha256": actual,
                    "bytes": path.stat().st_size if path.is_file() else 0,
                    "reason": (
                        "The run logger wrote its final completion line after "
                        "the original SHA256_MANIFEST.csv was generated. The "
                        "original manifest is preserved; this supplemental "
                        "entry provides the actual final-file hash."
                        if row.file == "EXECUTION_LOG.txt" and status == "hash_mismatch"
                        else "Mismatch found during independent verification."
                    ),
                }
            )


def _auc_row(
    results: list[dict],
    run: str,
    variant: str,
    y: np.ndarray,
    pred: np.ndarray,
) -> float:
    auc = float(roc_auc_score(y, pred))
    results.append(
        {
            "run": run,
            "variant": variant,
            "rows": len(y),
            "recomputed_pooled_auc": auc,
            "status": "pass" if np.isfinite(pred).all() else "nonfinite_predictions",
        }
    )
    return auc


def main() -> None:
    train = pd.read_csv(TRAIN_PATH, usecols=["SK_ID_CURR", "TARGET"])
    ids = train["SK_ID_CURR"].to_numpy()
    y = train["TARGET"].to_numpy(dtype=np.int8)
    phase13 = pd.read_csv(PHASE13_PATH)
    if not (
        len(phase13) == len(train)
        and np.array_equal(phase13["SK_ID_CURR"], ids)
        and np.array_equal(phase13["TARGET"], y)
    ):
        raise ValueError("Phase 13 reference does not align with train")

    registry: list[dict] = []
    corrections: list[dict] = []
    oof_results: list[dict] = []
    for name in RUN_NAMES:
        run_dir = RUNS / name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Expected Phase 15 run is missing: {run_dir}")
        _record_manifest(run_dir, registry, corrections)
        if name.endswith("123915Z"):
            continue
        if "native_catboost_preflight" in name:
            frame = pd.read_csv(run_dir / "PREFLIGHT_OOF.csv")
            indexes = frame["row_index"].to_numpy(dtype=np.int64)
            if not (
                np.array_equal(frame["SK_ID_CURR"], ids[indexes])
                and np.array_equal(frame["TARGET"], y[indexes])
            ):
                raise ValueError(f"{name} fold row alignment failed")
            for column in (
                "native_catboost_base_all8",
                "native_catboost_phase14_numeric",
                "native_catboost_phase14_plus_time",
            ):
                _auc_row(
                    oof_results,
                    name,
                    column,
                    frame["TARGET"].to_numpy(dtype=np.int8),
                    frame[column].to_numpy(dtype=np.float64),
                )
        elif "native_catboost_oof" in name:
            frame = pd.read_csv(run_dir / "OOF_PREDICTIONS.csv")
            for column in (
                "row_index",
                "SK_ID_CURR",
                "TARGET",
                "fold",
                "native_catboost_base_all8",
                "native_catboost_phase14_numeric",
                "native_catboost_phase14_plus_time",
            ):
                if column not in frame:
                    raise ValueError(f"{name} OOF is missing column {column}")
            if not (
                np.array_equal(frame["row_index"], np.arange(len(y)))
                and np.array_equal(frame["SK_ID_CURR"], ids)
                and np.array_equal(frame["TARGET"], y)
                and frame["fold"].notna().all()
            ):
                raise ValueError(f"{name} row/ID/target integrity failed")
            saved = pd.read_csv(run_dir / "MODEL_RESULTS.csv")
            for row in saved.itertuples(index=False):
                pred = frame[row.variant].to_numpy(dtype=np.float64)
                auc = _auc_row(oof_results, name, row.variant, y, pred)
                if abs(auc - row.pooled_oof_auc) > 1e-12:
                    raise ValueError(f"Independent AUC mismatch: {name}/{row.variant}")
        elif "multiseed" in name:
            frame = pd.read_csv(run_dir / "MULTISEED_OOF_PREDICTIONS.csv")
            if not (
                len(frame) == len(y)
                and np.array_equal(frame["row_index"], np.arange(len(y)))
                and np.array_equal(frame["SK_ID_CURR"], ids)
                and np.array_equal(frame["TARGET"], y)
            ):
                raise ValueError(f"{name} row/ID/target integrity failed")
            saved = pd.read_csv(run_dir / "MULTISEED_RESULTS.csv")
            for column in (
                "phase15_primary_seed_20261113",
                "phase15_secondary_seed_20261114",
                "phase15_secondary_seed_20261115",
                "phase15_three_seed_average",
            ):
                auc = _auc_row(
                    oof_results,
                    name,
                    column,
                    y,
                    frame[column].to_numpy(dtype=np.float64),
                )
                saved_name = (
                    "three_seed_oof_average"
                    if column == "phase15_three_seed_average"
                    else int(column.rsplit("_", 1)[1])
                )
                saved_row = saved[saved["seed"].astype(str) == str(saved_name)]
                if len(saved_row) != 1 or abs(
                    auc - float(saved_row.iloc[0]["pooled_oof_auc"])
                ) > 1e-12:
                    raise ValueError(f"Independent multiseed AUC mismatch: {column}")
        elif "org_ext_cross_oof" in name:
            frame = pd.read_csv(run_dir / "OOF_PREDICTIONS.csv")
            if not (
                len(frame) == len(y)
                and np.array_equal(frame["row_index"], np.arange(len(y)))
                and np.array_equal(frame["SK_ID_CURR"], ids)
                and np.array_equal(frame["TARGET"], y)
            ):
                raise ValueError(f"{name} row/ID/target integrity failed")
            for column in (
                "catboost_ctr1_time_base",
                "catboost_ctr1_time_plus_org_external_cross",
            ):
                _auc_row(
                    oof_results,
                    name,
                    column,
                    y,
                    frame[column].to_numpy(dtype=np.float64),
                )
        elif "xgboost_category" in name:
            frame = pd.read_csv(run_dir / "PREFLIGHT_PREDICTIONS.csv")
            indexes = frame["sample_row_index"].to_numpy(dtype=np.int64)
            if not (
                np.array_equal(frame["SK_ID_CURR"], ids[indexes])
                and np.array_equal(frame["TARGET"], y[indexes])
            ):
                raise ValueError(f"{name} sample row alignment failed")
            for column in (
                "xgboost_native_categories_no_product_keys",
                "xgboost_native_categories_four_constant_free_keys",
            ):
                _auc_row(
                    oof_results,
                    name,
                    column,
                    frame["TARGET"].to_numpy(dtype=np.int8),
                    frame[column].to_numpy(dtype=np.float64),
                )
        elif "ctr2_preflight" in name:
            frame = pd.read_csv(run_dir / "PREFLIGHT_OOF.csv")
            indexes = frame["row_index"].to_numpy(dtype=np.int64)
            if not (
                np.array_equal(frame["SK_ID_CURR"], ids[indexes])
                and np.array_equal(frame["TARGET"], y[indexes])
            ):
                raise ValueError(f"{name} fold row alignment failed")
            for column in ("ctr_complexity_1", "ctr_complexity_2"):
                _auc_row(
                    oof_results,
                    name,
                    column,
                    frame["TARGET"].to_numpy(dtype=np.int8),
                    frame[column].to_numpy(dtype=np.float64),
                )
        elif "org_ext_cross_preflight" in name:
            frame = pd.read_csv(run_dir / "PREFLIGHT_OOF.csv")
            indexes = frame["row_index"].to_numpy(dtype=np.int64)
            if not (
                np.array_equal(frame["SK_ID_CURR"], ids[indexes])
                and np.array_equal(frame["TARGET"], y[indexes])
            ):
                raise ValueError(f"{name} fold row alignment failed")
            for column in (
                "catboost_ctr1_time_base",
                "catboost_ctr1_time_plus_org_external_cross",
                "stored_primary",
            ):
                _auc_row(
                    oof_results,
                    name,
                    column,
                    frame["TARGET"].to_numpy(dtype=np.int8),
                    frame[column].to_numpy(dtype=np.float64),
                )

    registry_path = ROOT / "PHASE15_ARTIFACT_REGISTRY.csv"
    pd.DataFrame(registry).to_csv(registry_path, index=False)
    corrections_path = ROOT / "PHASE15_MANIFEST_CORRECTIONS.csv"
    pd.DataFrame(
        corrections,
        columns=(
            "run",
            "file",
            "declared_sha256",
            "actual_sha256",
            "bytes",
            "reason",
        ),
    ).to_csv(corrections_path, index=False)
    results_path = ROOT / "PHASE15_INDEPENDENT_AUC_VERIFICATION.csv"
    pd.DataFrame(oof_results).to_csv(results_path, index=False)
    report = {
        "status": "independent file/hash/OOF audit completed",
        "training_rows": len(train),
        "training_ids_unique": int(pd.Series(ids).nunique()),
        "training_sha256": sha256_file(TRAIN_PATH),
        "phase13_reference_sha256": sha256_file(PHASE13_PATH),
        "registry_entries": len(registry),
        "registry_pass_count": sum(row["status"] == "pass" for row in registry),
        "registry_mismatch_count": sum(
            row["status"] == "hash_mismatch" for row in registry
        ),
        "incomplete_runs_without_manifests": sum(
            row["status"] == "incomplete_run_no_manifest" for row in registry
        ),
        "manifest_corrections": corrections,
        "independently_recomputed_auc_rows": oof_results,
        "output_artifact_sha256": {
            registry_path.name: sha256_file(registry_path),
            corrections_path.name: sha256_file(corrections_path),
            results_path.name: sha256_file(results_path),
        },
    }
    (ROOT / "PHASE15_VERIFICATION_SUMMARY.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
