#!/usr/bin/env python3
"""Independently verify one completed corrected Phase 2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "runs" / "phase2_inner_stop_seed42"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    fold_ids_path = run_dir / "fold_ids.npy"
    for path in (train_path, test_path, sample_path, manifest_path, metrics_path, fold_ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required verification input missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    train = pd.read_csv(train_path, usecols=["SK_ID_CURR", "TARGET"])
    test_ids = pd.read_csv(test_path, usecols=["SK_ID_CURR"])
    sample = pd.read_csv(sample_path, usecols=["SK_ID_CURR", "TARGET"])
    if sample["SK_ID_CURR"].duplicated().any() or test_ids["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicate test/sample IDs")
    if not np.array_equal(sample["SK_ID_CURR"], test_ids["SK_ID_CURR"]):
        raise ValueError("Sample and test IDs/order differ")
    if set(train["SK_ID_CURR"]).intersection(test_ids["SK_ID_CURR"]):
        raise ValueError("Training and test IDs overlap")

    labels = train["TARGET"].to_numpy(dtype=np.int8)
    if train["SK_ID_CURR"].isna().any() or test_ids["SK_ID_CURR"].isna().any():
        raise ValueError("Missing train/test IDs")
    if train["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicate training IDs")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Training target is not binary")
    if np.isnan(labels).any():
        raise ValueError("Missing training labels")
    expected_fold_ids = np.full(len(labels), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_number, (_, valid_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels), start=1
    ):
        expected_fold_ids[valid_indices] = fold_number
    fold_ids = np.load(fold_ids_path, allow_pickle=False)
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Saved fold IDs do not match the specified seed-42 split")

    raw_inputs = {
        "train": train_path,
        "test": test_path,
        "sample": sample_path,
    }
    input_hashes = {name: sha256_file(path) for name, path in raw_inputs.items()}
    if input_hashes != manifest["raw_input_sha256"]:
        raise ValueError("Raw input hashes do not match the training manifest")
    if manifest["external_datasets_or_prediction_caches_used"] is not False:
        raise ValueError("Run manifest does not confirm raw-data-only provenance")
    source_paths = {
        "baseline": EXPERIMENT_DIR.parent.parent
        / "research_archive"
        / "exp85_phase2_source_snapshot"
        / "project"
        / "source"
        / "run_independent_baseline.py",
        "phase1": EXPERIMENT_DIR.parent.parent
        / "research_archive"
        / "exp85_phase2_source_snapshot"
        / "project"
        / "experiments"
        / "phase_01_fee_product_features"
        / "run_experiment.py",
        "phase2": EXPERIMENT_DIR.parent.parent
        / "research_archive"
        / "exp85_phase2_source_snapshot"
        / "project"
        / "experiments"
        / "phase_02_fold_safe_key_encodings"
        / "run_experiment.py",
        "runner": EXPERIMENT_DIR / "run_experiment_seed42_source.py",
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    if source_hashes != manifest["source_sha256"]:
        raise ValueError("Source hashes do not match the training manifest")

    variants = ("raw_baseline", "phase2")
    models = ("lightgbm", "catboost", "blend")
    verified_variants: dict[str, Any] = {}
    for variant in variants:
        oof = {
            model: np.load(
                run_dir / f"oof_{variant}_{model}.npy", allow_pickle=False
            )
            for model in models
        }
        test = {
            model: np.load(
                run_dir / f"test_{variant}_{model}.npy", allow_pickle=False
            )
            for model in models
        }
        for model, prediction in oof.items():
            if (
                prediction.shape != (len(labels),)
                or not np.isfinite(prediction).all()
                or (prediction < 0).any()
                or (prediction > 1).any()
            ):
                raise ValueError(f"Invalid OOF prediction array: {variant}/{model}")
        for model, prediction in test.items():
            if (
                prediction.shape != (len(test_ids),)
                or not np.isfinite(prediction).all()
                or (prediction < 0).any()
                or (prediction > 1).any()
            ):
                raise ValueError(f"Invalid test prediction array: {variant}/{model}")

        expected_oof_blend = (
            oof["lightgbm"].astype(np.float64)
            + oof["catboost"].astype(np.float64)
        ) / 2.0
        expected_test_blend = (
            test["lightgbm"].astype(np.float64)
            + test["catboost"].astype(np.float64)
        ) / 2.0
        if not np.array_equal(oof["blend"], expected_oof_blend.astype(np.float32)):
            raise ValueError(f"OOF blend mismatch for {variant}")
        if not np.array_equal(
            test["blend"], expected_test_blend.astype(np.float32)
        ):
            raise ValueError(f"Test blend mismatch for {variant}")

        pooled_auc = {
            model: float(roc_auc_score(labels, prediction))
            for model, prediction in oof.items()
        }
        saved = run_metrics["variants"][variant]
        for model in ("lightgbm", "catboost"):
            if abs(pooled_auc[model] - saved[f"{model}_oof_auc"]) > 1e-12:
                raise ValueError(f"Saved {variant}/{model} AUC mismatch")
        if abs(
            pooled_auc["blend"] - saved["fixed_50_50_blend_oof_auc"]
        ) > 1e-8:
            raise ValueError(f"Saved {variant} blend AUC mismatch")
        fold_auc = {
            model: [
                float(roc_auc_score(labels[fold_ids == fold], prediction[fold_ids == fold]))
                for fold in range(1, 6)
            ]
            for model, prediction in oof.items()
        }
        verified_variants[variant] = {
            "pooled_oof_auc": pooled_auc,
            "fold_oof_auc": fold_auc,
        }

    fold_records = run_metrics["folds"]
    if len(fold_records) != 5:
        raise ValueError("Expected five fold records")
    for fold_number, metric_record in enumerate(fold_records, start=1):
        record_path = run_dir / f"fold_{fold_number}_record.json"
        if not record_path.is_file():
            raise FileNotFoundError(f"Required fold record missing: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record != metric_record or record["fold"] != fold_number:
            raise ValueError(f"Fold {fold_number} record does not match metrics")
        if record["outer_validation_used_for_early_stopping"] is not False:
            raise ValueError("Outer validation fold was used for early stopping")
        if any(
            model_record["selected_iterations"] <= 0
            for model_record in record["models"].values()
        ):
            raise ValueError(f"Invalid selected iteration count in fold {fold_number}")
    if any(
        record["outer_validation_used_for_early_stopping"] is not False
        for record in fold_records
    ):
        raise ValueError("Outer validation fold was used for early stopping")
    if any((run_dir / name).is_file() for name in (
        "submission.csv",
        "submission_candidate.csv",
    )) or list(run_dir.glob("*.csv")):
        raise ValueError("Unexpected CSV/submission in run directory")

    blend_deltas = (
        np.asarray(verified_variants["phase2"]["fold_oof_auc"]["blend"])
        - np.asarray(verified_variants["raw_baseline"]["fold_oof_auc"]["blend"])
    )
    run_output_hashes = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    return {
        "experiment_id": run_dir.name,
        "raw_train_rows": len(labels),
        "raw_test_rows": len(test_ids),
        "raw_predictor_columns": len(pd.read_csv(train_path, nrows=0).columns) - 2,
        "outer_fold_seed": 42,
        "outer_folds": 5,
        "fold_assignments_match_seed42": True,
        "raw_input_sha256": {
            f"{name}.csv" if name != "sample" else "sample_submission.csv": digest
            for name, digest in input_hashes.items()
        },
        "executed_runner_sha256": source_hashes["runner"],
        "official_score": "NOT_EVALUATED",
        "submission_created": False,
        "variants": verified_variants,
        "paired_phase2_minus_baseline": {
            "pooled_blend_auc_delta": (
                verified_variants["phase2"]["pooled_oof_auc"]["blend"]
                - verified_variants["raw_baseline"]["pooled_oof_auc"]["blend"]
            ),
            "fold_blend_auc_deltas": blend_deltas.tolist(),
            "positive_blend_folds": int((blend_deltas > 0).sum()),
        },
        "checks": {
            "train_test_ids_and_sample_order_valid": True,
            "fold_ids_independently_recomputed": True,
            "all_oof_test_arrays_finite_in_unit_interval": True,
            "saved_blends_recomputed_from_component_arrays": True,
            "input_hashes_recomputed": True,
            "source_hashes_recomputed": True,
            "outer_validation_used_for_early_stopping": False,
            "external_data_or_prediction_cache_used": False,
            "csv_submission_present": False,
        },
        "run_output_sha256": run_output_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=EXPERIMENT_DIR / "VERIFIED_METRICS.json")
    args = parser.parse_args()
    result = verify(args.data_dir, args.run_dir)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
