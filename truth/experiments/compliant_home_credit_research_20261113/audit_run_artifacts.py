from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
ALLOWED_INPUTS = {"train.csv", "test.csv", "sample_submission.csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def audit_run(run_dir: Path, data_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.json"
    oof_path = run_dir / "oof_predictions.csv"
    if not metrics_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError(f"Run lacks metrics.json or oof_predictions.csv: {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_hashes = metrics.get("input_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != ALLOWED_INPUTS:
        raise AssertionError("Run must record hashes for exactly the three admitted inputs.")
    actual_hashes = {}
    for name in sorted(ALLOWED_INPUTS):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required organizer input is missing: {path}")
        actual_hashes[name] = sha256_file(path)
        if actual_hashes[name] != expected_hashes[name]:
            raise AssertionError(f"Input hash mismatch: {name}")

    train = pd.read_csv(data_dir / "train.csv", usecols=[ID_COLUMN, TARGET_COLUMN])
    test_ids = pd.read_csv(data_dir / "test.csv", usecols=[ID_COLUMN])[ID_COLUMN]
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise AssertionError("Sample submission schema differs from the official format.")
    if not np.array_equal(sample[ID_COLUMN].to_numpy(), test_ids.to_numpy()):
        raise AssertionError("Sample IDs/order differ from the organizer test file.")
    oof = pd.read_csv(oof_path)
    if not oof[ID_COLUMN].is_unique:
        raise AssertionError("OOF identifiers are not unique.")
    if not np.array_equal(oof[ID_COLUMN].to_numpy(), train[ID_COLUMN].to_numpy()):
        raise AssertionError("OOF IDs/order differ from the organizer training file.")
    prediction_columns = [column for column in oof if column.startswith("OOF_")]
    if not prediction_columns:
        raise AssertionError("No row-level OOF prediction columns were found.")
    predictions: dict[str, np.ndarray] = {}
    for column in prediction_columns:
        values = oof[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise AssertionError(f"OOF predictions contain non-finite values: {column}")
        predictions[column.removeprefix("OOF_")] = values

    seed = int(metrics["seed"])
    folds = int(metrics["folds"])
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    split_indices = list(
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )
    model_summaries = metrics.get("model_summary")
    if not isinstance(model_summaries, dict) or set(model_summaries) != set(predictions):
        raise AssertionError("Model metrics do not match the saved OOF columns.")
    checked_models = {}
    for name, values in predictions.items():
        fold_auc = [
            float(roc_auc_score(y[valid_idx], values[valid_idx]))
            for _, valid_idx in split_indices
        ]
        summary = model_summaries[name]
        pooled_auc = float(roc_auc_score(y, values))
        if not np.isclose(pooled_auc, summary["pooled_oof_auc"], atol=1e-12):
            raise AssertionError(f"Pooled OOF AUC does not reproduce for {name}.")
        if not np.allclose(fold_auc, summary["fold_auc"], atol=1e-12):
            raise AssertionError(f"Fold AUCs do not reproduce for {name}.")
        checked_models[name] = {
            "pooled_oof_auc": pooled_auc,
            "mean_fold_auc": float(np.mean(fold_auc)),
            "fold_auc": fold_auc,
        }

    recorded_oof_hash = metrics.get("oof_sha256")
    if recorded_oof_hash != sha256_file(oof_path):
        raise AssertionError("Saved OOF artifact hash differs from metrics.json.")

    source_hashes = metrics.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise AssertionError("Run has no source-code hashes.")
    source_root = run_dir.parent.parent
    for filename, expected_hash in source_hashes.items():
        source_path = source_root / filename
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise AssertionError(f"Source hash missing or changed: {filename}")

    listed_submissions = set(metrics.get("submission_files", []))
    found_submissions = {path.name for path in run_dir.glob("submission_*.csv")}
    if listed_submissions != found_submissions:
        raise AssertionError("Submission files do not match the recorded run manifest.")
    submission_hashes = {}
    for filename in sorted(listed_submissions):
        submission_path = run_dir / filename
        submission = pd.read_csv(submission_path)
        if list(submission.columns) != [ID_COLUMN, TARGET_COLUMN]:
            raise AssertionError(f"Submission schema mismatch: {filename}")
        if not submission[ID_COLUMN].is_unique or not sample[ID_COLUMN].is_unique:
            raise AssertionError(f"Submission/sample IDs are not unique: {filename}")
        if not np.array_equal(submission[ID_COLUMN].to_numpy(), test_ids.to_numpy()):
            raise AssertionError(f"Submission IDs/order mismatch: {filename}")
        target = submission[TARGET_COLUMN].to_numpy(dtype=np.float64)
        if not np.isfinite(target).all() or np.any((target < 0.0) | (target > 1.0)):
            raise AssertionError(f"Submission predictions are invalid: {filename}")
        submission_hashes[filename] = sha256_file(submission_path)

    return {
        "run_dir": str(run_dir.resolve()),
        "candidate_id": metrics["candidate_id"],
        "input_sha256": actual_hashes,
        "oof_sha256": recorded_oof_hash,
        "source_sha256": source_hashes,
        "n_train": len(train),
        "folds": folds,
        "checked_models": checked_models,
        "submission_sha256": submission_hashes,
        "result": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    results = [audit_run(run_dir, args.data_dir) for run_dir in args.run_dirs]
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
