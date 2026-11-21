from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth\experiments\exp02_financial_behaviour")
RUN = ROOT / "runs" / "financial_ratios_seed42"
DATA = Path(r"D:\gci\inpuy")
OUT = Path(r"D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery")
NOTEBOOK = ROOT / "EXP02_FINANCIAL_BEHAVIOUR.ipynb"
MANIFEST_PATH = ROOT / "SHA256_MANIFEST.json"
RESULTS_PATH = ROOT / "RESULTS.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        {
            "cell_type": cell.get("cell_type"),
            "source": "".join(cell.get("source", [])),
        }
        for cell in document.get("cells", [])
    ]
    payload = json.dumps(
        cells, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    input_hashes = manifest["raw_input_sha256"]
    for filename, expected in input_hashes.items():
        actual = sha256_file(DATA / filename)
        if actual != expected:
            raise ValueError(f"Input hash mismatch: {filename}")

    if source_sha256(NOTEBOOK) != manifest["notebook_source_sha256"]:
        raise ValueError("Canonical notebook source hash mismatch.")
    for filename, expected in manifest["root_artifact_sha256"].items():
        if sha256_file(ROOT / filename) != expected:
            raise ValueError(f"Root artifact hash mismatch: {filename}")
    for filename, expected in manifest["run_artifact_sha256"].items():
        if sha256_file(RUN / filename) != expected:
            raise ValueError(f"Run artifact hash mismatch: {filename}")

    labels = pd.read_csv(DATA / "train.csv", usecols=["TARGET"])["TARGET"].to_numpy()
    test_ids = pd.read_csv(DATA / "test.csv", usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()
    sample_ids = pd.read_csv(DATA / "sample_submission.csv", usecols=["SK_ID_CURR"])[
        "SK_ID_CURR"
    ].to_numpy()

    fold_ids = np.load(RUN / "fold_ids.npy", allow_pickle=False)
    raw_oof = np.load(RUN / "oof_raw_baseline.npy", allow_pickle=False)
    feature_oof = np.load(RUN / "oof_financial_features.npy", allow_pickle=False)
    raw_test = np.load(RUN / "test_raw_baseline.npy", allow_pickle=False)
    feature_test = np.load(RUN / "test_financial_features.npy", allow_pickle=False)
    saved_test_ids = np.load(RUN / "test_ids.npy", allow_pickle=False)

    if not (len(labels) == len(fold_ids) == len(raw_oof) == len(feature_oof)):
        raise ValueError("Training labels, folds, and OOF arrays are misaligned.")
    if not np.array_equal(test_ids, sample_ids) or not np.array_equal(
        test_ids, saved_test_ids
    ):
        raise ValueError("Test, sample submission, and saved test IDs are misaligned.")
    if set(np.unique(fold_ids)) != {1, 2, 3, 4, 5}:
        raise ValueError("Expected five non-empty outer folds numbered 1 through 5.")

    for name, prediction, expected_length in (
        ("raw OOF", raw_oof, len(labels)),
        ("financial OOF", feature_oof, len(labels)),
        ("raw test", raw_test, len(test_ids)),
        ("financial test", feature_test, len(test_ids)),
    ):
        if len(prediction) != expected_length:
            raise ValueError(f"Wrong row count for {name}.")
        if (
            not np.isfinite(prediction).all()
            or (prediction < 0).any()
            or (prediction > 1).any()
        ):
            raise ValueError(f"Invalid probability values in {name}.")

    fold_metrics = []
    for fold in range(1, 6):
        mask = fold_ids == fold
        raw_auc = float(roc_auc_score(labels[mask], raw_oof[mask]))
        feature_auc = float(roc_auc_score(labels[mask], feature_oof[mask]))
        saved = results["fold_metrics"][fold - 1]
        if not np.isclose(raw_auc, saved["raw_auc"], atol=1e-12):
            raise ValueError(f"Saved raw AUC mismatch in fold {fold}.")
        if not np.isclose(feature_auc, saved["financial_auc"], atol=1e-12):
            raise ValueError(f"Saved feature AUC mismatch in fold {fold}.")
        fold_metrics.append(
            {
                "fold": fold,
                "raw_auc": raw_auc,
                "financial_auc": feature_auc,
                "delta": feature_auc - raw_auc,
            }
        )

    pooled_raw = float(roc_auc_score(labels, raw_oof))
    pooled_features = float(roc_auc_score(labels, feature_oof))
    saved_pooled = results["pooled_oof_auc"]
    if not np.isclose(pooled_raw, saved_pooled["raw_baseline"], atol=1e-12):
        raise ValueError("Saved pooled raw AUC mismatch.")
    if not np.isclose(
        pooled_features, saved_pooled["financial_features"], atol=1e-12
    ):
        raise ValueError("Saved pooled feature AUC mismatch.")

    mean_delta = float(np.mean([row["delta"] for row in fold_metrics]))
    log_path = OUT / "EXP02_ARTIFACT_RECHECK_LOG.txt"
    lines = [
        f"timestamp_utc={timestamp}",
        f"command=python {Path(__file__)}",
        "status=PASS",
        "training=NOT_RUN; metrics_recomputed_from_existing_OOF_arrays=YES",
        f"notebook_source_sha256={manifest['notebook_source_sha256']}",
        f"verifier_sha256={sha256_file(Path(__file__))}",
        "input_and_artifact_hashes=PASS",
        "test_id_order_and_prediction_bounds=PASS",
        f"pooled_raw_auc={pooled_raw:.12f}",
        f"pooled_financial_auc={pooled_features:.12f}",
        f"pooled_delta={pooled_features - pooled_raw:+.12f}",
        f"mean_paired_fold_delta={mean_delta:+.12f}",
        "fold_deltas="
        + ",".join(f"{row['delta']:+.12f}" for row in fold_metrics),
        f"saved_decision={results['decision']}",
        f"log_path={log_path}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
