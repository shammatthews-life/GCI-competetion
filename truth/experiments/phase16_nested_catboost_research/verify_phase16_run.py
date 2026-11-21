from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE13 = (
    REPO
    / "experiments"
    / "phase13_matched_baseline"
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path) -> dict[str, object]:
    required = (
        run_dir / "SUMMARY.json",
        run_dir / "OOF_PREDICTIONS.csv",
        run_dir / "FOLD_METRICS.csv",
        run_dir / "INNER_CONFIG_SELECTION.csv",
        run_dir / "SHA256_MANIFEST.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Phase 16 outputs are missing: {missing}")

    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    reference = pd.read_csv(PHASE13)
    oof = pd.read_csv(run_dir / "OOF_PREDICTIONS.csv")
    folds = np.full(len(train), -1, dtype=np.int8)
    y = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    splitter = StratifiedKFold(
        n_splits=int(summary["outer_folds"]),
        shuffle=True,
        random_state=int(summary["outer_seed"]),
    )
    for fold, (_, apply) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        folds[apply] = fold

    if not (
        len(oof) == len(train) == len(reference)
        and np.array_equal(oof["row_index"].to_numpy(), np.arange(len(train)))
        and np.array_equal(oof["SK_ID_CURR"].to_numpy(), ids)
        and np.array_equal(oof["TARGET"].to_numpy(), y)
        and np.array_equal(oof["fold"].to_numpy(), folds)
        and np.array_equal(reference["primary_fold"].to_numpy(), folds)
        and np.array_equal(reference["TARGET"].to_numpy(), y)
        and np.array_equal(reference["SK_ID_CURR"].to_numpy(), ids)
    ):
        raise ValueError("Phase 16 OOF rows, IDs, targets, or folds do not align")

    prediction = oof["phase16_nested_tuned_catboost"].to_numpy(dtype=np.float64)
    baseline = oof["phase15_matched_seed_baseline"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
        raise ValueError("Phase 16 OOF predictions contain non-finite values")
    pooled_auc = float(roc_auc_score(y, prediction))
    baseline_auc = float(roc_auc_score(y, baseline))
    if abs(pooled_auc - float(summary["pooled_outer_auc"])) > 1e-12:
        raise ValueError("Pooled AUC does not match SUMMARY.json")
    if abs(baseline_auc - float(summary["phase15_baseline_pooled_auc"])) > 1e-12:
        raise ValueError("Baseline AUC does not match SUMMARY.json")

    fold_metrics = pd.read_csv(run_dir / "FOLD_METRICS.csv")
    if sorted(fold_metrics["fold"].astype(int).tolist()) != list(
        range(1, int(summary["outer_folds"]) + 1)
    ):
        raise ValueError("Fold metrics are missing or duplicate")
    for row in fold_metrics.itertuples(index=False):
        mask = folds == int(row.fold)
        auc = float(roc_auc_score(y[mask], prediction[mask]))
        base_fold_auc = float(roc_auc_score(y[mask], baseline[mask]))
        if (
            abs(auc - float(row.outer_auc)) > 1e-12
            or abs(base_fold_auc - float(row.phase15_baseline_auc)) > 1e-12
        ):
            raise ValueError(f"Independent fold AUC mismatch for fold {row.fold}")

    selections = pd.read_csv(run_dir / "INNER_CONFIG_SELECTION.csv")
    for fold in range(1, int(summary["outer_folds"]) + 1):
        rows = selections[selections["outer_fold"] == fold]
        if len(rows) != len(summary["configurations"]):
            raise ValueError(f"Unexpected inner config count for fold {fold}")
        winner = rows.loc[rows["pooled_inner_oof_auc"].idxmax(), "config"]
        selected = fold_metrics.loc[
            fold_metrics["fold"] == fold, "selected_config"
        ].iloc[0]
        if winner != selected:
            raise ValueError(f"Outer fold {fold} selection was not inner-OOF winner")

    manifest = pd.read_csv(run_dir / "SHA256_MANIFEST.csv")
    manifest_failures = []
    for row in manifest.itertuples(index=False):
        path = Path(row.file) if row.kind != "output" else run_dir / row.file
        if not path.is_file():
            manifest_failures.append((str(path), "missing"))
            continue
        if path.stat().st_size != int(row.bytes) or sha256_file(path) != row.sha256:
            manifest_failures.append((str(path), "hash_or_size_mismatch"))
    if manifest_failures:
        raise ValueError(f"Run manifest verification failed: {manifest_failures}")
    if sha256_file(TRAIN) != summary["training_sha256"]:
        raise ValueError("Training input hash differs from the recorded summary")
    if sha256_file(PHASE13) != summary["phase13_oof_sha256"]:
        raise ValueError("Phase 13 input hash differs from the recorded summary")

    return {
        "status": "pass",
        "run_id": run_dir.name,
        "rows": int(len(oof)),
        "finite_oof_predictions": bool(np.isfinite(prediction).all()),
        "pooled_auc": pooled_auc,
        "baseline_auc": baseline_auc,
        "delta": pooled_auc - baseline_auc,
        "folds_verified": int(len(fold_metrics)),
        "nested_config_selections_verified": int(summary["outer_folds"]),
        "manifest_entries_verified": int(len(manifest)),
        "train_sha256": sha256_file(TRAIN),
        "phase13_sha256": sha256_file(PHASE13),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
