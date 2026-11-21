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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path) -> dict[str, object]:
    required = (
        "SUMMARY.json",
        "OOF_PREDICTIONS.csv",
        "FOLD_METRICS.csv",
        "SHA256_MANIFEST.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 17 files: {missing}")
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    oof = pd.read_csv(run_dir / "OOF_PREDICTIONS.csv")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    folds = np.full(len(y), -1, dtype=np.int8)
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
        len(oof) == len(train)
        and np.array_equal(oof["row_index"].to_numpy(), np.arange(len(y)))
        and np.array_equal(oof["SK_ID_CURR"].to_numpy(), ids)
        and np.array_equal(oof["TARGET"].to_numpy(), y)
        and np.array_equal(oof["fold"].to_numpy(), folds)
    ):
        raise ValueError("Phase 17 OOF row, ID, target, or fold alignment failed")

    baseline = oof["baseline_constant_free_ohe"].to_numpy(dtype=np.float64)
    candidate = oof["nested_te_constant_free"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("Phase 17 OOF predictions contain non-finite values")
    baseline_auc = float(roc_auc_score(y, baseline))
    candidate_auc = float(roc_auc_score(y, candidate))
    if (
        abs(baseline_auc - float(summary["pooled_baseline_auc"])) > 1e-12
        or abs(candidate_auc - float(summary["pooled_target_encoded_auc"])) > 1e-12
    ):
        raise ValueError("Independent pooled AUC does not match SUMMARY.json")

    fold_metrics = pd.read_csv(run_dir / "FOLD_METRICS.csv")
    for row in fold_metrics.itertuples(index=False):
        mask = folds == int(row.fold)
        baseline_fold_auc = float(roc_auc_score(y[mask], baseline[mask]))
        candidate_fold_auc = float(roc_auc_score(y[mask], candidate[mask]))
        if (
            abs(baseline_fold_auc - float(row.baseline_auc)) > 1e-12
            or abs(candidate_fold_auc - float(row.target_encoded_auc)) > 1e-12
        ):
            raise ValueError(f"Independent fold AUC mismatch for fold {row.fold}")

    rng = np.random.default_rng(int(summary["bootstrap_seed"]))
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    deltas = np.empty(int(summary["bootstrap_repetitions"]), dtype=np.float64)
    for index in range(len(deltas)):
        sampled = np.concatenate(
            (
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            )
        )
        deltas[index] = roc_auc_score(y[sampled], candidate[sampled]) - roc_auc_score(
            y[sampled], baseline[sampled]
        )
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    if (
        abs(float(ci_low) - float(summary["paired_delta_ci95_low"])) > 1e-12
        or abs(float(ci_high) - float(summary["paired_delta_ci95_high"])) > 1e-12
    ):
        raise ValueError("Paired bootstrap interval does not match SUMMARY.json")

    manifest = pd.read_csv(run_dir / "SHA256_MANIFEST.csv")
    for row in manifest.itertuples(index=False):
        path = run_dir / row.file if row.kind == "output" else Path(row.file)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest entry is missing: {path}")
        if (
            path.stat().st_size != int(row.bytes)
            or sha256_file(path) != row.sha256
        ):
            raise ValueError(f"Manifest hash or size mismatch: {path}")
    if sha256_file(TRAIN) != summary["training_sha256"]:
        raise ValueError("Training hash differs from recorded input hash")

    return {
        "status": "pass",
        "run_id": run_dir.name,
        "rows": int(len(oof)),
        "baseline_auc": baseline_auc,
        "nested_te_auc": candidate_auc,
        "delta": candidate_auc - baseline_auc,
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "folds_verified": int(len(fold_metrics)),
        "manifest_entries_verified": int(len(manifest)),
        "training_sha256": sha256_file(TRAIN),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
