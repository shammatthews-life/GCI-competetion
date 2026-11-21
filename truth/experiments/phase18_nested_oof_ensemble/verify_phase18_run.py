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
PHASE15 = (
    REPO
    / "experiments"
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE17 = (
    REPO
    / "experiments"
    / "phase17_nested_constant_free_target_encoding"
    / "runs"
    / "phase17_nested_te_20261105T140620Z"
    / "OOF_PREDICTIONS.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_auc_interval(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        deltas[index] = roc_auc_score(y[sample], candidate[sample]) - roc_auc_score(
            y[sample], reference[sample]
        )
    return (
        float(np.quantile(deltas, 0.025)),
        float(np.quantile(deltas, 0.975)),
    )


def verify(run_dir: Path) -> dict[str, object]:
    required = (
        "SUMMARY.json",
        "OOF_PREDICTIONS.csv",
        "FOLD_METRICS.csv",
        "INNER_BLEND_SELECTION.csv",
        "SHA256_MANIFEST.csv",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 18 outputs: {missing}")
    summary = json.loads((run_dir / "SUMMARY.json").read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    p15 = pd.read_csv(PHASE15)
    p17 = pd.read_csv(PHASE17)
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
    for name, frame in (("Phase 15", p15), ("Phase 17", p17), ("Phase 18", oof)):
        if not (
            len(frame) == len(train)
            and np.array_equal(frame["row_index"].to_numpy(), np.arange(len(y)))
            and np.array_equal(frame["SK_ID_CURR"].to_numpy(), ids)
            and np.array_equal(frame["TARGET"].to_numpy(), y)
            and np.array_equal(frame["fold"].to_numpy(), folds)
        ):
            raise ValueError(f"{name} row/ID/target/fold alignment failed")

    p_cat = p15["native_catboost_phase14_plus_time"].to_numpy(dtype=np.float64)
    p_lgb = p17["nested_te_constant_free"].to_numpy(dtype=np.float64)
    p_blend = oof["phase18_nested_blend"].to_numpy(dtype=np.float64)
    fold_metrics = pd.read_csv(run_dir / "FOLD_METRICS.csv")
    if len(fold_metrics) != int(summary["outer_folds"]):
        raise ValueError("Unexpected outer fold metric count")
    weight_by_fold = {
        int(row.outer_fold): float(row.selected_catboost_weight)
        for row in fold_metrics.itertuples(index=False)
    }
    if set(weight_by_fold) != set(range(1, int(summary["outer_folds"]) + 1)):
        raise ValueError("Outer fold weights do not cover every fold")
    row_weights = np.array([weight_by_fold[int(fold)] for fold in folds])
    expected_blend = row_weights * p_cat + (1.0 - row_weights) * p_lgb
    if not np.allclose(p_blend, expected_blend, rtol=0.0, atol=1e-12):
        raise ValueError("Saved outer blend does not match saved components/weights")

    selections = pd.read_csv(run_dir / "INNER_BLEND_SELECTION.csv")
    if len(selections) != int(summary["outer_folds"]) * len(
        summary["candidate_catboost_weights"]
    ):
        raise ValueError("Inner weight grid is incomplete")
    for fold in range(1, int(summary["outer_folds"]) + 1):
        rows = selections[selections["outer_fold"] == fold]
        if len(rows) != len(summary["candidate_catboost_weights"]):
            raise ValueError(f"Unexpected candidate weight count for outer fold {fold}")
        selected = rows.loc[rows["pooled_inner_oof_auc"].idxmax()]
        metric = fold_metrics[fold_metrics["outer_fold"] == fold]
        if len(metric) != 1:
            raise ValueError(f"Missing or duplicate outer metric for fold {fold}")
        if abs(
            float(selected["catboost_weight"])
            - float(metric.iloc[0]["selected_catboost_weight"])
        ) > 1e-12:
            raise ValueError(f"Selected weight is not the inner OOF winner for fold {fold}")

    pooled_blend = float(roc_auc_score(y, p_blend))
    pooled_cat = float(roc_auc_score(y, p_cat))
    pooled_lgb = float(roc_auc_score(y, p_lgb))
    if (
        abs(pooled_blend - float(summary["nested_blend_auc"])) > 1e-12
        or abs(pooled_cat - float(summary["phase15_catboost_auc"])) > 1e-12
        or abs(pooled_lgb - float(summary["phase17_nested_te_lightgbm_auc"])) > 1e-12
    ):
        raise ValueError("Independent pooled AUC does not match SUMMARY.json")

    for row in fold_metrics.itertuples(index=False):
        mask = folds == int(row.outer_fold)
        auc = float(roc_auc_score(y[mask], p_blend[mask]))
        if abs(auc - float(row.outer_nested_blend_auc)) > 1e-12:
            raise ValueError(f"Independent blended fold AUC mismatch: {row.outer_fold}")

    for name, reference, recorded in (
        ("Phase 15", p_cat, summary["paired_interval_vs_catboost"]),
        ("Phase 17", p_lgb, summary["paired_interval_vs_phase17"]),
    ):
        low, high = paired_auc_interval(
            y,
            reference,
            p_blend,
            int(recorded["bootstrap_repetitions"]),
            int(recorded["bootstrap_seed"]),
        )
        if (
            abs(low - float(recorded["paired_delta_ci95_low"])) > 1e-12
            or abs(high - float(recorded["paired_delta_ci95_high"])) > 1e-12
        ):
            raise ValueError(f"Independent paired bootstrap mismatch vs {name}")

    manifest = pd.read_csv(run_dir / "SHA256_MANIFEST.csv")
    for row in manifest.itertuples(index=False):
        path = run_dir / row.file if row.kind == "output" else Path(row.file)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input missing: {path}")
        if path.stat().st_size != int(row.bytes) or sha256_file(path) != row.sha256:
            raise ValueError(f"Manifest mismatch: {path}")
    if (
        sha256_file(TRAIN) != summary["training_sha256"]
        or sha256_file(PHASE15) != summary["phase15_oof_sha256"]
        or sha256_file(PHASE17) != summary["phase17_oof_sha256"]
    ):
        raise ValueError("Recorded source input hash mismatch")

    return {
        "status": "pass",
        "run_id": run_dir.name,
        "rows": int(len(y)),
        "nested_blend_auc": pooled_blend,
        "phase15_catboost_auc": pooled_cat,
        "phase17_te_lightgbm_auc": pooled_lgb,
        "selected_outer_weights_verified": int(summary["outer_folds"]),
        "inner_weight_candidates_verified": int(len(selections)),
        "folds_verified": int(len(fold_metrics)),
        "manifest_entries_verified": int(len(manifest)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
