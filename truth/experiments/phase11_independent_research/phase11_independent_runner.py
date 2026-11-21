from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "phase10_catboost_product_robustness"
    ),
)
from phase10_independent_feature_builder import (
    build_product_keys,
    build_rowwise_features,
    categorical_columns,
)


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\gci\truth\experiments\phase11_independent_research\runs"
)
REPLICAS = 2
VARIANTS = ("raw", "product_keys")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(
    seed: int,
    iterations: int,
    depth: int = 6,
    learning_rate: float = 0.05,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.82,
        border_count=128,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )


def _stratified_bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    product: np.ndarray,
    seed: int,
    repetitions: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = np.concatenate(
            (
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            )
        )
        deltas[index] = roc_auc_score(
            y[sample], product[sample]
        ) - roc_auc_score(y[sample], baseline[sample])
    return {
        "paired_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "paired_delta_ci95_high": float(np.quantile(deltas, 0.975)),
        "paired_delta_bootstrap_repetitions": repetitions,
    }


def _summary_metrics(
    predictions: pd.DataFrame,
    y: np.ndarray,
    bootstrap_repetitions: int,
    replicas: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_rows: list[dict[str, Any]] = []
    for fold, data in predictions.groupby("fold", sort=True):
        target = data["target"].to_numpy(dtype=np.int8)
        row: dict[str, Any] = {"fold": int(fold)}
        for variant in VARIANTS:
            replica_columns = [
                column
                for column in data.columns
                if column.startswith(f"{variant}_replica_")
            ]
            for column in replica_columns:
                row[f"{column}_auc"] = float(
                    roc_auc_score(target, data[column].to_numpy())
                )
            row[f"{variant}_bagged_auc"] = float(
                roc_auc_score(
                    target,
                    data[f"{variant}_bagged"].to_numpy(),
                )
            )
        row["bagged_paired_delta"] = (
            row["product_keys_bagged_auc"] - row["raw_bagged_auc"]
        )
        fold_rows.append(row)

    baseline = predictions["raw_bagged"].to_numpy(dtype=np.float64)
    product = predictions["product_keys_bagged"].to_numpy(dtype=np.float64)
    summary: dict[str, Any] = {
        "n_rows": int(len(y)),
        "raw_bagged_oof_auc": float(roc_auc_score(y, baseline)),
        "product_keys_bagged_oof_auc": float(roc_auc_score(y, product)),
        "bagged_oof_auc_delta": float(
            roc_auc_score(y, product) - roc_auc_score(y, baseline)
        ),
        "prediction_pearson_correlation": float(np.corrcoef(baseline, product)[0, 1]),
        "prediction_spearman_correlation": float(
            pd.Series(baseline).corr(pd.Series(product), method="spearman")
        ),
    }
    summary.update(
        _stratified_bootstrap_delta(
            y,
            baseline,
            product,
            seed=20261105,
            repetitions=bootstrap_repetitions,
        )
    )
    for variant in VARIANTS:
        for replica in range(replicas):
            column = f"{variant}_replica_{replica + 1}"
            summary[f"{column}_pooled_oof_auc"] = float(
                roc_auc_score(y, predictions[column].to_numpy())
            )
    return pd.DataFrame(fold_rows), summary


def run(
    train_path: Path,
    output_dir: Path,
    folds_count: int,
    iterations: int,
    replicas: int,
    seed: int,
    smoke_rows: int | None,
    bootstrap_repetitions: int,
) -> None:
    started_at = datetime.now(timezone.utc)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    output_dir.mkdir(parents=True, exist_ok=False)

    train = pd.read_csv(train_path, low_memory=False)
    if smoke_rows is not None:
        if smoke_rows < folds_count * 2:
            raise ValueError("smoke_rows must be at least twice the fold count")
        train, _ = train_test_split(
            train,
            train_size=smoke_rows,
            stratify=train["TARGET"],
            random_state=seed,
        )
        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    if "TARGET" not in train or "SK_ID_CURR" not in train:
        raise ValueError("Training CSV must contain TARGET and SK_ID_CURR")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    raw = train.drop(columns=["TARGET"])
    row_features = build_rowwise_features(raw)
    keys = build_product_keys(raw)
    if row_features.columns.intersection(keys.columns).size:
        raise ValueError("Feature builder produced duplicate product-key columns")
    features_by_variant = {
        "raw": row_features,
        "product_keys": pd.concat([row_features, keys], axis=1),
    }
    fold_assignments = np.full(len(train), -1, dtype=np.int16)
    output: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        for replica in range(replicas):
            output[f"{variant}_replica_{replica + 1}"] = np.full(
                len(train), np.nan, dtype=np.float32
            )
        output[f"{variant}_bagged"] = np.full(
            len(train), np.nan, dtype=np.float32
        )

    splitter = StratifiedKFold(
        n_splits=folds_count, shuffle=True, random_state=seed
    )
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(raw, y), start=1):
        fold_assignments[valid_idx] = fold
        for variant in VARIANTS:
            frame = features_by_variant[variant]
            cat_features = categorical_columns(frame)
            replicas_predictions: list[np.ndarray] = []
            for replica in range(replicas):
                model_seed = seed + fold * 100 + replica
                model = make_model(model_seed, iterations)
                model.fit(
                    frame.iloc[fit_idx],
                    y[fit_idx],
                    cat_features=cat_features,
                )
                prediction = model.predict_proba(frame.iloc[valid_idx])[:, 1]
                column = f"{variant}_replica_{replica + 1}"
                output[column][valid_idx] = prediction
                replicas_predictions.append(prediction)
                print(
                    f"fold={fold}/{folds_count} variant={variant} "
                    f"replica={replica + 1}/{replicas} seed={model_seed} "
                    f"task_type=GPU",
                    flush=True,
                )
            output[f"{variant}_bagged"][valid_idx] = np.mean(
                replicas_predictions, axis=0
            )

    if np.any(fold_assignments < 0):
        raise AssertionError("Some rows have no out-of-fold assignment")
    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int32),
            "fold": fold_assignments,
            "target": y,
            **output,
        }
    )
    predictions_path = output_dir / "oof_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    fold_metrics, summary = _summary_metrics(
        predictions, y, bootstrap_repetitions, replicas
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    builder_path = (
        Path(__file__).parents[1]
        / "phase10_catboost_product_robustness"
        / "phase10_independent_feature_builder.py"
    )
    feature_manifest = {
        "row_feature_columns": row_features.columns.tolist(),
        "product_key_columns": keys.columns.tolist(),
        "categorical_columns": categorical_columns(row_features),
        "feature_builder": str(builder_path),
        "target_excluded_from_all_feature_builders": True,
        "identifier_excluded_from_model_features": True,
        "test_data_opened": False,
        "locked_holdout_used": False,
        "submission_created": False,
        "legacy_exp85_imported": False,
    }
    manifest = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "catboost": catboost.__version__,
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "train_rows_used": int(len(train)),
        "folds": folds_count,
        "fold_seed": seed,
        "replicas_per_fold": replicas,
        "iterations": iterations,
        "bootstrap_repetitions": bootstrap_repetitions,
        "model": {
            "task_type": "GPU",
            "devices": "0",
            "evaluation_metric": "external OOF ROC-AUC",
            "training_loss": "Logloss",
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "border_count": 128,
            "max_ctr_complexity": 1,
        },
        "features": feature_manifest,
        "predictions_sha256": sha256_file(predictions_path),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("phase11_candidate_search.py"),
                builder_path,
            )
        },
        "results": summary,
        "smoke_run_only": smoke_rows is not None,
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--replicas", type=int, default=REPLICAS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-rows", type=int)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT
        / f"phase11_catboost_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    if args.folds < 2 or args.iterations < 1 or args.replicas < 1:
        parser.error("folds, iterations, and replicas must be positive (folds >= 2)")
    run(
        args.train,
        output_dir,
        args.folds,
        args.iterations,
        args.replicas,
        args.seed,
        args.smoke_rows,
        args.bootstrap_repetitions,
    )


if __name__ == "__main__":
    main()
