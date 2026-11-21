from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import sys

PHASE30 = Path(__file__).resolve().parents[1] / "phase30_native_feature_scratch"
sys.path.insert(0, str(PHASE30))
import run_phase30_native_feature_scratch as phase30


TRAIN = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
DEPTHS = (6, 8)
FOLDS = 5
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catboost(
    depth: int,
    iterations: int,
    seed: int,
) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train:
        raise ValueError("Training data must include ID and TARGET")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, (0, 1)).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    missing_columns = tuple(
        column for column in raw if raw[column].isna().any()
    )
    features, categorical_columns = phase30.build_features(raw, missing_columns)
    cat_indices = [
        features.columns.get_loc(column) for column in categorical_columns
    ]
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    predictions = {
        depth: np.full(len(y), np.nan, dtype=np.float32) for depth in DEPTHS
    }
    fold_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_fit_idx, inner_valid_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_idx]
        inner_valid = fit_rows[inner_valid_idx]
        for depth in DEPTHS:
            selector = _catboost(depth, MAX_ITERATIONS, fold_seed)
            selector.fit(
                features.iloc[inner_fit],
                y[inner_fit],
                cat_features=cat_indices,
                eval_set=(features.iloc[inner_valid], y[inner_valid]),
                early_stopping_rounds=STOPPING_ROUNDS,
                use_best_model=True,
            )
            iterations = max(100, int(selector.get_best_iteration()) + 1)
            del selector
            model = _catboost(depth, iterations, fold_seed)
            model.fit(
                features.iloc[fit_rows],
                y[fit_rows],
                cat_features=cat_indices,
            )
            prediction = model.predict_proba(features.iloc[valid_rows])[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite prediction for depth={depth}, fold={fold_number}"
                )
            predictions[depth][valid_rows] = prediction.astype(np.float32)
            fold_records.append(
                {
                    "depth": depth,
                    "fold": fold_number,
                    "iterations": iterations,
                    "auc": float(roc_auc_score(y[valid_rows], prediction)),
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"depth={depth} fold={fold_number}/{FOLDS} "
                f"AUC={fold_records[-1]['auc']:.6f} iterations={iterations}",
                flush=True,
            )

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE32_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE32_OOF_PREDICTIONS.csv"
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    oof_frame = pd.DataFrame({ID: train[ID], TARGET: y})
    for depth, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF predictions incomplete for depth={depth}")
        oof_frame[f"oof_catboost_depth_{depth}"] = prediction
    oof_frame.to_csv(oof_path, index=False)

    metrics = {}
    for depth, prediction in predictions.items():
        metrics[str(depth)] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(
                pd.DataFrame(fold_records).query("depth == @depth")["auc"].mean()
            ),
            "log_loss": float(log_loss(y, prediction)),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 32 CatBoost native-category depth screen",
        "status": "exploratory; repeated-label model selection; not organizer-certified",
        "rows": int(len(train)),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": int(len(categorical_columns)),
        "depths": list(DEPTHS),
        "outer_folds": FOLDS,
        "seed": SEED,
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "rule_controls": {
            "external_data": "none",
            "custom_target_encoding": "none",
            "manual_prediction_edits": "none",
            "native_categorical_target_statistics": (
                "model-internal behavior; local rule interpretation unresolved"
            ),
            "validation": "five outer folds; early stopping uses an inner split",
        },
        "sha256": {
            str(path): sha256_file(path)
            for path in (train_path, Path(__file__).resolve(), fold_path, oof_path)
        },
    }
    summary_path = output_dir / "PHASE32_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase32_depth_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
