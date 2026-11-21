from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE34 = Path(__file__).resolve().parents[1] / "phase34_xgboost_rules_first"
sys.path[:0] = [str(PHASE33), str(PHASE34)]
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE33_OOF_DEFAULT = (
    PHASE33
    / "runs"
    / "phase33_rules_first_20261107T170931Z"
    / "PHASE33_OOF_PREDICTIONS.csv"
)
PHASE34_OOF_DEFAULT = (
    PHASE34
    / "runs"
    / "phase34_xgb_20261107T171823Z"
    / "PHASE34_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEEDS = (20261114, 20261115)
ALL_SEEDS = (20261113, *SEEDS)
FOLDS = 5
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(seed: int, iterations: int, stopping: bool) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=iterations,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=40,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=8,
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
        verbosity=0,
    )


def run(
    train_path: Path,
    phase33_oof_path: Path,
    phase34_oof_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase33_oof_path, phase34_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    lgbm_reference = pd.read_csv(phase33_oof_path)
    xgb_reference = pd.read_csv(phase34_oof_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must include unique IDs and TARGET")
    for label, frame in (
        ("Phase 33", lgbm_reference),
        ("Phase 34", xgb_reference),
    ):
        if (
            len(frame) != len(train)
            or not frame[ID].equals(train[ID])
            or not frame[TARGET].equals(train[TARGET])
        ):
            raise ValueError(f"{label} OOF rows do not align with training data")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    seed_predictions: dict[int, np.ndarray] = {}
    seed_fold_ids: dict[int, np.ndarray] = {}
    fold_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        folds = list(
            StratifiedKFold(
                n_splits=FOLDS, shuffle=True, random_state=seed
            ).split(np.zeros(len(y)), y)
        )
        prediction = np.full(len(y), np.nan, dtype=np.float32)
        fold_ids = np.full(len(y), -1, dtype=np.int8)
        for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
            fold_ids[valid_rows] = fold_number
            fold_seed = seed + fold_number * 100
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=0.12, random_state=fold_seed
            )
            inner_fit_idx, inner_valid_idx = next(
                inner.split(np.zeros(len(fit_rows)), y[fit_rows])
            )
            inner_fit = fit_rows[inner_fit_idx]
            inner_valid = fit_rows[inner_valid_idx]
            x_inner_fit, x_inner_valid = phase33.make_matrices(
                raw, blocks, inner_fit, inner_valid, "keys_frequency"
            )
            selector = _model(fold_seed, MAX_ITERATIONS, stopping=True)
            selector.fit(
                x_inner_fit,
                y[inner_fit],
                eval_set=[(x_inner_valid, y[inner_valid])],
                verbose=False,
            )
            rounds = max(100, int(selector.best_iteration) + 1)
            del selector, x_inner_fit, x_inner_valid

            x_fit, x_valid = phase33.make_matrices(
                raw, blocks, fit_rows, valid_rows, "keys_frequency"
            )
            model = _model(fold_seed, rounds, stopping=False)
            model.fit(x_fit, y[fit_rows], verbose=False)
            fold_prediction = model.predict_proba(x_valid)[:, 1]
            if not np.isfinite(fold_prediction).all():
                raise ValueError(
                    f"Nonfinite prediction for seed={seed}, fold={fold_number}"
                )
            prediction[valid_rows] = fold_prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], fold_prediction))
            fold_records.append(
                {
                    "seed": seed,
                    "fold": fold_number,
                    "auc": auc,
                    "iterations": rounds,
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"seed={seed} fold={fold_number}/{FOLDS} "
                f"AUC={auc:.6f} iterations={rounds}",
                flush=True,
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF predictions incomplete for seed={seed}")
        seed_predictions[seed] = prediction
        seed_fold_ids[seed] = fold_ids

    seed_predictions[20261113] = xgb_reference["oof_xgboost"].to_numpy(
        dtype=np.float32
    )
    expected_seed13_folds = lgbm_reference["fold"].to_numpy(dtype=np.int8)
    if not np.array_equal(
        expected_seed13_folds, xgb_reference["fold"].to_numpy(dtype=np.int8)
    ):
        raise ValueError("Phase 33 and Phase 34 seed-20261113 fold IDs differ")
    seed_fold_ids[20261113] = expected_seed13_folds
    xgb_average = np.mean(
        np.vstack([seed_predictions[seed] for seed in ALL_SEEDS]), axis=0
    ).astype(np.float32)
    lgbm = lgbm_reference["oof_keys_frequency"].to_numpy(dtype=np.float32)
    equal_cross_model = ((lgbm + xgb_average) / 2.0).astype(np.float32)
    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE37_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE37_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE37_SUMMARY.json"
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    output = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold_seed_20261113": seed_fold_ids[20261113],
            "fold_seed_20261114": seed_fold_ids[20261114],
            "fold_seed_20261115": seed_fold_ids[20261115],
            "oof_xgboost_seed_20261113": seed_predictions[20261113],
            "oof_xgboost_seed_20261114": seed_predictions[20261114],
            "oof_xgboost_seed_20261115": seed_predictions[20261115],
            "oof_xgboost_three_seed_average": xgb_average,
            "oof_equal_blend_lgbm_xgb_three_seeds": equal_cross_model,
        }
    )
    output.to_csv(oof_path, index=False)

    metrics: dict[str, dict[str, float]] = {}
    vector_map = {
        "xgboost_seed_20261113": seed_predictions[20261113],
        "xgboost_seed_20261114": seed_predictions[20261114],
        "xgboost_seed_20261115": seed_predictions[20261115],
        "xgboost_three_seed_average": xgb_average,
        "equal_blend_lgbm_and_xgb_three_seeds": equal_cross_model,
    }
    fold_assignment = seed_fold_ids[20261113]
    for name, vector in vector_map.items():
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, vector)),
            "pooled_log_loss": float(log_loss(y, vector)),
            "mean_auc_on_seed_20261113_folds": float(
                np.mean(
                    [
                        roc_auc_score(y[fold_assignment == fold], vector[fold_assignment == fold])
                        for fold in range(1, FOLDS + 1)
                    ]
                )
            ),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 37 repeated-seed clean XGBoost",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seeds": list(ALL_SEEDS),
        "outer_folds_per_seed": FOLDS,
        "feature_set": "Phase 33 constant-free keys with fold-fit frequencies",
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "seed_average": "each row is OOF in all three stratified seed runs",
            "cross_model_blend": "fixed equal probability average; no blend weights fitted",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                phase33_oof_path,
                phase34_oof_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                fold_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--phase33-oof", type=Path, default=PHASE33_OOF_DEFAULT)
    parser.add_argument("--phase34-oof", type=Path, default=PHASE34_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase37_multiseed_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase33_oof, args.phase34_oof, args.output_dir)
