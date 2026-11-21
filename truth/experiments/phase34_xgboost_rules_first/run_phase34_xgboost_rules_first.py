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
sys.path.insert(0, str(PHASE33))
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE33_OOF_DEFAULT = (
    PHASE33
    / "runs"
    / "phase33_rules_first_20261107T170931Z"
    / "PHASE33_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(seed: int, iterations: int) -> xgb.XGBClassifier:
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
        early_stopping_rounds=STOPPING_ROUNDS if iterations == MAX_ITERATIONS else None,
        verbosity=0,
    )


def run(
    train_path: Path,
    phase33_oof_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase33_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    reference = pd.read_csv(phase33_oof_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must include unique IDs and TARGET")
    if (
        len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or "fold" not in reference
        or not reference["fold"].between(1, FOLDS).all()
    ):
        raise ValueError("Phase 33 OOF artifact is not aligned with training rows")
    if "oof_keys_frequency" not in reference:
        raise ValueError("Phase 33 OOF artifact lacks the selected comparator")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    reference_fold = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        reference_fold[valid_rows] = fold_number
    if not np.array_equal(reference_fold, reference["fold"].to_numpy()):
        raise ValueError("Phase 33 OOF fold assignments do not match")

    oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
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
        x_inner_fit, x_inner_valid = phase33.make_matrices(
            raw, blocks, inner_fit, inner_valid, "keys_frequency"
        )
        selector = _model(fold_seed, MAX_ITERATIONS)
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
        model = _model(fold_seed, rounds)
        model.fit(x_fit, y[fit_rows], verbose=False)
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite Phase 34 prediction in fold {fold_number}")
        oof[valid_rows] = prediction.astype(np.float32)
        auc = float(roc_auc_score(y[valid_rows], prediction))
        records.append(
            {
                "fold": fold_number,
                "auc": auc,
                "iterations": rounds,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} AUC={auc:.6f} iterations={rounds}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions are incomplete")
    reference_oof = reference["oof_keys_frequency"].to_numpy(dtype=np.float32)
    blend = (0.5 * oof + 0.5 * reference_oof).astype(np.float32)

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE34_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE34_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE34_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    result = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": reference_fold,
            "oof_xgboost": oof,
            "oof_fixed_equal_blend_with_phase33_lgbm": blend,
        }
    )
    result.to_csv(oof_path, index=False)
    metrics: dict[str, dict[str, float]] = {}
    for name, prediction in (
        ("xgboost", oof),
        ("fixed_equal_blend", blend),
    ):
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(
                np.mean(
                    [
                        roc_auc_score(y[valid], prediction[valid])
                        for _, valid in folds
                    ]
                )
            ),
            "log_loss": float(log_loss(y, prediction)),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 34 rules-first XGBoost model-family comparison",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "feature_set": "Phase 33 constant-free ratio keys and fold-fit frequencies",
        "parameters": {
            "max_depth": 4,
            "learning_rate": 0.035,
            "min_child_weight": 40,
            "subsample": 0.82,
            "colsample_bytree": 0.82,
            "reg_lambda": 5.0,
            "max_iterations": MAX_ITERATIONS,
            "early_stopping_rounds": STOPPING_ROUNDS,
            "device": "CUDA",
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical columns are one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "blend": "predeclared equal 50/50 probability average for OOF diagnosis only",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                phase33_oof_path,
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase34_xgb_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase33_oof, args.output_dir)
