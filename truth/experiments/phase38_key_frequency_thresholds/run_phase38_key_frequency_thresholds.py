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
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE34 = Path(__file__).resolve().parents[1] / "phase34_xgboost_rules_first"
sys.path[:0] = [str(PHASE33), str(PHASE34)]
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE34_OOF_DEFAULT = (
    PHASE34
    / "runs"
    / "phase34_xgb_20261107T171823Z"
    / "PHASE34_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
THRESHOLDS = (10, 50)
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return csr_matrix(values)


def make_matrices(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    min_frequency: int,
) -> tuple[csr_matrix, csr_matrix]:
    numeric, categorical, rowwise, _, keys, _ = blocks
    fit_numeric = pd.concat(
        [
            numeric.iloc[fit_rows].reset_index(drop=True),
            rowwise.iloc[fit_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    apply_numeric = pd.concat(
        [
            numeric.iloc[apply_rows].reset_index(drop=True),
            rowwise.iloc[apply_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    fit_blocks = [_numeric_matrix(fit_numeric)]
    apply_blocks = [_numeric_matrix(apply_numeric)]
    raw_encoder = OneHotEncoder(
        handle_unknown="ignore", dtype=np.float32, sparse_output=True
    )
    fit_raw = raw_encoder.fit_transform(
        categorical.iloc[fit_rows].astype(str)
    ).tocsr()
    apply_raw = raw_encoder.transform(
        categorical.iloc[apply_rows].astype(str)
    ).tocsr()
    fit_blocks.append(fit_raw)
    apply_blocks.append(apply_raw)
    fit_key_values = keys.iloc[fit_rows].astype(str)
    apply_key_values = keys.iloc[apply_rows].astype(str)
    key_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=min_frequency,
        dtype=np.float32,
        sparse_output=True,
    )
    fit_key = key_encoder.fit_transform(fit_key_values).tocsr()
    apply_key = key_encoder.transform(apply_key_values).tocsr()
    fit_blocks.append(fit_key)
    apply_blocks.append(apply_key)
    fit_counts: list[np.ndarray] = []
    apply_counts: list[np.ndarray] = []
    for column in phase33.KEY_COLUMNS:
        counts = fit_key_values[column].value_counts()
        fit_counts.append(fit_key_values[column].map(counts).to_numpy(np.float32))
        apply_counts.append(
            apply_key_values[column].map(counts).fillna(0).to_numpy(np.float32)
        )
    fit_blocks.append(csr_matrix(np.column_stack(fit_counts), dtype=np.float32))
    apply_blocks.append(
        csr_matrix(np.column_stack(apply_counts), dtype=np.float32)
    )
    return (
        hstack(fit_blocks, format="csr", dtype=np.float32),
        hstack(apply_blocks, format="csr", dtype=np.float32),
    )


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
    phase34_oof_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase34_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    reference = pd.read_csv(phase34_oof_path)
    if (
        ID not in train
        or TARGET not in train
        or not train[ID].is_unique
        or len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or "oof_xgboost" not in reference
    ):
        raise ValueError("Training data or reference OOF is invalid")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if not np.array_equal(fold_ids, reference["fold"].to_numpy()):
        raise ValueError("Reference OOF fold assignment does not match")
    oof = {
        threshold: np.full(len(y), np.nan, dtype=np.float32)
        for threshold in THRESHOLDS
    }
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
        for threshold in THRESHOLDS:
            x_inner_fit, x_inner_valid = make_matrices(
                blocks, inner_fit, inner_valid, threshold
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

            x_fit, x_valid = make_matrices(
                blocks, fit_rows, valid_rows, threshold
            )
            model = _model(fold_seed, rounds, stopping=False)
            model.fit(x_fit, y[fit_rows], verbose=False)
            prediction = model.predict_proba(x_valid)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite prediction for threshold={threshold}, fold={fold_number}"
                )
            oof[threshold][valid_rows] = prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], prediction))
            records.append(
                {
                    "min_frequency": threshold,
                    "fold": fold_number,
                    "auc": auc,
                    "iterations": rounds,
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"min_frequency={threshold} fold={fold_number}/{FOLDS} "
                f"AUC={auc:.6f} iterations={rounds}",
                flush=True,
            )
    if any(not np.isfinite(value).all() for value in oof.values()):
        raise ValueError("One or more OOF vectors are incomplete")
    base = reference["oof_xgboost"].to_numpy(dtype=np.float32)

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE38_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE38_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE38_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    output = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_xgb_min_frequency_10": oof[10],
            "oof_xgb_min_frequency_50": oof[50],
            "oof_xgb_min_frequency_2_reference": base,
        }
    )
    output.to_csv(oof_path, index=False)
    metrics: dict[str, dict[str, float]] = {}
    for name, vector in (
        ("min_frequency_2_reference", base),
        ("min_frequency_10", oof[10]),
        ("min_frequency_50", oof[50]),
    ):
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, vector)),
            "pooled_log_loss": float(log_loss(y, vector)),
            "mean_fold_auc": float(
                np.mean(
                    [
                        roc_auc_score(y[valid], vector[valid])
                        for _, valid in folds
                    ]
                )
            ),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 38 fold-fit key frequency threshold comparison",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "feature_set": "Phase 33 constant-free ratio keys",
        "thresholds": list(THRESHOLDS),
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; one-hot numeric matrix",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "category_vocabularies_and_frequencies": "fit on each model training partition only",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
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
    parser.add_argument("--phase34-oof", type=Path, default=PHASE34_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase38_thresholds_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase34_oof, args.output_dir)
