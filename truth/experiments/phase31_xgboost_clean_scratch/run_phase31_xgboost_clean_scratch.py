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
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE29 = REPO / "experiments" / "phase29_clean_scratch"
sys.path.insert(0, str(PHASE29))
import run_phase29_clean_scratch as phase29


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
TEST_DEFAULT = Path(r"D:\gci\inpuy\test.csv")
SAMPLE_DEFAULT = Path(r"D:\gci\inpuy\sample_submission.csv")
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


def _matrices(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    test_numeric: pd.DataFrame,
    test_categorical: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    fit_cat = encoder.fit_transform(categorical.iloc[fit_rows])
    valid_cat = encoder.transform(categorical.iloc[valid_rows])
    test_cat = encoder.transform(test_categorical)
    fit = np.concatenate(
        [numeric.iloc[fit_rows].to_numpy(dtype=np.float32), fit_cat], axis=1
    )
    valid = np.concatenate(
        [numeric.iloc[valid_rows].to_numpy(dtype=np.float32), valid_cat], axis=1
    )
    test = np.concatenate(
        [test_numeric.to_numpy(dtype=np.float32), test_cat], axis=1
    )
    if fit.shape[1] != valid.shape[1] or fit.shape[1] != test.shape[1]:
        raise ValueError("Fold train/valid/test feature widths do not match")
    return fit, valid, test


def _select_rounds(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
) -> int:
    model = xgb.XGBClassifier(
        n_estimators=MAX_ITERATIONS,
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
        early_stopping_rounds=STOPPING_ROUNDS,
        verbosity=0,
    )
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )
    return max(100, int(model.best_iteration) + 1)


def run(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if TARGET not in train or ID not in train or TARGET in test or ID not in test:
        raise ValueError("Train/test input columns are invalid")
    if test.columns.tolist() != [
        column for column in train.columns if column != TARGET
    ]:
        raise ValueError("Train/test schemas differ")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("Sample submission schema must be ID,TARGET")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("IDs must be unique")
    if not test[ID].reset_index(drop=True).equals(
        sample[ID].reset_index(drop=True)
    ):
        raise ValueError("Test ID order differs from sample submission")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, [0, 1]).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)

    train_raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    missing_cols = tuple(
        column for column in train_raw if train_raw[column].isna().any()
    )
    train_numeric, train_categorical = phase29.make_rowwise_features(
        train_raw, missing_cols
    )
    test_numeric, test_categorical = phase29.make_rowwise_features(
        test_raw, missing_cols
    )
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    oof = np.full(len(y), np.nan, dtype=np.float32)
    test_fold: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_train_idx, inner_valid_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_train_rows = fit_rows[inner_train_idx]
        inner_valid_rows = fit_rows[inner_valid_idx]
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        cat_inner_train = encoder.fit_transform(
            train_categorical.iloc[inner_train_rows]
        )
        cat_inner_valid = encoder.transform(
            train_categorical.iloc[inner_valid_rows]
        )
        x_inner_train = np.concatenate(
            [
                train_numeric.iloc[inner_train_rows].to_numpy(dtype=np.float32),
                cat_inner_train,
            ],
            axis=1,
        )
        x_inner_valid = np.concatenate(
            [
                train_numeric.iloc[inner_valid_rows].to_numpy(dtype=np.float32),
                cat_inner_valid,
            ],
            axis=1,
        )
        rounds = _select_rounds(
            x_inner_train,
            y[inner_train_rows],
            x_inner_valid,
            y[inner_valid_rows],
            fold_seed,
        )
        x_fit, x_valid, x_test = _matrices(
            train_numeric,
            train_categorical,
            fit_rows,
            valid_rows,
            test_numeric,
            test_categorical,
        )
        model = xgb.XGBClassifier(
            n_estimators=rounds,
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
            random_state=fold_seed,
            n_jobs=8,
            verbosity=0,
        )
        model.fit(x_fit, y[fit_rows], verbose=False)
        valid_prediction = model.predict_proba(x_valid)[:, 1]
        test_prediction = model.predict_proba(x_test)[:, 1]
        if not np.isfinite(valid_prediction).all() or not np.isfinite(
            test_prediction
        ).all():
            raise ValueError("XGBoost generated non-finite predictions")
        oof[valid_rows] = valid_prediction.astype(np.float32)
        test_fold.append(test_prediction.astype(np.float64))
        auc = float(roc_auc_score(y[valid_rows], valid_prediction))
        records.append(
            {
                "fold": fold_number,
                "auc": auc,
                "best_iterations": rounds,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "feature_count": int(x_fit.shape[1]),
            }
        )
        print(
            f"FOLD {fold_number}/{FOLDS}: AUC={auc:.6f}, rounds={rounds}",
            flush=True,
        )
        del model

    if not np.isfinite(oof).all():
        raise ValueError("OOF vector is incomplete")
    test_prediction = np.mean(np.vstack(test_fold), axis=0)
    if not np.isfinite(test_prediction).all() or (
        (test_prediction < 0) | (test_prediction > 1)
    ).any():
        raise ValueError("Invalid test probability vector")

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE31_FOLD_RESULTS.csv"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    oof_path = output_dir / "PHASE31_OOF_PREDICTIONS.csv"
    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "oof_xgboost": oof,
            "fold": np.full(len(y), -1, dtype=np.int8),
        }
    )
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        oof_frame.loc[valid_rows, "fold"] = fold_number
    oof_frame.to_csv(oof_path, index=False)
    submission_path = output_dir / "submission_xgboost.csv"
    submission = pd.DataFrame({ID: test[ID], TARGET: test_prediction})
    if not submission[ID].equals(sample[ID]):
        raise ValueError("Submission ID order mismatch")
    submission.to_csv(submission_path, index=False)
    summary_path = output_dir / "PHASE31_SUMMARY.json"
    summary = {
        "phase": "Phase 31 clean-scope XGBoost scratch",
        "status": "exploratory; no external or label-derived features",
        "rows": int(len(y)),
        "outer_folds": FOLDS,
        "seed": SEED,
        "pooled_auc": float(roc_auc_score(y, oof)),
        "mean_fold_auc": float(np.mean([row["auc"] for row in records])),
        "std_fold_auc": float(np.std([row["auc"] for row in records])),
        "log_loss": float(log_loss(y, oof)),
        "runtime_seconds": time.perf_counter() - started,
        "rule_controls": {
            "external_data": "none",
            "custom_target_encoding": "none",
            "test_statistics": "none; only rowwise transforms and fold-fit one-hot vocabularies",
            "manual_prediction_edits": "none",
            "id_predictor": "excluded",
        },
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(phase29.__file__).resolve(),
                train_path,
                test_path,
                sample_path,
                fold_path,
                oof_path,
                submission_path,
            )
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / f"phase31_xgboost_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
    )
    args = parser.parse_args()
    run(args.train, args.test, args.sample, args.output_dir)


if __name__ == "__main__":
    main()
