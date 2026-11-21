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
from scipy.sparse import hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE38_OOF_DEFAULT = (
    PHASE38
    / "runs"
    / "phase38_thresholds_20261107T173546Z"
    / "PHASE38_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
EMPLOYMENT_SENTINEL = 365243.0
FEATURE_COLUMNS = (
    "SAFE_EXT_EMPLOYMENT_AGE_RATIO",
    "SAFE_EXT_ESTIMATED_PREEMPLOYMENT_AGE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_employment_features(
    raw: pd.DataFrame, rowwise: pd.DataFrame
) -> pd.DataFrame:
    age_days = pd.to_numeric(raw["DAYS_BIRTH"], errors="raise").to_numpy(
        dtype=np.float64
    )
    employed_days = pd.to_numeric(
        raw["DAYS_EMPLOYED"], errors="raise"
    ).to_numpy(dtype=np.float64, copy=True)
    employed_days[employed_days == EMPLOYMENT_SENTINEL] = np.nan
    age_years = -age_days / 365.25
    employed_years = -employed_days / 365.25
    ratio = np.full(len(raw), np.nan, dtype=np.float64)
    np.divide(
        employed_years,
        age_years,
        out=ratio,
        where=np.isfinite(age_years) & (age_years > 0),
    )
    rowwise = rowwise.copy()
    rowwise[FEATURE_COLUMNS[0]] = ratio
    rowwise[FEATURE_COLUMNS[1]] = age_years - employed_years
    return rowwise


def make_matrices(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[Any, Any]:
    base_fit, base_apply = phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    employment = blocks[2]
    fit_extra = phase38._numeric_matrix(
        employment.iloc[fit_rows][list(FEATURE_COLUMNS)].reset_index(drop=True)
    )
    apply_extra = phase38._numeric_matrix(
        employment.iloc[apply_rows][list(FEATURE_COLUMNS)].reset_index(drop=True)
    )
    return (
        hstack([base_fit, fit_extra], format="csr", dtype=np.float32),
        hstack([base_apply, apply_extra], format="csr", dtype=np.float32),
    )


def run(train_path: Path, phase38_oof_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase38_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    reference = pd.read_csv(phase38_oof_path)
    if (
        ID not in train
        or TARGET not in train
        or not train[ID].is_unique
        or len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or "fold" not in reference
        or "oof_xgb_min_frequency_50" not in reference
    ):
        raise ValueError("Training data or Phase 38 OOF is invalid")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    built_blocks = phase33.build_safe_blocks(raw)
    numeric, categorical, rowwise, financial, keys, peer = built_blocks
    rowwise = add_employment_features(raw, rowwise)
    blocks = (numeric, categorical, rowwise, financial, keys, peer)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if not np.array_equal(fold_ids, reference["fold"].to_numpy()):
        raise ValueError("Phase 38 fold assignment does not match")
    predictions = np.full(len(y), np.nan, dtype=np.float32)
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
        x_inner_fit, x_inner_valid = make_matrices(
            blocks, inner_fit, inner_valid
        )
        selector = phase38._model(
            fold_seed, MAX_ITERATIONS, stopping=True
        )
        selector.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_valid, y[inner_valid])],
            verbose=False,
        )
        rounds = max(100, int(selector.best_iteration) + 1)
        del selector, x_inner_fit, x_inner_valid

        x_fit, x_valid = make_matrices(blocks, fit_rows, valid_rows)
        model = phase38._model(fold_seed, rounds, stopping=False)
        model.fit(x_fit, y[fit_rows], verbose=False)
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite prediction in outer fold {fold_number}")
        predictions[valid_rows] = prediction.astype(np.float32)
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
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete")

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE41_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE41_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE41_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    baseline = reference["oof_xgb_min_frequency_50"].to_numpy(dtype=np.float32)
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_xgb_keys_frequency_50": baseline,
            "oof_xgb_keys_frequency_50_plus_employment": predictions,
        }
    ).to_csv(oof_path, index=False)
    metrics: dict[str, dict[str, float]] = {}
    for name, vector in (
        ("keys_frequency_50_baseline", baseline),
        ("keys_frequency_50_plus_employment", predictions),
    ):
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, vector)),
            "pooled_log_loss": float(log_loss(y, vector)),
            "mean_fold_auc": float(
                np.mean(
                    [roc_auc_score(y[valid], vector[valid]) for _, valid in folds]
                )
            ),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 41 employment trajectory feature ablation",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "added_features": list(FEATURE_COLUMNS),
        "feature_definition": {
            FEATURE_COLUMNS[0]: "(-DAYS_EMPLOYED / 365.25) / (-DAYS_BIRTH / 365.25); sentinel employment values become missing",
            FEATURE_COLUMNS[1]: "(-DAYS_BIRTH / 365.25) - (-DAYS_EMPLOYED / 365.25); sentinel employment values become missing",
            "interpretation": "Derived covariates only; estimated gap is not asserted to be actual age at first employment",
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
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
                phase38_oof_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
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
    parser.add_argument("--phase38-oof", type=Path, default=PHASE38_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase41_employment_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase38_oof, args.output_dir)
