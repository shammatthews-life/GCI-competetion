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
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_CALIBRATION_FOLDS = 3
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
MIN_EXPERT_FIT_ROWS = 5000
MIN_EXPERT_VALID_ROWS = 1000
CALIBRATION_C = 1.0e4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_membership(raw: pd.DataFrame) -> np.ndarray:
    if "NAME_CONTRACT_TYPE" not in raw:
        raise ValueError("NAME_CONTRACT_TYPE is required for routing")
    values = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(
        dtype=str
    )
    if set(np.unique(values)) != {"Cash loans", "Revolving loans"}:
        raise ValueError("Unexpected contract categories; refusing implicit routing")
    return (values == "Revolving loans").astype(np.int8)


def train_predict(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    if len(np.unique(y[fit_rows])) != 2:
        raise ValueError("Model training partition must contain both target classes")
    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.12, random_state=seed
    )
    inner_fit_idx, inner_valid_idx = next(
        inner.split(np.zeros(len(fit_rows)), y[fit_rows])
    )
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]
    x_inner_fit, x_inner_valid = phase38.make_matrices(
        blocks, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = phase38._model(seed, MAX_ITERATIONS, stopping=True)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_valid, y[inner_valid])],
        verbose=False,
    )
    iterations = max(100, int(selector.best_iteration) + 1)
    del selector, x_inner_fit, x_inner_valid

    x_fit, x_apply = phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    model = phase38._model(seed, iterations, stopping=False)
    model.fit(x_fit, y[fit_rows], verbose=False)
    prediction = model.predict_proba(x_apply)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced nonfinite predictions")
    return prediction.astype(np.float64), iterations


def fit_group_platt(
    blocks: tuple[pd.DataFrame, ...],
    group_rows: np.ndarray,
    y: np.ndarray,
    outer_fold: int,
    group: int,
) -> tuple[LogisticRegression, np.ndarray, list[int]]:
    group_y = y[group_rows]
    splitter = StratifiedKFold(
        n_splits=INNER_CALIBRATION_FOLDS,
        shuffle=True,
        random_state=SEED + outer_fold * 1000 + group + 71,
    )
    oof = np.full(len(group_rows), np.nan, dtype=np.float64)
    selected_iterations: list[int] = []
    for inner_fold, (fit_local, valid_local) in enumerate(
        splitter.split(np.zeros(len(group_rows)), group_y), start=1
    ):
        fit_rows = group_rows[fit_local]
        valid_rows = group_rows[valid_local]
        prediction, iterations = train_predict(
            blocks,
            fit_rows,
            valid_rows,
            y,
            SEED + outer_fold * 10000 + group * 100 + inner_fold + 900000,
        )
        oof[valid_local] = prediction
        selected_iterations.append(iterations)
    if not np.isfinite(oof).all():
        raise ValueError("Nested calibration OOF predictions are incomplete")
    clipped_logits = logit(np.clip(oof, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=CALIBRATION_C,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED + outer_fold * 100 + group,
    )
    calibrator.fit(clipped_logits, group_y)
    return calibrator, oof, selected_iterations


def apply_platt(
    calibrator: LogisticRegression, prediction: np.ndarray
) -> np.ndarray:
    clipped_logits = logit(
        np.clip(prediction, 1e-6, 1.0 - 1e-6)
    ).reshape(-1, 1)
    return calibrator.predict_proba(clipped_logits)[:, 1]


def _metrics(
    y: np.ndarray, pred: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    return {
        "pooled_auc": float(roc_auc_score(y, pred)),
        "pooled_log_loss": float(log_loss(y, pred)),
        "mean_fold_auc": float(
            np.mean(
                [roc_auc_score(y[valid], pred[valid]) for _, valid in folds]
            )
        ),
    }


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must contain unique IDs and TARGET")
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
    membership = contract_membership(raw)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float32)
        for name in (
            "global",
            "contract_type_hard",
            "contract_type_mixture_50_50",
            "contract_type_platt_mixture_50_50",
        )
    }
    records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        global_prediction, global_iterations = train_predict(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        expert_prediction = np.full(len(valid_rows), np.nan, dtype=np.float64)
        calibrated_expert = np.full(len(valid_rows), np.nan, dtype=np.float64)

        for group in (0, 1):
            group_fit = fit_rows[membership[fit_rows] == group]
            group_valid_mask = membership[valid_rows] == group
            group_valid = valid_rows[group_valid_mask]
            if (
                len(group_fit) < MIN_EXPERT_FIT_ROWS
                or len(group_valid) < MIN_EXPERT_VALID_ROWS
                or len(np.unique(y[group_fit])) != 2
            ):
                raise ValueError(
                    f"Insufficient support for contract group={group}, "
                    f"fold={fold_number}: fit={len(group_fit)}, "
                    f"valid={len(group_valid)}"
                )
            calibrator, calibration_oof, inner_iterations = fit_group_platt(
                blocks, group_fit, y, fold_number, group
            )
            group_prediction, expert_iterations = train_predict(
                blocks,
                group_fit,
                group_valid,
                y,
                fold_seed + group + 1,
            )
            expert_prediction[group_valid_mask] = group_prediction
            calibrated_expert[group_valid_mask] = apply_platt(
                calibrator, group_prediction
            )
            calibration_records.append(
                {
                    "fold": fold_number,
                    "group": group,
                    "fit_rows": int(len(group_fit)),
                    "outer_validation_rows": int(len(group_valid)),
                    "calibration_oof_rows": int(len(calibration_oof)),
                    "calibration_oof_auc": float(
                        roc_auc_score(y[group_fit], calibration_oof)
                    ),
                    "platt_slope": float(calibrator.coef_[0, 0]),
                    "platt_intercept": float(calibrator.intercept_[0]),
                    "inner_iterations": inner_iterations,
                    "expert_iterations": expert_iterations,
                }
            )

        if not (
            np.isfinite(expert_prediction).all()
            and np.isfinite(calibrated_expert).all()
        ):
            raise ValueError("Contract expert predictions are incomplete")
        values = {
            "global": global_prediction,
            "contract_type_hard": expert_prediction,
            "contract_type_mixture_50_50": (
                0.5 * global_prediction + 0.5 * expert_prediction
            ),
            "contract_type_platt_mixture_50_50": (
                0.5 * global_prediction + 0.5 * calibrated_expert
            ),
        }
        for name, value in values.items():
            predictions[name][valid_rows] = value.astype(np.float32)
        for name, value in values.items():
            records.append(
                {
                    "model": name,
                    "fold": fold_number,
                    "auc": float(roc_auc_score(y[valid_rows], value)),
                    "log_loss": float(log_loss(y[valid_rows], value)),
                    "validation_rows": int(len(valid_rows)),
                    "global_iterations": global_iterations,
                }
            )
        print(
            f"fold={fold_number}/{FOLDS} "
            f"global={roc_auc_score(y[valid_rows], values['global']):.6f} "
            f"fixed_mix={roc_auc_score(y[valid_rows], values['contract_type_mixture_50_50']):.6f} "
            f"platt_mix={roc_auc_score(y[valid_rows], values['contract_type_platt_mixture_50_50']):.6f}",
            flush=True,
        )

    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("OOF prediction vectors are incomplete")
    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE45_FOLD_RESULTS.csv"
    calibration_path = output_dir / "PHASE45_CALIBRATION_RESULTS.csv"
    oof_path = output_dir / "PHASE45_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE45_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    pd.DataFrame(calibration_records).to_csv(calibration_path, index=False)
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{f"oof_{name}": value for name, value in predictions.items()},
        }
    ).to_csv(oof_path, index=False)
    metrics = {
        name: _metrics(y, value, folds)
        for name, value in predictions.items()
    }
    summary: dict[str, Any] = {
        "phase": "Phase 45 nested cross-fitted contract-expert calibration",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_calibration_folds": INNER_CALIBRATION_FOLDS,
        "model": "Phase 38 XGBoost configuration; inner 12% stopping split; outer-fit refit",
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "gate": "NAME_CONTRACT_TYPE: cash=0, revolving=1",
        "fixed_mixture_weight": 0.5,
        "calibration": {
            "method": "per-contract-group Platt logistic regression on inner out-of-fold expert logits",
            "input_prediction_folds": INNER_CALIBRATION_FOLDS,
            "regularization_C": CALIBRATION_C,
            "validation_labels_used": False,
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "gate_membership": "deterministic supplied covariate only",
            "expert_training": "each expert fit only on its outer-training subgroup",
            "calibrator_training": "inner OOF expert predictions and labels within outer-training subgroup",
            "outer_validation": "never used for fitting, early stopping, gate selection, or calibration",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
        },
        "caveats": [
            "The calibrator can change cross-group ordering but is not expected to improve within-group ranking; judge only pooled outer OOF AUC and log-loss.",
            "Repeated reuse of the training labels creates selection bias; this is not independent confirmation or a leaderboard guarantee.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
                fold_path,
                calibration_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase45_calibrated_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
