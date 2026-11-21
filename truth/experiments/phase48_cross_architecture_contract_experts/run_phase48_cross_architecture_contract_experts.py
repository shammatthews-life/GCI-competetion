from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
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
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
MIN_EXPERT_FIT_ROWS = 5000
MIN_EXPERT_VALID_ROWS = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_contract_gate(raw: pd.DataFrame) -> np.ndarray:
    if "NAME_CONTRACT_TYPE" not in raw.columns:
        raise ValueError("NAME_CONTRACT_TYPE missing from input columns")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING")
    contract_values = contract.to_numpy(dtype=str)
    if set(np.unique(contract_values)) != {"Cash loans", "Revolving loans"}:
        raise ValueError("Unexpected contract categories; refusing implicit routing")
    return (contract_values == "Revolving loans").astype(np.int8)


def equal_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a + b) / 2.0).astype(np.float32)


def _lgb_model(seed: int, iterations: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=10.0,
        max_bin=255,
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def _xgb_model(seed: int, iterations: int, stopping: bool) -> xgb.XGBClassifier:
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


def train_predict_lgb(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    inner_fit_idx, inner_valid_idx = next(
        inner.split(np.zeros(len(fit_rows)), y[fit_rows])
    )
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]

    x_inner_fit, x_inner_valid = phase38.make_matrices(
        blocks, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = _lgb_model(seed, MAX_ITERATIONS)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_valid, y[inner_valid])],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iterations = max(100, int(selector.best_iteration_ or MAX_ITERATIONS))
    del selector, x_inner_fit, x_inner_valid

    x_fit, x_apply = phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    model = _lgb_model(seed, iterations)
    model.fit(x_fit, y[fit_rows])
    prediction = model.predict_proba(x_apply)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("LightGBM produced nonfinite predictions")
    return prediction.astype(np.float32), iterations


def train_predict_xgb(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    inner_fit_idx, inner_valid_idx = next(
        inner.split(np.zeros(len(fit_rows)), y[fit_rows])
    )
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]

    x_inner_fit, x_inner_valid = phase38.make_matrices(
        blocks, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = _xgb_model(seed, MAX_ITERATIONS, stopping=True)
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
    model = _xgb_model(seed, iterations, stopping=False)
    model.fit(x_fit, y[fit_rows], verbose=False)
    prediction = model.predict_proba(x_apply)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("XGBoost produced nonfinite predictions")
    return prediction.astype(np.float32), iterations


def _metric_record(
    y: np.ndarray,
    prediction: np.ndarray,
    folds_indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    fold_aucs = [
        float(roc_auc_score(y[valid], prediction[valid]))
        for _, valid in folds_indices
    ]
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "pooled_log_loss": float(log_loss(y, prediction)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
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
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, (0, 1)).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    gate = make_contract_gate(raw)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)

    # Prediction buffers
    preds: dict[str, np.ndarray] = {
        "lgb_global": np.full(len(y), np.nan, dtype=np.float32),
        "lgb_contract_hard": np.full(len(y), np.nan, dtype=np.float32),
        "lgb_contract_mixture_50_50": np.full(len(y), np.nan, dtype=np.float32),
        "xgb_global": np.full(len(y), np.nan, dtype=np.float32),
        "xgb_contract_hard": np.full(len(y), np.nan, dtype=np.float32),
        "xgb_contract_mixture_50_50": np.full(len(y), np.nan, dtype=np.float32),
        "cross_model_global_blend": np.full(len(y), np.nan, dtype=np.float32),
        "cross_model_expert_blend": np.full(len(y), np.nan, dtype=np.float32),
    }

    records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        print(f"\n--- Fold {fold_number}/{FOLDS} (Fit rows: {len(fit_rows):,}, Valid rows: {len(valid_rows):,}) ---", flush=True)

        # 1. LightGBM Global
        t0 = time.perf_counter()
        lgb_glob, lgb_glob_iter = train_predict_lgb(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        preds["lgb_global"][valid_rows] = lgb_glob
        lgb_auc = roc_auc_score(y[valid_rows], lgb_glob)
        print(f"  [LGBM Global]  AUC: {lgb_auc:.6f} ({lgb_glob_iter} trees, {time.perf_counter()-t0:.1f}s)", flush=True)

        # 2. LightGBM Contract Experts
        t0 = time.perf_counter()
        lgb_hard = np.full(len(valid_rows), np.nan, dtype=np.float32)
        for g in (0, 1):
            g_fit = fit_rows[gate[fit_rows] == g]
            g_valid_mask = gate[valid_rows] == g
            g_valid = valid_rows[g_valid_mask]
            pred_exp, it_exp = train_predict_lgb(
                blocks, g_fit, g_valid, y, fold_seed + g + 1
            )
            lgb_hard[g_valid_mask] = pred_exp
        preds["lgb_contract_hard"][valid_rows] = lgb_hard
        preds["lgb_contract_mixture_50_50"][valid_rows] = equal_blend(lgb_glob, lgb_hard)
        lgb_mix_auc = roc_auc_score(y[valid_rows], preds["lgb_contract_mixture_50_50"][valid_rows])
        print(f"  [LGBM Expert]  Hard AUC: {roc_auc_score(y[valid_rows], lgb_hard):.6f} | 50/50 Mix AUC: {lgb_mix_auc:.6f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        # 3. XGBoost Global
        t0 = time.perf_counter()
        xgb_glob, xgb_glob_iter = train_predict_xgb(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        preds["xgb_global"][valid_rows] = xgb_glob
        xgb_auc = roc_auc_score(y[valid_rows], xgb_glob)
        print(f"  [XGB  Global]  AUC: {xgb_auc:.6f} ({xgb_glob_iter} trees, {time.perf_counter()-t0:.1f}s)", flush=True)

        # 4. XGBoost Contract Experts
        t0 = time.perf_counter()
        xgb_hard = np.full(len(valid_rows), np.nan, dtype=np.float32)
        for g in (0, 1):
            g_fit = fit_rows[gate[fit_rows] == g]
            g_valid_mask = gate[valid_rows] == g
            g_valid = valid_rows[g_valid_mask]
            pred_exp, it_exp = train_predict_xgb(
                blocks, g_fit, g_valid, y, fold_seed + g + 1
            )
            xgb_hard[g_valid_mask] = pred_exp
        preds["xgb_contract_hard"][valid_rows] = xgb_hard
        preds["xgb_contract_mixture_50_50"][valid_rows] = equal_blend(xgb_glob, xgb_hard)
        xgb_mix_auc = roc_auc_score(y[valid_rows], preds["xgb_contract_mixture_50_50"][valid_rows])
        print(f"  [XGB  Expert]  Hard AUC: {roc_auc_score(y[valid_rows], xgb_hard):.6f} | 50/50 Mix AUC: {xgb_mix_auc:.6f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        # 5. Cross-Architecture Blends
        glob_blend = equal_blend(lgb_glob, xgb_glob)
        exp_blend = equal_blend(preds["lgb_contract_mixture_50_50"][valid_rows], preds["xgb_contract_mixture_50_50"][valid_rows])
        preds["cross_model_global_blend"][valid_rows] = glob_blend
        preds["cross_model_expert_blend"][valid_rows] = exp_blend
        print(f"  --> Cross Global Blend AUC: {roc_auc_score(y[valid_rows], glob_blend):.6f}", flush=True)
        print(f"  --> Cross Expert Blend AUC: {roc_auc_score(y[valid_rows], exp_blend):.6f}", flush=True)

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE48_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE48_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE48_SUMMARY.json"

    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{f"oof_{name}": values for name, values in preds.items()},
        }
    )
    oof_frame.to_csv(oof_path, index=False)

    metrics = {
        name: _metric_record(y, values, folds)
        for name, values in preds.items()
    }

    summary: dict[str, Any] = {
        "phase": "Phase 48 multi-architecture contract experts & cross-model synthesis",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "models": {
            "lightgbm": "num_leaves=31, lr=0.03, colsample=0.8, subsample=0.8",
            "xgboost": "max_depth=4, lr=0.035, colsample=0.82, subsample=0.82",
        },
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "gate": "contract_type: Cash loans=0, Revolving loans=1",
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "gate_membership": "deterministic supplied covariates only; no TARGET-based routing",
            "expert_training": "each expert fit only on its outer-training subgroup",
            "iteration_selection": "inner split within the expert's outer-training subgroup",
            "outer_validation": "never used for fitting, early stopping, or gate selection",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 90)
    print("  PHASE 48 COMPLETE: POOLED OOF METRICS")
    print("=" * 90)
    ref_auc = metrics["xgb_global"]["pooled_auc"]
    for name, m in metrics.items():
        delta_auc = (m["pooled_auc"] - ref_auc) * 10000.0
        print(f"  {name:<32} Pooled AUC: {m['pooled_auc']:.6f} ({delta_auc:+.1f} bps) | LogLoss: {m['pooled_log_loss']:.6f} | Mean Fold: {m['mean_fold_auc']:.6f} (±{m['std_fold_auc']:.6f})")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase48_cross_experts_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)

