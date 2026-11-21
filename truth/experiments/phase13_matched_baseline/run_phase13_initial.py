from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier, early_stopping
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from phase13_features import (
    ID_COLUMN,
    PRODUCT_KEYS,
    TARGET_COLUMN,
    build_label_free_decompositions,
    build_product_keys,
    build_raw_feature_blocks,
    category_support,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
PRIMARY_SEED = 20261113
SECONDARY_SEED = 20261114
OUTER_FOLDS = 5
INNER_FOLDS = 3
MAX_ITERATIONS = 1200
EARLY_STOPPING_ROUNDS = 80
BOOTSTRAP_REPETITIONS = 1000
CONTROL_SEEDS = (20261121, 20261122, 20261123, 20261124, 20261125)
CONFIGS = (
    {"name": "leaves31_child80", "num_leaves": 31, "min_child_samples": 80},
    {"name": "leaves63_child100", "num_leaves": 63, "min_child_samples": 100},
)
VARIANTS = (
    "A_raw_numeric",
    "B_raw_numeric_categories",
    "C_complete_raw_rowwise",
    "D_C_plus_keys_ohe",
    "E_C_plus_keys_frequency_ohe",
    "F_C_plus_supported_numeric_decompositions",
)
KEY_GROUPS: dict[str, tuple[str, ...]] = {
    "all_eight": PRODUCT_KEYS,
    "single_Q3": ("PRODUCT_KEY_Q3",),
    "single_Q4": ("PRODUCT_KEY_Q4",),
    "single_Q5": ("PRODUCT_KEY_Q5",),
    "single_KQ3": ("PRODUCT_KEY_KQ3",),
    "single_KQ4": ("PRODUCT_KEY_KQ4",),
    "single_P4": ("PRODUCT_KEY_P4",),
    "single_P5": ("PRODUCT_KEY_P5",),
    "single_AG3": ("PRODUCT_KEY_AG3",),
    "leave_out_Q3": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_Q3"),
    "leave_out_Q4": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_Q4"),
    "leave_out_Q5": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_Q5"),
    "leave_out_KQ3": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_KQ3"),
    "leave_out_KQ4": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_KQ4"),
    "leave_out_P4": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_P4"),
    "leave_out_P5": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_P5"),
    "leave_out_AG3": tuple(k for k in PRODUCT_KEYS if k != "PRODUCT_KEY_AG3"),
    "preregistered_Q3_Q4_Q5": (
        "PRODUCT_KEY_Q3",
        "PRODUCT_KEY_Q4",
        "PRODUCT_KEY_Q5",
    ),
    "preregistered_P4_P5_Q4": (
        "PRODUCT_KEY_P4",
        "PRODUCT_KEY_P5",
        "PRODUCT_KEY_Q4",
    ),
    "preregistered_P4_Q4_AG3": (
        "PRODUCT_KEY_P4",
        "PRODUCT_KEY_Q4",
        "PRODUCT_KEY_AG3",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_mb() -> float:
    if os.name != "nt":
        raise OSError("Phase 13 memory telemetry currently requires Windows")
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_process = kernel32.GetCurrentProcess
    get_process.argtypes = []
    get_process.restype = ctypes.c_void_p
    get_info = psapi.GetProcessMemoryInfo
    get_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    get_info.restype = ctypes.c_int
    if not get_info(get_process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return float(counters.PeakWorkingSetSize / (1024**2))


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase13")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(output_dir / "EXECUTION_LOG.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def make_model(config: dict[str, Any], seed: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=int(config["num_leaves"]),
        max_depth=-1,
        min_child_samples=int(config["min_child_samples"]),
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return csr_matrix(values)


def _encode_categories(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    columns: tuple[str, ...],
    minimum_frequency: int | None,
) -> tuple[csr_matrix, csr_matrix, int]:
    if not columns:
        return (
            csr_matrix((len(fit), 0), dtype=np.float32),
            csr_matrix((len(apply), 0), dtype=np.float32),
            0,
        )
    encoder = OneHotEncoder(
        handle_unknown=(
            "infrequent_if_exist" if minimum_frequency is not None else "ignore"
        ),
        min_frequency=minimum_frequency,
        dtype=np.float32,
        sparse_output=True,
    )
    fit_matrix = encoder.fit_transform(fit.loc[:, columns].astype(str)).tocsr()
    apply_matrix = encoder.transform(apply.loc[:, columns].astype(str)).tocsr()
    return fit_matrix, apply_matrix, int(fit_matrix.shape[1])


def make_matrices(
    indices_fit: np.ndarray,
    indices_apply: np.ndarray,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rowwise: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    variant: str,
    key_columns: tuple[str, ...] = PRODUCT_KEYS,
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    fit_blocks: list[csr_matrix] = []
    apply_blocks: list[csr_matrix] = []
    fit_numeric = numeric.iloc[indices_fit].reset_index(drop=True).copy()
    apply_numeric = numeric.iloc[indices_apply].reset_index(drop=True).copy()
    fit_categories: list[str] = []
    minimum_key_count: int | None = None

    if variant in {"A_raw_numeric", "B_raw_numeric_categories"}:
        fit_blocks.append(_numeric_matrix(fit_numeric))
        apply_blocks.append(_numeric_matrix(apply_numeric))
    else:
        fit_numeric = pd.concat(
            [fit_numeric, rowwise.iloc[indices_fit].reset_index(drop=True)], axis=1
        )
        apply_numeric = pd.concat(
            [apply_numeric, rowwise.iloc[indices_apply].reset_index(drop=True)], axis=1
        )
        fit_blocks.append(_numeric_matrix(fit_numeric))
        apply_blocks.append(_numeric_matrix(apply_numeric))

    if variant != "A_raw_numeric":
        fit_categories = list(categorical.columns)
        fit_cat, apply_cat, raw_category_count = _encode_categories(
            categorical.iloc[indices_fit],
            categorical.iloc[indices_apply],
            tuple(fit_categories),
            None,
        )
        fit_blocks.append(fit_cat)
        apply_blocks.append(apply_cat)
    else:
        raw_category_count = 0

    encoded_key_count = 0
    frequency_count = 0
    if variant in {"D_C_plus_keys_ohe", "E_C_plus_keys_frequency_ohe"}:
        minimum_key_count = 2 if variant.startswith("E_") else None
        fit_key, apply_key, encoded_key_count = _encode_categories(
            keys.iloc[indices_fit],
            keys.iloc[indices_apply],
            key_columns,
            minimum_key_count,
        )
        fit_blocks.append(fit_key)
        apply_blocks.append(apply_key)
        if variant.startswith("E_"):
            fit_frequency = keys.iloc[indices_fit][list(key_columns)].astype(str).copy()
            apply_frequency = keys.iloc[indices_apply][list(key_columns)].astype(str).copy()
            frequency_rows = []
            apply_frequency_rows = []
            for column in key_columns:
                counts = fit_frequency[column].value_counts()
                frequency_rows.append(
                    fit_frequency[column].map(counts).to_numpy(dtype=np.float32)
                )
                apply_frequency_rows.append(
                    apply_frequency[column]
                    .map(counts)
                    .fillna(0)
                    .to_numpy(dtype=np.float32)
                )
            fit_counts = pd.DataFrame(
                np.column_stack(frequency_rows),
                columns=[f"FIT_COUNT_{column}" for column in key_columns],
            )
            apply_counts = pd.DataFrame(
                np.column_stack(apply_frequency_rows),
                columns=[f"FIT_COUNT_{column}" for column in key_columns],
            )
            fit_blocks.append(_numeric_matrix(fit_counts))
            apply_blocks.append(_numeric_matrix(apply_counts))
            frequency_count = len(key_columns)

    if variant == "F_C_plus_supported_numeric_decompositions":
        fit_blocks.append(_numeric_matrix(decompositions.iloc[indices_fit]))
        apply_blocks.append(_numeric_matrix(decompositions.iloc[indices_apply]))

    fit_matrix = hstack(fit_blocks, format="csr")
    apply_matrix = hstack(apply_blocks, format="csr")
    return fit_matrix, apply_matrix, {
        "feature_count": int(fit_matrix.shape[1]),
        "fit_nonzero_count": int(fit_matrix.nnz),
        "apply_nonzero_count": int(apply_matrix.nnz),
        "raw_category_feature_count": raw_category_count,
        "encoded_key_feature_count": encoded_key_count,
        "frequency_feature_count": frequency_count,
    }


def outer_splits(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y), dtype=np.int8), y))


def inner_splits(
    outer_fit: np.ndarray, y: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return [
        (outer_fit[fit], outer_fit[apply])
        for fit, apply in splitter.split(
            np.zeros(len(outer_fit), dtype=np.int8), y[outer_fit]
        )
    ]


def _model_seed(seed: int, outer_fold: int, inner_fold: int, config: int) -> int:
    return int(seed + outer_fold * 10000 + inner_fold * 100 + config)


def _early_stopped_predict(
    fit_matrix: csr_matrix,
    apply_matrix: csr_matrix,
    y_fit: np.ndarray,
    y_apply: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int, float]:
    model = make_model(config, seed, MAX_ITERATIONS)
    started = time.perf_counter()
    model.fit(
        fit_matrix,
        y_fit,
        eval_X=apply_matrix,
        eval_y=y_apply,
        eval_metric="auc",
        callbacks=[
            early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            )
        ],
    )
    return (
        model.predict_proba(apply_matrix)[:, 1],
        int(model.best_iteration_ or MAX_ITERATIONS),
        time.perf_counter() - started,
    )


def _fit_predict(
    fit_matrix: csr_matrix,
    apply_matrix: csr_matrix,
    y_fit: np.ndarray,
    config: dict[str, Any],
    seed: int,
    rounds: int,
) -> tuple[np.ndarray, float]:
    model = make_model(config, seed, rounds)
    started = time.perf_counter()
    model.fit(fit_matrix, y_fit)
    return model.predict_proba(apply_matrix)[:, 1], time.perf_counter() - started


def tune_variant(
    variant: str,
    outer_fit: np.ndarray,
    y: np.ndarray,
    blocks: tuple[pd.DataFrame, ...],
    seed: int,
    fold: int,
    support_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], int, dict[str, float], float]:
    numeric, categorical, rowwise, keys, decompositions = blocks
    config_oof = {
        str(config["name"]): np.full(len(outer_fit), np.nan, dtype=np.float64)
        for config in CONFIGS
    }
    rounds: dict[str, list[int]] = {name: [] for name in config_oof}
    runtimes = {name: 0.0 for name in config_oof}
    for inner_fold, (fit, apply) in enumerate(
        inner_splits(outer_fit, y, seed + fold), start=1
    ):
        fit_matrix, apply_matrix, details = make_matrices(
            fit,
            apply,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            variant,
        )
        if variant in {"D_C_plus_keys_ohe", "E_C_plus_keys_frequency_ohe"}:
            support_rows.extend(
                category_support(
                    keys.iloc[fit],
                    keys.iloc[apply],
                    PRODUCT_KEYS,
                    f"outer{fold}_inner{inner_fold}",
                    2 if variant.startswith("E_") else 1,
                    variant,
                )
            )
        for config_index, config in enumerate(CONFIGS, start=1):
            prediction, best_rounds, elapsed = _early_stopped_predict(
                fit_matrix,
                apply_matrix,
                y[fit],
                y[apply],
                config,
                _model_seed(seed, fold, inner_fold, config_index),
            )
            local_apply = np.searchsorted(outer_fit, apply)
            config_oof[str(config["name"])][local_apply] = prediction
            rounds[str(config["name"])].append(best_rounds)
            runtimes[str(config["name"])] += elapsed
            selection_rows.append(
                {
                    "seed": seed,
                    "outer_fold": fold,
                    "inner_fold": inner_fold,
                    "variant": variant,
                    "config": config["name"],
                    "inner_apply_auc": float(roc_auc_score(y[apply], prediction)),
                    "best_iteration": best_rounds,
                    "runtime_seconds": elapsed,
                    "feature_count": details["feature_count"],
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
        logger.info(
            "inner fit seed=%s outer=%s/%s inner=%s/%s variant=%s features=%s",
            seed,
            fold,
            OUTER_FOLDS,
            inner_fold,
            INNER_FOLDS,
            variant,
            details["feature_count"],
        )
    scores = {
        name: float(roc_auc_score(y[outer_fit], prediction))
        for name, prediction in config_oof.items()
    }
    best_config_name = max(scores, key=scores.get)
    best_config = next(c for c in CONFIGS if c["name"] == best_config_name)
    best_rounds = max(1, int(np.rint(np.median(rounds[best_config_name]))))
    for config in CONFIGS:
        name = str(config["name"])
        selection_rows.append(
            {
                "seed": seed,
                "outer_fold": fold,
                "inner_fold": "pooled_inner_oof",
                "variant": variant,
                "config": name,
                "inner_apply_auc": scores[name],
                "best_iteration": int(np.rint(np.median(rounds[name]))),
                "runtime_seconds": runtimes[name],
                "feature_count": None,
                "peak_working_set_mb": peak_working_set_mb(),
                "selected": name == best_config_name,
            }
        )
    return best_config, best_rounds, scores, sum(runtimes.values())


def run_main(
    y: np.ndarray,
    blocks: tuple[pd.DataFrame, ...],
    variants: tuple[str, ...],
    seed: int,
    logger: logging.Logger,
    support_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, np.ndarray],
    dict[int, dict[str, tuple[dict[str, Any], int]]],
    list[dict[str, Any]],
]:
    numeric, categorical, rowwise, keys, decompositions = blocks
    predictions = {name: np.full(len(y), np.nan) for name in variants}
    selected: dict[int, dict[str, tuple[dict[str, Any], int]]] = {}
    fit_rows: list[dict[str, Any]] = []
    for fold, (outer_fit, outer_apply) in enumerate(outer_splits(y, seed), start=1):
        selected[fold] = {}
        for variant in variants:
            config, rounds, inner_scores, tuning_runtime = tune_variant(
                variant,
                outer_fit,
                y,
                blocks,
                seed,
                fold,
                support_rows,
                selection_rows,
                logger,
            )
            selected[fold][variant] = (config, rounds)
            fit_matrix, apply_matrix, details = make_matrices(
                outer_fit,
                outer_apply,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                variant,
            )
            if variant in {"D_C_plus_keys_ohe", "E_C_plus_keys_frequency_ohe"}:
                support_rows.extend(
                    category_support(
                        keys.iloc[outer_fit],
                        keys.iloc[outer_apply],
                        PRODUCT_KEYS,
                        f"outer{fold}",
                        2 if variant.startswith("E_") else 1,
                        variant,
                    )
                )
            pred, fit_runtime = _fit_predict(
                fit_matrix,
                apply_matrix,
                y[outer_fit],
                config,
                seed + fold * 1000 + 500,
                rounds,
            )
            predictions[variant][outer_apply] = pred
            fit_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "variant": variant,
                    "config": config["name"],
                    "selected_rounds": rounds,
                    "inner_oof_config_auc": json.dumps(inner_scores, sort_keys=True),
                    "inner_tuning_runtime_seconds": tuning_runtime,
                    "outer_fit_runtime_seconds": fit_runtime,
                    **details,
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
            logger.info(
                "outer fit seed=%s fold=%s/%s variant=%s auc=%.6f rounds=%s "
                "features=%s runtime=%.1fs",
                seed,
                fold,
                OUTER_FOLDS,
                variant,
                roc_auc_score(y[outer_apply], pred),
                rounds,
                details["feature_count"],
                fit_runtime,
            )
    return predictions, selected, fit_rows


def run_ablations(
    y: np.ndarray,
    blocks: tuple[pd.DataFrame, ...],
    selected: dict[int, dict[str, tuple[dict[str, Any], int]]],
    seed: int,
    logger: logging.Logger,
    support_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    numeric, categorical, rowwise, keys, decompositions = blocks
    predictions = {name: np.full(len(y), np.nan) for name in KEY_GROUPS}
    rows: list[dict[str, Any]] = []
    for fold, (outer_fit, outer_apply) in enumerate(outer_splits(y, seed), start=1):
        config, rounds = selected[fold]["D_C_plus_keys_ohe"]
        for name, key_columns in KEY_GROUPS.items():
            fit_matrix, apply_matrix, details = make_matrices(
                outer_fit,
                outer_apply,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                "D_C_plus_keys_ohe",
                key_columns,
            )
            support_rows.extend(
                category_support(
                    keys.iloc[outer_fit],
                    keys.iloc[outer_apply],
                    key_columns,
                    f"ablation_outer{fold}",
                    1,
                    f"ablation_{name}",
                )
            )
            pred, elapsed = _fit_predict(
                fit_matrix,
                apply_matrix,
                y[outer_fit],
                config,
                seed + fold * 5000 + len(name),
                rounds,
            )
            predictions[name][outer_apply] = pred
            rows.append(
                {
                    "variant": name,
                    "fold": fold,
                    "key_count": len(key_columns),
                    "keys": "|".join(key_columns),
                    "fold_auc": float(roc_auc_score(y[outer_apply], pred)),
                    "config": config["name"],
                    "rounds": rounds,
                    "runtime_seconds": elapsed,
                    **details,
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
            logger.info(
                "ablation fold=%s/%s variant=%s auc=%.6f",
                fold,
                OUTER_FOLDS,
                name,
                roc_auc_score(y[outer_apply], pred),
            )
    result = pd.DataFrame(rows)
    pooled = {name: roc_auc_score(y, pred) for name, pred in predictions.items()}
    result["pooled_oof_auc"] = result["variant"].map(pooled)
    all_fold_auc = result.loc[result["variant"] == "all_eight"].set_index("fold")[
        "fold_auc"
    ]
    result["paired_fold_delta_vs_all_eight"] = result.apply(
        lambda row: row["fold_auc"] - float(all_fold_auc.loc[row["fold"]]), axis=1
    )
    return predictions, result


def run_controls(
    y: np.ndarray,
    blocks: tuple[pd.DataFrame, ...],
    selected: dict[int, dict[str, tuple[dict[str, Any], int]]],
    complete_raw_oof: np.ndarray,
    genuine_oof: np.ndarray,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    numeric, categorical, rowwise, keys, decompositions = blocks
    rows: list[dict[str, Any]] = []
    control_oof: dict[str, np.ndarray] = {}
    for repetition, control_seed in enumerate(CONTROL_SEEDS, start=1):
        oof = np.full(len(y), np.nan)
        for fold, (outer_fit, outer_apply) in enumerate(outer_splits(y, seed), start=1):
            rng = np.random.default_rng(control_seed + fold)
            fit_keys = keys.iloc[outer_fit].loc[:, PRODUCT_KEYS].copy().reset_index(drop=True)
            apply_keys = keys.iloc[outer_apply].loc[:, PRODUCT_KEYS].copy().reset_index(drop=True)
            for column in PRODUCT_KEYS:
                fit_values = fit_keys[column].astype(str).to_numpy(copy=True)
                apply_values = apply_keys[column].astype(str).to_numpy(copy=True)
                rng.shuffle(fit_values)
                rng.shuffle(apply_values)
                fit_keys[column] = fit_values
                apply_keys[column] = apply_values
            config, rounds = selected[fold]["D_C_plus_keys_ohe"]
            fit_base, apply_base, details = make_matrices(
                outer_fit,
                outer_apply,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                "C_complete_raw_rowwise",
            )
            key_fit, key_apply, key_count = _encode_categories(
                fit_keys, apply_keys, PRODUCT_KEYS, None
            )
            fit_matrix = hstack([fit_base, key_fit], format="csr")
            apply_matrix = hstack([apply_base, key_apply], format="csr")
            pred, elapsed = _fit_predict(
                fit_matrix,
                apply_matrix,
                y[outer_fit],
                config,
                control_seed + fold * 5000,
                rounds,
            )
            oof[outer_apply] = pred
            auc = float(roc_auc_score(y[outer_apply], pred))
            rows.append(
                {
                    "control": "frequency_preserving_shuffled_keys",
                    "repetition": repetition,
                    "seed": control_seed,
                    "fold": fold,
                    "fold_auc": auc,
                    "paired_delta_vs_C": auc
                    - float(roc_auc_score(y[outer_apply], complete_raw_oof[outer_apply])),
                    "paired_delta_vs_genuine_D": auc
                    - float(roc_auc_score(y[outer_apply], genuine_oof[outer_apply])),
                    "runtime_seconds": elapsed,
                    "feature_count": fit_matrix.shape[1],
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
            logger.info(
                "negative control repetition=%s seed=%s fold=%s/%s auc=%.6f",
                repetition,
                control_seed,
                fold,
                OUTER_FOLDS,
                auc,
            )
        if not np.isfinite(oof).all():
            raise AssertionError("A negative-control OOF prediction is missing")
        for row in rows:
            if row["repetition"] == repetition:
                row["pooled_oof_auc"] = float(roc_auc_score(y, oof))
        control_oof[f"control_{repetition}"] = oof
    return pd.DataFrame(rows), control_oof


def paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)

    def ranks(scores: np.ndarray) -> tuple[np.ndarray, int]:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
        ids_ordered = np.repeat(
            np.arange(len(starts), dtype=np.int32),
            np.diff(np.r_[starts, len(scores)]),
        )
        ids = np.empty(len(scores), dtype=np.int32)
        ids[order] = ids_ordered
        return ids, len(starts)

    base_ids, base_groups = ranks(baseline)
    cand_ids, cand_groups = ranks(candidate)
    deltas = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for i in range(BOOTSTRAP_REPETITIONS):
        pos = rng.choice(positive, size=len(positive), replace=True)
        neg = rng.choice(negative, size=len(negative), replace=True)
        aucs = []
        for ids, groups in ((base_ids, base_groups), (cand_ids, cand_groups)):
            pos_counts = np.bincount(ids[pos], minlength=groups)
            neg_counts = np.bincount(ids[neg], minlength=groups)
            below = np.cumsum(neg_counts) - neg_counts
            aucs.append(
                np.sum(pos_counts * (below + 0.5 * neg_counts), dtype=np.float64)
                / (len(pos) * len(neg))
            )
        deltas[i] = aucs[1] - aucs[0]
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def formula_provenance_audit() -> str:
    return """# Phase 13 formula provenance audit

## Scope and evidence searched

The audit reviewed the local competition materials review, compliance matrix,
Phase 9 final execution plan/results and fee-gap audit, the independent Phase
10 product-key specification, Phase 11 results, Phase 12 formula audit, and
the current master ledger. No external web material, external dataset, or
undocumented source repository was used to manufacture provenance.

## Historical constant 130.5

**Status: UNRESOLVED; not an established fee.** The Phase 9 adjacent-gap audit
found 6,792 of 27,255 adjacent annuity gaps (24.9%) equal to 130.5 within exact
cash-loan (goods-price, credit) groups, with nearby placebo gaps rare/absent.
That is evidence of a repeated numeric gap in the supplied training rows only.
It does not establish a recurring charge, fee, repayment schedule, contractual
rule, causal mechanism, or organizer permission. The independent Phase 10
specification explicitly labels 130.5 an exploratory hypothesis. P4/P5 remain
research-only / eligibility-pending.

## Historical constant 0.0066

**Status: UNRESOLVED; not an established financing rate or tier.** The
independent Phase 10 specification records 0.0066 only as an exploratory
normalization step in a cash-loan credit-to-goods coordinate. No derivation
from an official rule, supplied data dictionary, or permitted independent
source was located in the inspected repository materials. The coordinate may
be evaluated as a predictive transformation but must not be described as a
real interest rate, rate step, payment term, financing multiplier, or
contract attribute. KQ3/KQ4 remain research-only / eligibility-pending.

## Phase 13 handling and eligibility

The matched-baseline models D/E intentionally reproduce the historical keys
to measure their incremental predictive value; their use does not certify
eligibility. Track F excludes both undocumented constants. It adds only the
rowwise, label-free values `AMT_ANNUITY / AMT_GOODS_PRICE`,
`AMT_CREDIT - AMT_GOODS_PRICE`, and `AMT_ANNUITY - AMT_CREDIT`. The local
competition audit supports deterministic arithmetic from supplied columns in
general, but no organizer approval for these specific engineered fields is
asserted. All transformations fit or calculate only within supplied training
rows; no test rows are opened.
"""


def _eligibility_matrix() -> pd.DataFrame:
    rows = [
        ("Raw supplied numeric predictors", "label-free", "No learned transform; model fit uses outer-fit labels only", "No", "No", "Local course README: supplied train/test only; external data prohibited", "No separate organizer ruling"),
        ("Raw supplied categorical predictors", "label-free", "One-hot categories fit on each training partition; unknown apply values ignored", "No", "No", "Local course README: supplied train/test only", "No separate organizer ruling"),
        ("Rowwise missingness/sentinel/age features", "label-free", "Deterministic rowwise; no pooled statistics", "No", "No", "Local course tutorial permits constructing features from supplied columns", "365243 treated as numeric sentinel only; no semantic claim"),
        ("Raw financial rowwise ratios", "label-free", "Deterministic rowwise, guarded denominators", "No", "No", "Local competition audit says deterministic formulas from provided columns are allowed", "General rule support is not organizer certification"),
        ("Q3/Q4/Q5/AG3 product keys", "label-free", "Rowwise formula; OHE fit on train folds", "No", "No", "Supplied-column mathematical features supported generally by tutorial guidance", "Specific categorical discretization not explicitly ruled on"),
        ("P4/P5 product keys", "label-free arithmetic; fixed constant", "Rowwise formula; OHE fit on train folds", "No", "No", "General formula guidance only", "130.5 provenance unresolved; not an established fee"),
        ("KQ3/KQ4 product keys", "label-free arithmetic; fixed constant", "Rowwise formula; OHE fit on train folds", "No", "No", "General formula guidance only", "0.0066 provenance unresolved; not an established rate/tier"),
        ("Frequency-supported keys", "label-free", "Counts and encoder fitted on each inner/outer fit partition only", "No", "No", "Uses only supplied training partition", "Eligibility inherits source-formula limitations"),
        ("Track F direct numeric components", "label-free", "Rowwise; no pooled fitting", "No", "No", "General supplied-column formula guidance", "No organizer approval claimed"),
        ("Target encoding / target-rate hidden-key scan", "label-dependent", "Separate exploratory nested fit-fold-only scan; never used in A-F", "No", "No", "Local rules audit says nested target encoding is unresolved by README", "Exploratory; not Phase 13 eligible track and not promotable"),
        ("CatBoost CTRs", "label-dependent internal category statistics", "Not used in Phase 13", "No", "No", "Not explicitly settled in competition materials", "Excluded; eligibility unresolved"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "feature_or_transform",
            "label_status",
            "fit_scope",
            "test_information",
            "external_information",
            "official_rule_support",
            "unresolved_interpretation",
        ],
    )


def _write_report(
    out: Path,
    summary: pd.DataFrame,
    incremental: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ablation: pd.DataFrame,
    controls: pd.DataFrame,
    secondary: pd.DataFrame,
    input_path: Path,
    input_hash: str,
    start: datetime,
    finish: datetime,
    seed: int,
) -> None:
    by_name = summary.set_index("variant")
    key_increment = incremental[
        incremental["reference_variant"] == "C_complete_raw_rowwise"
    ]
    best_increment = key_increment.sort_values(
        "delta_vs_reference_pooled_auc", ascending=False
    ).iloc[0]
    best_single = (
        ablation[ablation["variant"].str.startswith("single_")]
        .drop_duplicates("variant")
        .sort_values("pooled_oof_auc", ascending=False)
        .iloc[0]
    )
    controls_mean = controls.groupby("control")["pooled_oof_auc"].mean().to_dict()
    lines = [
        "# Phase 13 matched-baseline execution report",
        "",
        f"- Run folder: `{out}`",
        f"- UTC interval: `{start.isoformat()}` to `{finish.isoformat()}`",
        f"- Training file: `{input_path}`; SHA-256 `{input_hash}`.",
        f"- Rows: {int(summary['n_rows'].iloc[0]):,}; primary seed: {seed}; 5 stratified outer folds; 3 inner folds.",
        "- CPU LightGBM; all transformations/encoders fit in the relevant training partition. No test, external data, target encoding, target-derived category statistics, holdout, submission, or champion change.",
        "",
        "## Matched primary benchmark",
        "",
        "| Variant | Pooled OOF AUC | Delta vs A | Positive folds vs A |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['variant']} | {row['pooled_oof_auc']:.6f} | "
            f"{row['delta_vs_A_pooled_auc']:+.6f} | "
            f"{int(row['positive_folds_vs_A'])}/5 |"
        )
    lines.extend(
        [
            "",
            "## Product-key incremental value over complete raw baseline C",
            "",
            f"- D: C plus ordinary one-hot keys: {by_name.loc['D_C_plus_keys_ohe', 'pooled_oof_auc']:.6f}; delta vs C {by_name.loc['D_C_plus_keys_ohe', 'delta_vs_C_pooled_auc']:+.6f}.",
            f"- E: C plus fit-fold frequency-supported one-hot keys: {by_name.loc['E_C_plus_keys_frequency_ohe', 'pooled_oof_auc']:.6f}; delta vs C {by_name.loc['E_C_plus_keys_frequency_ohe', 'delta_vs_C_pooled_auc']:+.6f}.",
            f"- Highest primary feature set by incremental OOF AUC over C was `{best_increment['variant']}` ({best_increment['delta_vs_reference_pooled_auc']:+.6f}); primary fold and bootstrap results are also reported.",
            "- Compare `MATCHED_BASELINE_RESULTS.csv`, `PRODUCT_KEY_INCREMENTAL_VALUE.csv`, and `PAIRED_BOOTSTRAP_REPORT.md`. These are selection-sensitive CV estimates on previously examined labels, not independent confirmation.",
            "",
            "## Key ablation and controls",
            "",
            f"- Strongest single-key ablation was `{best_single['variant']}` at pooled OOF AUC {best_single['pooled_oof_auc']:.6f}. All ablations share the D configuration/rounds selected inside each outer fit and remain exploratory.",
            f"- Mean control pooled OOF AUCs: `{json.dumps(controls_mean, sort_keys=True)}`.",
            "- Do not remove any key on pooled rank alone. Inspect every fold and paired contrast.",
            "",
            "## Formula provenance and eligibility",
            "",
            "The 130.5 annuity gap is a repeated numerical gap in the Phase 9 audit, not an established fee. The 0.0066 step has no located permitted-data derivation and is not an established financing rate/tier. Both remain unresolved; D/E reproduce these hypotheses only to measure prediction lift. F excludes both. See `FORMULA_PROVENANCE_AUDIT.md` and `COMPETITION_ELIGIBILITY_MATRIX.csv`.",
            "",
            "## Secondary-seed robustness",
            "",
        ]
    )
    for row in secondary.to_dict(orient="records"):
        lines.append(
            f"- {row['variant']} seed {row['seed']}: AUC {row['pooled_oof_auc']:.6f}, "
            f"delta vs C {row['delta_vs_C']:+.6f}, {row['positive_folds_vs_C']}/5 positive folds."
        )
    lines.extend(
        [
            "This is robustness replication using the same applicants/labels, not independent confirmation.",
            "",
            "## Decisions",
            "",
            "1. True incremental key contribution is the D/E paired contrast against complete raw+rowwise C; report the observed result without comparing it to the numeric-only Phase 12 delta.",
            "2. Frequency support is preferred only if E reliably beats D across folds and the secondary seed; category counts use the fit partition only.",
            "3. Key consistency is descriptive across the fixed-fold ablation, not a basis for a final key subset.",
            "4. Genuine versus shuffled-control performance is a negative-control check, not an untouched validation set.",
            "5. No track is organizer-certified. P4/P5/KQ3/KQ4 remain eligibility-pending; F excludes their undocumented constants.",
            "6. Advance only the best matched feature representation after a separately predeclared robustness replication; do not promote the model from these exposed OOF rows.",
            "",
        ]
    )
    (out / "PHASE13_EXECUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(train_path: Path, output_root: Path, seed: int) -> Path:
    started_at = datetime.now(timezone.utc)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    out = output_root / f"phase13_{stamp}"
    out.mkdir(exist_ok=False)
    logger = configure_logger(out)
    input_hash = sha256_file(train_path)
    logger.info("Starting Phase 13 matched baseline; output=%s", out)
    logger.info(
        "UTC start=%s local=%s; seed=%s; input_sha256=%s",
        started_at.isoformat(),
        started_at.astimezone().isoformat(),
        seed,
        input_hash,
    )
    train = pd.read_csv(train_path, low_memory=False)
    if ID_COLUMN not in train or TARGET_COLUMN not in train:
        raise ValueError(f"Training data must include {ID_COLUMN} and {TARGET_COLUMN}")
    if train[ID_COLUMN].isna().any() or train[ID_COLUMN].duplicated().any():
        raise ValueError("Training IDs must be nonmissing and unique")
    if train[TARGET_COLUMN].isna().any() or set(train[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training target must be complete and binary")
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET_COLUMN])
    numeric, categorical, rowwise = build_raw_feature_blocks(raw)
    keys = build_product_keys(raw)
    decompositions = build_label_free_decompositions(raw)
    blocks = (numeric, categorical, rowwise, keys, decompositions)
    if not all(
        len(frame) == len(train)
        for frame in (numeric, categorical, rowwise, keys, decompositions)
    ):
        raise AssertionError("Feature builders changed the training row count")
    if sha256_file(train_path) != input_hash:
        raise AssertionError("Input file changed after it was read")
    logger.info(
        "Loaded train only rows=%s raw_numeric=%s raw_categorical=%s rowwise=%s "
        "key_count=%s decomposition_count=%s Python=%s pandas=%s numpy=%s "
        "sklearn=%s lightgbm=%s scipy=%s device=CPU n_jobs=4",
        len(train),
        numeric.shape[1],
        categorical.shape[1],
        rowwise.shape[1],
        keys.shape[1],
        decompositions.shape[1],
        platform.python_version(),
        pd.__version__,
        np.__version__,
        sklearn.__version__,
        lightgbm.__version__,
        scipy.__version__,
    )
    logger.info(
        "CONFIGS=%s MAX_ITERATIONS=%s EARLY_STOPPING_ROUNDS=%s OUTER=%s INNER=%s",
        json.dumps(CONFIGS, sort_keys=True),
        MAX_ITERATIONS,
        EARLY_STOPPING_ROUNDS,
        OUTER_FOLDS,
        INNER_FOLDS,
    )

    support_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    primary, selected, outer_rows = run_main(
        y,
        blocks,
        VARIANTS,
        seed,
        logger,
        support_rows,
        selection_rows,
    )
    ablation_predictions, ablation = run_ablations(
        y, blocks, selected, seed, logger, support_rows
    )
    controls, control_predictions = run_controls(
        y,
        blocks,
        selected,
        primary["C_complete_raw_rowwise"],
        primary["D_C_plus_keys_ohe"],
        seed,
        logger,
    )

    def fold_auc_map(name: str, preds: np.ndarray) -> dict[int, float]:
        return {
            fold: float(roc_auc_score(y[apply], preds[apply]))
            for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1)
        }

    base_auc = float(roc_auc_score(y, primary["A_raw_numeric"]))
    complete_auc = float(roc_auc_score(y, primary["C_complete_raw_rowwise"]))
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for variant, prediction in primary.items():
        auc = float(roc_auc_score(y, prediction))
        folds = fold_auc_map(variant, prediction)
        baseline_folds = fold_auc_map("A_raw_numeric", primary["A_raw_numeric"])
        complete_folds = fold_auc_map(
            "C_complete_raw_rowwise", primary["C_complete_raw_rowwise"]
        )
        summary_rows.append(
            {
                "variant": variant,
                "n_rows": len(y),
                "pooled_oof_auc": auc,
                "delta_vs_A_pooled_auc": auc - base_auc,
                "delta_vs_C_pooled_auc": auc - complete_auc,
                "positive_folds_vs_A": sum(
                    folds[f] > baseline_folds[f] for f in folds
                ),
                "positive_folds_vs_C": sum(
                    folds[f] > complete_folds[f] for f in folds
                ),
                "prediction_pearson_vs_C": float(
                    np.corrcoef(primary["C_complete_raw_rowwise"], prediction)[0, 1]
                ),
                "prediction_spearman_vs_C": float(
                    pd.Series(primary["C_complete_raw_rowwise"]).corr(
                        pd.Series(prediction), method="spearman"
                    )
                ),
                "per_fold_auc": json.dumps(folds, sort_keys=True),
                "device": "CPU",
            }
        )
        for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
            fold_rows.append(
                {
                    "experiment_group": "main",
                    "variant": variant,
                    "fold": fold,
                    "auc": folds[fold],
                    "paired_delta_vs_A": folds[fold] - baseline_folds[fold],
                    "paired_delta_vs_C": folds[fold] - complete_folds[fold],
                    "positive_delta_vs_C": folds[fold] > complete_folds[fold],
                    "rows": len(apply),
                }
            )
    summary = pd.DataFrame(summary_rows)

    incremental = pd.DataFrame(
        [
            {
                "comparison": "D_C_plus_keys_ohe_vs_C",
                "variant": "D_C_plus_keys_ohe",
                "reference_variant": "C_complete_raw_rowwise",
                "pooled_oof_auc": float(
                    roc_auc_score(y, primary["D_C_plus_keys_ohe"])
                ),
                "reference_pooled_auc": complete_auc,
                "delta_vs_reference_pooled_auc": float(
                    roc_auc_score(y, primary["D_C_plus_keys_ohe"])
                    - complete_auc
                ),
                "delta_vs_C_pooled_auc": float(
                    roc_auc_score(y, primary["D_C_plus_keys_ohe"]) - complete_auc
                ),
                "positive_folds_vs_reference": int(
                    summary.loc[
                        summary["variant"] == "D_C_plus_keys_ohe",
                        "positive_folds_vs_C",
                    ].iloc[0]
                ),
                "prediction_spearman_vs_reference": float(
                    summary.loc[
                        summary["variant"] == "D_C_plus_keys_ohe",
                        "prediction_spearman_vs_C",
                    ].iloc[0]
                ),
            },
            {
                "comparison": "E_C_plus_keys_frequency_ohe_vs_C",
                "variant": "E_C_plus_keys_frequency_ohe",
                "reference_variant": "C_complete_raw_rowwise",
                "pooled_oof_auc": float(
                    roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"])
                ),
                "reference_pooled_auc": complete_auc,
                "delta_vs_reference_pooled_auc": float(
                    roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"])
                    - complete_auc
                ),
                "delta_vs_C_pooled_auc": float(
                    roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"])
                    - complete_auc
                ),
                "positive_folds_vs_reference": int(
                    summary.loc[
                        summary["variant"] == "E_C_plus_keys_frequency_ohe",
                        "positive_folds_vs_C",
                    ].iloc[0]
                ),
                "prediction_spearman_vs_reference": float(
                    summary.loc[
                        summary["variant"] == "E_C_plus_keys_frequency_ohe",
                        "prediction_spearman_vs_C",
                    ].iloc[0]
                ),
            },
            {
                "comparison": "E_frequency_ohe_vs_D_ordinary_ohe",
                "variant": "E_C_plus_keys_frequency_ohe",
                "reference_variant": "D_C_plus_keys_ohe",
                "pooled_oof_auc": float(
                    roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"])
                ),
                "reference_pooled_auc": float(
                    roc_auc_score(y, primary["D_C_plus_keys_ohe"])
                ),
                "delta_vs_reference_pooled_auc": float(
                    roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"])
                    - roc_auc_score(y, primary["D_C_plus_keys_ohe"])
                ),
                "delta_vs_C_pooled_auc": np.nan,
                "positive_folds_vs_reference": int(
                    sum(
                        roc_auc_score(y[apply], primary["E_C_plus_keys_frequency_ohe"][apply])
                        > roc_auc_score(y[apply], primary["D_C_plus_keys_ohe"][apply])
                        for _, apply in outer_splits(y, seed)
                    )
                ),
                "prediction_spearman_vs_reference": float(
                    pd.Series(primary["E_C_plus_keys_frequency_ohe"]).corr(
                        pd.Series(primary["D_C_plus_keys_ohe"]), method="spearman"
                    )
                ),
            },
        ]
    )
    bootstrap_rows = []
    for name in (
        "B_raw_numeric_categories",
        "C_complete_raw_rowwise",
        "D_C_plus_keys_ohe",
        "E_C_plus_keys_frequency_ohe",
        "F_C_plus_supported_numeric_decompositions",
    ):
        for reference_name in (
            ("A_raw_numeric", "C_complete_raw_rowwise")
            if name in {"D_C_plus_keys_ohe", "E_C_plus_keys_frequency_ohe"}
            else ("A_raw_numeric",)
        ):
            low, high = paired_bootstrap(
                y,
                primary[reference_name],
                primary[name],
                seed + 800000 + len(bootstrap_rows),
            )
            delta = float(
                roc_auc_score(y, primary[name])
                - roc_auc_score(y, primary[reference_name])
            )
            bootstrap_rows.append(
                {
                    "candidate": name,
                    "reference": reference_name,
                    "reference_auc": float(roc_auc_score(y, primary[reference_name])),
                    "candidate_auc": float(roc_auc_score(y, primary[name])),
                    "paired_delta": delta,
                    "bootstrap_95_low": low,
                    "bootstrap_95_high": high,
                    "repetitions": BOOTSTRAP_REPETITIONS,
                    "interpretation": "Conditional on fitted OOF predictions; not independent confirmation.",
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    complete_fold_auc = {
        row["fold"]: row["auc"]
        for row in fold_rows
        if row["experiment_group"] == "main"
        and row["variant"] == "C_complete_raw_rowwise"
    }
    ablation["paired_delta_vs_C"] = ablation.apply(
        lambda row: float(row["fold_auc"] - complete_fold_auc[row["fold"]]),
        axis=1,
    )
    for variant, group in ablation.groupby("variant"):
        pred = ablation_predictions[variant]
        value = float(roc_auc_score(y, pred))
        ablation.loc[ablation["variant"] == variant, "pooled_oof_auc"] = value
    for row in ablation.itertuples():
        fold_rows.append(
            {
                "experiment_group": "product_key_ablation",
                "variant": row.variant,
                "fold": row.fold,
                "auc": row.fold_auc,
                "paired_delta_vs_A": np.nan,
                "paired_delta_vs_C": row.paired_delta_vs_C,
                "positive_delta_vs_C": row.paired_delta_vs_C > 0,
                "rows": "",
            }
        )
    controls["paired_delta_vs_C"] = np.nan
    for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
        base_fold = roc_auc_score(y[apply], primary["C_complete_raw_rowwise"][apply])
        mask = controls["fold"] == fold
        controls.loc[mask, "paired_delta_vs_C"] = controls.loc[mask, "fold_auc"] - base_fold
        for row in controls.loc[mask].itertuples():
            fold_rows.append(
                {
                    "experiment_group": "negative_control",
                    "variant": row.control,
                    "fold": row.fold,
                    "auc": row.fold_auc,
                    "paired_delta_vs_A": np.nan,
                    "paired_delta_vs_C": row.paired_delta_vs_C,
                    "positive_delta_vs_C": row.paired_delta_vs_C > 0,
                    "rows": len(apply),
                }
            )

    # Select the single key representation only after the primary comparison.
    secondary_variant = (
        "E_C_plus_keys_frequency_ohe"
        if summary.set_index("variant").loc[
            "E_C_plus_keys_frequency_ohe", "pooled_oof_auc"
        ]
        > summary.set_index("variant").loc["D_C_plus_keys_ohe", "pooled_oof_auc"]
        else "D_C_plus_keys_ohe"
    )
    secondary_predictions, secondary_selected, secondary_fit_rows = run_main(
        y,
        blocks,
        ("C_complete_raw_rowwise", secondary_variant),
        SECONDARY_SEED,
        logger,
        [],
        [],
    )
    secondary_rows = []
    for variant, pred in secondary_predictions.items():
        c_pred = secondary_predictions["C_complete_raw_rowwise"]
        fold_wins = sum(
            roc_auc_score(y[apply], pred[apply])
            > roc_auc_score(y[apply], c_pred[apply])
            for _, apply in outer_splits(y, SECONDARY_SEED)
        )
        secondary_rows.append(
            {
                "seed": SECONDARY_SEED,
                "variant": variant,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "delta_vs_C": float(
                    roc_auc_score(y, pred) - roc_auc_score(y, c_pred)
                ),
                "positive_folds_vs_C": int(fold_wins),
                "runtime_seconds": sum(
                    row["inner_tuning_runtime_seconds"]
                    + row["outer_fit_runtime_seconds"]
                    for row in secondary_fit_rows
                    if row["variant"] == variant
                ),
                "interpretation": "Robustness replication on same labels; not independent confirmation.",
            }
        )
    secondary = pd.DataFrame(secondary_rows)
    for variant, pred in secondary_predictions.items():
        path_data = pd.DataFrame(
            {
                "row_index": np.arange(len(y)),
                ID_COLUMN: train[ID_COLUMN].to_numpy(),
                TARGET_COLUMN: y,
                variant: pred,
            }
        )
        if variant == "C_complete_raw_rowwise":
            secondary_oof = path_data
        else:
            secondary_oof[variant] = pred
    secondary_oof["fold"] = 0
    for fold, (_, apply) in enumerate(outer_splits(y, SECONDARY_SEED), start=1):
        secondary_oof.loc[apply, "fold"] = fold

    # Build formula and eligibility evidence before writing the final manifest.
    (out / "FORMULA_PROVENANCE_AUDIT.md").write_text(
        formula_provenance_audit(), encoding="utf-8"
    )
    _eligibility_matrix().to_csv(
        out / "COMPETITION_ELIGIBILITY_MATRIX.csv", index=False
    )
    summary.to_csv(out / "MATCHED_BASELINE_RESULTS.csv", index=False)
    incremental.to_csv(out / "PRODUCT_KEY_INCREMENTAL_VALUE.csv", index=False)
    ablation.to_csv(out / "PRODUCT_KEY_ABLATION.csv", index=False)
    controls.to_csv(out / "NEGATIVE_CONTROL_RESULTS.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "FOLD_LEVEL_METRICS.csv", index=False)
    bootstrap.to_csv(out / "PAIRED_BOOTSTRAP_REPORT.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(out / "INNER_MODEL_SELECTION.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(out / "OUTER_MODEL_FITS.csv", index=False)
    pd.DataFrame(support_rows).to_csv(out / "CATEGORY_SUPPORT_REPORT.csv", index=False)
    pd.DataFrame(secondary_fit_rows).to_csv(
        out / "SECONDARY_OUTER_MODEL_FITS.csv", index=False
    )
    secondary.to_csv(out / "SECONDARY_SEED_REPLICATION.csv", index=False)
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
            "primary_fold": 0,
        }
    )
    for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
        oof.loc[apply, "primary_fold"] = fold
    for variant, pred in primary.items():
        oof[variant] = pred
    for variant, pred in ablation_predictions.items():
        oof[f"ablation_{variant}"] = pred
    oof.to_csv(out / "OOF_PREDICTIONS.csv", index=False)
    for name, pred in control_predictions.items():
        oof[f"{name}"] = pred
    oof.to_csv(out / "OOF_PREDICTIONS.csv", index=False)
    secondary_oof.to_csv(out / "SECONDARY_SEED_OOF_PREDICTIONS.csv", index=False)

    issues = []
    if len(oof) != len(train) or oof[ID_COLUMN].duplicated().any():
        issues.append("primary OOF rows or IDs do not align")
    if not np.array_equal(oof[TARGET_COLUMN].to_numpy(), y):
        issues.append("primary OOF target does not align")
    pred_cols = list(primary) + [f"ablation_{name}" for name in ablation_predictions]
    pred_cols += list(control_predictions)
    if oof[pred_cols].isna().any().any():
        issues.append("primary, ablation or control OOF has missing predictions")
    if not oof[pred_cols].apply(lambda s: s.between(0, 1).all()).all():
        issues.append("primary, ablation or control OOF is outside [0,1]")
    if secondary_oof.isna().any().any() or not np.array_equal(
        secondary_oof[TARGET_COLUMN].to_numpy(), y
    ):
        issues.append("secondary OOF rows, targets or predictions do not align")
    if sha256_file(train_path) != input_hash:
        issues.append("training source changed during execution")
    if issues:
        raise AssertionError("; ".join(issues))
    (out / "OOF_INTEGRITY_REPORT.md").write_text(
        "\n".join(
            [
                "# Phase 13 OOF integrity report",
                "",
                f"- Input: `{train_path}` SHA-256 `{input_hash}`.",
                f"- Rows/unique IDs: {len(train):,}/{train[ID_COLUMN].nunique():,}; positives: {int(y.sum()):,}.",
                f"- Primary seed `{seed}` and secondary seed `{SECONDARY_SEED}`; each has complete five-fold OOF coverage.",
                f"- Primary OOF columns: {len(primary)}; ablation columns: {len(ablation_predictions)}; shuffled-control columns: {len(control_predictions)}.",
                f"- Missing or out-of-range primary/control predictions: 0; secondary prediction missing count: {int(secondary_oof.isna().sum().sum())}.",
                "- IDs and targets align to original input row order; input SHA-256 was unchanged.",
                "- Test, sample submission, external data, holdout, and submissions were not opened/used.",
                "- Status: PASS.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    markdown = [
        "# Paired bootstrap report",
        "",
        "Paired, stratified row bootstrap intervals are conditional on fitted OOF predictions and are not independent confirmation.",
        "",
        "| Candidate | Reference | Delta | 95% low | 95% high |",
        "|---|---|---:|---:|---:|",
    ]
    for row in bootstrap.to_dict(orient="records"):
        markdown.append(
            f"| {row['candidate']} | {row['reference']} | {row['paired_delta']:+.6f} | "
            f"{row['bootstrap_95_low']:+.6f} | {row['bootstrap_95_high']:+.6f} |"
        )
    (out / "PAIRED_BOOTSTRAP_REPORT.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    _write_report(
        out,
        summary,
        incremental,
        bootstrap,
        ablation,
        controls,
        secondary,
        train_path,
        input_hash,
        started_at,
        datetime.now(timezone.utc),
        seed,
    )
    best = secondary_variant
    (out / "NEXT_PHASE_RECOMMENDATION.md").write_text(
        "\n".join(
            [
                "# Next phase recommendation",
                "",
                f"- The representation advanced to secondary-seed robustness testing was `{best}` because it had the higher primary pooled AUC among ordinary and frequency-supported keys.",
                "- Treat the secondary seed as robustness replication, not independent confirmation.",
                "- Do not claim Phase 11 lift preservation until the matched C baseline comparison is reviewed alongside the historical Phase 11 raw-category baseline.",
                "- Resolve the provenance of 130.5 and 0.0066; exclude formulas depending on them from any competition candidate unless verified or approved.",
                "- No submission, champion promotion, or leaderboard tuning.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.info(
        "All Phase 13 core outputs complete; A=%.6f C=%.6f D=%.6f E=%.6f secondary=%s",
        base_auc,
        complete_auc,
        roc_auc_score(y, primary["D_C_plus_keys_ohe"]),
        roc_auc_score(y, primary["E_C_plus_keys_frequency_ohe"]),
        best,
    )
    for handler in logger.handlers:
        handler.flush()
    logger.handlers.clear()
    logging.shutdown()
    manifest_path = out / "SHA256_MANIFEST.csv"
    files = sorted(path for path in out.iterdir() if path.is_file() and path != manifest_path)
    manifest_rows = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    for source in (
        Path(__file__),
        ROOT / "phase13_features.py",
        ROOT / "test_phase13_features.py",
    ):
        manifest_rows.append(
            {
                "file": f"source:{source.name}",
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 13 matched-baseline research.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args()
    run(args.train, args.output_root, args.seed)


if __name__ == "__main__":
    main()
