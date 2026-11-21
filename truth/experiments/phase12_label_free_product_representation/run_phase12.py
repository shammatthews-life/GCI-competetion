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
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from phase12_feature_definitions import (
    FEE_OFFSET_HYPOTHESIS,
    FINANCING_STEP_HYPOTHESIS,
    ID_COLUMN,
    PRODUCT_KEYS,
    TARGET_COLUMN,
    build_decomposed_features,
    build_numeric_predictors,
    build_product_keys,
    build_selected_interactions,
    category_support,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
PRIMARY_SEED = 20261105
SECONDARY_SEED = 20261106
INNER_FOLDS = 3
OUTER_FOLDS = 5
MAX_ITERATIONS = 1200
EARLY_STOPPING_ROUNDS = 80
BOOTSTRAP_REPETITIONS = 1000
HASH_BITS_PER_KEY = 13
HASH_FEATURES_PER_KEY = 2**HASH_BITS_PER_KEY
CONTROL_SEEDS = (20261201, 20261202, 20261203, 20261204, 20261205)

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
    "leave_out_Q3": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_Q3"),
    "leave_out_Q4": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_Q4"),
    "leave_out_Q5": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_Q5"),
    "leave_out_KQ3": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_KQ3"),
    "leave_out_KQ4": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_KQ4"),
    "leave_out_P4": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_P4"),
    "leave_out_P5": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_P5"),
    "leave_out_AG3": tuple(key for key in PRODUCT_KEYS if key != "PRODUCT_KEY_AG3"),
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

MAIN_VARIANTS = (
    "raw_numeric",
    "raw_plus_keys_ohe",
    "raw_plus_keys_frequency_ohe",
    "raw_plus_decompositions",
    "raw_plus_interactions",
    "raw_plus_keys_decompositions_interactions",
    "raw_plus_keys_hash",
)

CONFIGS = (
    {
        "name": "leaves31_child80",
        "num_leaves": 31,
        "min_child_samples": 80,
    },
    {
        "name": "leaves63_child100",
        "num_leaves": 63,
        "min_child_samples": 100,
    },
)


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
        raise OSError("Phase 12 memory telemetry currently requires Windows")
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_int
    handle = get_current_process()
    succeeded = get_memory_info(handle, ctypes.byref(counters), counters.cb)
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    return float(counters.PeakWorkingSetSize / (1024**2))


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase12")
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


def _feature_keys(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    key_columns: tuple[str, ...],
    minimum_frequency: int | None = None,
) -> tuple[csr_matrix, csr_matrix, int]:
    encoder = OneHotEncoder(
        handle_unknown=(
            "infrequent_if_exist" if minimum_frequency is not None else "ignore"
        ),
        dtype=np.float32,
        sparse_output=True,
        min_frequency=minimum_frequency,
    )
    fit_values = fit.loc[:, key_columns].astype(str)
    apply_values = apply.loc[:, key_columns].astype(str)
    fit_encoded = encoder.fit_transform(fit_values).tocsr()
    apply_encoded = encoder.transform(apply_values).tocsr()
    return fit_encoded, apply_encoded, int(fit_encoded.shape[1])


def _hash_keys(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    key_columns: tuple[str, ...],
) -> tuple[csr_matrix, csr_matrix, int, list[dict[str, int | str]]]:
    fit_blocks: list[csr_matrix] = []
    apply_blocks: list[csr_matrix] = []
    collision_rows: list[dict[str, int | str]] = []
    for column in key_columns:
        hasher = FeatureHasher(
            n_features=HASH_FEATURES_PER_KEY,
            input_type="string",
            dtype=np.float32,
            alternate_sign=False,
        )
        fit_values = fit[column].astype(str).tolist()
        apply_values = apply[column].astype(str).tolist()
        fit_block = hasher.transform([[value] for value in fit_values]).tocsr()
        apply_block = hasher.transform([[value] for value in apply_values]).tocsr()
        unique_values = sorted(set(fit_values))
        buckets = hasher.transform([[value] for value in unique_values]).tocoo().col
        unique_buckets = int(np.unique(buckets).size)
        fit_blocks.append(fit_block)
        apply_blocks.append(apply_block)
        collision_rows.append(
            {
                "key": column,
                "fit_unique_categories": len(unique_values),
                "distinct_hash_buckets": unique_buckets,
                "observed_category_bucket_collisions": len(unique_values)
                - unique_buckets,
                "hash_buckets_per_key": HASH_FEATURES_PER_KEY,
            }
        )
    return (
        hstack(fit_blocks, format="csr"),
        hstack(apply_blocks, format="csr"),
        int(len(key_columns) * HASH_FEATURES_PER_KEY),
        collision_rows,
    )


def make_matrices(
    base_fit: pd.DataFrame,
    base_apply: pd.DataFrame,
    keys_fit: pd.DataFrame,
    keys_apply: pd.DataFrame,
    decompositions_fit: pd.DataFrame,
    decompositions_apply: pd.DataFrame,
    interactions_fit: pd.DataFrame,
    interactions_apply: pd.DataFrame,
    variant: str,
    key_columns: tuple[str, ...] = PRODUCT_KEYS,
) -> tuple[csr_matrix, csr_matrix, dict[str, Any]]:
    fit_blocks = [_numeric_matrix(base_fit)]
    apply_blocks = [_numeric_matrix(base_apply)]
    details: dict[str, Any] = {
        "encoded_key_feature_count": 0,
        "hash_collisions": [],
    }

    if variant in {
        "raw_plus_decompositions",
        "raw_plus_keys_decompositions_interactions",
    }:
        fit_blocks.append(_numeric_matrix(decompositions_fit))
        apply_blocks.append(_numeric_matrix(decompositions_apply))
    if variant in {
        "raw_plus_interactions",
        "raw_plus_keys_decompositions_interactions",
    }:
        fit_blocks.append(_numeric_matrix(interactions_fit))
        apply_blocks.append(_numeric_matrix(interactions_apply))
    if variant in {
        "raw_plus_keys_ohe",
        "raw_plus_keys_frequency_ohe",
        "raw_plus_keys_decompositions_interactions",
    }:
        fit_encoded, apply_encoded, feature_count = _feature_keys(
            keys_fit,
            keys_apply,
            key_columns,
            minimum_frequency=(
                2
                if variant
                in {
                    "raw_plus_keys_frequency_ohe",
                    "raw_plus_keys_decompositions_interactions",
                }
                else None
            ),
        )
        fit_blocks.append(fit_encoded)
        apply_blocks.append(apply_encoded)
        details["encoded_key_feature_count"] = feature_count
    if variant == "raw_plus_keys_frequency_ohe":
        counts = keys_fit.loc[:, key_columns].astype(str).apply(
            lambda column: column.map(column.value_counts()).astype(np.float32)
        )
        apply_counts = keys_apply.loc[:, key_columns].astype(str).copy()
        for column in key_columns:
            fit_frequency = keys_fit[column].astype(str).value_counts()
            apply_counts[column] = (
                apply_counts[column]
                .map(fit_frequency)
                .fillna(0)
                .astype(np.float32)
            )
        counts.columns = [f"FIT_FREQUENCY_{column}" for column in key_columns]
        apply_counts.columns = [f"FIT_FREQUENCY_{column}" for column in key_columns]
        fit_blocks.append(_numeric_matrix(counts))
        apply_blocks.append(_numeric_matrix(apply_counts))
    if variant == "raw_plus_keys_hash":
        fit_encoded, apply_encoded, feature_count, collisions = _hash_keys(
            keys_fit, keys_apply, key_columns
        )
        fit_blocks.append(fit_encoded)
        apply_blocks.append(apply_encoded)
        details["encoded_key_feature_count"] = feature_count
        details["hash_collisions"] = collisions

    fit_matrix = hstack(fit_blocks, format="csr")
    apply_matrix = hstack(apply_blocks, format="csr")
    details["feature_count"] = int(fit_matrix.shape[1])
    details["fit_nonzero_count"] = int(fit_matrix.nnz)
    details["apply_nonzero_count"] = int(apply_matrix.nnz)
    return fit_matrix, apply_matrix, details


def model_seed(seed: int, outer_fold: int, inner_fold: int, config_index: int) -> int:
    return int(seed + outer_fold * 10000 + inner_fold * 100 + config_index)


def _early_stopped_prediction(
    matrix_fit: csr_matrix,
    matrix_apply: csr_matrix,
    y_fit: np.ndarray,
    y_apply: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int, float]:
    model = make_model(config, seed, MAX_ITERATIONS)
    started = time.perf_counter()
    model.fit(
        matrix_fit,
        y_fit,
        eval_X=matrix_apply,
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
    prediction = model.predict_proba(matrix_apply)[:, 1]
    best_iteration = int(model.best_iteration_ or MAX_ITERATIONS)
    return prediction, best_iteration, time.perf_counter() - started


def _fit_prediction(
    matrix_fit: csr_matrix,
    matrix_apply: csr_matrix,
    y_fit: np.ndarray,
    config: dict[str, Any],
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, float]:
    model = make_model(config, seed, max(1, int(iterations)))
    started = time.perf_counter()
    model.fit(matrix_fit, y_fit)
    prediction = model.predict_proba(matrix_apply)[:, 1]
    return prediction, time.perf_counter() - started


def _inner_splits(
    fit_indices: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    return [
        (fit_indices[inner_fit], fit_indices[inner_apply])
        for inner_fit, inner_apply in splitter.split(
            np.zeros(len(fit_indices), dtype=np.int8), y[fit_indices]
        )
    ]


def tune_variant(
    variant: str,
    key_columns: tuple[str, ...],
    outer_fit: np.ndarray,
    y: np.ndarray,
    numeric: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    interactions: pd.DataFrame,
    seed: int,
    outer_fold: int,
    support_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[dict[str, Any], int, dict[str, float], float]:
    config_oof = {
        str(config["name"]): np.full(len(outer_fit), np.nan, dtype=np.float64)
        for config in CONFIGS
    }
    config_iterations: dict[str, list[int]] = {
        str(config["name"]): [] for config in CONFIGS
    }
    config_runtimes: dict[str, float] = {
        str(config["name"]): 0.0 for config in CONFIGS
    }
    inner_splits = _inner_splits(outer_fit, y, seed + outer_fold)

    for inner_fold, (inner_fit, inner_apply) in enumerate(inner_splits, start=1):
        fit_matrix, apply_matrix, details = make_matrices(
            numeric.iloc[inner_fit],
            numeric.iloc[inner_apply],
            keys.iloc[inner_fit],
            keys.iloc[inner_apply],
            decompositions.iloc[inner_fit],
            decompositions.iloc[inner_apply],
            interactions.iloc[inner_fit],
            interactions.iloc[inner_apply],
            variant,
            key_columns,
        )
        if variant in {
            "raw_plus_keys_ohe",
            "raw_plus_keys_frequency_ohe",
            "raw_plus_keys_decompositions_interactions",
        }:
            support_rows.extend(
                category_support(
                    keys.iloc[inner_fit],
                    keys.iloc[inner_apply],
                    key_columns,
                    2
                    if variant
                    in {
                        "raw_plus_keys_frequency_ohe",
                        "raw_plus_keys_decompositions_interactions",
                    }
                    else 1,
                    outer_fold * 100 + inner_fold,
                    variant,
                )
            )
        for config_index, config in enumerate(CONFIGS, start=1):
            prediction, best_iteration, elapsed = _early_stopped_prediction(
                fit_matrix,
                apply_matrix,
                y[inner_fit],
                y[inner_apply],
                config,
                model_seed(seed, outer_fold, inner_fold, config_index),
            )
            config_oof[str(config["name"])][
                np.searchsorted(outer_fit, inner_apply)
            ] = prediction
            config_iterations[str(config["name"])].append(best_iteration)
            config_runtimes[str(config["name"])] += elapsed
            selection_rows.append(
                {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "variant": variant,
                    "config": config["name"],
                    "inner_auc": float(roc_auc_score(y[inner_apply], prediction)),
                    "best_iteration": best_iteration,
                    "runtime_seconds": elapsed,
                    "feature_count": details["feature_count"],
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
        logger.info(
            "tuned seed=%s outer=%s/%s inner=%s/%s variant=%s features=%s",
            seed,
            outer_fold,
            OUTER_FOLDS,
            inner_fold,
            INNER_FOLDS,
            variant,
            details["feature_count"],
        )

    aucs = {
        name: float(roc_auc_score(y[outer_fit], predictions))
        for name, predictions in config_oof.items()
    }
    selected_name = max(aucs, key=aucs.get)
    selected_config = next(config for config in CONFIGS if config["name"] == selected_name)
    rounds = max(
        1,
        int(np.rint(np.median(config_iterations[selected_name]))),
    )
    for config in CONFIGS:
        config_name = str(config["name"])
        selection_rows.append(
            {
                "seed": seed,
                "outer_fold": outer_fold,
                "inner_fold": "pooled_inner_oof",
                "variant": variant,
                "config": config_name,
                "inner_auc": aucs[config_name],
                "best_iteration": int(
                    np.rint(np.median(config_iterations[config_name]))
                ),
                "runtime_seconds": config_runtimes[config_name],
                "feature_count": None,
                "peak_working_set_mb": peak_working_set_mb(),
                "selected": config_name == selected_name,
            }
        )
    return selected_config, rounds, aucs, sum(config_runtimes.values())


def _outer_fold_splits(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y), dtype=np.int8), y))


def run_nested_variants(
    variants: tuple[str, ...],
    seed: int,
    y: np.ndarray,
    numeric: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    interactions: pd.DataFrame,
    logger: logging.Logger,
    support_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, tuple[dict[str, Any], int]]], list[dict[str, Any]]]:
    predictions = {
        variant: np.full(len(y), np.nan, dtype=np.float64) for variant in variants
    }
    selected_by_fold: dict[int, dict[str, tuple[dict[str, Any], int]]] = {}
    fit_rows: list[dict[str, Any]] = []
    for outer_fold, (outer_fit, outer_apply) in enumerate(
        _outer_fold_splits(y, seed), start=1
    ):
        selected_by_fold[outer_fold] = {}
        for variant in variants:
            key_columns = PRODUCT_KEYS
            config, rounds, inner_auc, tune_runtime = tune_variant(
                variant,
                key_columns,
                outer_fit,
                y,
                numeric,
                keys,
                decompositions,
                interactions,
                seed,
                outer_fold,
                support_rows,
                selection_rows,
                logger,
            )
            selected_by_fold[outer_fold][variant] = (config, rounds)
            fit_matrix, apply_matrix, details = make_matrices(
                numeric.iloc[outer_fit],
                numeric.iloc[outer_apply],
                keys.iloc[outer_fit],
                keys.iloc[outer_apply],
                decompositions.iloc[outer_fit],
                decompositions.iloc[outer_apply],
                interactions.iloc[outer_fit],
                interactions.iloc[outer_apply],
                variant,
                key_columns,
            )
            if variant in {
                "raw_plus_keys_ohe",
                "raw_plus_keys_frequency_ohe",
                "raw_plus_keys_decompositions_interactions",
            }:
                support_rows.extend(
                    category_support(
                        keys.iloc[outer_fit],
                        keys.iloc[outer_apply],
                        key_columns,
                        2
                        if variant
                        in {
                            "raw_plus_keys_frequency_ohe",
                            "raw_plus_keys_decompositions_interactions",
                        }
                        else 1,
                        outer_fold,
                        variant,
                    )
                )
            prediction, fit_runtime = _fit_prediction(
                fit_matrix,
                apply_matrix,
                y[outer_fit],
                config,
                seed + outer_fold * 1000 + 500,
                rounds,
            )
            predictions[variant][outer_apply] = prediction
            fit_rows.append(
                {
                    "seed": seed,
                    "fold": outer_fold,
                    "variant": variant,
                    "config": config["name"],
                    "selected_rounds": rounds,
                    "inner_selected_pooled_auc": json.dumps(inner_auc, sort_keys=True),
                    "inner_tuning_runtime_seconds": tune_runtime,
                    "outer_fit_runtime_seconds": fit_runtime,
                    "feature_count": details["feature_count"],
                    "fit_nonzero_count": details["fit_nonzero_count"],
                    "apply_nonzero_count": details["apply_nonzero_count"],
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
            logger.info(
                "outer fit seed=%s fold=%s/%s variant=%s auc=%.6f rounds=%s "
                "features=%s runtime=%.1fs",
                seed,
                outer_fold,
                OUTER_FOLDS,
                variant,
                roc_auc_score(y[outer_apply], prediction),
                rounds,
                details["feature_count"],
                fit_runtime,
            )
    return predictions, selected_by_fold, fit_rows


def run_ablation_grid(
    y: np.ndarray,
    numeric: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    interactions: pd.DataFrame,
    selected_by_fold: dict[int, dict[str, tuple[dict[str, Any], int]]],
    seed: int,
    logger: logging.Logger,
    support_rows: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float64) for name in KEY_GROUPS
    }
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_fit, outer_apply) in enumerate(
        _outer_fold_splits(y, seed), start=1
    ):
        config, rounds = selected_by_fold[outer_fold]["raw_plus_keys_ohe"]
        for variant, key_columns in KEY_GROUPS.items():
            fit_matrix, apply_matrix, details = make_matrices(
                numeric.iloc[outer_fit],
                numeric.iloc[outer_apply],
                keys.iloc[outer_fit],
                keys.iloc[outer_apply],
                decompositions.iloc[outer_fit],
                decompositions.iloc[outer_apply],
                interactions.iloc[outer_fit],
                interactions.iloc[outer_apply],
                "raw_plus_keys_ohe",
                key_columns,
            )
            support_rows.extend(
                category_support(
                    keys.iloc[outer_fit],
                    keys.iloc[outer_apply],
                    key_columns,
                    1,
                    outer_fold,
                    f"ablation_{variant}",
                )
            )
            prediction, elapsed = _fit_prediction(
                fit_matrix,
                apply_matrix,
                y[outer_fit],
                config,
                seed + outer_fold * 5000 + len(variant),
                rounds,
            )
            predictions[variant][outer_apply] = prediction
            rows.append(
                {
                    "variant": variant,
                    "fold": outer_fold,
                    "key_count": len(key_columns),
                    "keys": "|".join(key_columns),
                    "fold_auc": float(roc_auc_score(y[outer_apply], prediction)),
                    "paired_delta_vs_all_eight": None,
                    "config": config["name"],
                    "rounds": rounds,
                    "feature_count": details["feature_count"],
                    "runtime_seconds": elapsed,
                    "peak_working_set_mb": peak_working_set_mb(),
                }
            )
            logger.info(
                "ablation fold=%s/%s variant=%s auc=%.6f",
                outer_fold,
                OUTER_FOLDS,
                variant,
                roc_auc_score(y[outer_apply], prediction),
            )
    return predictions, rows


def _randomized_key_frames(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    rng: np.random.Generator,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_control = pd.DataFrame(index=np.arange(len(fit)))
    apply_control = pd.DataFrame(index=np.arange(len(apply)))
    for column in PRODUCT_KEYS:
        fit_values = fit[column].astype(str).to_numpy(copy=True)
        if method == "frequency_preserving_shuffle":
            apply_values = apply[column].astype(str).to_numpy(copy=True)
            rng.shuffle(fit_values)
            rng.shuffle(apply_values)
        elif method == "randomized_assignment":
            counts = fit[column].astype(str).value_counts(normalize=True)
            categories = counts.index.to_numpy(dtype=str)
            probabilities = counts.to_numpy(dtype=np.float64)
            fit_values = rng.choice(categories, size=len(fit), p=probabilities)
            apply_values = rng.choice(categories, size=len(apply), p=probabilities)
        else:
            raise ValueError(f"Unknown negative-control method: {method}")
        fit_control[column] = fit_values
        apply_control[column] = apply_values
    return fit_control, apply_control


def run_negative_controls(
    y: np.ndarray,
    numeric: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    interactions: pd.DataFrame,
    selected_by_fold: dict[int, dict[str, tuple[dict[str, Any], int]]],
    raw_oof: np.ndarray,
    keys_oof: np.ndarray,
    seed: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, np.ndarray]]:
    control_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    control_predictions: dict[str, np.ndarray] = {}
    for method in ("frequency_preserving_shuffle", "randomized_assignment"):
        for repetition, control_seed in enumerate(CONTROL_SEEDS, start=1):
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for outer_fold, (outer_fit, outer_apply) in enumerate(
                _outer_fold_splits(y, seed), start=1
            ):
                rng = np.random.default_rng(control_seed + outer_fold)
                keys_fit, keys_apply = _randomized_key_frames(
                    keys.iloc[outer_fit],
                    keys.iloc[outer_apply],
                    rng,
                    method,
                )
                config, rounds = selected_by_fold[outer_fold]["raw_plus_keys_ohe"]
                fit_matrix, apply_matrix, details = make_matrices(
                    numeric.iloc[outer_fit],
                    numeric.iloc[outer_apply],
                    keys_fit,
                    keys_apply,
                    decompositions.iloc[outer_fit],
                    decompositions.iloc[outer_apply],
                    interactions.iloc[outer_fit],
                    interactions.iloc[outer_apply],
                    "raw_plus_keys_ohe",
                )
                prediction, elapsed = _fit_prediction(
                    fit_matrix,
                    apply_matrix,
                    y[outer_fit],
                    config,
                    control_seed + outer_fold * 5000,
                    rounds,
                )
                oof[outer_apply] = prediction
                control_rows.append(
                    {
                        "control": method,
                        "repetition": repetition,
                        "control_seed": control_seed,
                        "fold": outer_fold,
                        "fold_auc": float(roc_auc_score(y[outer_apply], prediction)),
                        "pooled_oof_auc": None,
                        "paired_delta_vs_raw_numeric": float(
                            roc_auc_score(y[outer_apply], prediction)
                            - roc_auc_score(y[outer_apply], raw_oof[outer_apply])
                        ),
                        "paired_delta_vs_all_eight_ohe": float(
                            roc_auc_score(y[outer_apply], prediction)
                            - roc_auc_score(y[outer_apply], keys_oof[outer_apply])
                        ),
                        "feature_count": details["feature_count"],
                        "runtime_seconds": elapsed,
                        "peak_working_set_mb": peak_working_set_mb(),
                    }
                )
                logger.info(
                    "negative control=%s repetition=%s fold=%s/%s auc=%.6f",
                    method,
                    repetition,
                    outer_fold,
                    OUTER_FOLDS,
                    roc_auc_score(y[outer_apply], prediction),
                )
            if not np.isfinite(oof).all():
                raise AssertionError(f"Control OOF predictions are incomplete: {method}")
            pooled_auc = float(roc_auc_score(y, oof))
            control_predictions[f"{method}_rep_{repetition}"] = oof
            for row in control_rows:
                if (
                    row["control"] == method
                    and row["repetition"] == repetition
                ):
                    row["pooled_oof_auc"] = pooled_auc
            collision_rows.append(
                {
                    "control": method,
                    "repetition": repetition,
                    "control_seed": control_seed,
                    "pooled_oof_auc": pooled_auc,
                }
            )
    return pd.DataFrame(control_rows), collision_rows, control_predictions


def _fold_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = predictions["raw_numeric"]
    for name, prediction in predictions.items():
        for fold, (_, apply) in enumerate(
            _outer_fold_splits(y, seed), start=1
        ):
            fold_auc = float(roc_auc_score(y[apply], prediction[apply]))
            baseline_auc = float(roc_auc_score(y[apply], baseline[apply]))
            rows.append(
                {
                    "variant": name,
                    "fold": fold,
                    "auc": fold_auc,
                    "paired_delta_vs_raw_numeric": fold_auc - baseline_auc,
                    "positive_delta": bool(fold_auc > baseline_auc),
                }
            )
    return pd.DataFrame(rows)


def _paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    score_map = {"raw_numeric": baseline, **predictions}

    def group_ids(scores: np.ndarray) -> tuple[np.ndarray, int]:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
        ordered_ids = np.repeat(np.arange(len(boundaries), dtype=np.int32), np.diff(
            np.r_[boundaries, len(scores)]
        ))
        ids = np.empty(len(scores), dtype=np.int32)
        ids[order] = ordered_ids
        return ids, len(boundaries)

    grouped = {
        name: group_ids(np.asarray(score, dtype=np.float64))
        for name, score in score_map.items()
    }
    deltas_by_name = {
        name: np.empty(repetitions, dtype=np.float64)
        for name in predictions
    }
    for index in range(repetitions):
        pos_sample = rng.choice(positive, size=len(positive), replace=True)
        neg_sample = rng.choice(negative, size=len(negative), replace=True)
        aucs: dict[str, float] = {}
        for name, (ids, groups) in grouped.items():
            positive_weights = np.bincount(ids[pos_sample], minlength=groups)
            negative_weights = np.bincount(ids[neg_sample], minlength=groups)
            negative_below = np.cumsum(negative_weights) - negative_weights
            numerator = np.sum(
                positive_weights
                * (negative_below + 0.5 * negative_weights),
                dtype=np.float64,
            )
            aucs[name] = float(
                numerator / (len(pos_sample) * len(neg_sample))
            )
        for name in predictions:
            deltas_by_name[name][index] = aucs[name] - aucs["raw_numeric"]
    rows: list[dict[str, Any]] = []
    baseline_auc = roc_auc_score(y, baseline)
    for name, prediction in predictions.items():
        deltas = deltas_by_name[name]
        rows.append(
            {
                "comparison": f"{name}_vs_raw_numeric",
                "reference_auc": float(baseline_auc),
                "candidate_auc": float(roc_auc_score(y, prediction)),
                "paired_delta": float(
                    roc_auc_score(y, prediction) - baseline_auc
                ),
                "paired_stratified_bootstrap_95_low": float(
                    np.quantile(deltas, 0.025)
                ),
                "paired_stratified_bootstrap_95_high": float(
                    np.quantile(deltas, 0.975)
                ),
                "bootstrap_repetitions": repetitions,
                "seed": seed,
                "interpretation": "Conditional on the fitted OOF predictions; not independent confirmation.",
            }
        )
    return pd.DataFrame(rows)


def _summaries(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    fold_metrics: pd.DataFrame,
    fit_rows: list[dict[str, Any]],
    selected_by_fold: dict[int, dict[str, tuple[dict[str, Any], int]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_auc = float(roc_auc_score(y, predictions["raw_numeric"]))
    baseline = predictions["raw_numeric"]
    for name, prediction in predictions.items():
        fold_data = fold_metrics.loc[fold_metrics["variant"] == name]
        fits = [row for row in fit_rows if row["variant"] == name]
        configs = {
            str(selected_by_fold[fold][name][0]["name"])
            for fold in selected_by_fold
            if name in selected_by_fold[fold]
        }
        rows.append(
            {
                "variant": name,
                "pooled_oof_auc": float(roc_auc_score(y, prediction)),
                "delta_vs_raw_numeric": float(
                    roc_auc_score(y, prediction) - baseline_auc
                ),
                "positive_folds_vs_raw_numeric": int(
                    fold_data["positive_delta"].sum()
                ),
                "fold_count": int(len(fold_data)),
                "prediction_pearson_correlation_vs_raw": float(
                    np.corrcoef(baseline, prediction)[0, 1]
                ),
                "prediction_spearman_correlation_vs_raw": float(
                    pd.Series(baseline).corr(
                        pd.Series(prediction), method="spearman"
                    )
                ),
                "feature_count_by_outer_fold": json.dumps(
                    {
                        str(row["fold"]): row["feature_count"]
                        for row in fits
                    },
                    sort_keys=True,
                ),
                "selected_configs_by_outer_fold": "|".join(sorted(configs)),
                "total_tuning_runtime_seconds": float(
                    sum(row["inner_tuning_runtime_seconds"] for row in fits)
                ),
                "total_outer_fit_runtime_seconds": float(
                    sum(row["outer_fit_runtime_seconds"] for row in fits)
                ),
                "peak_process_working_set_mb": float(
                    max((row["peak_working_set_mb"] for row in fits), default=0.0)
                ),
                "device": "CPU",
                "label_derived_feature_statistics": False,
            }
        )
    return pd.DataFrame(rows)


def _formula_audit(keys: pd.DataFrame, keys_raw: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "PRODUCT_KEY_Q3": (
            "round(AMT_ANNUITY / AMT_CREDIT, 3 significant decimal places via rint(x*10^3))",
            "AMT_ANNUITY; AMT_CREDIT",
            "MISSING when either input is non-finite or denominator is zero",
            "decimal scaling precision 10^3; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
        "PRODUCT_KEY_Q4": (
            "round(AMT_ANNUITY / AMT_CREDIT, 4 significant decimal places via rint(x*10^4))",
            "AMT_ANNUITY; AMT_CREDIT",
            "MISSING when either input is non-finite or denominator is zero",
            "decimal scaling precision 10^4; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
        "PRODUCT_KEY_Q5": (
            "round(AMT_ANNUITY / AMT_CREDIT, 5 significant decimal places via rint(x*10^5))",
            "AMT_ANNUITY; AMT_CREDIT",
            "MISSING when either input is non-finite or denominator is zero",
            "decimal scaling precision 10^5; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
        "PRODUCT_KEY_KQ3": (
            "cross(round((AMT_CREDIT/AMT_GOODS_PRICE-1)/0.0066), Q3) for eligible Cash loans",
            "NAME_CONTRACT_TYPE; AMT_CREDIT; AMT_GOODS_PRICE; Q3",
            "NONCASH_OR_INVALID unless Cash loans, credit>goods>0 and components finite; invalid Q3 also maps to sentinel",
            "financing coordinate rounded to integer; Q3 precision 10^3",
            "Cross tuple is exact text; identical coordinate/Q3 pairs intentionally collide",
        ),
        "PRODUCT_KEY_KQ4": (
            "cross(round((AMT_CREDIT/AMT_GOODS_PRICE-1)/0.0066), Q4) for eligible Cash loans",
            "NAME_CONTRACT_TYPE; AMT_CREDIT; AMT_GOODS_PRICE; Q4",
            "NONCASH_OR_INVALID unless Cash loans, credit>goods>0 and components finite; invalid Q4 also maps to sentinel",
            "financing coordinate rounded to integer; Q4 precision 10^4",
            "Cross tuple is exact text; identical coordinate/Q4 pairs intentionally collide",
        ),
        "PRODUCT_KEY_P4": (
            "round((AMT_ANNUITY-130.5)/AMT_CREDIT, 4 decimal places via rint(x*10^4))",
            "AMT_ANNUITY; AMT_CREDIT",
            "MISSING when either input is non-finite or denominator is zero",
            "fee offset 130.5; decimal scaling precision 10^4; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
        "PRODUCT_KEY_P5": (
            "round((AMT_ANNUITY-130.5)/AMT_CREDIT, 5 decimal places via rint(x*10^5))",
            "AMT_ANNUITY; AMT_CREDIT",
            "MISSING when either input is non-finite or denominator is zero",
            "fee offset 130.5; decimal scaling precision 10^5; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
        "PRODUCT_KEY_AG3": (
            "round(AMT_ANNUITY / AMT_GOODS_PRICE, 3 decimal places via rint(x*10^3))",
            "AMT_ANNUITY; AMT_GOODS_PRICE",
            "MISSING when either input is non-finite or denominator is zero",
            "decimal scaling precision 10^3; NumPy round-to-even",
            "Equal rounded values intentionally collide; no hash collision",
        ),
    }
    rows: list[dict[str, Any]] = []
    for key in PRODUCT_KEYS:
        formula, columns, missing, precision, collisions = definitions[key]
        rows.append(
            {
                "feature": key,
                "exact_formula": formula,
                "input_columns": columns,
                "missing_value_handling": missing,
                "numerical_precision": precision,
                "train_category_cardinality": int(keys[key].nunique(dropna=False)),
                "collision_behavior": collisions,
                "label_free_formula": True,
                "train_fold_category_support": "Measured per outer and inner fit partition in CATEGORY_SUPPORT_REPORT.csv.",
                "uses_test_information": False,
                "uses_external_information": False,
                "official_rule_support": "Supplied-column mathematical features are supported by the course tutorial guidance; see reports/COMPETITION_MATERIALS_REVIEW.md.",
                "eligibility_interpretation": (
                    "UNRESOLVED: historical offset 130.5 provenance is not established."
                    if key in {"PRODUCT_KEY_P4", "PRODUCT_KEY_P5"}
                    else (
                        "UNRESOLVED: historical financing step 0.0066 provenance is not established."
                        if key in {"PRODUCT_KEY_KQ3", "PRODUCT_KEY_KQ4"}
                        else "Formula uses only supplied columns; organizer approval is not documented."
                    )
                ),
            }
        )
    rows.extend(
        [
            {
                "feature": "RAW_NUMERIC",
                "exact_formula": "Use supplied numeric predictors without target-derived transformation.",
                "input_columns": "; ".join(build_numeric_predictors(keys_raw).columns),
                "missing_value_handling": "Preserve missing values for LightGBM native missing-value handling; infinities mapped to missing.",
                "numerical_precision": "Original supplied numeric precision; model matrix float32.",
                "train_category_cardinality": "",
                "collision_behavior": "Not applicable.",
                "label_free_formula": True,
                "train_fold_category_support": "No categories.",
                "uses_test_information": False,
                "uses_external_information": False,
                "official_rule_support": "Supplied training predictors; see competition rules audit.",
                "eligibility_interpretation": "Consistent with supplied-data-only rule; not organizer approval.",
            },
            {
                "feature": "NUMERIC_DECOMPOSITIONS",
                "exact_formula": "Annuity/credit; annuity/goods; (annuity-130.5)/credit; financing coordinate; financing eligibility indicator.",
                "input_columns": "NAME_CONTRACT_TYPE; AMT_CREDIT; AMT_ANNUITY; AMT_GOODS_PRICE",
                "missing_value_handling": "Safe ratios return NaN; ineligible financing coordinate becomes NaN.",
                "numerical_precision": "float64 construction, float32 model matrix; offset 130.5 and step 0.0066.",
                "train_category_cardinality": "",
                "collision_behavior": "No categorization or hashing.",
                "label_free_formula": True,
                "train_fold_category_support": "No learned category support.",
                "uses_test_information": False,
                "uses_external_information": False,
                "official_rule_support": "Supplied-column mathematical features are supported by course tutorial guidance.",
                "eligibility_interpretation": "UNRESOLVED for terms using historical constants.",
            },
            {
                "feature": "SELECTED_INTERACTIONS",
                "exact_formula": "Credit*region population; annuity*region population; EXT_SOURCE_1*EXT_SOURCE_2; EXT_SOURCE_2*EXT_SOURCE_3.",
                "input_columns": "AMT_CREDIT; AMT_ANNUITY; REGION_POPULATION_RELATIVE; EXT_SOURCE_1; EXT_SOURCE_2; EXT_SOURCE_3",
                "missing_value_handling": "Propagate missing values; infinities mapped to missing.",
                "numerical_precision": "float64 construction, float32 model matrix.",
                "train_category_cardinality": "",
                "collision_behavior": "No categorization or hashing.",
                "label_free_formula": True,
                "train_fold_category_support": "No learned category support.",
                "uses_test_information": False,
                "uses_external_information": False,
                "official_rule_support": "Supplied-column mathematical features are supported by course tutorial guidance.",
                "eligibility_interpretation": "Formula-only; not organizer approval.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _eligibility_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_or_transform": "Raw numeric supplied predictors",
                "label_free_or_dependent": "Label-free transformation; model fitting uses fit-fold TARGET only.",
                "fit_scope": "No fitted feature transform; model trained within each outer fit fold.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Provided train/test only; see competition materials review.",
                "unresolved_interpretation": "No organizer determination is claimed.",
            },
            {
                "feature_or_transform": "Q3, Q4, Q5, AG3 formulas",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Deterministic rowwise transforms; no pooled statistics.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Supplied-column feature construction is supported by course tutorial slide 22.",
                "unresolved_interpretation": "Deterministic rounded categories are not explicitly ruled on by organizer.",
            },
            {
                "feature_or_transform": "KQ3, KQ4 formulas",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Deterministic rowwise transforms; no pooled statistics.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Supplied-column feature construction is supported by course tutorial slide 22.",
                "unresolved_interpretation": "The fixed 0.0066 financing step's permitted-data provenance is unresolved.",
            },
            {
                "feature_or_transform": "P4, P5 formulas",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Deterministic rowwise transforms; no pooled statistics.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Supplied-column feature construction is supported by course tutorial slide 22.",
                "unresolved_interpretation": "The fixed 130.5 fee offset's permitted-data provenance is unresolved.",
            },
            {
                "feature_or_transform": "One-hot / frequency-supported one-hot",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Encoder and category frequencies fit on each inner/outer fit partition only.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "No target statistics or external data; supplied training features only.",
                "unresolved_interpretation": "Any unresolved source-formula issue carries through its encoded representation.",
            },
            {
                "feature_or_transform": "Feature hashing",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Fixed hashing map; no data-fitted mapping.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "No target statistics or external data; supplied training features only.",
                "unresolved_interpretation": "Any unresolved source-formula issue carries through its hashed representation.",
            },
            {
                "feature_or_transform": "Frequency-preserving / randomized controls",
                "label_free_or_dependent": "Label-free",
                "fit_scope": "Randomization uses fixed recorded seeds and fit-fold category support only.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Negative controls are validation diagnostics, not submission features.",
                "unresolved_interpretation": "No organizer approval is claimed.",
            },
            {
                "feature_or_transform": "Target encoding, target rates, CatBoost CTRs",
                "label_free_or_dependent": "Label-dependent",
                "fit_scope": "Not used.",
                "test_information": "No",
                "external_information": "No",
                "official_rule_support": "Excluded by Phase 12 protocol.",
                "unresolved_interpretation": "Eligibility not evaluated; prohibited from this eligible-track experiment.",
            },
        ]
    )


def _category_support_hash_collisions(
    support_rows: list[dict[str, Any]],
    collision_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = list(support_rows)
    for row in collision_rows:
        rows.append(
            {
                "variant": "raw_plus_keys_hash",
                "fold": row.get("fold", ""),
                "key": row["key"],
                "fit_rows": row.get("fit_rows", ""),
                "apply_rows": row.get("apply_rows", ""),
                "fit_unique_categories": row["fit_unique_categories"],
                "fit_singleton_categories": "",
                "fit_rare_categories_below_minimum": "",
                "fit_categories_at_or_above_minimum": "",
                "apply_unseen_categories": "",
                "minimum_fit_count": "",
                "distinct_hash_buckets": row["distinct_hash_buckets"],
                "observed_category_bucket_collisions": row[
                    "observed_category_bucket_collisions"
                ],
                "hash_buckets_per_key": row["hash_buckets_per_key"],
            }
        )
    return pd.DataFrame(rows)


def _run_secondary_seed(
    y: np.ndarray,
    numeric: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    interactions: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    seed_predictions, _, fit_rows = run_nested_variants(
        ("raw_numeric", "raw_plus_keys_ohe"),
        SECONDARY_SEED,
        y,
        numeric,
        keys,
        decompositions,
        interactions,
        logger,
        [],
        [],
    )
    rows: list[dict[str, Any]] = []
    baseline = seed_predictions["raw_numeric"]
    for variant, prediction in seed_predictions.items():
        rows.append(
            {
                "seed": SECONDARY_SEED,
                "variant": variant,
                "pooled_oof_auc": float(roc_auc_score(y, prediction)),
                "delta_vs_secondary_raw_numeric": float(
                    roc_auc_score(y, prediction) - roc_auc_score(y, baseline)
                ),
                "positive_fold_count": int(
                    sum(
                        roc_auc_score(y[apply], prediction[apply])
                        > roc_auc_score(y[apply], baseline[apply])
                        for _, apply in _outer_fold_splits(y, SECONDARY_SEED)
                    )
                ),
                "runtime_seconds": float(
                    sum(row["inner_tuning_runtime_seconds"] + row["outer_fit_runtime_seconds"] for row in fit_rows if row["variant"] == variant)
                ),
                "interpretation": "Robustness replication on a second seed; not independent confirmation.",
            }
        )
    return pd.DataFrame(rows), seed_predictions


def _write_report(
    output_dir: Path,
    input_path: Path,
    input_hash: str,
    n_rows: int,
    n_columns: int,
    seed: int,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    ablation_rows: list[dict[str, Any]],
    negative: pd.DataFrame,
    replication: pd.DataFrame,
    started_at: datetime,
    completed_at: datetime,
) -> None:
    result_by_variant = summary.set_index("variant")
    raw_auc = float(result_by_variant.loc["raw_numeric", "pooled_oof_auc"])
    ohe_auc = float(result_by_variant.loc["raw_plus_keys_ohe", "pooled_oof_auc"])
    ohe_delta = ohe_auc - raw_auc
    fold_deltas = (
        summary.loc[summary["variant"] == "raw_plus_keys_ohe", "positive_folds_vs_raw_numeric"]
        .astype(int)
        .iloc[0]
    )
    single_key_summary: dict[str, float] = {}
    for row in ablation_rows:
        if str(row["variant"]).startswith("single_"):
            single_key_summary[str(row["variant"])] = float(row["pooled_oof_auc"])
    top_single_name = max(single_key_summary, key=single_key_summary.get)
    control_summary = (
        negative.groupby("control", as_index=False)["pooled_oof_auc"]
        .mean()
        .to_dict(orient="records")
    )
    lines = [
        "# Phase 12 execution report",
        "",
        "## Executive status",
        "",
        f"- Run folder: `{output_dir}`",
        f"- Execution window (UTC): {started_at.isoformat()} to {completed_at.isoformat()}",
        f"- Input: `{input_path}`; SHA-256 `{input_hash}`; {n_rows:,} rows and {n_columns} columns.",
        f"- Primary seed: `{seed}`; one fixed stratified five-fold assignment shared by all primary variants.",
        f"- Raw numeric OOF AUC: **{raw_auc:.6f}**.",
        f"- Raw numeric plus eight-key one-hot OOF AUC: **{ohe_auc:.6f}** (paired change {ohe_delta:+.6f}).",
        f"- Eight-key one-hot positive folds: **{fold_deltas}/5**.",
        "- All key formulas are label-free, but the historical 130.5 offset and 0.0066 step have unresolved provenance; eligibility is therefore not conclusively established.",
        "- CPU LightGBM was used. No test data, target encoding, target-derived category statistics, external data, or submission was used.",
        "",
        "## Representation results",
        "",
        "| Representation | Pooled OOF AUC | Delta vs raw numeric | Positive folds |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['pooled_oof_auc']:.6f} | "
            f"{row['delta_vs_raw_numeric']:+.6f} | "
            f"{int(row['positive_folds_vs_raw_numeric'])}/5 |"
        )
    lines.extend(
        [
            "",
            "## Statistical reporting",
            "",
            "Paired fold differences and the conditional stratified bootstrap intervals are in `FOLD_LEVEL_METRICS.csv` and `PAIRED_BOOTSTRAP_REPORT.md`. The bootstrap resamples the OOF rows conditional on these fitted models and is not independent confirmation.",
            "",
            "## Ablation and negative-control interpretation",
            "",
            f"- The strongest single key by pooled OOF AUC was `{top_single_name}` ({single_key_summary[top_single_name]:.6f}); the full fold-level `PRODUCT_KEY_INDIVIDUAL_ABLATION.csv` remains exploratory.",
            "- Individual and leave-one-out ablations reuse the exposed outer folds and share the all-key-selected configuration/rounds. They are diagnostic, not independent key selection evidence.",
            f"- Mean negative-control pooled AUCs by control type: `{json.dumps(control_summary, sort_keys=True)}`. Every fixed-seed repetition and fold is retained in `NEGATIVE_CONTROL_RESULTS.csv`.",
            "",
            "## Secondary-seed replication",
            "",
        ]
    )
    for _, row in replication.iterrows():
        lines.append(
            f"- `{row['variant']}` seed {int(row['seed'])}: pooled OOF AUC "
            f"{row['pooled_oof_auc']:.6f}; paired change vs that seed's raw numeric "
            f"baseline {row['delta_vs_secondary_raw_numeric']:+.6f}; "
            f"{int(row['positive_fold_count'])}/5 positive folds."
        )
    lines.extend(
        [
            "",
            "This is robustness replication on a new seed/fold assignment, not independent confirmation.",
            "",
            "## Competition eligibility gate",
            "",
            "No model in this run uses target-derived feature statistics or test-fitted transforms. The 130.5 fee offset and 0.0066 financing step are preserved solely to reproduce the Phase 11 definitions; the available audit does not establish their permitted-data derivation. Accordingly, P4/P5/KQ3/KQ4 and decompositions containing these constants remain eligibility-pending. Formula-level label freedom does not prove organizer approval.",
            "",
            "## Integrity, runtime, and limits",
            "",
            "OOF row identity, target alignment, fold coverage, finite prediction values, row counts, source hashes, and artifact hashes are checked in `OOF_INTEGRITY_REPORT.md` and `SHA256_MANIFEST.csv`. Per-fit CPU runtime, sparse feature count, nonzero count, and process peak working set are recorded in the comparison outputs.",
            "",
            "The primary comparison is a research result on previously examined competition training labels. The OOF metrics and bootstrap do not provide an untouched holdout, external confirmation, official score, competition approval, or a champion promotion.",
            "",
            "## Required final decisions",
            "",
            "1. **Can label-free representations preserve most of the historical eight-key gain?** Phase 12 measures their gains against its own raw-numeric baseline; it is not directly comparable to Phase 11's CatBoost baseline or its LightGBM model with rowwise features and raw categorical one-hot inputs.",
            "2. **Which individual keys are strongest?** Treat the ablation ranking as exploratory; the report records the full pooled and fold-level results, and Q3 is not removed solely from the prior diagnostic.",
            "3. **Is the improvement stable across folds and seeds?** Use the primary five-fold paired results and the explicitly labeled secondary-seed replication; neither is independent confirmation.",
            "4. **Do genuine keys outperform matched controls?** Compare genuine all-key OOF performance to all five fixed-seed repetitions of both controls; the control artifacts retain every fold.",
            "5. **Which model is demonstrably eligible?** The feature transforms avoid target-derived category statistics and test fitting, but undocumented key constants prevent a definitive eligibility assertion. No model is promoted as competition-ready.",
            "6. **Single best next experiment:** obtain documented provenance or organizer clarification for the 130.5/0.0066 constants, then independently replicate the best fully supported representation on a predeclared clean evaluation protocol.",
            "",
        ]
    )
    (output_dir / "PHASE12_EXECUTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(train_path: Path, output_root: Path, seed: int) -> Path:
    started_at = datetime.now(timezone.utc)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"phase12_{run_stamp}"
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(output_dir)
    logger.info("Starting Phase 12; output=%s", output_dir)
    logger.info(
        "Clock UTC=%s local=%s; Phase 11 primary manifest records started_at_utc=2026-11-04T16:18:30.309282+00:00 although its folder suffix is 20261105; the timestamp confirms Phase 11 preceded this run and the suffix is only a folder label.",
        started_at.isoformat(),
        started_at.astimezone().isoformat(),
    )
    input_hash_before = sha256_file(train_path)
    train = pd.read_csv(train_path, low_memory=False)
    if ID_COLUMN not in train or TARGET_COLUMN not in train:
        raise ValueError(f"Training CSV must contain {ID_COLUMN} and {TARGET_COLUMN}")
    if train[ID_COLUMN].isna().any() or train[ID_COLUMN].duplicated().any():
        raise ValueError("Training IDs must be present and unique")
    if train[TARGET_COLUMN].isna().any() or set(train[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("Training target must be complete and binary")
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET_COLUMN])
    numeric = build_numeric_predictors(raw)
    keys = build_product_keys(raw)
    decompositions = build_decomposed_features(raw)
    interactions = build_selected_interactions(raw)
    if not (
        len(numeric)
        == len(keys)
        == len(decompositions)
        == len(interactions)
        == len(y)
    ):
        raise AssertionError("Feature builders changed row counts")
    if input_hash_before != sha256_file(train_path):
        raise AssertionError("Training input changed during initial read")
    input_hash = input_hash_before
    logger.info(
        "Loaded permitted training data only: rows=%s predictors=%s numeric=%s input_sha256=%s",
        len(train),
        len(raw.columns) - 1,
        len(numeric.columns),
        input_hash,
    )
    logger.info(
        "Dependencies: Python=%s pandas=%s numpy=%s sklearn=%s LightGBM=%s scipy=%s; device=CPU; n_jobs=4",
        platform.python_version(),
        pd.__version__,
        np.__version__,
        sklearn.__version__,
        lightgbm.__version__,
        scipy.__version__,
    )
    logger.info(
        "Primary config grid=%s; outer=%s inner=%s; max_iterations=%s early_stopping_rounds=%s",
        json.dumps(CONFIGS, sort_keys=True),
        OUTER_FOLDS,
        INNER_FOLDS,
        MAX_ITERATIONS,
        EARLY_STOPPING_ROUNDS,
    )

    support_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    main_predictions, selected_by_fold, fit_rows = run_nested_variants(
        MAIN_VARIANTS,
        seed,
        y,
        numeric,
        keys,
        decompositions,
        interactions,
        logger,
        support_rows,
        selection_rows,
    )
    ablation_predictions, ablation_rows = run_ablation_grid(
        y,
        numeric,
        keys,
        decompositions,
        interactions,
        selected_by_fold,
        seed,
        logger,
        support_rows,
    )
    control_results, control_summary, control_predictions = run_negative_controls(
        y,
        numeric,
        keys,
        decompositions,
        interactions,
        selected_by_fold,
        main_predictions["raw_numeric"],
        main_predictions["raw_plus_keys_ohe"],
        seed,
        logger,
    )
    negative = control_results.copy()
    for row in control_summary:
        match = (
            (negative["control"] == row["control"])
            & (negative["repetition"] == row["repetition"])
        )
        negative.loc[match, "pooled_oof_auc"] = row["pooled_oof_auc"]

    logger.info("Primary comparison, ablations, and controls are complete.")
    replication, secondary_predictions = _run_secondary_seed(
        y,
        numeric,
        keys,
        decompositions,
        interactions,
        logger,
    )
    logger.info("Secondary-seed robustness replication is complete.")

    primary_fold_metrics = _fold_metrics(y, main_predictions, seed)
    summary = _summaries(
        y, main_predictions, primary_fold_metrics, fit_rows, selected_by_fold
    )
    baseline_auc = float(roc_auc_score(y, main_predictions["raw_numeric"]))
    for variant, prediction in main_predictions.items():
        fold_auc_by_fold = {
            fold: roc_auc_score(
                y[apply],
                prediction[apply],
            )
            for fold, (_, apply) in enumerate(_outer_fold_splits(y, seed), start=1)
        }
        summary.loc[summary["variant"] == variant, "per_fold_auc"] = json.dumps(
            fold_auc_by_fold, sort_keys=True
        )
    bootstrap = _paired_bootstrap(
        y,
        main_predictions["raw_numeric"],
        main_predictions,
        seed + 900_000,
    )
    model_comparison = summary.copy()
    inner_selection = pd.DataFrame(selection_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    all_key_auc = float(roc_auc_score(y, ablation_predictions["all_eight"]))
    ablation_df["pooled_oof_auc"] = ablation_df["variant"].map(
        {
            name: float(roc_auc_score(y, prediction))
            for name, prediction in ablation_predictions.items()
        }
    )
    all_eight_by_fold = (
        ablation_df.loc[ablation_df["variant"] == "all_eight"]
        .set_index("fold")["fold_auc"]
    )
    ablation_df["paired_delta_vs_all_eight"] = ablation_df.apply(
        lambda row: row["fold_auc"] - float(all_eight_by_fold.loc[row["fold"]]),
        axis=1,
    )
    ablation_rows = ablation_df.to_dict(orient="records")
    main_fold_output = primary_fold_metrics.rename(
        columns={
            "paired_delta_vs_raw_numeric": "paired_fold_delta",
        }
    ).assign(
        reference_variant="raw_numeric",
        experiment_group="main_representation",
        repetition=None,
        control_seed=None,
    )
    ablation_fold_output = ablation_df[
        ["variant", "fold", "fold_auc", "paired_delta_vs_all_eight"]
    ].rename(
        columns={
            "fold_auc": "auc",
            "paired_delta_vs_all_eight": "paired_fold_delta",
        }
    ).assign(
        reference_variant="all_eight",
        positive_delta=lambda frame: frame["paired_fold_delta"] > 0,
        experiment_group="product_key_ablation",
        repetition=None,
        control_seed=None,
    )
    negative_fold_output = negative.rename(
        columns={
            "control": "variant",
            "fold_auc": "auc",
            "paired_delta_vs_raw_numeric": "paired_fold_delta",
        }
    ).assign(
        reference_variant="raw_numeric",
        positive_delta=lambda frame: frame["paired_fold_delta"] > 0,
        experiment_group="negative_control",
    )
    combined_fold_metrics = pd.concat(
        [
            main_fold_output,
            ablation_fold_output,
            negative_fold_output,
        ],
        ignore_index=True,
        sort=False,
    )
    individual = ablation_df.loc[
        ablation_df["variant"].str.startswith("single_")
        | (ablation_df["variant"] == "all_eight")
    ].copy()
    leave_one_out = ablation_df.loc[
        ablation_df["variant"].str.startswith("leave_out_")
    ].copy()
    individual.to_csv(output_dir / "PRODUCT_KEY_INDIVIDUAL_ABLATION.csv", index=False)
    leave_one_out.to_csv(output_dir / "PRODUCT_KEY_LEAVE_ONE_OUT.csv", index=False)
    ablation_df.to_csv(output_dir / "PRODUCT_KEY_ABLATION_ALL_VARIANTS.csv", index=False)
    negative.to_csv(output_dir / "NEGATIVE_CONTROL_RESULTS.csv", index=False)
    model_comparison.to_csv(output_dir / "MODEL_COMPARISON.csv", index=False)
    summary.to_csv(output_dir / "LABEL_FREE_REPRESENTATION_RESULTS.csv", index=False)
    combined_fold_metrics.to_csv(output_dir / "FOLD_LEVEL_METRICS.csv", index=False)
    bootstrap.to_csv(output_dir / "PAIRED_BOOTSTRAP_REPORT.csv", index=False)
    replication.to_csv(output_dir / "SECONDARY_SEED_REPLICATION.csv", index=False)
    inner_selection.to_csv(output_dir / "INNER_MODEL_SELECTION.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "OUTER_MODEL_FITS.csv", index=False)
    _eligibility_matrix().to_csv(
        output_dir / "COMPETITION_ELIGIBILITY_MATRIX.csv", index=False
    )
    _formula_audit(keys, raw).to_csv(
        output_dir / "PHASE12_FEATURE_FORMULA_AUDIT.csv", index=False
    )

    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
            "primary_fold": np.full(len(train), -1, dtype=np.int8),
        }
    )
    for fold, (_, apply) in enumerate(_outer_fold_splits(y, seed), start=1):
        oof.loc[apply, "primary_fold"] = fold
    for name, prediction in main_predictions.items():
        oof[name] = prediction
    for name, prediction in ablation_predictions.items():
        oof[f"ablation_{name}"] = prediction
    oof.to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    control_oof = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
        }
    )
    for name, prediction in control_predictions.items():
        control_oof[name] = prediction
    control_oof.to_csv(
        output_dir / "NEGATIVE_CONTROL_OOF_PREDICTIONS.csv.gz",
        index=False,
        compression="gzip",
    )
    secondary_oof = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int64),
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
            "raw_numeric": secondary_predictions["raw_numeric"],
            "raw_plus_keys_ohe": secondary_predictions["raw_plus_keys_ohe"],
        }
    )
    for fold, (_, apply) in enumerate(
        _outer_fold_splits(y, SECONDARY_SEED), start=1
    ):
        secondary_oof.loc[apply, "secondary_fold"] = fold
    secondary_oof.to_csv(output_dir / "SECONDARY_SEED_OOF_PREDICTIONS.csv", index=False)

    category_report = _category_support_hash_collisions(support_rows, [])
    for outer_fold, (outer_fit, outer_apply) in enumerate(
        _outer_fold_splits(y, seed), start=1
    ):
        for column in PRODUCT_KEYS:
            hasher = FeatureHasher(
                n_features=HASH_FEATURES_PER_KEY,
                input_type="string",
                dtype=np.float32,
                alternate_sign=False,
            )
            fit_values = keys.iloc[outer_fit][column].astype(str).tolist()
            unique_values = sorted(set(fit_values))
            buckets = hasher.transform([[value] for value in unique_values]).tocoo().col
            category_report = pd.concat(
                [
                    category_report,
                    pd.DataFrame(
                        [
                            {
                                "variant": "raw_plus_keys_hash",
                                "fold": outer_fold,
                                "key": column,
                                "fit_rows": len(outer_fit),
                                "apply_rows": len(outer_apply),
                                "fit_unique_categories": len(unique_values),
                                "fit_singleton_categories": int(
                                    (keys.iloc[outer_fit][column].astype(str).value_counts() == 1).sum()
                                ),
                                "fit_rare_categories_below_minimum": "",
                                "fit_categories_at_or_above_minimum": "",
                                "apply_unseen_categories": int(
                                    len(
                                        set(keys.iloc[outer_apply][column].astype(str))
                                        - set(fit_values)
                                    )
                                ),
                                "minimum_fit_count": "",
                                "distinct_hash_buckets": int(np.unique(buckets).size),
                                "observed_category_bucket_collisions": len(unique_values)
                                - int(np.unique(buckets).size),
                                "hash_buckets_per_key": HASH_FEATURES_PER_KEY,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    category_report.to_csv(output_dir / "CATEGORY_SUPPORT_REPORT.csv", index=False)
    for _, row in summary.iterrows():
        logger.info(
            "summary variant=%s pooled_auc=%.6f delta=%+.6f",
            row["variant"],
            row["pooled_oof_auc"],
            row["delta_vs_raw_numeric"],
        )

    integrity_issues: list[str] = []
    if len(oof) != len(train):
        integrity_issues.append("OOF row count differs from training rows")
    if oof[ID_COLUMN].duplicated().any():
        integrity_issues.append("OOF identifiers are not unique")
    if not np.array_equal(oof[TARGET_COLUMN].to_numpy(), y):
        integrity_issues.append("OOF target does not align with training order")
    if (oof["primary_fold"] < 1).any():
        integrity_issues.append("Primary fold assignment is incomplete")
    all_prediction_columns = list(main_predictions) + [
        f"ablation_{name}" for name in ablation_predictions
    ]
    if not oof.loc[:, all_prediction_columns].notna().all().all():
        integrity_issues.append("At least one main or ablation OOF prediction is missing")
    if not oof.loc[:, all_prediction_columns].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        integrity_issues.append("At least one main or ablation OOF prediction is outside [0,1]")
    if set(oof["primary_fold"].unique()) != set(range(1, OUTER_FOLDS + 1)):
        integrity_issues.append("Primary fold labels are not exactly 1..5")
    if control_oof[ID_COLUMN].duplicated().any():
        integrity_issues.append("Negative-control OOF identifiers are not unique")
    if not np.array_equal(control_oof[TARGET_COLUMN].to_numpy(), y):
        integrity_issues.append("Negative-control target does not align to training order")
    if not control_oof.loc[:, list(control_predictions)].notna().all().all():
        integrity_issues.append("At least one negative-control OOF prediction is missing")
    if not control_oof.loc[:, list(control_predictions)].apply(
        lambda column: column.between(0, 1).all()
    ).all():
        integrity_issues.append("A negative-control OOF prediction is outside [0,1]")
    if secondary_oof[ID_COLUMN].duplicated().any():
        integrity_issues.append("Secondary-seed OOF identifiers are not unique")
    if not np.array_equal(secondary_oof[TARGET_COLUMN].to_numpy(), y):
        integrity_issues.append("Secondary-seed target does not align to training order")
    if secondary_oof["secondary_fold"].isna().any():
        integrity_issues.append("Secondary-seed fold assignment is incomplete")
    if not secondary_oof.loc[
        :, ["raw_numeric", "raw_plus_keys_ohe"]
    ].notna().all().all():
        integrity_issues.append("A secondary-seed OOF prediction is missing")
    input_hash_after = sha256_file(train_path)
    if input_hash != input_hash_after:
        integrity_issues.append("Input hash changed during execution")
    if integrity_issues:
        raise AssertionError("; ".join(integrity_issues))
    integrity_text = [
        "# OOF integrity report",
        "",
        f"- Training file: `{train_path}`",
        f"- Training SHA-256 before/after run: `{input_hash}` / `{input_hash_after}`",
        f"- Rows: {len(train):,}; unique IDs: {oof[ID_COLUMN].nunique():,}; target positives: {int(y.sum()):,}.",
        f"- Primary folds: {OUTER_FOLDS}; seed `{seed}`; each row receives exactly one outer validation prediction.",
        f"- Secondary seed: `{SECONDARY_SEED}`; its OOF prediction file includes a distinct complete fold assignment.",
        f"- Main-model OOF predictions present: {len(main_predictions)}; all finite and within [0,1].",
        f"- Ablation-model OOF predictions present: {len(ablation_predictions)}; all finite and within [0,1].",
        f"- Negative-control OOF predictions present: {len(control_predictions)}; all finite and within [0,1], with IDs and targets aligned.",
        f"- Missing prediction count across primary, ablation, control, and secondary OOF outputs: {int(oof.loc[:, all_prediction_columns].isna().sum().sum() + control_oof.loc[:, list(control_predictions)].isna().sum().sum() + secondary_oof.loc[:, ['raw_numeric', 'raw_plus_keys_ohe']].isna().sum().sum())}.",
        "- ID order and target alignment were checked against the exact training input row order.",
        "- The runner did not open test.csv, sample_submission.csv, external files, or a holdout.",
        "- Negative-control predictions were validated for completeness before pooled metrics; each control repetition and fold is recorded in NEGATIVE_CONTROL_RESULTS.csv.",
        "- No target encoding, target-derived category statistics, CatBoost CTRs, or test-fitted transforms were used.",
        "- Input file hash remained unchanged during execution.",
        "- Status: PASS.",
        "",
    ]
    (output_dir / "OOF_INTEGRITY_REPORT.md").write_text(
        "\n".join(integrity_text), encoding="utf-8"
    )
    paired_markdown = [
        "# Paired bootstrap report",
        "",
        "Intervals are paired, stratified row bootstraps of the saved OOF predictions. They are conditional on fitted models and previously examined labels, not independent confirmation.",
        "",
        "| Comparison | Paired delta | 95% low | 95% high | Repetitions |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in bootstrap.iterrows():
        paired_markdown.append(
            f"| {row['comparison']} | {row['paired_delta']:+.6f} | "
            f"{row['paired_stratified_bootstrap_95_low']:+.6f} | "
            f"{row['paired_stratified_bootstrap_95_high']:+.6f} | "
            f"{int(row['bootstrap_repetitions'])} |"
        )
    paired_markdown.append("")
    (output_dir / "PAIRED_BOOTSTRAP_REPORT.md").write_text(
        "\n".join(paired_markdown), encoding="utf-8"
    )
    _write_report(
        output_dir,
        train_path,
        input_hash,
        len(train),
        len(train.columns),
        seed,
        summary,
        bootstrap,
        ablation_rows,
        negative,
        replication,
        started_at,
        datetime.now(timezone.utc),
    )

    best_variant = str(
        summary.loc[summary["pooled_oof_auc"].idxmax(), "variant"]
    )
    recommendation = [
        "# Next phase recommendation",
        "",
        f"- Highest primary OOF result among the predeclared representations: `{best_variant}`.",
        "- Do not declare a champion from this exposed training-label comparison.",
        "- First establish permitted-data provenance for the fixed 130.5 and 0.0066 constants, or obtain written organizer guidance.",
        "- Then run a predeclared, independent robustness experiment for the best eligible representation. Keep the primary and secondary seed outputs intact.",
        "- Do not launch another broad candidate scan, use leaderboard feedback, create a submission, or promote a model based on this phase alone.",
        "",
    ]
    (output_dir / "NEXT_PHASE_RECOMMENDATION.md").write_text(
        "\n".join(recommendation), encoding="utf-8"
    )

    logger.info(
        "All Phase 12 model, validation, report, and integrity artifacts are written; "
        "finalizing hashes. elapsed=%s raw_auc=%.6f all_key_auc=%.6f",
        datetime.now(timezone.utc) - started_at,
        baseline_auc,
        all_key_auc,
    )
    for handler in logger.handlers:
        handler.flush()
    logger.handlers.clear()
    logging.shutdown()
    manifest_path = output_dir / "SHA256_MANIFEST.csv"
    manifest_files = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != manifest_path.name
    ]
    manifest_rows = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(manifest_files)
    ]
    manifest_rows.extend(
        [
            {
                "file": f"source:{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                Path(__file__),
                ROOT / "phase12_feature_definitions.py",
                ROOT / "test_phase12_feature_definitions.py",
            )
        ]
    )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 12 label-free product representation research."
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = parser.parse_args()
    run(args.train, args.output_root, args.seed)


if __name__ == "__main__":
    main()
