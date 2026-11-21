from __future__ import annotations

import argparse
import csv
import gc
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
import xgboost
from lightgbm import LGBMClassifier, early_stopping
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

PHASE13_DIR = Path(__file__).resolve().parents[1] / "phase13_matched_baseline"
sys.path.insert(0, str(PHASE13_DIR))
import run_phase13 as phase13

from phase14_features import build_numeric_enhancements


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_OOF = (
    PHASE13_DIR
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
MAX_ITERATIONS = 1200
EARLY_STOPPING_ROUNDS = 80
BOOTSTRAP_REPETITIONS = 1000
ALL_KEYS = phase13.PRODUCT_KEYS
CONSTANT_FREE_KEYS = (
    "PRODUCT_KEY_Q3",
    "PRODUCT_KEY_Q4",
    "PRODUCT_KEY_Q5",
    "PRODUCT_KEY_AG3",
)

VARIANTS: dict[str, dict[str, Any]] = {
    "lgbm_all8_e": {
        "model": "lightgbm",
        "features": "phase13_E",
        "keys": ALL_KEYS,
        "augmented": False,
    },
    "lgbm_all8_augmented": {
        "model": "lightgbm",
        "features": "phase14_numeric",
        "keys": ALL_KEYS,
        "augmented": True,
    },
    "xgb_all8_e": {
        "model": "xgboost",
        "features": "phase13_E",
        "keys": ALL_KEYS,
        "augmented": False,
    },
    "xgb_all8_augmented": {
        "model": "xgboost",
        "features": "phase14_numeric",
        "keys": ALL_KEYS,
        "augmented": True,
    },
    "lgbm_constant_free_keys": {
        "model": "lightgbm",
        "features": "phase14_numeric",
        "keys": CONSTANT_FREE_KEYS,
        "augmented": True,
    },
}
LGB_CONFIGS = (
    {"name": "leaves31_child80", "num_leaves": 31, "min_child_samples": 80},
    {"name": "leaves63_child100", "num_leaves": 63, "min_child_samples": 100},
)
XGB_CONFIGS = (
    {"name": "depth4_child10", "max_depth": 4, "min_child_weight": 10},
    {"name": "depth6_child20", "max_depth": 6, "min_child_weight": 20},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase14")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    file_handler = logging.FileHandler(
        output_dir / "EXECUTION_LOG.txt", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def verify_inputs(
    train: pd.DataFrame, phase13_oof: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if "TARGET" not in train or "SK_ID_CURR" not in train:
        raise ValueError("Training input must include TARGET and SK_ID_CURR")
    if len(train) != len(phase13_oof):
        raise ValueError("Phase 13 OOF row count differs from the training input")
    expected_indices = np.arange(len(train))
    if not np.array_equal(phase13_oof["row_index"].to_numpy(), expected_indices):
        raise ValueError("Phase 13 OOF rows are not in original input order")
    if not np.array_equal(
        phase13_oof["SK_ID_CURR"].to_numpy(), train["SK_ID_CURR"].to_numpy()
    ):
        raise ValueError("Phase 13 OOF IDs do not align to training input")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    if not np.array_equal(phase13_oof["TARGET"].to_numpy(dtype=np.int8), y):
        raise ValueError("Phase 13 OOF targets do not align to training input")
    required_oof = {
        "primary_fold",
        "C_complete_raw_rowwise",
        "D_C_plus_keys_ohe",
        "E_C_plus_keys_frequency_ohe",
    }
    missing = required_oof.difference(phase13_oof.columns)
    if missing:
        raise ValueError(f"Phase 13 OOF file lacks required columns: {sorted(missing)}")
    if not phase13_oof[
        ["C_complete_raw_rowwise", "D_C_plus_keys_ohe", "E_C_plus_keys_frequency_ohe"]
    ].apply(np.isfinite).all().all():
        raise ValueError("Phase 13 reference OOF predictions contain invalid values")

    recomputed = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, apply) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        recomputed[apply] = fold
    if not np.array_equal(
        phase13_oof["primary_fold"].to_numpy(dtype=np.int8), recomputed
    ):
        raise ValueError("Phase 13 primary folds do not match the preregistered seed")
    return y, recomputed


def _inner_splits(
    outer_fit: np.ndarray, y: np.ndarray, fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold
    )
    return [
        (outer_fit[fit], outer_fit[apply])
        for fit, apply in splitter.split(
            np.zeros(len(outer_fit), dtype=np.int8), y[outer_fit]
        )
    ]


def _feature_blocks(
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rowwise: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    enhancements: pd.DataFrame,
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del raw
    if not variant["augmented"]:
        return numeric, categorical, rowwise, keys, decompositions
    expanded_rowwise = pd.concat(
        [
            rowwise.reset_index(drop=True),
            decompositions.reset_index(drop=True),
            enhancements.reset_index(drop=True),
        ],
        axis=1,
    )
    return numeric, categorical, expanded_rowwise, keys, decompositions


def _matrix_pair(
    fit: np.ndarray,
    apply: np.ndarray,
    blocks: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    key_columns: tuple[str, ...],
):
    return phase13.make_matrices(
        fit,
        apply,
        *blocks,
        "E_C_plus_keys_frequency_ohe",
        key_columns=key_columns,
    )


def _lgb_model(config: dict[str, Any], seed: int, rounds: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
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


def _xgb_model(config: dict[str, Any], seed: int, rounds: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=rounds,
        learning_rate=0.03,
        max_depth=int(config["max_depth"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        max_bin=127,
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        verbosity=0,
    )


def _fit_inner(
    model_name: str,
    config: dict[str, Any],
    x_fit,
    y_fit: np.ndarray,
    x_apply,
    y_apply: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    started = time.perf_counter()
    if model_name == "lightgbm":
        model = _lgb_model(config, seed, MAX_ITERATIONS)
        model.fit(
            x_fit,
            y_fit,
            eval_X=x_apply,
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
        rounds = int(model.best_iteration_ or MAX_ITERATIONS)
    else:
        model = _xgb_model(config, seed, MAX_ITERATIONS)
        model.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(x_fit, y_fit, eval_set=[(x_apply, y_apply)], verbose=False)
        rounds = int(model.best_iteration or MAX_ITERATIONS - 1) + 1
    prediction = model.predict_proba(x_apply)[:, 1]
    return prediction, rounds, time.perf_counter() - started


def _fit_final(
    model_name: str,
    config: dict[str, Any],
    x_fit,
    y_fit: np.ndarray,
    rounds: int,
    seed: int,
) -> tuple[Any, float]:
    model = (
        _lgb_model(config, seed, rounds)
        if model_name == "lightgbm"
        else _xgb_model(config, seed, rounds)
    )
    started = time.perf_counter()
    model.fit(x_fit, y_fit)
    return model, time.perf_counter() - started


def _config_seed(fold: int, inner_fold: int, config_index: int, model_index: int) -> int:
    return int(SEED + fold * 10000 + inner_fold * 100 + config_index * 10 + model_index)


def _tune_and_predict(
    variant_name: str,
    variant: dict[str, Any],
    outer_fit: np.ndarray,
    outer_apply: np.ndarray,
    y: np.ndarray,
    fold: int,
    blocks: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
    selection_rows: list[dict[str, Any]],
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_name = str(variant["model"])
    configs = LGB_CONFIGS if model_name == "lightgbm" else XGB_CONFIGS
    key_columns = tuple(variant["keys"])
    config_predictions = {
        str(config["name"]): np.full(len(outer_fit), np.nan, dtype=np.float64)
        for config in configs
    }
    best_rounds: dict[str, list[int]] = {
        str(config["name"]): [] for config in configs
    }
    config_runtime = {str(config["name"]): 0.0 for config in configs}

    for inner_fold, (fit, apply) in enumerate(
        _inner_splits(outer_fit, y, fold), start=1
    ):
        x_fit, x_apply, matrix_info = _matrix_pair(
            fit, apply, blocks, key_columns
        )
        local_apply = np.searchsorted(outer_fit, apply)
        for config_index, config in enumerate(configs, start=1):
            prediction, rounds, elapsed = _fit_inner(
                model_name,
                config,
                x_fit,
                y[fit],
                x_apply,
                y[apply],
                _config_seed(fold, inner_fold, config_index, 1 if model_name == "lightgbm" else 2),
            )
            name = str(config["name"])
            config_predictions[name][local_apply] = prediction
            best_rounds[name].append(rounds)
            config_runtime[name] += elapsed
            selection_rows.append(
                {
                    "variant": variant_name,
                    "model": model_name,
                    "outer_fold": fold,
                    "inner_fold": inner_fold,
                    "configuration": name,
                    "best_iteration": rounds,
                    "inner_rows": len(apply),
                    "feature_count": matrix_info["feature_count"],
                    "runtime_seconds": elapsed,
                }
            )
            del prediction
            gc.collect()
        del x_fit, x_apply

    scores = {
        name: roc_auc_score(y[outer_fit], predictions)
        for name, predictions in config_predictions.items()
        if np.isfinite(predictions).all()
    }
    if len(scores) != len(configs):
        raise RuntimeError(f"Incomplete inner OOF coverage for {variant_name}, fold {fold}")
    selected_name = max(scores, key=scores.get)
    selected_config = next(
        config for config in configs if config["name"] == selected_name
    )
    rounds = max(
        1, int(np.median(np.asarray(best_rounds[selected_name], dtype=np.int64)))
    )
    x_fit, x_apply, matrix_info = _matrix_pair(
        outer_fit, outer_apply, blocks, key_columns
    )
    final_model, elapsed = _fit_final(
        model_name,
        selected_config,
        x_fit,
        y[outer_fit],
        rounds,
        SEED + fold * 10000 + (31 if model_name == "lightgbm" else 32),
    )
    prediction = final_model.predict_proba(x_apply)[:, 1]
    logger.info(
        "%s fold=%d selected=%s inner_auc=%.6f rounds=%d features=%d fit_seconds=%.1f",
        variant_name,
        fold,
        selected_name,
        scores[selected_name],
        rounds,
        matrix_info["feature_count"],
        elapsed,
    )
    return prediction, {
        "variant": variant_name,
        "model": model_name,
        "outer_fold": fold,
        "selected_configuration": selected_name,
        "selected_inner_pooled_auc": float(scores[selected_name]),
        "final_rounds": rounds,
        "final_fit_seconds": elapsed,
        "feature_count": matrix_info["feature_count"],
        "inner_config_auc": json.dumps(scores, sort_keys=True),
        "inner_rounds": json.dumps(best_rounds[selected_name]),
    }


def _stratified_bootstrap_deltas(
    y: np.ndarray,
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    delta_samples = {
        name: np.empty(repetitions, dtype=np.float64) for name in candidates
    }
    reference_auc_samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        reference_auc_samples[index] = roc_auc_score(y[sample], reference[sample])
        for name, prediction in candidates.items():
            delta_samples[name][index] = (
                roc_auc_score(y[sample], prediction[sample])
                - reference_auc_samples[index]
            )
    rows = []
    for name, prediction in candidates.items():
        deltas = delta_samples[name]
        rows.append(
            {
                "candidate": name,
                "reference": "phase13_E",
                "delta_auc": roc_auc_score(y, prediction)
                - roc_auc_score(y, reference),
                "bootstrap_repetitions": repetitions,
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "interpretation": "Conditional on fitted OOF predictions; not independent confirmation.",
            }
        )
    return rows


def _subgroup_rows(
    y: np.ndarray,
    fold_ids: np.ndarray,
    raw: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(1, FOLDS + 1):
        apply = np.flatnonzero(fold_ids == fold)
        fit = np.flatnonzero(fold_ids != fold)
        age_fit = -pd.to_numeric(raw.iloc[fit]["DAYS_BIRTH"]).to_numpy(
            dtype=np.float64
        ) / 365.25
        income_fit = pd.to_numeric(raw.iloc[fit]["AMT_INCOME_TOTAL"]).to_numpy(
            dtype=np.float64
        )
        age_cuts = np.unique(
            np.nanquantile(age_fit[np.isfinite(age_fit)], [0.25, 0.5, 0.75])
        )
        income_cuts = np.unique(
            np.nanquantile(
                income_fit[np.isfinite(income_fit)], [0.25, 0.5, 0.75]
            )
        )
        age = -pd.to_numeric(raw.iloc[apply]["DAYS_BIRTH"]).to_numpy(
            dtype=np.float64
        ) / 365.25
        income = pd.to_numeric(raw.iloc[apply]["AMT_INCOME_TOTAL"]).to_numpy(
            dtype=np.float64
        )
        groups = {
            "CODE_GENDER": raw.iloc[apply]["CODE_GENDER"].astype("string").fillna(
                "MISSING"
            ).to_numpy(dtype=str),
            "NAME_CONTRACT_TYPE": raw.iloc[apply]["NAME_CONTRACT_TYPE"]
            .astype("string")
            .fillna("MISSING")
            .to_numpy(dtype=str),
            "AGE_TRAIN_QUARTILE": np.where(
                np.isfinite(age),
                np.digitize(age, age_cuts, right=True).astype(str),
                "MISSING",
            ),
            "INCOME_TRAIN_QUARTILE": np.where(
                np.isfinite(income),
                np.digitize(income, income_cuts, right=True).astype(str),
                "MISSING",
            ),
        }
        for group_name, labels in groups.items():
            for label in np.unique(labels):
                mask = labels == label
                group_y = y[apply][mask]
                positives = int(group_y.sum())
                negatives = int(len(group_y) - positives)
                for model_name, prediction in predictions.items():
                    score = (
                        float(roc_auc_score(group_y, prediction[apply][mask]))
                        if positives and negatives
                        else float("nan")
                    )
                    rows.append(
                        {
                            "variant": model_name,
                            "outer_fold": fold,
                            "group": group_name,
                            "group_value": str(label),
                            "rows": int(mask.sum()),
                            "positives": positives,
                            "negatives": negatives,
                            "auc": score,
                        }
                    )
    return rows


def _write_manifest(output_dir: Path, source_files: list[Path]) -> None:
    entries = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            entries.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in source_files:
        entries.append(
            {
                "file": f"source:{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with (output_dir / "SHA256_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(entries)


def run(args: argparse.Namespace) -> Path:
    train_path = Path(args.train).resolve()
    phase13_path = Path(args.phase13_oof).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("phase14_%Y%m%dT%H%M%SZ")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(output_dir)
    started = time.perf_counter()
    logger.info("Starting Phase 14 label-free model development")
    logger.info("train=%s", train_path)
    logger.info("phase13_oof=%s", phase13_path)
    logger.info("test, submissions, and external data are out of scope and not opened")

    train = pd.read_csv(train_path)
    phase13_oof = pd.read_csv(phase13_path)
    y, fold_ids = verify_inputs(train, phase13_oof)
    raw = train.drop(columns=["TARGET"]).copy()
    numeric, categorical, rowwise = phase13.build_raw_feature_blocks(raw)
    keys = phase13.build_product_keys(raw)
    decompositions = phase13.build_label_free_decompositions(raw)
    enhancements = build_numeric_enhancements(raw.drop(columns=["SK_ID_CURR"]))
    if not (
        len(numeric)
        == len(categorical)
        == len(rowwise)
        == len(keys)
        == len(decompositions)
        == len(enhancements)
        == len(y)
    ):
        raise RuntimeError("Feature block row counts differ from the target")

    reference_predictions = {
        "phase13_C": phase13_oof["C_complete_raw_rowwise"].to_numpy(dtype=float),
        "phase13_D": phase13_oof["D_C_plus_keys_ohe"].to_numpy(dtype=float),
        "phase13_E": phase13_oof[
            "E_C_plus_keys_frequency_ohe"
        ].to_numpy(dtype=float),
    }
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(y), np.nan, dtype=np.float64)
        for name in VARIANTS
    }
    selection_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []

    for fold in range(1, FOLDS + 1):
        outer_fit = np.flatnonzero(fold_ids != fold)
        outer_apply = np.flatnonzero(fold_ids == fold)
        for variant_name, variant in VARIANTS.items():
            blocks = _feature_blocks(
                raw,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                enhancements,
                variant,
            )
            fold_prediction, fit_info = _tune_and_predict(
                variant_name,
                variant,
                outer_fit,
                outer_apply,
                y,
                fold,
                blocks,
                selection_rows,
                logger,
            )
            predictions[variant_name][outer_apply] = fold_prediction
            outer_rows.append(fit_info)
            del blocks, fold_prediction
            gc.collect()

    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete or invalid OOF predictions for {name}")

    # This equal-weight blend is preregistered; weights are not fit on the OOF labels.
    predictions["blend_lgbm_xgb_augmented_equal"] = (
        predictions["lgbm_all8_augmented"]
        + predictions["xgb_all8_augmented"]
    ) / 2.0
    all_predictions = {**reference_predictions, **predictions}

    model_rows = []
    fold_rows = []
    for name, prediction in all_predictions.items():
        pooled = float(roc_auc_score(y, prediction))
        fold_scores = []
        for fold in range(1, FOLDS + 1):
            apply = fold_ids == fold
            score = float(roc_auc_score(y[apply], prediction[apply]))
            fold_scores.append(score)
            fold_rows.append(
                {
                    "variant": name,
                    "outer_fold": fold,
                    "auc": score,
                    "delta_vs_phase13_E": float(
                        score
                        - roc_auc_score(
                            y[apply], reference_predictions["phase13_E"][apply]
                        )
                    ),
                    "validation_rows": int(apply.sum()),
                }
            )
        deltas = [
            fold_rows[-FOLDS + index]["delta_vs_phase13_E"]
            for index in range(FOLDS)
        ]
        model_rows.append(
            {
                "variant": name,
                "pooled_oof_auc": pooled,
                "mean_fold_auc": float(np.mean(fold_scores)),
                "fold_auc_sd": float(np.std(fold_scores, ddof=1)),
                "delta_vs_phase13_E": float(
                    pooled - roc_auc_score(y, reference_predictions["phase13_E"])
                ),
                "positive_folds_vs_phase13_E": int(sum(delta > 0 for delta in deltas)),
                "fold_auc_values": json.dumps(fold_scores),
            }
        )

    development_predictions = {
        name: prediction
        for name, prediction in predictions.items()
        if name != "blend_lgbm_xgb_augmented_equal"
    }
    bootstrap_rows = _stratified_bootstrap_deltas(
        y,
        reference_predictions["phase13_E"],
        {**development_predictions, "blend_lgbm_xgb_augmented_equal": predictions[
            "blend_lgbm_xgb_augmented_equal"
        ]},
        BOOTSTRAP_REPETITIONS,
        SEED + 100,
    )

    correlation_names = list(all_predictions)
    correlation_rows = []
    for left_index, left in enumerate(correlation_names):
        for right in correlation_names[left_index + 1 :]:
            correlation_rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman_correlation": float(
                        spearmanr(all_predictions[left], all_predictions[right]).statistic
                    ),
                }
            )

    subgroup_rows = _subgroup_rows(y, fold_ids, raw, all_predictions)
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            "SK_ID_CURR": train["SK_ID_CURR"].to_numpy(),
            "TARGET": y,
            "primary_fold": fold_ids,
            **reference_predictions,
            **predictions,
        }
    )
    oof.to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(model_rows).sort_values(
        "pooled_oof_auc", ascending=False
    ).to_csv(output_dir / "MODEL_RESULTS.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "FOLD_LEVEL_METRICS.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "PAIRED_BOOTSTRAP.csv", index=False
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "INNER_MODEL_SELECTION.csv", index=False
    )
    pd.DataFrame(outer_rows).to_csv(
        output_dir / "OUTER_MODEL_FITS.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "PREDICTION_CORRELATIONS.csv", index=False
    )
    pd.DataFrame(subgroup_rows).to_csv(
        output_dir / "SUBGROUP_STABILITY.csv", index=False
    )

    eligibility = pd.DataFrame(
        [
            {
                "feature_or_method": "Supplied raw predictors",
                "label_free_feature": "yes",
                "fit_scope": "Model fits use only outer-fit labels",
                "test_rows_used": "no",
                "local_rule_status": "Supplied-column use allowed by local materials",
                "eligibility_status": "No organizer certification claimed",
            },
            {
                "feature_or_method": "Phase 13 rowwise features and guarded ratios",
                "label_free_feature": "yes",
                "fit_scope": "Deterministic rowwise transforms",
                "test_rows_used": "no",
                "local_rule_status": "General deterministic transforms supported",
                "eligibility_status": "No organizer certification claimed",
            },
            {
                "feature_or_method": "Q3/Q4/Q5/AG3 rounded ratio keys",
                "label_free_feature": "yes",
                "fit_scope": "One-hot and frequency support fit within each training partition",
                "test_rows_used": "no",
                "local_rule_status": "General supplied-column formula guidance; no specific ruling",
                "eligibility_status": "Constant-free research track; organizer certification absent",
            },
            {
                "feature_or_method": "P4/P5/KQ3/KQ4 keys",
                "label_free_feature": "yes",
                "fit_scope": "One-hot and frequency support fit within each training partition",
                "test_rows_used": "no",
                "local_rule_status": "130.5 and 0.0066 provenance unresolved",
                "eligibility_status": "Research-only / eligibility-pending",
            },
            {
                "feature_or_method": "Phase 14 numeric enhancements",
                "label_free_feature": "yes",
                "fit_scope": "Deterministic rowwise transforms; no pooled cut points",
                "test_rows_used": "no",
                "local_rule_status": "Derived only from supplied columns",
                "eligibility_status": "No organizer certification claimed",
            },
            {
                "feature_or_method": "LightGBM/XGBoost training",
                "label_free_feature": "Not applicable; labels train model only",
                "fit_scope": "Nested model selection inside outer-fit rows",
                "test_rows_used": "no",
                "local_rule_status": "No unresolved target-derived feature encoding",
                "eligibility_status": "Local-method status only; no organizer certification",
            },
            {
                "feature_or_method": "Hidden-key target-rate candidates",
                "label_free_feature": "no",
                "fit_scope": "Excluded from Phase 14",
                "test_rows_used": "no",
                "local_rule_status": "Prior exploratory screen only",
                "eligibility_status": "Not used",
            },
        ]
    )
    eligibility.to_csv(output_dir / "ELIGIBILITY_MATRIX.csv", index=False)

    elapsed = time.perf_counter() - started
    best = max(model_rows, key=lambda row: row["pooled_oof_auc"])
    report = f"""# Phase 14 label-free model-development report

## Status and scope

- Run: `{run_name}`
- Start UTC: `{datetime.fromtimestamp(time.time() - elapsed, timezone.utc).isoformat()}`
- End UTC: `{datetime.now(timezone.utc).isoformat()}`
- Rows: {len(train):,}; positives: {int(y.sum()):,}; primary folds: {FOLDS}; seed: {SEED}.
- Training SHA-256: `{sha256_file(train_path)}`.
- Phase 13 reference OOF SHA-256: `{sha256_file(phase13_path)}`.
- Runtime: {elapsed / 60:.1f} minutes; peak process working set: {phase13.peak_working_set_mb():.2f} MiB.
- Models: LightGBM {lightgbm.__version__}, XGBoost {xgboost.__version__}; Python {platform.python_version()}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}, SciPy {scipy.__version__}.

This is nested out-of-fold model development on previously exposed training labels. It is **not** an untouched confirmation: prior project work evaluated these labels across all applicants, so another random split cannot restore label-level independence. The results are exploratory model-development evidence only. No test rows, external data, submissions, or champion files were used or opened.

The Phase 13 feature definitions and primary folds were frozen. Hidden-screen target-rate candidates were excluded. All eight keys were retained in the research track; an additional constant-free track used only Q3/Q4/Q5/AG3. P4/P5/KQ3/KQ4 remain unresolved for competition eligibility because their historical constants lack verified provenance. The local materials do not certify any model or specific engineered formula.

## Preregistered candidate configurations

- Phase 13 E baseline: full raw, rowwise and all-eight-key feature representation; fold-fitted frequency-supported one-hot.
- Numeric enhancement block: log1p amounts/age/employment, external-score summaries/pair terms, financial per-family-member ratios, selected interactions, and missingness/support summaries. All are deterministic rowwise transforms from supplied columns; no labels or test statistics define features.
- Constant-free track: same raw and numeric-enhancement blocks, with only Q3/Q4/Q5/AG3 keys.
- Model families: CPU LightGBM and CPU XGBoost, each with two fixed configurations selected on three inner folds within each outer-fit partition; early stopping is confined to inner folds. The outer model is refit on all outer-fit rows using the median selected iteration.
- Fixed blend: equal-weight average of the all-eight-key augmented LightGBM and XGBoost OOF predictions. Blend weights were not fit on labels.

## Results

Best row by pooled OOF AUC: `{best['variant']}` at **{best['pooled_oof_auc']:.6f}**. Its delta vs Phase 13 E is **{best['delta_vs_phase13_E']:+.6f}**, positive folds {best['positive_folds_vs_phase13_E']}/5. Historical score `0.768240` is shown only as context and is not an independent evaluation threshold.

See `MODEL_RESULTS.csv`, `FOLD_LEVEL_METRICS.csv`, and `PAIRED_BOOTSTRAP.csv` for every candidate. Bootstrap intervals condition on fitted OOF predictions and do not account for model-selection or prior label exposure. Subgroup rows are descriptive fold-level diagnostics with age/income cut points fit on each outer training partition; they are not fairness certification or independent subgroup validation.

## Integrity and outcome limits

Input IDs, targets, row order, OOF predictions, and the Phase 13 fold assignment are programmatically checked before fitting. Every candidate must predict each row once out of fold; the run aborts on incomplete or non-finite coverage. Inner tuning uses only outer-fit labels. No test information or target-derived feature candidates enter the pipeline.

The score target does not override provenance/eligibility. Even if a candidate exceeds `0.768240`, that would not establish a competition-ready model or independent confirmation. A genuinely untouched labeled cohort or organizer-approved external evaluation protocol is unavailable in this workspace; obtaining one is required before claiming confirmation.

## Decision

Do not promote a champion or submit. If a candidate improves the matched Phase 13 E result reproducibly, preserve it as research and resolve the four constant-based key formulas before considering competition eligibility. No target-derived hidden-key feature is recommended for this next step.
"""
    (output_dir / "PHASE14_EXECUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    logger.info(
        "Completed in %.1f minutes; best=%s auc=%.6f",
        elapsed / 60,
        best["variant"],
        best["pooled_oof_auc"],
    )

    source_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "phase14_features.py",
        Path(__file__).resolve().parent / "test_phase14_features.py",
        PHASE13_DIR / "phase13_features.py",
        PHASE13_DIR / "run_phase13.py",
    ]
    _write_manifest(output_dir, source_files)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, default=DEFAULT_PHASE13_OOF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
