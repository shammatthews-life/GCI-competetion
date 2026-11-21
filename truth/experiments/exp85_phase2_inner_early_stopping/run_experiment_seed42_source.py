#!/usr/bin/env python3
"""Train the Phase 2 feature stack with fold-local transforms and inner stopping."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


TRUTH_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROJECT = (
    TRUTH_ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
)
SOURCE_DIR = SOURCE_PROJECT / "source"
PHASE_01_DIR = SOURCE_PROJECT / "experiments" / "phase_01_fee_product_features"
PHASE_02_DIR = (
    SOURCE_PROJECT / "experiments" / "phase_02_fold_safe_key_encodings"
)
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(PHASE_01_DIR))

import run_independent_baseline as baseline  # noqa: E402


def _load_phase_02_module():
    module_path = PHASE_02_DIR / "run_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "truth_exp85_phase2_source", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load archived Phase 2 source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase_02 = _load_phase_02_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_target(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[baseline.TARGET_COLUMN], errors="ignore").copy()


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def build_fold_local_fee_features(
    reference_raw: pd.DataFrame, apply_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Derive historical fee/product features using reference rows only."""
    reference = _without_target(reference_raw).reset_index(drop=True)
    apply = _without_target(apply_raw).reset_index(drop=True)
    required = {
        baseline.ID_COLUMN,
        "NAME_CONTRACT_TYPE",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    }
    if required - set(reference.columns) or required - set(apply.columns):
        raise ValueError("Fee feature inputs are missing required supplied columns")

    reference_credit = reference["AMT_CREDIT"].to_numpy(dtype=np.float64)
    reference_annuity = reference["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    reference_raw_ratio = _ratio(reference_annuity, reference_credit)
    reference_fee_ratio = _ratio(
        reference_annuity - phase_02._phase_01_module.FEE_AMOUNT,
        reference_credit,
    )
    raw_counts = (
        pd.Series(np.round(reference_raw_ratio, 5))
        .value_counts(dropna=True)
        .to_dict()
    )
    fee_counts = (
        pd.Series(np.round(reference_fee_ratio, 5))
        .value_counts(dropna=True)
        .to_dict()
    )

    combined = pd.concat([reference, apply], ignore_index=True)
    credit = combined["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = combined["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = combined["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    contract = combined["NAME_CONTRACT_TYPE"].astype("string").to_numpy()
    raw_ratio = _ratio(annuity, credit)
    fee_ratio = _ratio(
        annuity - phase_02._phase_01_module.FEE_AMOUNT,
        credit,
    )

    raw_count = (
        pd.Series(np.round(raw_ratio, 5))
        .map(raw_counts)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    fee_count = (
        pd.Series(np.round(fee_ratio, 5))
        .map(fee_counts)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    fee_decoded = (
        (contract == "Cash loans") & (fee_count > raw_count) & np.isfinite(raw_ratio)
    )
    decoded_ratio = np.where(fee_decoded, fee_ratio, raw_ratio)

    credit_goods_ratio = _ratio(credit, goods)
    implied_product = (
        credit_goods_ratio - 1.0
    ) / phase_02._phase_01_module.PRODUCT_RATE
    nearest_product = np.rint(implied_product)
    product_valid = (
        (contract == "Cash loans")
        & np.isfinite(implied_product)
        & (credit > goods)
        & (np.abs(implied_product - nearest_product) < 0.1)
    )
    features = pd.DataFrame(
        {
            "FEE_DECODED": fee_decoded.astype(np.float32),
            "FEE_RAW_RATIO_COUNT": np.log1p(raw_count).astype(np.float32),
            "FEE_ADJUSTED_RATIO_COUNT": np.log1p(fee_count).astype(np.float32),
            "ANNUITY_CREDIT_RATIO_FEE_ADJUSTED": fee_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED": decoded_ratio,
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_3": np.round(decoded_ratio, 3),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_4": np.round(decoded_ratio, 4),
            "ANNUITY_CREDIT_RATIO_DECODED_ROUND_5": np.round(decoded_ratio, 5),
            "CREDIT_GOODS_RATIO": credit_goods_ratio,
            "PRODUCT_KEY_VALID": product_valid.astype(np.float32),
            "PRODUCT_KEY_K": np.where(product_valid, nearest_product, np.nan),
            "PRODUCT_KEY_DISTANCE": np.where(
                product_valid, implied_product - nearest_product, np.nan
            ),
        }
    )
    if np.isinf(features.to_numpy(dtype=np.float64, copy=False)).any():
        raise ValueError("Fold-local fee features contain infinite values")

    n_reference = len(reference)
    cash_reference = int((contract[:n_reference] == "Cash loans").sum())
    diagnostics = {
        "frequency_reference_rows": n_reference,
        "fee_decoded_reference_rows": int(fee_decoded[:n_reference].sum()),
        "cash_loan_reference_rows": cash_reference,
        "product_key_reference_rows": int(product_valid[:n_reference].sum()),
        "reference_ids_sha256": hashlib.sha256(
            np.sort(reference[baseline.ID_COLUMN].to_numpy()).tobytes()
        ).hexdigest(),
        "frequency_reference_scope": "reference_raw_rows_only",
        "target_used": False,
    }
    return (
        features.iloc[:n_reference].reset_index(drop=True),
        features.iloc[n_reference:].reset_index(drop=True),
        diagnostics,
    )


def _phase2_feature_frames(
    fit_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    fit_fee: pd.DataFrame,
    apply_fee: pd.DataFrame,
    fit_labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_labels = np.asarray(fit_labels, dtype=np.int8)
    if len(fit_raw) != len(fit_labels):
        raise ValueError("Feature fit rows and labels differ in length")
    keys = phase_02.key_arrays(fit_fee, apply_fee)
    fit_indices = np.arange(len(fit_raw), dtype=np.int64)
    apply_indices = np.arange(
        len(fit_raw), len(fit_raw) + len(apply_raw), dtype=np.int64
    )
    fit_te = phase_02.nested_training_encodings(
        keys, fit_labels, fit_indices
    )
    apply_te = phase_02.apply_training_encodings(
        keys, fit_labels, fit_indices, apply_indices
    )
    fit_features = pd.concat(
        [
            baseline.build_rowwise_features(_without_target(fit_raw)),
            fit_fee.reset_index(drop=True),
            fit_te.reset_index(drop=True),
        ],
        axis=1,
    )
    apply_features = pd.concat(
        [
            baseline.build_rowwise_features(_without_target(apply_raw)),
            apply_fee.reset_index(drop=True),
            apply_te.reset_index(drop=True),
        ],
        axis=1,
    )
    if list(fit_features.columns) != list(apply_features.columns):
        raise ValueError("Phase 2 fit/apply feature schemas differ")
    numeric = fit_features.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isinf(numeric).any():
        raise ValueError("Phase 2 feature matrix contains infinite numeric values")
    return fit_features, apply_features


def _raw_feature_frames(
    fit_raw: pd.DataFrame, apply_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = baseline.build_rowwise_features(_without_target(fit_raw))
    apply = baseline.build_rowwise_features(_without_target(apply_raw))
    if list(fit.columns) != list(apply.columns):
        raise ValueError("Raw feature schemas differ")
    return fit, apply


def _inner_stop_split(
    outer_fit_indices: np.ndarray, labels: np.ndarray, fold_number: int
) -> tuple[np.ndarray, np.ndarray]:
    outer_fit_indices = np.asarray(outer_fit_indices, dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=20261102 + fold_number,
    )
    local_fit, local_stop = next(
        splitter.split(
            np.zeros(len(outer_fit_indices)),
            labels[outer_fit_indices],
        )
    )
    fit_indices = outer_fit_indices[local_fit]
    stop_indices = outer_fit_indices[local_stop]
    if np.intersect1d(fit_indices, stop_indices).size:
        raise AssertionError("Inner fit and stopping indices overlap")
    return fit_indices, stop_indices


def _model_params(model_name: str, fold_number: int) -> dict[str, Any]:
    settings = baseline.MODEL_CONFIG[model_name]
    seed = baseline.FOLD_SEED + fold_number
    if model_name == "lightgbm":
        params = {
            key: value
            for key, value in settings.items()
            if key not in ("device", "num_boost_round", "early_stopping_rounds")
        }
        params["seed"] = seed
        return params
    if model_name == "catboost":
        return {
            "iterations": settings["iterations"],
            "depth": settings["depth"],
            "learning_rate": settings["learning_rate"],
            "loss_function": settings["loss_function"],
            "eval_metric": settings["eval_metric"],
            "l2_leaf_reg": settings["l2_leaf_reg"],
            "random_strength": settings["random_strength"],
            "bootstrap_type": settings["bootstrap_type"],
            "subsample": settings["subsample"],
            "border_count": settings["border_count"],
            "task_type": "GPU",
            "devices": "0",
            "random_seed": seed,
            "allow_writing_files": False,
            "verbose": False,
            "thread_count": baseline.CPU_THREADS,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_with_inner_stopping(
    model_name: str,
    fit_features: pd.DataFrame,
    fit_labels: np.ndarray,
    stop_features: pd.DataFrame,
    stop_labels: np.ndarray,
    outer_fit_features: pd.DataFrame,
    outer_fit_labels: np.ndarray,
    outer_apply_features: pd.DataFrame,
    fold_number: int,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    settings = baseline.MODEL_CONFIG[model_name]
    params = _model_params(model_name, fold_number)

    if model_name == "lightgbm":
        tune_fit, tune_stop = baseline.make_lightgbm_frames(
            fit_features, stop_features
        )
        tune_train_set = lgb.Dataset(tune_fit, label=fit_labels)
        tune_stop_set = lgb.Dataset(
            tune_stop, label=stop_labels, reference=tune_train_set
        )
        selection_model = lgb.train(
            params,
            tune_train_set,
            num_boost_round=settings["num_boost_round"],
            valid_sets=[tune_stop_set],
            callbacks=[
                lgb.early_stopping(
                    settings["early_stopping_rounds"], verbose=False
                )
            ],
        )
        selected_iterations = int(
            selection_model.best_iteration or selection_model.current_iteration()
        )
        if selected_iterations < 1:
            raise RuntimeError("LightGBM did not select a positive iteration count")

        final_fit, final_apply = baseline.make_lightgbm_frames(
            outer_fit_features, outer_apply_features
        )
        final_train_set = lgb.Dataset(final_fit, label=outer_fit_labels)
        final_model = lgb.train(
            params, final_train_set, num_boost_round=selected_iterations
        )
        apply_predictions = final_model.predict(
            final_apply, num_iteration=selected_iterations
        )
        importance = pd.DataFrame(
            {
                "feature": final_fit.columns,
                "gain": final_model.feature_importance(importance_type="gain"),
                "split": final_model.feature_importance(importance_type="split"),
                "fold": fold_number,
            }
        )
        return (
            apply_predictions.astype(np.float32),
            selected_iterations,
            importance,
        )

    selection_model = CatBoostClassifier(**params)
    selection_model.fit(
        fit_features,
        fit_labels,
        cat_features=list(baseline.CATEGORICAL_COLUMNS),
        eval_set=(stop_features, stop_labels),
        early_stopping_rounds=settings["early_stopping_rounds"],
        verbose=False,
    )
    selected_iterations = int(selection_model.best_iteration_) + 1
    if selected_iterations < 1:
        raise RuntimeError("CatBoost did not select a positive iteration count")

    params["iterations"] = selected_iterations
    final_model = CatBoostClassifier(**params)
    final_model.fit(
        outer_fit_features,
        outer_fit_labels,
        cat_features=list(baseline.CATEGORICAL_COLUMNS),
        verbose=False,
    )
    apply_predictions = final_model.predict_proba(outer_apply_features)[:, 1]
    importance = pd.DataFrame(
        {
            "feature": outer_fit_features.columns,
            "importance": final_model.get_feature_importance(),
            "fold": fold_number,
        }
    )
    predictions = apply_predictions.astype(np.float32)
    return predictions, selected_iterations, importance


def _fit_predict_pair(
    model_name: str,
    outer_fit_indices: np.ndarray,
    outer_valid_indices: np.ndarray,
    inner_fit_indices: np.ndarray,
    inner_stop_indices: np.ndarray,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    labels: np.ndarray,
    fold_number: int,
    phase2: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    apply_raw = pd.concat(
        [
            train_raw.iloc[outer_valid_indices].drop(
                columns=[baseline.TARGET_COLUMN], errors="ignore"
            ),
            test_raw,
        ],
        ignore_index=True,
    )
    outer_fit_raw = train_raw.iloc[outer_fit_indices]
    outer_fit_labels = labels[outer_fit_indices]

    inner_fit_raw = train_raw.iloc[inner_fit_indices]
    inner_stop_raw = train_raw.iloc[inner_stop_indices]
    inner_fit_labels = labels[inner_fit_indices]
    inner_stop_labels = labels[inner_stop_indices]

    if phase2:
        selection_apply_raw = inner_stop_raw.drop(
            columns=[baseline.TARGET_COLUMN]
        ).reset_index(drop=True)
        inner_fee_fit, inner_fee_apply, _ = build_fold_local_fee_features(
            inner_fit_raw, selection_apply_raw
        )
        inner_fit_features, inner_apply_features = _phase2_feature_frames(
            inner_fit_raw,
            selection_apply_raw,
            inner_fee_fit,
            inner_fee_apply,
            inner_fit_labels,
        )
        inner_stop_features = inner_apply_features
        if len(inner_stop_features) != len(inner_stop_raw):
            raise AssertionError("Inner stopping feature row count mismatch")

        outer_fee_fit, outer_fee_apply, _ = build_fold_local_fee_features(
            outer_fit_raw, apply_raw
        )
        outer_fit_features, outer_apply_features = _phase2_feature_frames(
            outer_fit_raw,
            apply_raw,
            outer_fee_fit,
            outer_fee_apply,
            outer_fit_labels,
        )
    else:
        inner_fit_features, inner_stop_features = _raw_feature_frames(
            inner_fit_raw, inner_stop_raw
        )
        outer_fit_features, outer_apply_features = _raw_feature_frames(
            outer_fit_raw, apply_raw
        )

    apply_predictions, selected_iterations, _ = _fit_with_inner_stopping(
        model_name=model_name,
        fit_features=inner_fit_features,
        fit_labels=inner_fit_labels,
        stop_features=inner_stop_features,
        stop_labels=inner_stop_labels,
        outer_fit_features=outer_fit_features,
        outer_fit_labels=outer_fit_labels,
        outer_apply_features=outer_apply_features,
        fold_number=fold_number,
    )
    n_valid = len(outer_valid_indices)
    valid_predictions = apply_predictions[:n_valid]
    test_predictions = apply_predictions[n_valid:]
    if len(test_predictions) != len(test_raw):
        raise AssertionError("Test prediction row count mismatch")
    return valid_predictions, test_predictions, selected_iterations


def _save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
    temporary.replace(path)


def _gpu_check() -> dict[str, Any]:
    snapshot = baseline.gpu_snapshot()
    if not snapshot.get("available"):
        raise RuntimeError(
            "CatBoost GPU was required but nvidia-smi is unavailable: "
            f"{snapshot.get('error', 'unknown error')}"
        )
    query = snapshot.get("query", "")
    if "RTX 4050" not in query:
        raise RuntimeError(
            "Expected the authorized RTX 4050 GPU; refusing training on "
            f"unexpected device report: {query}"
        )
    return snapshot


def run(data_dir: Path, run_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Required supplied input is missing: {path}")
    run_dir.mkdir(parents=True, exist_ok=True)

    started_utc = datetime.now(timezone.utc).isoformat()
    gpu_before = _gpu_check()
    train_raw = pd.read_csv(paths["train"])
    test_raw = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    if list(sample.columns) != [baseline.ID_COLUMN, baseline.TARGET_COLUMN]:
        raise ValueError("Unexpected sample submission schema")
    if baseline.TARGET_COLUMN not in train_raw or baseline.TARGET_COLUMN in test_raw:
        raise ValueError("Train/test target boundary is invalid")
    if list(train_raw.drop(columns=[baseline.TARGET_COLUMN]).columns) != list(
        test_raw.columns
    ):
        raise ValueError("Train predictors and test columns do not match exactly")
    if not np.array_equal(sample[baseline.ID_COLUMN], test_raw[baseline.ID_COLUMN]):
        raise ValueError("Sample and test IDs/order differ")
    if train_raw[baseline.ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate training IDs")
    if test_raw[baseline.ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate test IDs")
    if set(train_raw[baseline.ID_COLUMN]).intersection(test_raw[baseline.ID_COLUMN]):
        raise ValueError("Train/test IDs overlap")

    labels = train_raw[baseline.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("TARGET must contain only binary 0/1 values")
    folds = list(
        StratifiedKFold(
            n_splits=baseline.N_SPLITS,
            shuffle=True,
            random_state=baseline.FOLD_SEED,
        ).split(np.zeros(len(labels)), labels)
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for fold_number, (fit_indices, valid_indices) in enumerate(folds, 1):
        if np.intersect1d(fit_indices, valid_indices).size:
            raise AssertionError("Outer fit/validation indices overlap")
        if np.any(fold_ids[valid_indices] != -1):
            raise AssertionError("An observation appears in multiple validation folds")
        fold_ids[valid_indices] = fold_number
    if np.any(fold_ids < 1):
        raise AssertionError("Some training rows are missing an outer fold")

    n_train = len(train_raw)
    outputs: dict[str, dict[str, np.ndarray]] = {
        variant: {
            model: np.full(n_train, np.nan, dtype=np.float32)
            for model in ("lightgbm", "catboost")
        }
        for variant in ("raw_baseline", "phase2")
    }
    test_outputs: dict[str, dict[str, list[np.ndarray]]] = {
        variant: {model: [] for model in ("lightgbm", "catboost")}
        for variant in ("raw_baseline", "phase2")
    }
    fold_records = []
    for fold_number, (outer_fit_indices, outer_valid_indices) in enumerate(folds, 1):
        inner_fit_indices, inner_stop_indices = _inner_stop_split(
            outer_fit_indices, labels, fold_number
        )
        if not set(inner_fit_indices).issubset(set(outer_fit_indices)):
            raise AssertionError("Inner fit escaped the outer fit partition")
        if not set(inner_stop_indices).issubset(set(outer_fit_indices)):
            raise AssertionError("Inner stop escaped the outer fit partition")
        fold_record: dict[str, Any] = {
            "fold": fold_number,
            "outer_fit_rows": int(len(outer_fit_indices)),
            "outer_validation_rows": int(len(outer_valid_indices)),
            "inner_fit_rows": int(len(inner_fit_indices)),
            "inner_stopping_rows": int(len(inner_stop_indices)),
            "outer_validation_used_for_early_stopping": False,
            "inner_stopping_used_for_final_refit": True,
            "models": {},
        }
        for variant in ("raw_baseline", "phase2"):
            for model_name in ("lightgbm", "catboost"):
                valid_prediction, test_prediction, iterations = _fit_predict_pair(
                    model_name=model_name,
                    outer_fit_indices=outer_fit_indices,
                    outer_valid_indices=outer_valid_indices,
                    inner_fit_indices=inner_fit_indices,
                    inner_stop_indices=inner_stop_indices,
                    train_raw=train_raw,
                    test_raw=test_raw,
                    labels=labels,
                    fold_number=fold_number,
                    phase2=(variant == "phase2"),
                )
                outputs[variant][model_name][outer_valid_indices] = valid_prediction
                test_outputs[variant][model_name].append(test_prediction)
                fold_record["models"][f"{variant}_{model_name}"] = {
                    "selected_iterations": int(iterations),
                    "auc": float(
                        roc_auc_score(labels[outer_valid_indices], valid_prediction)
                    ),
                }
                gc.collect()
        fold_records.append(fold_record)
        _save_array(run_dir / "fold_ids.npy", fold_ids)
        (run_dir / f"fold_{fold_number}_record.json").write_text(
            json.dumps(fold_record, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    if any(np.isnan(prediction).any() for group in outputs.values() for prediction in group.values()):
        raise ValueError("OOF predictions contain uncovered rows")
    if any(not np.isfinite(prediction).all() for group in test_outputs.values() for predictions in group.values() for prediction in predictions):
        raise ValueError("Test predictions contain non-finite values")

    test_predictions = {
        variant: {
            model: np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(
                np.float32
            )
            for model, predictions in models.items()
        }
        for variant, models in test_outputs.items()
    }
    metrics: dict[str, Any] = {"variants": {}, "folds": fold_records}
    for variant in ("raw_baseline", "phase2"):
        oof_blend = (
            outputs[variant]["lightgbm"].astype(np.float64)
            + outputs[variant]["catboost"].astype(np.float64)
        ) / 2.0
        test_blend = (
            test_predictions[variant]["lightgbm"].astype(np.float64)
            + test_predictions[variant]["catboost"].astype(np.float64)
        ) / 2.0
        outputs[variant]["blend"] = oof_blend.astype(np.float32)
        test_predictions[variant]["blend"] = test_blend.astype(np.float32)
        metrics["variants"][variant] = {
            "lightgbm_oof_auc": float(
                roc_auc_score(labels, outputs[variant]["lightgbm"])
            ),
            "catboost_oof_auc": float(
                roc_auc_score(labels, outputs[variant]["catboost"])
            ),
            "fixed_50_50_blend_oof_auc": float(roc_auc_score(labels, oof_blend)),
            "fold_auc": [
                float(fold["models"][f"{variant}_lightgbm"]["auc"])
                for fold in fold_records
            ],
        }
        for model, prediction in outputs[variant].items():
            _save_array(run_dir / f"oof_{variant}_{model}.npy", prediction)
        for model, prediction in test_predictions[variant].items():
            _save_array(run_dir / f"test_{variant}_{model}.npy", prediction)

    baseline_fold_auc = np.asarray(
        metrics["variants"]["raw_baseline"]["fold_auc"], dtype=np.float64
    )
    phase2_fold_auc = np.asarray(
        metrics["variants"]["phase2"]["fold_auc"], dtype=np.float64
    )
    metrics["paired_phase2_minus_baseline_fold_auc"] = (
        phase2_fold_auc - baseline_fold_auc
    ).tolist()
    metrics["paired_phase2_minus_baseline_positive_folds"] = int(
        np.sum(phase2_fold_auc > baseline_fold_auc)
    )
    metrics["official_competition_score"] = "NOT_EVALUATED"
    metrics["submission_created"] = False
    metrics["started_utc"] = started_utc
    metrics["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    source_paths = {
        "baseline": SOURCE_DIR / "run_independent_baseline.py",
        "phase1": PHASE_01_DIR / "run_experiment.py",
        "phase2": PHASE_02_DIR / "run_experiment.py",
        "runner": Path(__file__).resolve(),
    }
    manifest = {
        "experiment_id": run_dir.name,
        "lineage": "RAW_SUPPLIED_DATA_ONLY_PHASE2_FEATURES_WITH_NESTED_INNER_STOPPING",
        "raw_input_sha256": input_hashes,
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
        "fold_seed": baseline.FOLD_SEED,
        "outer_folds": baseline.N_SPLITS,
        "inner_stop_fraction": 0.1,
        "inner_stop_seed_rule": "20261102 + 1-based outer fold number",
        "frequency_reference_policy": "inner-fit for iteration selection; outer-fit for final refit",
        "target_encoding_policy": (
            "inner-fit labels only for the iteration-selection model; "
            "outer-fit labels only for final outer-fold predictions"
        ),
        "external_datasets_or_prediction_caches_used": False,
        "gpu_before_training": gpu_before,
        "gpu_after_training": baseline.gpu_snapshot(),
        "model_config": baseline.MODEL_CONFIG,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_dir, args.run_dir)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
