from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
PHASE14 = REPO / "experiments" / "phase14_label_free_model_development"
PHASE15 = REPO / "experiments" / "phase15_autonomous_research"
PHASE20 = REPO / "experiments" / "phase20_segmentation_expert_research"
sys.path[:0] = [str(PHASE15), str(PHASE14), str(PHASE13), str(PHASE20)]

import phase13_features as features13
import run_phase14 as features14
import run_phase15_catboost_preflight as phase15
import run_phase20_segmentation_study as phase20
from phase15_features import build_time_lifecycle_features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
PHASE15_OOF = (
    PHASE15
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE15_MULTISEED = (
    PHASE15
    / "runs"
    / "phase15_native_catboost_multiseed_20261105T125826Z"
    / "MULTISEED_OOF_PREDICTIONS.csv"
)
SEED = 20261113
FOLDS = 5
ITERATIONS = 700
LEARNING_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
SEED_COUNTS = (1, 3, 5, 10, 20)
SEED_OFFSETS = tuple(index * 10000 for index in range(max(SEED_COUNTS)))
MIN_EXPERT_FIT_ROWS = 2000
MIN_EXPERT_CLASS_ROWS = 100
MIN_EXPERT_VALID_ROWS = 500
DEFAULT_OUTPUT = ROOT / "runs" / "phase21_focused_20261106"
TARGET = features13.TARGET_COLUMN
ID = features13.ID_COLUMN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monitor_gpu(
    samples: list[dict[str, float]],
    errors: list[str],
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            fields = [float(value.strip()) for value in result.stdout.split(",")]
            if len(fields) != 4:
                errors.append(f"Unexpected nvidia-smi output: {result.stdout!r}")
                return
            samples.append(
                {
                    "gpu_utilization_percent": fields[0],
                    "memory_used_mib": fields[1],
                    "memory_total_mib": fields[2],
                    "power_draw_watts": fields[3],
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            errors.append(f"{type(error).__name__}: {error}")
            return
        stop.wait(2.0)


def make_model(seed: int, *, one_hot_max_size: int = 10) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=ITERATIONS,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.78,
        border_count=128,
        one_hot_max_size=one_hot_max_size,
        max_ctr_complexity=1,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )


def fit_predict(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_apply: pd.DataFrame,
    cat_columns: list[str],
    seed: int,
    *,
    one_hot_max_size: int = 10,
) -> tuple[np.ndarray, float]:
    if len(np.unique(y_fit)) != 2:
        raise ValueError("A model fit partition contains only one target class")
    if x_fit.columns.tolist() != x_apply.columns.tolist():
        raise ValueError("Training and application feature schemas differ")
    cat_indices = [x_fit.columns.get_loc(column) for column in cat_columns]
    model = make_model(seed, one_hot_max_size=one_hot_max_size)
    started = time.perf_counter()
    model.fit(x_fit, y_fit, cat_features=cat_indices)
    prediction = model.predict_proba(x_apply)[:, 1].astype(np.float64)
    elapsed = time.perf_counter() - started
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("CatBoost did not execute on GPU")
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced nonfinite predictions")
    del model
    gc.collect()
    return prediction, elapsed


def build_feature_blocks(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    numeric, categorical, rowwise = features13.build_raw_feature_blocks(raw)
    keys = features13.build_product_keys(raw)
    decompositions = features13.build_label_free_decompositions(raw)
    enhancements = features14.build_numeric_enhancements(
        raw.drop(columns=[ID])
    )
    time_features = build_time_lifecycle_features(raw.drop(columns=[ID]))
    if enhancements.columns.intersection(time_features.columns).size:
        raise ValueError("Phase 14 and Phase 15 feature names collide")
    enhancements_and_time = pd.concat(
        [enhancements.reset_index(drop=True), time_features.reset_index(drop=True)],
        axis=1,
    )
    document_flags = raw.filter(regex=r"^FLAG_DOCUMENT_")
    return (
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        enhancements_and_time,
        document_flags,
    )


def phase15_matrix(
    blocks: tuple[pd.DataFrame, ...],
    fit: np.ndarray,
    apply: np.ndarray,
    *,
    key_columns: tuple[str, ...] = features13.PRODUCT_KEYS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    numeric, categorical, rowwise, keys, decompositions, enhancements, _ = blocks
    feature_blocks = features14._feature_blocks(
        pd.DataFrame(index=np.arange(len(numeric))),
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        enhancements,
        {"augmented": True},
    )
    x_fit, x_apply, info = features14._matrix_pair(
        fit, apply, feature_blocks, key_columns, "catboost"
    )
    return x_fit, x_apply, list(info["cat_features"])


def track_a_matrix(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    fit: np.ndarray,
    apply: np.ndarray,
    *,
    add_constant_free_frequencies: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    x_fit, x_apply, cat_columns = phase15_matrix(
        blocks, fit, apply, key_columns=()
    )
    if add_constant_free_frequencies:
        direct_keys = phase20.build_constant_free_keys(raw)
        fit_counts = pd.DataFrame(index=np.arange(len(fit)))
        apply_counts = pd.DataFrame(index=np.arange(len(apply)))
        for column in direct_keys.columns:
            counts = direct_keys.iloc[fit][column].astype(str).value_counts()
            name = f"FOLD_COUNT_{column}"
            fit_counts[name] = (
                direct_keys.iloc[fit][column]
                .astype(str)
                .map(counts)
                .to_numpy(dtype=np.float32)
            )
            apply_counts[name] = (
                direct_keys.iloc[apply][column]
                .astype(str)
                .map(counts)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )
        x_fit = pd.concat(
            [x_fit.reset_index(drop=True), fit_counts], axis=1
        )
        x_apply = pd.concat(
            [x_apply.reset_index(drop=True), apply_counts], axis=1
        )
    if not x_fit.columns.equals(x_apply.columns):
        raise ValueError("Track A feature schemas do not match")
    return x_fit, x_apply, cat_columns


def drop_feature_group(
    x_fit: pd.DataFrame,
    x_apply: pd.DataFrame,
    cat_columns: list[str],
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    def remove(column: str) -> bool:
        upper = column.upper()
        if group == "external_scores":
            return "EXT_SOURCE" in upper or upper.startswith("EXT_")
        if group == "product_keys":
            return upper.startswith("PRODUCT_KEY_") or upper.startswith(
                "FIT_COUNT_PRODUCT_KEY_"
            )
        if group == "demographics":
            return upper in {
                "CNT_CHILDREN",
                "CNT_FAM_MEMBERS",
                "DAYS_BIRTH",
                "AGE_YEARS_FROM_DAYS_BIRTH",
                "CODE_GENDER",
                "NAME_FAMILY_STATUS",
                "NAME_EDUCATION_TYPE",
                "NAME_HOUSING_TYPE",
                "REGION_POPULATION_RELATIVE",
                "REGION_RATING_CLIENT",
            } or upper.startswith("AGE_YEARS_")
        if group == "day_columns":
            return upper.startswith("DAYS_") or upper.startswith(
                "AGE_YEARS_AT_"
            ) or upper.endswith("_TO_AGE_FRACTION") or upper in {
                "EMPLOYMENT_TO_AGE_RATIO",
                "LOG1P_AGE_DAYS",
                "LOG1P_EMPLOYED_DAYS_CLEAN",
            }
        raise ValueError(f"Unknown feature group: {group}")

    removed = [column for column in x_fit.columns if remove(column)]
    if not removed:
        raise ValueError(f"Feature group {group} matched no model columns")
    keep = [column for column in x_fit.columns if column not in removed]
    return (
        x_fit[keep].copy(),
        x_apply[keep].copy(),
        [column for column in cat_columns if column in keep],
    )


def feature_flags(raw: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    selected = raw.iloc[rows]
    contract = selected["NAME_CONTRACT_TYPE"].astype(str).eq("Cash loans")
    ext_complete = selected[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].notna().all(axis=1)
    credit = pd.to_numeric(selected["AMT_CREDIT"], errors="raise")
    goods = pd.to_numeric(selected["AMT_GOODS_PRICE"], errors="raise")
    goods_to_credit = goods.div(credit.replace(0, np.nan))
    return pd.DataFrame(
        {
            "P21_IS_CASH_LOAN": contract.astype(np.int8).to_numpy(),
            "P21_EXT_ALL_COMPLETE": ext_complete.astype(np.int8).to_numpy(),
            "P21_GOODS_CREDIT_BELOW_090": (
                goods_to_credit < 0.90
            ).fillna(False).astype(np.int8).to_numpy(),
            "P21_CASH_X_EXT_COMPLETE": (
                contract & ext_complete
            ).astype(np.int8).to_numpy(),
        }
    )


def nested_stratified_subsets(
    fit_rows: np.ndarray,
    y: np.ndarray,
    fold: int,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(SEED + 70000 + fold)
    shuffled: list[np.ndarray] = []
    for label in (0, 1):
        rows = fit_rows[y[fit_rows] == label].copy()
        rng.shuffle(rows)
        shuffled.append(rows)
    result: dict[float, np.ndarray] = {}
    for fraction in LEARNING_FRACTIONS:
        selected = [
            rows[: max(1, int(round(len(rows) * fraction)))]
            for rows in shuffled
        ]
        result[fraction] = np.concatenate(selected)
        if len(np.unique(y[result[fraction]])) != 2:
            raise ValueError(f"Learning-curve sample {fraction} lost a class")
    return result


def learning_curve_matrix(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return phase15_matrix(blocks, fit_rows, apply_rows)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def prepare_run_directory(
    output_dir: Path, resume: bool, identity: dict[str, Any]
) -> None:
    identity_path = output_dir / "PHASE21_RUN_IDENTITY.json"
    finalized = (
        output_dir / "PHASE21_RUN_SUMMARY.json"
    ).is_file() and (output_dir / "PHASE21_SHA256_MANIFEST.csv").is_file()
    existing = output_dir.exists() and any(output_dir.iterdir())
    if existing:
        if not resume:
            raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
        if finalized:
            raise FileExistsError(f"Run is already finalized: {output_dir}")
        if not identity_path.is_file():
            raise ValueError(
                f"Cannot resume a run without its identity file: {identity_path}"
            )
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded != identity:
            raise ValueError(
                "Resume identity differs from the original run inputs or code"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(identity_path, identity)


def save_fold_checkpoint(
    checkpoint_dir: Path,
    fold: int,
    valid_rows: np.ndarray,
    global_prediction: np.ndarray,
    seed_predictions: dict[int, np.ndarray],
    predictions: dict[str, np.ndarray],
    fold_runtime: list[dict[str, Any]],
    specialist_metrics: list[dict[str, Any]],
    identity: dict[str, Any],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    array_names = sorted(predictions)
    seed_indices = sorted(seed_predictions)
    expected_length = len(valid_rows)
    if (
        np.asarray(global_prediction).shape != (expected_length,)
        or any(
            np.asarray(predictions[name]).shape != (expected_length,)
            for name in array_names
        )
        or any(
            np.asarray(seed_predictions[index]).shape != (expected_length,)
            for index in seed_indices
        )
    ):
        raise ValueError("Fold checkpoint predictions do not match validation rows")
    if (
        not np.isfinite(global_prediction).all()
        or any(not np.isfinite(predictions[name]).all() for name in array_names)
        or any(
            not np.isfinite(seed_predictions[index]).all()
            for index in seed_indices
        )
    ):
        raise ValueError("Fold checkpoint contains nonfinite predictions")
    metadata = {
        **identity,
        "fold": fold,
        "valid_rows_sha256": hashlib.sha256(
            np.asarray(valid_rows, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    checkpoint_path = checkpoint_dir / f"fold_{fold:02d}.npz"
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            valid_rows=np.asarray(valid_rows, dtype=np.int64),
            global_prediction=np.asarray(global_prediction, dtype=np.float64),
            array_names=np.asarray(array_names, dtype=str),
            array_predictions=np.vstack(
                [predictions[name] for name in array_names]
            ),
            seed_indices=np.asarray(seed_indices, dtype=np.int16),
            seed_predictions=np.vstack(
                [seed_predictions[index] for index in seed_indices]
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            fold_runtime_json=np.asarray(json.dumps(fold_runtime, sort_keys=True)),
            specialist_metrics_json=np.asarray(
                json.dumps(specialist_metrics, sort_keys=True)
            ),
        )
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_fold_checkpoint(
    checkpoint_path: Path,
    fold: int,
    valid_rows: np.ndarray,
    array_names: list[str],
    seed_indices: list[int],
    identity: dict[str, Any],
) -> dict[str, Any]:
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        expected_metadata = {
            **identity,
            "fold": fold,
            "valid_rows_sha256": hashlib.sha256(
                np.asarray(valid_rows, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
        if metadata != expected_metadata:
            raise ValueError(f"Fold {fold} checkpoint identity mismatch")
        saved_rows = archive["valid_rows"]
        saved_array_names = archive["array_names"].astype(str).tolist()
        saved_seed_indices = archive["seed_indices"].astype(int).tolist()
        if not np.array_equal(saved_rows, valid_rows):
            raise ValueError(f"Fold {fold} checkpoint row alignment mismatch")
        if saved_array_names != sorted(array_names):
            raise ValueError(f"Fold {fold} checkpoint array schema mismatch")
        if saved_seed_indices != sorted(seed_indices):
            raise ValueError(f"Fold {fold} checkpoint seed schema mismatch")
        global_prediction = archive["global_prediction"].astype(np.float64)
        array_values = archive["array_predictions"].astype(np.float64)
        seed_values = archive["seed_predictions"].astype(np.float64)
        if (
            array_values.shape != (len(saved_array_names), len(valid_rows))
            or seed_values.shape != (len(saved_seed_indices), len(valid_rows))
            or not np.isfinite(global_prediction).all()
            or not np.isfinite(array_values).all()
            or not np.isfinite(seed_values).all()
        ):
            raise ValueError(f"Fold {fold} checkpoint prediction data is invalid")
        return {
            "global_prediction": global_prediction,
            "predictions": dict(zip(saved_array_names, array_values)),
            "seed_predictions": dict(zip(saved_seed_indices, seed_values)),
            "fold_runtime": json.loads(
                str(archive["fold_runtime_json"].item())
            ),
            "specialist_metrics": json.loads(
                str(archive["specialist_metrics_json"].item())
            ),
        }


def contract_route(raw: pd.DataFrame) -> np.ndarray:
    return raw["NAME_CONTRACT_TYPE"].astype(str).eq("Cash loans").to_numpy()


def external_completeness_route(raw: pd.DataFrame) -> np.ndarray:
    return raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().all(
        axis=1
    ).to_numpy()


def combined_route(raw: pd.DataFrame) -> np.ndarray:
    contract = contract_route(raw)
    complete = external_completeness_route(raw)
    return (contract.astype(np.int8) * 2 + complete.astype(np.int8)).astype(
        np.int8
    )


def support_check(
    y: np.ndarray, route: np.ndarray, fit: np.ndarray, valid: np.ndarray, gate: str
) -> None:
    for group in np.unique(route):
        fit_y = y[fit[route[fit] == group]]
        valid_group = valid[route[valid] == group]
        counts = np.bincount(fit_y, minlength=2)
        if (
            int(counts.min()) < MIN_EXPERT_CLASS_ROWS
            or len(fit_y) < MIN_EXPERT_FIT_ROWS
            or len(valid_group) < MIN_EXPERT_VALID_ROWS
        ):
            raise ValueError(
                f"{gate} group={group} fails preregistered support: "
                f"fit={len(fit_y)}, class_counts={counts.tolist()}, "
                f"valid={len(valid_group)}"
            )


def fit_platt_oof(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Invalid OOF calibration data")
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        linear = np.clip(slope * logits + intercept, -40, 40)
        probability = 1.0 / (1.0 + np.exp(-linear))
        loss = -np.mean(
            labels * np.log(np.clip(probability, 1e-12, 1.0))
            + (1.0 - labels)
            * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
        )
        residual = probability - labels
        gradient = np.array(
            [np.mean(residual * logits), np.mean(residual)], dtype=np.float64
        )
        return float(loss), gradient

    result = minimize(
        objective,
        np.array([1.0, 0.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.01, 20.0), (-20.0, 20.0)),
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Platt calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def apply_platt(scores: np.ndarray, parameters: tuple[float, float]) -> np.ndarray:
    slope, intercept = parameters
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    linear = np.clip(slope * logits + intercept, -40, 40)
    return 1.0 / (1.0 + np.exp(-linear))


def fit_expert_oof_calibration(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    route_fit: np.ndarray,
    cat_columns: list[str],
    fold: int,
    gate: str,
) -> tuple[dict[int, tuple[float, float]], list[dict[str, Any]]]:
    parameters: dict[int, tuple[float, float]] = {}
    runtimes: list[dict[str, Any]] = []
    for group in np.unique(route_fit):
        group_rows = np.flatnonzero(route_fit == group)
        group_y = y_fit[group_rows]
        inner = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED + fold * 1000 + int(group) + 71,
        )
        predictions = np.full(len(group_rows), np.nan, dtype=np.float64)
        for inner_fold, (inner_fit, inner_valid) in enumerate(
            inner.split(np.zeros(len(group_rows)), group_y), start=1
        ):
            pred, elapsed = fit_predict(
                x_fit.iloc[group_rows[inner_fit]].reset_index(drop=True),
                group_y[inner_fit],
                x_fit.iloc[group_rows[inner_valid]].reset_index(drop=True),
                cat_columns,
                SEED + fold * 10000 + int(group) * 100 + inner_fold + 900000,
            )
            predictions[inner_valid] = pred
            runtimes.append(
                {
                    "experiment": f"calibration_inner_{gate}_group_{int(group)}",
                    "fold": fold,
                    "inner_fold": inner_fold,
                    "fit_rows": len(inner_fit),
                    "fit_seconds": elapsed,
                    "gpu": True,
                }
            )
        if not np.isfinite(predictions).all():
            raise ValueError(f"{gate} group {group} calibration OOF is incomplete")
        parameters[int(group)] = fit_platt_oof(predictions, group_y)
    return parameters, runtimes


def fit_specialists(
    gate: str,
    route: np.ndarray,
    x_fit: pd.DataFrame,
    x_valid: pd.DataFrame,
    cat_columns: list[str],
    y: np.ndarray,
    fit: np.ndarray,
    valid: np.ndarray,
    global_valid: np.ndarray,
    fold: int,
    *,
    calibrate: bool,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    support_check(y, route, fit, valid, gate)
    route_fit = route[fit]
    calibration: dict[int, tuple[float, float]] = {}
    runtimes: list[dict[str, Any]] = []
    if calibrate:
        calibration, calibration_runtimes = fit_expert_oof_calibration(
            x_fit, y[fit], route_fit, cat_columns, fold, gate
        )
        runtimes.extend(calibration_runtimes)
    hard = np.full(len(valid), np.nan, dtype=np.float64)
    calibrated = np.full(len(valid), np.nan, dtype=np.float64)
    expert = np.full(len(valid), np.nan, dtype=np.float64)
    for group in np.unique(route):
        fit_rows = np.flatnonzero(route_fit == group)
        valid_rows = np.flatnonzero(route[valid] == group)
        prediction, elapsed = fit_predict(
            x_fit.iloc[fit_rows].reset_index(drop=True),
            y[fit[fit_rows]],
            x_valid.iloc[valid_rows].reset_index(drop=True),
            cat_columns,
            SEED + fold * 100 + int(group) + 400000,
        )
        hard[valid_rows] = prediction
        expert[valid_rows] = prediction
        runtimes.append(
            {
                "experiment": f"specialist_outer_{gate}_group_{int(group)}",
                "fold": fold,
                "fit_rows": len(fit_rows),
                "fit_seconds": elapsed,
                "gpu": True,
            }
        )
        calibrated[valid_rows] = (
            apply_platt(prediction, calibration[int(group)])
            if calibrate
            else prediction
        )
    if not (
        np.isfinite(hard).all()
        and np.isfinite(calibrated).all()
        and np.isfinite(expert).all()
    ):
        raise ValueError(f"{gate} expert predictions are incomplete")
    mixture = 0.5 * global_valid + 0.5 * expert
    calibrated_mixture = 0.5 * global_valid + 0.5 * calibrated
    metrics: dict[str, Any] = {
        "gate": gate,
        "calibrated": calibrate,
        "global_auc": float(roc_auc_score(y[valid], global_valid)),
        "hard_route_auc": float(roc_auc_score(y[valid], hard)),
        "fixed_50_50_auc": float(roc_auc_score(y[valid], mixture)),
        "calibrated_50_50_auc": float(
            roc_auc_score(y[valid], calibrated_mixture)
        ),
        "fixed_50_50_logloss": float(
            log_loss(y[valid], np.clip(mixture, 1e-7, 1.0 - 1e-7))
        ),
        "calibrated_50_50_logloss": float(
            log_loss(
                y[valid], np.clip(calibrated_mixture, 1e-7, 1.0 - 1e-7)
            )
        ),
        "fixed_50_50_brier": float(brier_score_loss(y[valid], mixture)),
        "calibrated_50_50_brier": float(
            brier_score_loss(y[valid], calibrated_mixture)
        ),
        "calibration_parameters": json.dumps(calibration, sort_keys=True),
        "group_metrics": {},
    }
    for group in np.unique(route):
        mask = route[valid] == group
        if len(np.unique(y[valid][mask])) == 2:
            metrics["group_metrics"][str(int(group))] = {
                "rows": int(mask.sum()),
                "positives": int(y[valid][mask].sum()),
                "global_auc": float(
                    roc_auc_score(y[valid][mask], global_valid[mask])
                ),
                "hard_expert_auc": float(roc_auc_score(y[valid][mask], hard[mask])),
                "mixture_auc": float(
                    roc_auc_score(y[valid][mask], mixture[mask])
                ),
                "calibrated_mixture_auc": float(
                    roc_auc_score(y[valid][mask], calibrated_mixture[mask])
                ),
            }
    return metrics, hard, runtimes


def categorical_and_feature_schema(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y), dtype=np.int8), y
        )
    )
    first_fit, first_valid = folds[0]
    x_fit, x_valid, cats = phase15_matrix(blocks, first_fit, first_valid)
    if len(x_fit.columns) != 99:
        raise ValueError(
            f"Phase 15 feature schema changed: expected 99, got {len(x_fit.columns)}"
        )
    support: dict[str, Any] = {}
    routes = {
        "contract_type": contract_route(raw).astype(np.int8),
        "external_completeness": external_completeness_route(raw).astype(np.int8),
        "contract_x_external": combined_route(raw),
    }
    for name, route in routes.items():
        support[name] = {}
        for group in np.unique(route):
            mask = route == group
            support[name][str(int(group))] = {
                "rows": int(mask.sum()),
                "positives": int(y[mask].sum()),
            }
    return {
        "rows": len(raw),
        "predictors": len(raw.columns) - 1,
        "feature_count": int(x_fit.shape[1]),
        "categorical_count": len(cats),
        "categorical_columns": cats,
        "document_flag_columns_present": [
            column for column in raw.columns if column.startswith("FLAG_DOCUMENT_")
        ],
        "support": support,
    }


def run_gpu_smoke(
    blocks: tuple[pd.DataFrame, ...], raw: pd.DataFrame, y: np.ndarray
) -> dict[str, Any]:
    train_rows = np.arange(min(12000, len(y)))
    if len(train_rows) < 11000:
        raise ValueError("Supplied training data is too small for GPU smoke test")
    fit = train_rows[:10000]
    apply = train_rows[10000:12000]
    x_fit, x_apply, cats = phase15_matrix(blocks, fit, apply)
    pred, seconds = fit_predict(
        x_fit, y[fit], x_apply, cats, SEED + 999, one_hot_max_size=10
    )
    return {
        "gpu_catboost_smoke": "passed",
        "fit_rows": len(fit),
        "apply_rows": len(apply),
        "features": x_fit.shape[1],
        "categorical_features": len(cats),
        "fit_seconds": seconds,
        "finite_predictions": bool(np.isfinite(pred).all()),
    }


def auc_summary(
    name: str, predictions: np.ndarray, y: np.ndarray, fold_ids: np.ndarray
) -> dict[str, Any]:
    if not np.isfinite(predictions).all():
        raise ValueError(f"{name} OOF predictions are incomplete")
    fold_auc = [
        float(roc_auc_score(y[fold_ids == fold], predictions[fold_ids == fold]))
        for fold in range(1, FOLDS + 1)
    ]
    probability_metrics: dict[str, float | str] = {
        "logloss": "",
        "brier": "",
    }
    probability_metrics = {
        "logloss": float(log_loss(y, np.clip(predictions, 1e-7, 1.0 - 1e-7))),
        "brier": float(brier_score_loss(y, predictions)),
    }
    return {
        "experiment": name,
        "pooled_auc": float(roc_auc_score(y, predictions)),
        "fold_auc": json.dumps(fold_auc),
        "positive_folds_vs_reference": "",
        "delta_vs_phase15_single_seed": "",
        "mean_fold_auc": float(np.mean(fold_auc)),
        "prediction_correlation_with_phase15": "",
        **probability_metrics,
    }


def score_diagnostics(
    name: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    summary = auc_summary(name, prediction, y, fold_ids)
    summary["positive_folds_vs_reference"] = int(
        sum(
            roc_auc_score(y[fold_ids == fold], prediction[fold_ids == fold])
            > roc_auc_score(y[fold_ids == fold], reference[fold_ids == fold])
            for fold in range(1, FOLDS + 1)
        )
    )
    summary["delta_vs_phase15_single_seed"] = float(
        summary["pooled_auc"] - roc_auc_score(y, reference)
    )
    summary["prediction_correlation_with_phase15"] = float(
        np.corrcoef(prediction, reference)[0, 1]
    )
    return summary


def subset_pairwise_diagnostics(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, seed: int = 21
) -> dict[str, int]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    count = min(1_000_000, len(positive) * len(negative))
    positive_rows = rng.choice(positive, size=count, replace=True)
    negative_rows = rng.choice(negative, size=count, replace=True)
    base_correct = baseline[positive_rows] > baseline[negative_rows]
    candidate_correct = candidate[positive_rows] > candidate[negative_rows]
    return {
        "sampled_positive_negative_pairs": int(count),
        "global_wrong_candidate_correct": int(
            np.sum(~base_correct & candidate_correct)
        ),
        "global_correct_candidate_wrong": int(
            np.sum(base_correct & ~candidate_correct)
        ),
        "both_wrong": int(np.sum(~base_correct & ~candidate_correct)),
        "both_correct": int(np.sum(base_correct & candidate_correct)),
    }


def paired_stratified_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    repetitions: int = 300,
) -> dict[str, Any]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        sampled_y = y[sample]
        deltas[repetition] = roc_auc_score(
            sampled_y, candidate[sample]
        ) - roc_auc_score(sampled_y, reference[sample])
    return {
        "bootstrap_repetitions": repetitions,
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "interpretation": (
            "Paired row-stratified uncertainty conditional on these OOF vectors; "
            "does not account for training-fold variability or repeated model selection."
        ),
    }


def aggregate_specialist_metrics(
    fold_metrics: list[dict[str, Any]], y: np.ndarray
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for gate in sorted({str(row["gate"]) for row in fold_metrics}):
        rows = [row for row in fold_metrics if row["gate"] == gate]
        for metric in (
            "global_auc",
            "hard_route_auc",
            "fixed_50_50_auc",
            "calibrated_50_50_auc",
            "fixed_50_50_logloss",
            "calibrated_50_50_logloss",
            "fixed_50_50_brier",
            "calibrated_50_50_brier",
        ):
            values = [float(row[metric]) for row in rows]
            output.append(
                {
                    "gate": gate,
                    "metric": metric,
                    "mean_fold_value": float(np.mean(values)),
                    "fold_values": json.dumps(values),
                    "calibration": any(bool(row["calibrated"]) for row in rows),
                }
            )
    return output


def run_suite(
    train_path: Path, oof_path: Path, output_dir: Path, *, resume: bool = False
) -> None:
    if not train_path.is_file() or not oof_path.is_file():
        raise FileNotFoundError("Training or Phase 15 reference OOF file is missing")
    started = time.perf_counter()
    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(oof_path)
    if TARGET not in train or ID not in train:
        raise ValueError("Train data must include TARGET and SK_ID_CURR")
    y = train[TARGET].to_numpy(dtype=np.int8)
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("TARGET must be binary")
    if len(reference) != len(train):
        raise ValueError("Phase 15 OOF length does not match training rows")
    if not np.array_equal(train[ID].to_numpy(), reference[ID].to_numpy()):
        raise ValueError("Phase 15 OOF applicant IDs do not align")
    if not np.array_equal(y, reference[TARGET].to_numpy(dtype=np.int8)):
        raise ValueError("Phase 15 OOF targets do not align")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y), dtype=np.int8), y))
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = fold
    if not np.array_equal(fold_ids, reference["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Phase 15 folds do not match the frozen full-data fold")
    phase15_reference = reference["native_catboost_phase14_plus_time"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(phase15_reference).all():
        raise ValueError("Phase 15 OOF reference contains nonfinite predictions")
    if not PHASE15_MULTISEED.is_file():
        raise FileNotFoundError(f"Historical Phase 15 bagged OOF is missing: {PHASE15_MULTISEED}")
    multiseed = pd.read_csv(PHASE15_MULTISEED)
    if (
        len(multiseed) != len(train)
        or not np.array_equal(multiseed[ID].to_numpy(), train[ID].to_numpy())
        or not np.array_equal(multiseed[TARGET].to_numpy(dtype=np.int8), y)
    ):
        raise ValueError("Historical Phase 15 multiseed OOF alignment failed")
    historical_bagged = multiseed["phase15_three_seed_average"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(historical_bagged).all():
        raise ValueError("Historical Phase 15 three-seed OOF is nonfinite")
    historical_reference = {
        "phase15_three_seed_average_pooled_auc": float(
            roc_auc_score(y, historical_bagged)
        ),
        "phase15_three_seed_average_correlation_with_single_seed": float(
            np.corrcoef(historical_bagged, phase15_reference)[0, 1]
        ),
        "folds_match_this_run": False,
        "meaning": (
            "Historical rowwise mean of three OOF vectors with different fold "
            "assignments; contextual only, not a fold-matched comparison."
        ),
    }

    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = build_feature_blocks(raw)
    schema = categorical_and_feature_schema(blocks, raw, y)
    source_paths = [
        train_path,
        oof_path,
        ROOT / "run_phase21_focused_gpu.py",
        PHASE13 / "phase13_features.py",
        PHASE14 / "run_phase14.py",
        PHASE15 / "run_phase15_catboost_preflight.py",
        PHASE15 / "phase15_features.py",
        PHASE15_MULTISEED,
        PHASE20 / "run_phase20_segmentation_study.py",
    ]
    identity = {
        "train_sha256": sha256_file(train_path),
        "phase15_oof_sha256": sha256_file(oof_path),
        "phase15_multiseed_oof_sha256": sha256_file(PHASE15_MULTISEED),
        "source_sha256": {
            str(path): sha256_file(path) for path in source_paths
        },
        "feature_schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "iterations": ITERATIONS,
        "depth": 6,
        "learning_rate": 0.05,
        "task_type": "GPU",
        "one_hot_max_size": 10,
        "max_ctr_complexity": 1,
        "gpu_ram_part": 0.78,
        "border_count": 128,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "seed_counts": list(SEED_COUNTS),
        "learning_fractions": list(LEARNING_FRACTIONS),
    }
    prepare_run_directory(output_dir, resume, identity)
    (output_dir / "PHASE21_FEATURE_SCHEMA.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8"
    )
    smoke = run_gpu_smoke(blocks, raw, y)
    (output_dir / "PHASE21_GPU_PREFLIGHT.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"GPU_PREFLIGHT {json.dumps(smoke, sort_keys=True)}", flush=True)
    identity_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    progress_path = output_dir / "PHASE21_GPU_TELEMETRY_PROGRESS.json"
    prior_elapsed_seconds = 0.0
    session_count = 1
    telemetry_history_complete = True
    if resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity_sha256") != identity_sha256:
            raise ValueError("GPU telemetry checkpoint identity mismatch")
        gpu_samples = list(progress.get("samples", []))
        prior_elapsed_seconds = float(
            progress.get("active_runtime_seconds", 0.0)
        )
        session_count = int(progress.get("session_count", 0)) + 1
        telemetry_history_complete = (
            bool(progress.get("telemetry_history_complete", True))
            and bool(progress.get("active_session_complete", False))
        )
    else:
        gpu_samples = []
        if resume and any((output_dir / "fold_checkpoints").glob("fold_*.npz")):
            telemetry_history_complete = False
    completed_folds: set[int] = set()
    gpu_errors: list[str] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(gpu_samples, gpu_errors, stop_monitor),
        daemon=True,
    )
    monitor.start()

    checkpoint_dir = output_dir / "fold_checkpoints"
    if resume:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    fold_runtime: list[dict[str, Any]] = []
    specialist_fold_metrics: list[dict[str, Any]] = []
    global_oof = np.full(len(y), np.nan, dtype=np.float64)
    seed_oof = {
        index: np.full(len(y), np.nan, dtype=np.float64)
        for index in range(max(SEED_COUNTS))
    }
    seed_oof[0] = global_oof
    for name in (
        "track_a_raw",
        "track_a_constant_free_key_frequency",
        "no_external_scores",
        "no_product_keys",
        "no_demographics",
        "no_day_columns",
        "add_cash_flag",
        "add_ext_complete_flag",
        "add_goods_credit_flag",
        "add_all_domain_flags",
        "learning_20",
        "learning_40",
        "learning_60",
        "learning_80",
        "specialist_contract_hard",
        "specialist_contract_fixed",
        "specialist_contract_calibrated",
        "specialist_external_hard",
        "specialist_external_fixed",
        "specialist_combined_hard",
        "specialist_combined_fixed",
    ):
        arrays[name] = np.full(len(y), np.nan, dtype=np.float64)

    write_json_atomic(
        progress_path,
        {
            "identity_sha256": identity_sha256,
            "completed_folds": [],
            "session_count": session_count,
            "active_runtime_seconds": prior_elapsed_seconds,
            "samples": gpu_samples,
            "telemetry_history_complete": telemetry_history_complete,
            "active_session_complete": False,
        },
    )

    for fold, (fit, valid) in enumerate(folds, start=1):
        print(
            f"FOLD {fold}/{FOLDS} fit={len(fit)} valid={len(valid)}",
            flush=True,
        )
        checkpoint_path = checkpoint_dir / f"fold_{fold:02d}.npz"
        if checkpoint_path.is_file():
            if not resume:
                raise FileExistsError(
                    f"Unexpected existing fold checkpoint: {checkpoint_path}"
                )
            checkpoint = load_fold_checkpoint(
                checkpoint_path,
                fold,
                valid,
                list(arrays),
                list(seed_oof)[1:],
                identity,
            )
            global_oof[valid] = checkpoint["global_prediction"]
            for name, prediction in checkpoint["predictions"].items():
                arrays[name][valid] = prediction
            for index, prediction in checkpoint["seed_predictions"].items():
                seed_oof[int(index)][valid] = prediction
            fold_runtime.extend(checkpoint["fold_runtime"])
            specialist_fold_metrics.extend(checkpoint["specialist_metrics"])
            completed_folds.add(fold)
            print(f"RESUMED_FOLD {fold}/{FOLDS}", flush=True)
            continue
        fold_runtime_start = len(fold_runtime)
        specialist_metrics_start = len(specialist_fold_metrics)
        x_fit, x_valid, cat_columns = phase15_matrix(blocks, fit, valid)
        baseline_seed = SEED + fold * 100
        baseline_prediction, elapsed = fit_predict(
            x_fit, y[fit], x_valid, cat_columns, baseline_seed
        )
        baseline_elapsed = elapsed
        global_oof[valid] = baseline_prediction
        fold_runtime.append(
            {
                "experiment": "phase15_track_b_global",
                "fold": fold,
                "fit_rows": len(fit),
                "fit_seconds": elapsed,
                "seed": baseline_seed,
            }
        )
        for seed_index in range(1, max(SEED_COUNTS)):
            seed_value = baseline_seed + SEED_OFFSETS[seed_index]
            prediction, elapsed = fit_predict(
                x_fit, y[fit], x_valid, cat_columns, seed_value
            )
            seed_oof[seed_index][valid] = prediction
            fold_runtime.append(
                {
                    "experiment": "seed_scaling",
                    "fold": fold,
                    "seed_index": seed_index + 1,
                    "fit_rows": len(fit),
                    "fit_seconds": elapsed,
                    "seed": seed_value,
                }
            )

        subsets = nested_stratified_subsets(fit, y, fold)
        for fraction in LEARNING_FRACTIONS:
            selected = subsets[fraction]
            x_learning_fit, x_learning_valid, learning_cat_columns = (
                learning_curve_matrix(blocks, selected, valid)
            )
            if learning_cat_columns != cat_columns:
                raise ValueError("Learning-curve categorical schema changed")
            prediction, elapsed = fit_predict(
                x_learning_fit,
                y[selected],
                x_learning_valid,
                learning_cat_columns,
                SEED + fold * 100 + int(fraction * 100000),
            )
            arrays[f"learning_{int(fraction * 100)}"][valid] = prediction
            del x_learning_fit, x_learning_valid
            fold_runtime.append(
                {
                    "experiment": f"learning_{int(fraction * 100)}",
                    "fold": fold,
                    "fit_rows": len(selected),
                    "fit_seconds": elapsed,
                }
            )

        for group, result_name in (
            ("external_scores", "no_external_scores"),
            ("product_keys", "no_product_keys"),
            ("demographics", "no_demographics"),
            ("day_columns", "no_day_columns"),
        ):
            xa_fit, xa_valid, xa_cat = drop_feature_group(
                x_fit, x_valid, cat_columns, group
            )
            prediction, elapsed = fit_predict(
                xa_fit,
                y[fit],
                xa_valid,
                xa_cat,
                SEED + fold * 100 + len(result_name) * 31,
            )
            arrays[result_name][valid] = prediction
            fold_runtime.append(
                {
                    "experiment": result_name,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "feature_count": xa_fit.shape[1],
                    "fit_seconds": elapsed,
                }
            )

        for flag_name, selected_flags in (
            ("add_cash_flag", ["P21_IS_CASH_LOAN"]),
            ("add_ext_complete_flag", ["P21_EXT_ALL_COMPLETE"]),
            ("add_goods_credit_flag", ["P21_GOODS_CREDIT_BELOW_090"]),
            (
                "add_all_domain_flags",
                [
                    "P21_IS_CASH_LOAN",
                    "P21_EXT_ALL_COMPLETE",
                    "P21_GOODS_CREDIT_BELOW_090",
                    "P21_CASH_X_EXT_COMPLETE",
                ],
            ),
        ):
            train_flags = feature_flags(raw, fit)[selected_flags].reset_index(drop=True)
            valid_flags = feature_flags(raw, valid)[selected_flags].reset_index(drop=True)
            xf = pd.concat([x_fit.reset_index(drop=True), train_flags], axis=1)
            xv = pd.concat([x_valid.reset_index(drop=True), valid_flags], axis=1)
            prediction, elapsed = fit_predict(
                xf,
                y[fit],
                xv,
                cat_columns,
                SEED + fold * 100 + len(flag_name) * 41,
            )
            arrays[flag_name][valid] = prediction
            fold_runtime.append(
                {
                    "experiment": flag_name,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "feature_count": xf.shape[1],
                    "fit_seconds": elapsed,
                }
            )

        for add_keys, result_name in (
            (False, "track_a_raw"),
            (True, "track_a_constant_free_key_frequency"),
        ):
            ta_fit, ta_valid, ta_cat = track_a_matrix(
                blocks,
                raw,
                fit,
                valid,
                add_constant_free_frequencies=add_keys,
            )
            prediction, elapsed = fit_predict(
                ta_fit,
                y[fit],
                ta_valid,
                ta_cat,
                SEED + fold * 100 + (21 if add_keys else 11),
                one_hot_max_size=256,
            )
            arrays[result_name][valid] = prediction
            fold_runtime.append(
                {
                    "experiment": result_name,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "feature_count": ta_fit.shape[1],
                    "fit_seconds": elapsed,
                    "one_hot_max_size": 256,
                }
            )

        for gate, route, names, calibrate in (
            (
                "contract_type",
                contract_route(raw),
                (
                    "specialist_contract_hard",
                    "specialist_contract_fixed",
                    "specialist_contract_calibrated",
                ),
                True,
            ),
            (
                "external_completeness",
                external_completeness_route(raw),
                (
                    "specialist_external_hard",
                    "specialist_external_fixed",
                    None,
                ),
                False,
            ),
            (
                "contract_x_external_completeness",
                combined_route(raw),
                (
                    "specialist_combined_hard",
                    "specialist_combined_fixed",
                    None,
                ),
                False,
            ),
        ):
            specialist_metrics, hard, specialist_runtimes = fit_specialists(
                gate,
                route,
                x_fit,
                x_valid,
                cat_columns,
                y,
                fit,
                valid,
                baseline_prediction,
                fold,
                calibrate=calibrate,
            )
            specialist_fold_metrics.append(
                {"fold": fold, **specialist_metrics}
            )
            fold_runtime.extend(specialist_runtimes)
            arrays[names[0]][valid] = hard
            arrays[names[1]][valid] = (
                0.5 * baseline_prediction + 0.5 * hard
            )
            if names[2] is not None:
                # The calibrated mixture is accumulated by the dedicated
                # calibrated prediction path computed with the same fold fit.
                calibrated_route = route[valid]
                calibrated_values = np.full(len(valid), np.nan, dtype=np.float64)
                for group_key, params in json.loads(
                    specialist_metrics["calibration_parameters"]
                ).items():
                    group = int(group_key)
                    mask = calibrated_route == group
                    calibrated_values[mask] = apply_platt(
                        hard[mask], (float(params[0]), float(params[1]))
                    )
                arrays[names[2]][valid] = (
                    0.5 * baseline_prediction + 0.5 * calibrated_values
                )

        del x_fit, x_valid
        gc.collect()
        save_fold_checkpoint(
            checkpoint_dir,
            fold,
            valid,
            global_oof[valid],
            {
                index: seed_oof[index][valid]
                for index in range(1, max(SEED_COUNTS))
            },
            {name: prediction[valid] for name, prediction in arrays.items()},
            fold_runtime[fold_runtime_start:],
            specialist_fold_metrics[specialist_metrics_start:],
            identity,
        )
        completed_folds.add(fold)
        write_json_atomic(
            progress_path,
            {
                "identity_sha256": identity_sha256,
                "completed_folds": sorted(completed_folds),
                "session_count": session_count,
                "active_runtime_seconds": (
                    prior_elapsed_seconds + time.perf_counter() - started
                ),
                "samples": list(gpu_samples),
                "telemetry_history_complete": telemetry_history_complete,
                "active_session_complete": False,
            },
        )
        print(
            f"FOLD {fold} baseline_auc="
            f"{roc_auc_score(y[valid], baseline_prediction):.6f} "
            f"baseline_seconds={baseline_elapsed:.1f}",
            flush=True,
        )

    stop_monitor.set()
    monitor.join(timeout=15)
    if monitor.is_alive():
        raise RuntimeError("GPU telemetry monitor did not stop cleanly")
    if gpu_errors:
        raise RuntimeError(
            "GPU telemetry collection failed: " + "; ".join(gpu_errors)
        )
    expected_folds = set(range(1, FOLDS + 1))
    if completed_folds != expected_folds:
        raise ValueError(
            f"Completed fold checkpoints differ from protocol: {sorted(completed_folds)}"
        )
    active_runtime_seconds = (
        prior_elapsed_seconds + time.perf_counter() - started
    )
    write_json_atomic(
        progress_path,
        {
            "identity_sha256": identity_sha256,
            "completed_folds": sorted(completed_folds),
            "session_count": session_count,
            "active_runtime_seconds": active_runtime_seconds,
            "samples": list(gpu_samples),
            "telemetry_history_complete": telemetry_history_complete,
            "active_session_complete": True,
        },
    )
    telemetry = {
        "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "sampling_interval_seconds": 2.0,
        "sample_count": len(gpu_samples),
        "session_count": session_count,
        "completed_folds": sorted(completed_folds),
        "history_complete": telemetry_history_complete,
        "coverage_note": (
            "Samples are checkpointed after each fold; an interrupted session's "
            "uncheckpointed tail may be absent."
        ),
        "max_utilization_percent": max(
            (row["gpu_utilization_percent"] for row in gpu_samples), default=None
        ),
        "max_memory_used_mib": max(
            (row["memory_used_mib"] for row in gpu_samples), default=None
        ),
        "samples": gpu_samples,
    }
    (output_dir / "PHASE21_GPU_TELEMETRY.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8"
    )

    if not np.isfinite(global_oof).all():
        raise ValueError("Global OOF predictions do not cover all training rows")
    for name, prediction in arrays.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete OOF coverage for {name}")
    for index, prediction in seed_oof.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete OOF coverage for seed index {index}")

    baseline_summary = score_diagnostics(
        "phase15_track_b_global_reproduced", global_oof, phase15_reference, y, fold_ids
    )
    baseline_summary["reference_phase15_auc"] = float(
        roc_auc_score(y, phase15_reference)
    )
    results = [baseline_summary]
    for name, prediction in arrays.items():
        results.append(score_diagnostics(name, prediction, global_oof, y, fold_ids))

    for seed_count in SEED_COUNTS:
        averaged = np.mean(
            np.column_stack([seed_oof[index] for index in range(seed_count)]),
            axis=1,
        )
        summary = score_diagnostics(
            f"seed_average_{seed_count}", averaged, global_oof, y, fold_ids
        )
        summary["seed_count"] = seed_count
        results.append(summary)

    pairwise = {
        name: subset_pairwise_diagnostics(y, global_oof, prediction)
        for name, prediction in arrays.items()
        if name.startswith("specialist_") or name.startswith("add_")
    }
    (output_dir / "PHASE21_ERROR_CORRECTIONS.json").write_text(
        json.dumps(pairwise, indent=2, sort_keys=True), encoding="utf-8"
    )
    bootstrap_candidates = (
        "specialist_contract_hard",
        "specialist_contract_fixed",
        "specialist_contract_calibrated",
        "specialist_external_fixed",
        "specialist_combined_fixed",
        "track_a_constant_free_key_frequency",
    )
    bootstrap_rows = []
    for index, name in enumerate(bootstrap_candidates):
        result = paired_stratified_bootstrap(
            y,
            global_oof,
            arrays[name],
            seed=SEED + 900000 + index,
        )
        bootstrap_rows.append({"candidate": name, **result})
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "PHASE21_PAIRED_BOOTSTRAP.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(results).to_csv(
        output_dir / "PHASE21_MODEL_RESULTS.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(fold_runtime).to_csv(
        output_dir / "PHASE21_FIT_RUNTIME.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(specialist_fold_metrics).to_json(
        output_dir / "PHASE21_SPECIALIST_FOLD_METRICS.json",
        orient="records",
        indent=2,
    )
    pd.DataFrame(
        aggregate_specialist_metrics(specialist_fold_metrics, y)
    ).to_csv(
        output_dir / "PHASE21_SPECIALIST_SUMMARY.csv",
        index=False,
        lineterminator="\n",
    )
    seed_rows = []
    for seed_count in SEED_COUNTS:
        prediction = np.mean(
            np.column_stack([seed_oof[index] for index in range(seed_count)]),
            axis=1,
        )
        seed_rows.append(
            {
                "seed_count": seed_count,
                "pooled_auc": float(roc_auc_score(y, prediction)),
                "correlation_with_single_seed": float(
                    np.corrcoef(prediction, global_oof)[0, 1]
                ),
                "mean_single_seed_correlation": float(
                    np.mean(
                        [
                            np.corrcoef(prediction, seed_oof[index])[0, 1]
                            for index in range(seed_count)
                        ]
                    )
                ),
            }
        )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "PHASE21_SEED_SCALING.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(
        [
            {
                "fraction": fraction,
                "pooled_auc": float(
                    roc_auc_score(y, arrays[f"learning_{int(fraction*100)}"])
                ),
                "fold_auc": json.dumps(
                    [
                        float(
                            roc_auc_score(
                                y[fold_ids == fold],
                                arrays[f"learning_{int(fraction*100)}"][
                                    fold_ids == fold
                                ],
                            )
                        )
                        for fold in range(1, FOLDS + 1)
                    ]
                ),
            }
            for fraction in LEARNING_FRACTIONS
        ]
    ).to_csv(
        output_dir / "PHASE21_LEARNING_CURVE.csv",
        index=False,
        lineterminator="\n",
    )
    oof_columns: dict[str, Any] = {
        ID: train[ID].to_numpy(),
        TARGET: y,
        "fold": fold_ids,
        "phase15_track_b_global_reproduced": global_oof,
    }
    oof_columns.update(arrays)
    oof_columns.update(
        {
            f"seed_{index + 1}": prediction
            for index, prediction in seed_oof.items()
        }
    )
    pd.DataFrame(oof_columns).to_csv(
        output_dir / "PHASE21_OOF_PREDICTIONS.csv",
        index=False,
        lineterminator="\n",
    )
    integrity = {
        "train_rows": len(train),
        "unique_ids": int(train[ID].nunique()),
        "train_sha256": sha256_file(train_path),
        "phase15_oof_sha256": sha256_file(oof_path),
        "phase15_multiseed_oof_sha256": sha256_file(PHASE15_MULTISEED),
        "historical_phase15_reference": historical_reference,
        "fold_seed": SEED,
        "fold_sizes": {
            str(fold): int(np.sum(fold_ids == fold))
            for fold in range(1, FOLDS + 1)
        },
        "target_alignment": True,
        "id_alignment": True,
        "all_new_oof_predictions_finite": True,
        "no_test_or_external_data_read": True,
        "prediction_cache_written": True,
        "prediction_cache_path": "PHASE21_OOF_PREDICTIONS.csv",
        "runtime_seconds": active_runtime_seconds,
        "feature_schema": schema,
        "gpu_preflight": smoke,
        "completed_folds": sorted(completed_folds),
        "resume_session_count": session_count,
        "gpu_telemetry_history_complete": telemetry_history_complete,
        "ranker_screening_pilot": {
            "outer_fold": 1,
            "gpu_auc": 0.7102218572566733,
            "objective": "YetiRankPairwise",
            "status": "excluded_from_full_suite",
            "reason": (
                "Weak one-fold result; CatBoost's default PFound metric is "
                "computed on CPU, so this is not GPU-only end to end."
            ),
        },
        "fit_count_including_smoke": len(fold_runtime) + 1,
        "fit_count_excluding_smoke": len(fold_runtime),
        "gpu_telemetry_sample_count": len(gpu_samples),
        "unsupported_or_deferred": {
            "monotone_constraints": (
                "GPU CatBoost preflight rejected monotone_constraints as unimplemented."
            ),
            "logistic_stack": (
                "No CUDA logistic-regression package or CUDA-enabled PyTorch is installed; "
                "not silently run on CPU."
            ),
            "hierarchical_target_encoding": (
                "Deferred pending organizer guidance on nested target statistics; "
                "not mixed into Track A."
            ),
            "test_key_coverage": (
                "Not inspected by this training run; any separate test-feature "
                "diagnostic must be documented independently."
            ),
            "document_flags": (
                "No FLAG_DOCUMENT_* columns exist in supplied train schema."
            ),
        },
    }
    (output_dir / "PHASE21_RUN_SUMMARY.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_reports(output_dir, results, specialist_fold_metrics, integrity, pairwise)
    write_manifest(output_dir, source_paths)


def write_reports(
    output_dir: Path,
    results: list[dict[str, Any]],
    specialist_metrics: list[dict[str, Any]],
    integrity: dict[str, Any],
    pairwise: dict[str, Any],
) -> None:
    frame = pd.DataFrame(results)
    best = frame.sort_values("pooled_auc", ascending=False).iloc[0]
    baseline = frame.loc[
        frame.experiment.eq("phase15_track_b_global_reproduced")
    ].iloc[0]
    telemetry_statement = (
        "GPU telemetry covers every active suite process"
        if integrity["gpu_telemetry_history_complete"]
        else (
            "GPU telemetry is checkpointed per fold, but an interrupted "
            "process may have an unrecorded tail"
        )
    )
    specialists = pd.DataFrame(specialist_metrics)
    specialist_summary = (
        specialists.groupby("gate")[
            [
                "hard_route_auc",
                "fixed_50_50_auc",
                "calibrated_50_50_auc",
            ]
        ]
        .mean()
        .to_string()
        if not specialists.empty
        else "No specialist results."
    )
    report = f"""# Phase 21 — focused full-data validation

## Status and protocol

This is a focused, full-training-data follow-up on the same five stratified folds
as Phase 15 (`random_state={SEED}`). Each candidate is compared with the
reproduced Phase 15 GPU CatBoost single-seed baseline on identical rows. It is
not independent confirmation: these labels have been examined in prior phases.
No test data, external data, submissions, or champion files were read or changed.

The Phase 15 Track B baseline uses its original 99-column feature representation,
including native CatBoost category statistics and historical product keys. Their
competition eligibility remains unresolved. Track A comparisons use GPU
CatBoost with one-hot categories only (`one_hot_max_size=256`) and no historical
product-key categories; a second Track A model adds fold-local frequencies of
four direct, constant-free ratios. All models use 700 iterations, depth 6,
learning rate 0.05, and no outer-fold early stopping.
For each learning-curve fraction, product-key frequency features are recomputed
from that fraction's selected training rows and applied to the unchanged outer
validation fold.

## Main findings

- Reproduced single-seed full-data AUC: `{float(baseline.pooled_auc):.9f}`;
  saved Phase 15 reference: `{float(baseline.reference_phase15_auc):.9f}`.
- Historical Phase 15 three-seed OOF AUC:
  `{integrity['historical_phase15_reference']['phase15_three_seed_average_pooled_auc']:.9f}`.
  Those component OOF vectors used different fold assignments, so this is
  context only; the new 1/3/5/10/20 seed curve holds the same folds fixed.
- Best result in this preregistered follow-up: `{best.experiment}` at
  `{float(best.pooled_auc):.9f}`. This is a research result on previously
  examined labels, not a promotion or score claim.
- See `PHASE21_LEARNING_CURVE.csv`, `PHASE21_SEED_SCALING.csv`,
  `PHASE21_SPECIALIST_SUMMARY.csv`, and `PHASE21_MODEL_RESULTS.csv` for
  fold-level detail.

## Specialist error and calibration analysis

Full-data five-fold specialist means:

```
{specialist_summary}
```

Hard routing, the fixed 50/50 global/expert mixture, and the contract-type
Platt-calibrated mixture are evaluated on each outer fold. Calibration
parameters are fitted only to inner-fold OOF expert scores within each outer
training partition. The sampled positive-negative pair counts showing where
specialists reversed global ordering errors are in `PHASE21_ERROR_CORRECTIONS.json`.
Complete row-level predictions, IDs, targets, and fold assignments are saved in
`PHASE21_OOF_PREDICTIONS.csv` for independent AUC, correlation, and error analysis.
`PHASE21_PAIRED_BOOTSTRAP.csv` gives 300-replicate row-stratified intervals
conditional on these OOF vectors; they do not capture training-fold
variability or the effects of earlier model selection.

## Eligibility, blocked directions, and data inventory

- Only supplied `train.csv` and the saved Phase 15 train OOF reference were read.
- All heavy model fits use CatBoost GPU. The tiny two-parameter Platt mapping is
  fitted on CPU with deterministic numerical optimization; no CPU classifier
  or general model fallback is used.
- GPU monotonic constraints were tested and rejected by CatBoost as
  unimplemented for GPU.
- A one-fold GPU `YetiRankPairwise` screen scored
  `{integrity['ranker_screening_pilot']['gpu_auc']:.9f}`. Its default PFound
  metric is CPU-computed, so the objective was excluded from the full suite
  both for the weak result and to avoid CPU work.
- The logistic-regression stack was not run because no CUDA logistic solver is
  installed; CPU fallback is intentionally prohibited.
- Nested target encodings/hierarchical target-statistic shrinkage are deferred
  pending an organizer ruling. No TARGET encodings are used here.
- No `FLAG_DOCUMENT_*` columns exist in the supplied training schema.
- Test product-key coverage was not inspected to preserve the established
  no-test-data development protocol.
- A genuinely untouched labeled holdout is unavailable after prior phases;
  this run cannot establish independent generalization.

## Completion

Run summary: `{integrity['runtime_seconds']:.1f}` seconds, with
`{integrity['fit_count_including_smoke']}` GPU model fits including smoke test,
on `{integrity['train_rows']}` rows across
`{integrity['resume_session_count']}` process session(s). {telemetry_statement}:
`{integrity['gpu_telemetry_sample_count']}` samples at two-second intervals.
Completed folds: `{integrity['completed_folds']}`.
Completed fold predictions are atomically checkpointed and resumable only when
input, code, and feature-schema hashes match.
Train SHA-256: `{integrity['train_sha256']}`.
"""
    (output_dir / "PHASE21_EXECUTION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    organizer_questions = """# Organizer questions — eligibility clarification

These are prepared questions, not messages sent to organizers.

1. **Nested target statistics:** Are supervised encodings (category means,
   smoothed rates, counts conditioned on TARGET, and hierarchical/shrunk
   product-key encodings) allowed when every statistic is computed solely from
   supplied training labels and is nested/cross-fitted so each scored row's
   label is excluded?
2. **Data-derived constants:** Are deterministic constants inferred only from
   the supplied training columns (for example a repeated fee gap or financing
   increment) allowed in rowwise feature formulas when the complete estimator
   is reproducible and is applied without applicant-specific manual decisions?
   Are constants estimated inside each training fold treated differently from
   constants fixed once from all supplied training rows?
3. **Train/test covariate pooling:** May unsupervised transforms such as
   category-frequency maps or quantile bins be fit on the combined supplied
   train and test covariates, without using test labels, or must every transform
   be fit on training rows only?

Until written answers are received, the runner treats target-derived features
as unresolved Track B and avoids all test-covariate fitting.
"""
    (output_dir / "PHASE21_ORGANIZER_QUESTIONS.md").write_text(
        organizer_questions, encoding="utf-8"
    )


def write_manifest(output_dir: Path, sources: list[Path]) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "PHASE21_SHA256_MANIFEST.csv"
            and path.suffix != ".tmp"
        ):
            rows.append(
                {
                    "kind": "output",
                    "path": str(path.relative_to(REPO)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in sources:
        rows.append(
            {
                "kind": "source_or_input",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "PHASE21_SHA256_MANIFEST.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--phase15-oof", type=Path, default=PHASE15_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_suite(
        args.train.resolve(),
        args.phase15_oof.resolve(),
        args.output_dir.resolve(),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
