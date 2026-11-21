from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
PHASE15 = REPO / "experiments" / "phase15_autonomous_research"
PHASE21 = REPO / "experiments" / "phase21_focused_validation"
PHASE22 = REPO / "experiments" / "phase22_pairwise_auc_diagnostics"
sys.path[:0] = [str(PHASE15), str(PHASE13), str(PHASE21), str(PHASE22)]

import phase13_features as features13
import run_phase21_focused_gpu as phase21
import run_pairwise_auc_diagnostics as pairwise22


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
PHASE15_OOF = (
    PHASE15
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT = ROOT / "runs" / "phase23_targeted_20261106"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
ITERATIONS = 700
TARGET = features13.TARGET_COLUMN
ID = features13.ID_COLUMN
MISSING_COLUMNS = (
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)
PROFILE_PAIRS = (
    ("OCCUPATION_TYPE", "NAME_EDUCATION_TYPE"),
    ("NAME_INCOME_TYPE", "DAYS_EMPLOYED"),
    ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    ("NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"),
)
FAMILY_NAMES = (
    "application_process",
    "external_disagreement",
    "financial_residuals",
    "applicant_consistency",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)


def _category_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype("string").fillna("<MISSING>").astype(str).to_numpy()


def _pair_keys(frame: pd.DataFrame, left: str, right: str) -> np.ndarray:
    a = _category_values(frame, left)
    b = _category_values(frame, right)
    return np.char.add(np.char.add(a.astype(str), "\x1f"), b.astype(str))


def application_process_features(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = raw.loc[:, MISSING_COLUMNS].isna()
    flag_values: dict[str, np.ndarray] = {
        f"P23_MISSING_{column}": missing[column].to_numpy(dtype=np.int8)
        for column in MISSING_COLUMNS
    }
    employed = _numeric(raw, "DAYS_EMPLOYED")
    flag_values["P23_EMPLOYMENT_SENTINEL"] = (employed == 365243.0).astype(np.int8)
    fit_flags = pd.DataFrame(flag_values).iloc[fit_rows].reset_index(drop=True)
    apply_flags = pd.DataFrame(flag_values).iloc[apply_rows].reset_index(drop=True)

    pattern_codes = np.zeros(len(raw), dtype=np.int16)
    for bit, column in enumerate(MISSING_COLUMNS):
        pattern_codes |= missing[column].to_numpy(dtype=np.int16) << bit
    fit_codes = pattern_codes[fit_rows]
    apply_codes = pattern_codes[apply_rows]
    levels = np.unique(fit_codes)
    pattern_fit = pd.DataFrame(
        {
            f"P23_MISSING_PROFILE_{int(level):03d}": (fit_codes == level).astype(
                np.int8
            )
            for level in levels
        }
    )
    pattern_apply = pd.DataFrame(
        {
            f"P23_MISSING_PROFILE_{int(level):03d}": (
                apply_codes == level
            ).astype(np.int8)
            for level in levels
        }
    )
    known_codes = set(levels.tolist())
    pattern_fit["P23_MISSING_PROFILE_UNSEEN"] = 0
    pattern_apply["P23_MISSING_PROFILE_UNSEEN"] = np.asarray(
        [code not in known_codes for code in apply_codes], dtype=np.int8
    )
    pattern_fit["P23_MISSING_COUNT"] = fit_flags[
        [f"P23_MISSING_{column}" for column in MISSING_COLUMNS]
    ].sum(axis=1)
    pattern_apply["P23_MISSING_COUNT"] = apply_flags[
        [f"P23_MISSING_{column}" for column in MISSING_COLUMNS]
    ].sum(axis=1)

    age = -_numeric(raw, "DAYS_BIRTH") / 365.25
    cash = raw["NAME_CONTRACT_TYPE"].astype(str).eq("Cash loans").to_numpy(
        dtype=np.int8
    )
    conditional_fit: dict[str, np.ndarray] = {}
    conditional_apply: dict[str, np.ndarray] = {}
    for column in MISSING_COLUMNS:
        flag = missing[column].to_numpy(dtype=np.float64)
        name = f"P23_MISSING_{column}"
        conditional_fit[f"{name}_X_AGE"] = flag[fit_rows] * age[fit_rows]
        conditional_apply[f"{name}_X_AGE"] = flag[apply_rows] * age[apply_rows]
        conditional_fit[f"{name}_X_CASH"] = flag[fit_rows] * cash[fit_rows]
        conditional_apply[f"{name}_X_CASH"] = flag[apply_rows] * cash[apply_rows]
    fit_out = pd.concat(
        [
            fit_flags.reset_index(drop=True),
            pattern_fit.reset_index(drop=True),
            pd.DataFrame(conditional_fit),
        ],
        axis=1,
    )
    apply_out = pd.concat(
        [
            apply_flags.reset_index(drop=True),
            pattern_apply.reset_index(drop=True),
            pd.DataFrame(conditional_apply),
        ],
        axis=1,
    )
    if fit_out.columns.tolist() != apply_out.columns.tolist():
        raise ValueError("Application-process feature schemas differ")
    return fit_out, apply_out


def external_disagreement_features(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    available = np.isfinite(values)
    count = available.sum(axis=1)
    mean = np.divide(
        np.nansum(values, axis=1),
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    )
    ordered = np.sort(np.where(available, values, np.inf), axis=1)
    median = np.full(len(values), np.nan, dtype=np.float64)
    median[count == 1] = ordered[count == 1, 0]
    median[count == 2] = ordered[count == 2, :2].mean(axis=1)
    median[count == 3] = ordered[count == 3, 1]
    minimum = np.where(count > 0, ordered[:, 0], np.nan)
    maximum = np.where(count > 0, ordered[:, 2], np.nan)
    maximum[count == 1] = ordered[count == 1, 0]
    maximum[count == 2] = ordered[count == 2, 1]
    variance = np.divide(
        np.sum(np.where(available, (values - mean[:, None]) ** 2, 0.0), axis=1),
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    )
    std = np.sqrt(variance)
    std[count < 2] = np.nan
    pair_differences: list[np.ndarray] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        valid = available[:, left] & available[:, right]
        diff = np.full(len(values), np.nan, dtype=np.float64)
        diff[valid] = np.abs(values[valid, left] - values[valid, right])
        pair_differences.append(diff)
    diff_matrix = np.column_stack(pair_differences)
    diff_count = np.isfinite(diff_matrix).sum(axis=1)
    pair_mean = np.divide(
        np.nansum(diff_matrix, axis=1),
        diff_count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=diff_count > 0,
    )
    out = pd.DataFrame(
        {
            "P23_EXT_AVAILABLE_COUNT": count.astype(np.int8),
            "P23_EXT_MEAN": mean,
            "P23_EXT_MEDIAN": median,
            "P23_EXT_RANGE": maximum - minimum,
            "P23_EXT_STD": std,
            "P23_EXT_ABS_DIFF_12": pair_differences[0],
            "P23_EXT_ABS_DIFF_13": pair_differences[1],
            "P23_EXT_ABS_DIFF_23": pair_differences[2],
            "P23_EXT_MEAN_PAIRWISE_ABS_DIFF": pair_mean,
        }
    )
    out.loc[count < 2, ["P23_EXT_RANGE", "P23_EXT_STD"]] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.array([], dtype=np.float64)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, bins + 1)))


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) < 2:
        return np.zeros(len(values), dtype=np.int16)
    ids = np.searchsorted(edges[1:-1], values, side="right").astype(np.int16)
    ids[~np.isfinite(values)] = -1
    return ids


def _residual_prediction(
    fit_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    target_column: str,
    left_column: str,
    right_column: str,
    bins: int = 8,
    shrinkage: float = 10.0,
) -> np.ndarray:
    target = _numeric(fit_raw, target_column)
    left_fit = np.log1p(np.maximum(_numeric(fit_raw, left_column), 0.0))
    right_fit = np.log1p(np.maximum(_numeric(fit_raw, right_column), 0.0))
    left_apply = np.log1p(np.maximum(_numeric(apply_raw, left_column), 0.0))
    right_apply = np.log1p(np.maximum(_numeric(apply_raw, right_column), 0.0))
    target_log = np.where(np.isfinite(target) & (target >= 0), np.log1p(target), np.nan)
    valid_fit = (
        np.isfinite(target_log) & np.isfinite(left_fit) & np.isfinite(right_fit)
    )
    global_median = (
        float(np.nanmedian(target_log[valid_fit])) if valid_fit.any() else np.nan
    )
    if not np.isfinite(global_median):
        return np.full(len(apply_raw), np.nan, dtype=np.float64)
    left_edges = _bin_edges(left_fit, bins)
    right_edges = _bin_edges(right_fit, bins)
    left_ids = _bin_ids(left_fit, left_edges)
    right_ids = _bin_ids(right_fit, right_edges)
    apply_left_ids = _bin_ids(left_apply, left_edges)
    apply_right_ids = _bin_ids(right_apply, right_edges)
    keys = left_ids.astype(np.int32) * (bins + 1) + right_ids.astype(np.int32)
    apply_keys = (
        apply_left_ids.astype(np.int32) * (bins + 1)
        + apply_right_ids.astype(np.int32)
    )
    table = pd.DataFrame(
        {"key": keys[valid_fit], "target": target_log[valid_fit]}
    ).groupby("key")["target"].agg(["median", "count"])
    smoothed = (
        table["median"] * table["count"]
        + global_median * shrinkage
    ) / (table["count"] + shrinkage)
    prediction = (
        pd.Series(apply_keys)
        .map(smoothed)
        .fillna(global_median)
        .to_numpy(dtype=np.float64)
    )
    return prediction


def financial_residual_features(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definitions = (
        (
            "P23_LOG_ANNUITY_RESID_CREDIT_INCOME",
            "AMT_ANNUITY",
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL",
        ),
        (
            "P23_LOG_GOODS_RESID_CREDIT_INCOME",
            "AMT_GOODS_PRICE",
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL",
        ),
    )
    fit_features = pd.DataFrame(
        index=pd.RangeIndex(len(fit_rows))
    )
    apply_features = pd.DataFrame(
        index=pd.RangeIndex(len(apply_rows))
    )
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for name, target_column, left_column, right_column in definitions:
        target = _numeric(raw.iloc[fit_rows], target_column)
        log_target = np.where(
            np.isfinite(target) & (target >= 0), np.log1p(target), np.nan
        )
        oof_expectation = np.full(len(fit_rows), np.nan, dtype=np.float64)
        for inner_fit, inner_apply in inner.split(fit_rows):
            prediction = _residual_prediction(
                raw.iloc[fit_rows[inner_fit]],
                raw.iloc[fit_rows[inner_apply]],
                target_column,
                left_column,
                right_column,
            )
            oof_expectation[inner_apply] = prediction
        if not np.isfinite(oof_expectation).all():
            raise ValueError(f"{name} inner cross-fit is incomplete")
        fit_features[name] = log_target - oof_expectation
        apply_expectation = _residual_prediction(
            raw.iloc[fit_rows],
            raw.iloc[apply_rows],
            target_column,
            left_column,
            right_column,
        )
        apply_target = _numeric(raw.iloc[apply_rows], target_column)
        apply_log_target = np.where(
            np.isfinite(apply_target) & (apply_target >= 0),
            np.log1p(apply_target),
            np.nan,
        )
        apply_features[name] = apply_log_target - apply_expectation
    return fit_features.reset_index(drop=True), apply_features.reset_index(drop=True)


def applicant_consistency_features(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = {
        f"P23_PROFILE_{left}_X_{right}": _pair_keys(raw, left, right)
        for left, right in PROFILE_PAIRS
    }
    fit_out = pd.DataFrame(index=pd.RangeIndex(len(fit_rows)))
    apply_out = pd.DataFrame(index=pd.RangeIndex(len(apply_rows)))
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for name, values in keys.items():
        fit_values = values[fit_rows]
        apply_values = values[apply_rows]
        oof_frequency = np.full(len(fit_rows), np.nan, dtype=np.float64)
        for inner_fit, inner_apply in inner.split(fit_rows):
            counts = pd.Series(fit_values[inner_fit]).value_counts()
            oof_frequency[inner_apply] = (
                pd.Series(fit_values[inner_apply])
                .map(counts)
                .fillna(0)
                .to_numpy(dtype=np.float64)
                / len(inner_fit)
            )
        if not np.isfinite(oof_frequency).all():
            raise ValueError(f"{name} inner cross-fit is incomplete")
        fit_out[f"{name}_LOG_FREQUENCY"] = np.log1p(oof_frequency * len(fit_rows))
        counts = pd.Series(fit_values).value_counts()
        apply_count = (
            pd.Series(apply_values)
            .map(counts)
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )
        apply_out[f"{name}_LOG_FREQUENCY"] = np.log1p(apply_count)
        fit_out[f"{name}_TRAIN_FREQUENCY"] = oof_frequency
        apply_out[f"{name}_TRAIN_FREQUENCY"] = apply_count / len(fit_rows)
    return fit_out.reset_index(drop=True), apply_out.reset_index(drop=True)


def _append_block(
    x_fit: pd.DataFrame,
    x_apply: pd.DataFrame,
    feature_fit: pd.DataFrame,
    feature_apply: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_fit = feature_fit.reset_index(drop=True)
    feature_apply = feature_apply.reset_index(drop=True)
    if feature_fit.columns.tolist() != feature_apply.columns.tolist():
        raise ValueError("Added feature schemas do not match")
    if x_fit.columns.intersection(feature_fit.columns).size:
        raise ValueError("Feature names collide with the baseline matrix")
    return (
        pd.concat([x_fit.reset_index(drop=True), feature_fit], axis=1),
        pd.concat([x_apply.reset_index(drop=True), feature_apply], axis=1),
    )


def fit_weighted_predict(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_apply: pd.DataFrame,
    cat_columns: list[str],
    seed: int,
    scale_pos_weight: float,
) -> tuple[np.ndarray, float]:
    if not np.isfinite(scale_pos_weight) or scale_pos_weight <= 0:
        raise ValueError("scale_pos_weight must be finite and positive")
    model = phase21.make_model(seed)
    model.set_params(scale_pos_weight=float(scale_pos_weight))
    cat_indices = [x_fit.columns.get_loc(column) for column in cat_columns]
    started = time.perf_counter()
    model.fit(x_fit, y_fit, cat_features=cat_indices)
    prediction = model.predict_proba(x_apply)[:, 1].astype(np.float64)
    elapsed = time.perf_counter() - started
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("Weighted CatBoost did not execute on GPU")
    if not np.isfinite(prediction).all():
        raise ValueError("Weighted model produced nonfinite predictions")
    del model
    gc.collect()
    return prediction, elapsed


def _auc_row(
    name: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    fold_aucs = [
        float(roc_auc_score(y[fold_ids == fold], prediction[fold_ids == fold]))
        for fold in range(1, FOLDS + 1)
    ]
    ref_fold_aucs = [
        float(roc_auc_score(y[fold_ids == fold], reference[fold_ids == fold]))
        for fold in range(1, FOLDS + 1)
    ]
    auc = float(roc_auc_score(y, prediction))
    baseline_auc = float(roc_auc_score(y, reference))
    return {
        "candidate": name,
        "pooled_auc": auc,
        "delta_vs_internal_baseline": auc - baseline_auc,
        "fold_auc": json.dumps(fold_aucs),
        "fold_deltas": json.dumps(
            [current - base for current, base in zip(fold_aucs, ref_fold_aucs)]
        ),
        "positive_folds": int(
            sum(current > base for current, base in zip(fold_aucs, ref_fold_aucs))
        ),
    }


def _paired_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
    repetitions: int = 200,
) -> dict[str, Any]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        rows = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[rows], candidate[rows]) - roc_auc_score(
            y[rows], baseline[rows]
        )
    return {
        "repetitions": repetitions,
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "interpretation": "Conditional on these OOF vectors; does not capture model-training variability or prior selection.",
    }


def _data_profile(train: pd.DataFrame) -> dict[str, Any]:
    predictors = train.drop(columns=[TARGET, ID])
    missing_cols = [column for column in predictors if predictors[column].isna().any()]
    missing = predictors.loc[:, missing_cols].isna()
    pattern = missing.astype(np.uint8).astype(str).agg("".join, axis=1)
    grouped = (
        pd.DataFrame({"pattern": pattern, TARGET: train[TARGET]})
        .groupby("pattern")[TARGET]
        .agg(["size", "mean"])
        .sort_values("size", ascending=False)
    )
    ext_count = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(
        axis=1
    )
    target_rate_by_ext_count = (
        pd.DataFrame({"count": ext_count, TARGET: train[TARGET]})
        .groupby("count")[TARGET]
        .agg(["size", "mean"])
    )
    profile_hash = pd.util.hash_pandas_object(predictors, index=False)
    duplicates = int(profile_hash.duplicated(keep=False).sum())
    id_deciles = (
        pd.DataFrame(
            {
                "id_decile": pd.qcut(train[ID], 10, labels=False, duplicates="drop"),
                TARGET: train[TARGET],
            }
        )
        .groupby("id_decile")[TARGET]
        .agg(["size", "mean"])
    )
    return {
        "rows": int(len(train)),
        "predictors": int(predictors.shape[1]),
        "target_rate": float(train[TARGET].mean()),
        "missing_columns": {
            column: int(predictors[column].isna().sum()) for column in missing_cols
        },
        "missing_profile_count": int(len(grouped)),
        "top_missing_profiles": [
            {
                "pattern": str(index),
                "rows": int(row["size"]),
                "target_rate": float(row["mean"]),
            }
            for index, row in grouped.head(15).iterrows()
        ],
        "external_score_availability": [
            {
                "available": int(index),
                "rows": int(row["size"]),
                "target_rate": float(row["mean"]),
            }
            for index, row in target_rate_by_ext_count.iterrows()
        ],
        "exact_duplicate_predictor_rows": duplicates,
        "id_decile_target_rates": [
            {
                "id_decile": int(index) + 1,
                "rows": int(row["size"]),
                "target_rate": float(row["mean"]),
            }
            for index, row in id_deciles.iterrows()
        ],
        "interpretation": "Descriptive, post-hoc diagnostics on previously examined labels; not feature validation.",
    }


def run(
    train_path: Path,
    phase15_path: Path,
    output_dir: Path,
) -> None:
    if not train_path.is_file() or not phase15_path.is_file():
        raise FileNotFoundError("Training data or Phase 15 OOF reference is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(phase15_path)
    if not {ID, TARGET}.issubset(train.columns):
        raise ValueError("Train file must contain applicant ID and target")
    y = train[TARGET].to_numpy(dtype=np.int8)
    if len(reference) != len(train):
        raise ValueError("Phase 15 OOF row count differs from train")
    if not np.array_equal(train[ID].to_numpy(), reference[ID].to_numpy()):
        raise ValueError("Phase 15 OOF IDs do not align with train")
    if not np.array_equal(y, reference[TARGET].to_numpy(dtype=np.int8)):
        raise ValueError("Phase 15 OOF targets do not align with train")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y), dtype=np.int8), y))
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = fold
    if not np.array_equal(fold_ids, reference["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Phase 15 fold IDs differ from the preregistered split")

    profile = _data_profile(train)
    (output_dir / "PHASE23_TRAIN_ONLY_DATA_PROFILE.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase21.build_feature_blocks(raw)
    ext_all = external_disagreement_features(raw)
    baseline_oof = np.full(len(y), np.nan, dtype=np.float64)
    predictions: dict[str, np.ndarray] = {
        "application_process": np.full(len(y), np.nan),
        "external_disagreement": np.full(len(y), np.nan),
        "financial_residuals": np.full(len(y), np.nan),
        "applicant_consistency": np.full(len(y), np.nan),
        "all_four_feature_families": np.full(len(y), np.nan),
        "scale_pos_weight_2": np.full(len(y), np.nan),
        "scale_pos_weight_4": np.full(len(y), np.nan),
        "low_score_hard_routed_expert": np.full(len(y), np.nan),
        "low_score_fixed_50_50_expert": np.full(len(y), np.nan),
    }
    fit_log: list[dict[str, Any]] = []
    for fold, (fit, valid) in enumerate(folds, start=1):
        print(
            f"FOLD {fold}/{FOLDS} fit={len(fit)} valid={len(valid)}",
            flush=True,
        )
        x_fit, x_valid, cat_columns = phase21.phase15_matrix(blocks, fit, valid)
        seed = SEED + fold * 100
        base_prediction, elapsed = phase21.fit_predict(
            x_fit, y[fit], x_valid, cat_columns, seed
        )
        baseline_oof[valid] = base_prediction
        fit_log.append(
            {
                "candidate": "matched_baseline",
                "fold": fold,
                "fit_rows": len(fit),
                "fit_seconds": elapsed,
                "gpu": True,
            }
        )

        process_fit, process_valid = application_process_features(raw, fit, valid)
        ext_fit, ext_valid = ext_all.iloc[fit].reset_index(drop=True), ext_all.iloc[
            valid
        ].reset_index(drop=True)
        residual_fit, residual_valid = financial_residual_features(
            raw, fit, valid, seed + 11
        )
        consistency_fit, consistency_valid = applicant_consistency_features(
            raw, fit, valid, seed + 17
        )
        family_blocks = {
            "application_process": (process_fit, process_valid),
            "external_disagreement": (ext_fit, ext_valid),
            "financial_residuals": (residual_fit, residual_valid),
            "applicant_consistency": (consistency_fit, consistency_valid),
        }
        augmented_matrices: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for name, (new_fit, new_valid) in family_blocks.items():
            augmented_matrices[name] = _append_block(
                x_fit, x_valid, new_fit, new_valid
            )
        all_fit = pd.concat(
            [block[0].reset_index(drop=True) for block in family_blocks.values()],
            axis=1,
        )
        all_valid = pd.concat(
            [block[1].reset_index(drop=True) for block in family_blocks.values()],
            axis=1,
        )
        augmented_matrices["all_four_feature_families"] = _append_block(
            x_fit, x_valid, all_fit, all_valid
        )

        for name, (aug_fit, aug_valid) in augmented_matrices.items():
            prediction, fit_seconds = phase21.fit_predict(
                aug_fit,
                y[fit],
                aug_valid,
                cat_columns,
                seed,
            )
            predictions[name][valid] = prediction
            fit_log.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "feature_count": aug_fit.shape[1],
                    "fit_seconds": fit_seconds,
                    "gpu": True,
                }
            )

        for weight in (2.0, 4.0):
            prediction, fit_seconds = fit_weighted_predict(
                x_fit, y[fit], x_valid, cat_columns, seed, weight
            )
            name = f"scale_pos_weight_{int(weight)}"
            predictions[name][valid] = prediction
            fit_log.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "feature_count": x_fit.shape[1],
                    "scale_pos_weight": weight,
                    "fit_seconds": fit_seconds,
                    "gpu": True,
                }
            )

        inner_scores = np.full(len(fit), np.nan, dtype=np.float64)
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=seed + 31,
        )
        for inner_index, (inner_fit, inner_valid) in enumerate(
            inner_splitter.split(np.zeros(len(fit)), y[fit]), start=1
        ):
            prediction, fit_seconds = phase21.fit_predict(
                x_fit.iloc[inner_fit].reset_index(drop=True),
                y[fit[inner_fit]],
                x_fit.iloc[inner_valid].reset_index(drop=True),
                cat_columns,
                seed + 10000 + inner_index,
            )
            inner_scores[inner_valid] = prediction
            fit_log.append(
                {
                    "candidate": "low_score_gate_inner_oof",
                    "fold": fold,
                    "inner_fold": inner_index,
                    "fit_rows": len(inner_fit),
                    "fit_seconds": fit_seconds,
                    "gpu": True,
                }
            )
        if not np.isfinite(inner_scores).all():
            raise ValueError("Low-score gate inner OOF predictions are incomplete")
        threshold = float(np.quantile(inner_scores, 0.50))
        low_rows = np.flatnonzero(inner_scores <= threshold)
        if len(np.unique(y[fit[low_rows]])) != 2:
            raise ValueError("Low-score expert training subset lost a target class")
        expert_prediction, fit_seconds = phase21.fit_predict(
            x_fit.iloc[low_rows].reset_index(drop=True),
            y[fit[low_rows]],
            x_valid,
            cat_columns,
            seed + 50000,
        )
        valid_low = base_prediction <= threshold
        hard = base_prediction.copy()
        hard[valid_low] = expert_prediction[valid_low]
        fixed = base_prediction.copy()
        fixed[valid_low] = (
            0.5 * base_prediction[valid_low] + 0.5 * expert_prediction[valid_low]
        )
        predictions["low_score_hard_routed_expert"][valid] = hard
        predictions["low_score_fixed_50_50_expert"][valid] = fixed
        fit_log.append(
            {
                "candidate": "low_score_specialist",
                "fold": fold,
                "fit_rows": len(low_rows),
                "validation_routed_rows": int(valid_low.sum()),
                "score_threshold": threshold,
                "fit_seconds": fit_seconds,
                "gpu": True,
            }
        )
        print(
            f"FOLD {fold} baseline_auc={roc_auc_score(y[valid], base_prediction):.6f} "
            f"seconds={elapsed:.1f}",
            flush=True,
        )
        del x_fit, x_valid, augmented_matrices
        gc.collect()

    if not np.isfinite(baseline_oof).all():
        raise ValueError("Baseline OOF prediction coverage is incomplete")
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Candidate OOF prediction coverage is incomplete: {name}")

    candidate_predictions: dict[str, np.ndarray] = {
        "matched_baseline": baseline_oof,
        **predictions,
    }
    for family in FAMILY_NAMES:
        candidate_predictions[f"{family}_fixed_50_50_blend"] = (
            0.5 * baseline_oof + 0.5 * predictions[family]
        )
    result_rows = [
        _auc_row(name, prediction, baseline_oof, y, fold_ids)
        for name, prediction in candidate_predictions.items()
    ]
    results = pd.DataFrame(result_rows)
    bootstrap_rows = []
    for index, (name, prediction) in enumerate(candidate_predictions.items()):
        if name == "matched_baseline":
            continue
        bootstrap_rows.append(
            {
                "candidate": name,
                **_paired_bootstrap(y, baseline_oof, prediction, SEED + 1000 + index),
            }
        )
    pairwise_rows = []
    baseline_bands = pairwise22.score_deciles(baseline_oof)
    for name, prediction in candidate_predictions.items():
        if name == "matched_baseline":
            continue
        counts = pairwise22.pairwise_cell_counts(y, baseline_bands, baseline_oof)
        candidate_counts = pairwise22.pairwise_cell_counts(
            y, baseline_bands, prediction
        )
        for band in range(pairwise22.N_BANDS):
            pairs = int(
                np.sum((y == 1) & (baseline_bands == band))
                * np.sum(y == 0)
            )
            if pairs == 0:
                continue
            pairwise_rows.append(
                {
                    "candidate": name,
                    "baseline_positive_risk_decile": band + 1,
                    "positive_rows": int(np.sum((y == 1) & (baseline_bands == band))),
                    "all_negative_pairs": pairs,
                    "baseline_concordant_pairs": float(counts[band].sum()),
                    "candidate_concordant_pairs": float(
                        candidate_counts[band].sum()
                    ),
                    "net_pair_change": float(
                        candidate_counts[band].sum() - counts[band].sum()
                    ),
                    "delta_auc_contribution": float(
                        (candidate_counts[band].sum() - counts[band].sum())
                        / (int(np.sum(y == 1)) * int(np.sum(y == 0)))
                    ),
                }
            )
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "matched_baseline": baseline_oof,
            **candidate_predictions,
        }
    )
    oof.to_csv(output_dir / "PHASE23_OOF_PREDICTIONS.csv", index=False, lineterminator="\n")
    results.to_csv(output_dir / "PHASE23_AUC_RESULTS.csv", index=False, lineterminator="\n")
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "PHASE23_PAIRED_BOOTSTRAP.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "PHASE23_PAIRWISE_RISK_BANDS.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(fit_log).to_csv(
        output_dir / "PHASE23_FIT_LOG.csv", index=False, lineterminator="\n"
    )
    baseline_reference = reference[
        "native_catboost_phase14_plus_time"
    ].to_numpy(dtype=np.float64)
    summary = {
        "rows": len(train),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "fold_ids_match_phase15": True,
        "phase15_oof_auc": float(roc_auc_score(y, baseline_reference)),
        "matched_baseline_oof_auc": float(roc_auc_score(y, baseline_oof)),
        "matched_baseline_delta_vs_phase15": float(
            roc_auc_score(y, baseline_oof) - roc_auc_score(y, baseline_reference)
        ),
        "train_sha256": sha256_file(train_path),
        "phase15_oof_sha256": sha256_file(phase15_path),
        "fit_count": len(fit_log),
        "gpu_model_fit_count": int(sum(bool(row.get("gpu")) for row in fit_log)),
        "gpu_device": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "runtime_seconds": time.perf_counter() - started,
        "test_data_read": False,
        "external_data_read": False,
        "all_model_fits_gpu": bool(all(bool(row.get("gpu")) for row in fit_log)),
        "independent_confirmation": False,
        "feature_family_descriptions": {
            "application_process": "Raw missingness flags, fit-fold-vocabulary one-hot missingness profiles, and missingness-by-age/contract interactions.",
            "external_disagreement": "Availability-adjusted spread, standard deviation, median/range, and pairwise absolute differences among EXT_SOURCE_1..3.",
            "financial_residuals": "Cross-fitted, fold-local shrunk median residuals of log annuity and log goods price conditional on binned log credit and income.",
            "applicant_consistency": "Cross-fitted, train-fold-only frequencies of four predeclared categorical applicant-profile pairs.",
            "low_score_expert": "50th-percentile score gate derived from inner OOF CatBoost scores; specialist fit uses only gated outer-fit rows.",
        },
        "class_weight_candidates": [1.0, 2.0, 4.0],
        "interpretation": "All candidates exploratory on historically exposed labels; bootstrap intervals are conditional on saved OOF vectors and do not account for training variability or prior selection.",
    }
    (output_dir / "PHASE23_RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(output_dir, summary, results, bootstrap_rows, profile)
    _write_manifest(output_dir, [train_path, phase15_path, Path(__file__).resolve()])


def _write_report(
    output_dir: Path,
    summary: dict[str, Any],
    results: pd.DataFrame,
    bootstrap_rows: list[dict[str, Any]],
    profile: dict[str, Any],
) -> None:
    ordered = results.sort_values("pooled_auc", ascending=False)
    lines = [
        "# Phase 23 - targeted AUC feature research",
        "",
        "## Protocol and evidence boundary",
        "",
        "Train-only, five matched stratified folds (seed 20261113), CatBoost GPU, 700 iterations, depth 6, learning rate 0.05. Each feature family was evaluated separately, together, and as a fixed 50/50 OOF blend with the matched global model. Positive-class weights 2 and 4 and a cross-fitted low-score specialist were included as predeclared AUC-directed tests. No test or external data were read. Every label has prior exposure, so this is exploratory rather than independent confirmation.",
        "",
        f"- Rows: {summary['rows']:,}; positives: {summary['positives']:,} ({summary['positive_rate']:.3%}).",
        f"- Phase 15 reference AUC: {summary['phase15_oof_auc']:.9f}.",
        f"- Refit matched baseline AUC: {summary['matched_baseline_oof_auc']:.9f} (delta {summary['matched_baseline_delta_vs_phase15']:+.9f}).",
        f"- GPU fits: {summary['gpu_model_fit_count']}; runtime: {summary['runtime_seconds']:.1f} seconds.",
        "",
        "## OOF results",
        "",
        "| Candidate | Pooled AUC | Delta vs matched baseline | Positive folds |",
        "|---|---:|---:|---:|",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.pooled_auc:.9f} | "
            f"{row.delta_vs_internal_baseline:+.9f} | {row.positive_folds}/5 |"
        )
    lines.extend(
        [
            "",
            "Fixed 50/50 blends and class weights are exploratory diagnostics; no weight or family was promoted. See `PHASE23_AUC_RESULTS.csv` for fold AUC/deltas and `PHASE23_PAIRED_BOOTSTRAP.csv` for conditional row-stratified intervals.",
            "",
            "## Pairwise interpretation",
            "",
            "Candidate ranking changes are decomposed over all positive-negative pairs, and by baseline positive-risk decile, in `PHASE23_PAIRWISE_RISK_BANDS.csv`. A subgroup-only AUC is not a promotion criterion.",
            "",
            "## Train-only discovery profile",
            "",
            f"- Distinct missingness profiles: {profile['missing_profile_count']}.",
            f"- Exact duplicate predictor rows: {profile['exact_duplicate_predictor_rows']}.",
            "- Missingness and target-rate summaries are descriptive on previously examined labels; they do not establish incremental feature value.",
            "",
            "## Interpretation limits",
            "",
            "Financial residuals and applicant-pair frequencies are constructed without TARGET: their training-side values are inner-fold cross-fitted, and mappings are fit only on each outer training fold. The process-pattern vocabulary is fit on outer training rows. The low-score gate comes from inner OOF scores and is never selected on outer labels. Bootstrap intervals condition on the saved OOF predictions and do not account for model-training variation or accumulated research selection. Competition eligibility of inherited Phase 15 features remains unresolved.",
        ]
    )
    (output_dir / "PHASE23_EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_manifest(output_dir: Path, inputs: list[Path]) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "PHASE23_SHA256_MANIFEST.csv":
            rows.append(
                {
                    "kind": "output",
                    "path": str(path.relative_to(REPO)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in inputs:
        rows.append(
            {
                "kind": "source_or_input",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "PHASE23_SHA256_MANIFEST.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--phase15-oof", type=Path, default=PHASE15_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.train.resolve(), args.phase15_oof.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
