from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
PHASE3B = ROOT / "experiments" / "phase3b_deep_validation"
OUT = ROOT / "experiments" / "phase3c_model_intelligence"
PHASE3_PRODUCT_REFERENCE = (
    PHASE3
    / "batch1_family_ablation_02"
    / "BATCH1_FAMILY_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261103
CAT_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
]
PRODUCT_VARIANTS = {
    "raw_plus_product": "all_product",
    "raw_plus_nonredundant": "no_redundant",
}
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE3B_RUNNER = PHASE3B / "phase3b_ablation.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_module("phase3c_preserved_baseline", BASELINE_SOURCE)
phase3b = load_module("phase3c_phase3b_runner", PHASE3B_RUNNER)
shared = phase3b.shared


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mem_snapshot() -> dict[str, int | None]:
    if os.name != "nt":
        return {"working_set_bytes": None, "peak_working_set_bytes": None}
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return {"working_set_bytes": None, "peak_working_set_bytes": None}
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
    }


def validate_reference(
    reference: pd.DataFrame,
    ids: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    prediction_column: str,
    name: str,
) -> np.ndarray:
    if reference[ID].duplicated().any():
        raise ValueError(f"{name} has duplicated IDs")
    aligned = reference.set_index(ID).reindex(ids)
    if aligned[TARGET].isna().any() or not np.array_equal(
        aligned[TARGET].to_numpy(dtype=np.int8), y
    ):
        raise ValueError(f"{name} target alignment failed")
    if "fold" in aligned and not np.array_equal(aligned["fold"].to_numpy(), folds):
        raise ValueError(f"{name} fold alignment failed")
    if not np.isfinite(aligned[prediction_column].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} contains nonfinite predictions")
    return aligned[prediction_column].to_numpy(dtype=np.float64)


def summarize_oof(
    name: str, prediction: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> tuple[dict, list[dict]]:
    pooled = float(roc_auc_score(y, prediction))
    per_fold = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        per_fold.append(
            {
                "model_variant": name,
                "fold": int(fold),
                "n_validation": int(mask.sum()),
                "fold_auc": float(roc_auc_score(y[mask], prediction[mask])),
            }
        )
    fold_auc = np.array([row["fold_auc"] for row in per_fold])
    return (
        {
            "model_variant": name,
            "pooled_oof_auc": pooled,
            "mean_fold_auc": float(fold_auc.mean()),
            "std_fold_auc": float(fold_auc.std(ddof=1)),
        },
        per_fold,
    )


def model_feature_frames(raw_fit, raw_valid, variant: str, product_params=None):
    fit_base = baseline.build_rowwise_features(raw_fit)
    valid_base = baseline.build_rowwise_features(raw_valid)
    if variant == "raw":
        return fit_base, valid_base
    family = PRODUCT_VARIANTS[variant]
    fit_extra, valid_extra, _ = phase3b.make_family(
        raw_fit,
        raw_valid,
        feature_params=product_params,
        family_name=family,
    )
    return (
        pd.concat([fit_base, fit_extra.reset_index(drop=True)], axis=1),
        pd.concat([valid_base, valid_extra.reset_index(drop=True)], axis=1),
    )


def sklearn_transformer(train_frame: pd.DataFrame):
    categorical = [column for column in CAT_COLUMNS if column in train_frame]
    numeric = [column for column in train_frame.columns if column not in categorical]
    transformer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return transformer


def as_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def select_rounds(
    model_name: str,
    fit_frame: pd.DataFrame,
    stop_frame: pd.DataFrame,
    y_fit: np.ndarray,
    y_stop: np.ndarray,
    fold: int,
):
    started = time.perf_counter()
    if model_name == "lightgbm":
        x_fit, x_stop = baseline.make_lightgbm_frames(fit_frame, stop_frame)
        config = baseline.MODEL_CONFIG["lightgbm"]
        params = {
            key: value
            for key, value in config.items()
            if key not in ("device", "num_boost_round", "early_stopping_rounds")
        }
        selector = lgb.train(
            {**params, "seed": baseline.FOLD_SEED + fold},
            lgb.Dataset(x_fit, label=y_fit),
            num_boost_round=config["num_boost_round"],
            valid_sets=[lgb.Dataset(x_stop, label=y_stop)],
            callbacks=[
                lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        rounds = int(selector.best_iteration or config["num_boost_round"])
    elif model_name == "catboost":
        model = CatBoostClassifier(
            iterations=1800,
            depth=7,
            learning_rate=0.045,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=5.0,
            random_strength=0.5,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            border_count=128,
            task_type="CPU",
            random_seed=SEED + fold,
            allow_writing_files=False,
            verbose=False,
            thread_count=min(8, os.cpu_count() or 1),
        )
        model.fit(
            fit_frame,
            y_fit,
            cat_features=[c for c in CAT_COLUMNS if c in fit_frame],
            eval_set=(stop_frame, y_stop),
            early_stopping_rounds=120,
            verbose=False,
        )
        rounds = max(1, int(model.best_iteration_) + 1)
    elif model_name in ("xgboost", "hist_gradient_boosting"):
        transformer = sklearn_transformer(fit_frame)
        x_fit = transformer.fit_transform(fit_frame)
        x_stop = transformer.transform(stop_frame)
        if model_name == "xgboost":
            model = XGBClassifier(
                n_estimators=1400,
                max_depth=4,
                learning_rate=0.03,
                min_child_weight=40,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=min(8, os.cpu_count() or 1),
                random_state=SEED + fold,
                early_stopping_rounds=100,
            )
            model.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)], verbose=False)
            rounds = max(1, int(model.best_iteration) + 1)
        else:
            x_fit = as_dense(x_fit).astype(np.float32, copy=False)
            x_stop = as_dense(x_stop).astype(np.float32, copy=False)
            model = HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=5.0,
                early_stopping=False,
                random_state=SEED + fold,
            )
            model.fit(x_fit, y_fit)
            best_auc, rounds = -1.0, 100
            for iteration, probabilities in enumerate(model.staged_predict_proba(x_stop), 1):
                if iteration % 10 == 0 or iteration == 1:
                    auc = roc_auc_score(y_stop, probabilities[:, 1])
                    if auc > best_auc:
                        best_auc, rounds = auc, iteration
    elif model_name == "random_forest":
        rounds = 250
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return rounds, time.perf_counter() - started


def fit_predict(
    model_name: str,
    fit_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    y_fit: np.ndarray,
    rounds: int,
    fold: int,
):
    if model_name == "lightgbm":
        x_fit, x_valid = baseline.make_lightgbm_frames(fit_frame, valid_frame)
        config = baseline.MODEL_CONFIG["lightgbm"]
        params = {
            key: value
            for key, value in config.items()
            if key not in ("device", "num_boost_round", "early_stopping_rounds")
        }
        model = lgb.train(
            {**params, "seed": baseline.FOLD_SEED + fold},
            lgb.Dataset(x_fit, label=y_fit),
            num_boost_round=rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        return model.predict(x_valid, num_iteration=rounds)
    if model_name == "catboost":
        model = CatBoostClassifier(
            iterations=rounds,
            depth=7,
            learning_rate=0.045,
            loss_function="Logloss",
            eval_metric="AUC",
            l2_leaf_reg=5.0,
            random_strength=0.5,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            border_count=128,
            task_type="CPU",
            random_seed=SEED + fold,
            allow_writing_files=False,
            verbose=False,
            thread_count=min(8, os.cpu_count() or 1),
        )
        model.fit(
            fit_frame,
            y_fit,
            cat_features=[c for c in CAT_COLUMNS if c in fit_frame],
            verbose=False,
        )
        return model.predict_proba(valid_frame)[:, 1]
    transformer = sklearn_transformer(fit_frame)
    x_fit = transformer.fit_transform(fit_frame)
    x_valid = transformer.transform(valid_frame)
    if model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=rounds,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=min(8, os.cpu_count() or 1),
            random_state=SEED + fold,
        )
        model.fit(x_fit, y_fit, verbose=False)
        return model.predict_proba(x_valid)[:, 1]
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=rounds,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=SEED + fold,
        )
        model.fit(as_dense(x_fit).astype(np.float32, copy=False), y_fit)
        return model.predict_proba(as_dense(x_valid).astype(np.float32, copy=False))[:, 1]
    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=rounds,
            max_features="sqrt",
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=min(8, os.cpu_count() or 1),
            random_state=SEED + fold,
        )
        model.fit(x_fit, y_fit)
        return model.predict_proba(x_valid)[:, 1]
    raise ValueError(f"Unknown model: {model_name}")


def metric_tables(oof_predictions, y, folds):
    summary_rows, fold_rows = [], []
    for name, prediction in oof_predictions.items():
        summary, fold_metrics = summarize_oof(name, prediction, y, folds)
        summary_rows.append(summary)
        fold_rows.extend(fold_metrics)
    summary = pd.DataFrame(summary_rows)
    summary[["model", "feature_variant"]] = summary["model_variant"].str.split(
        "__", n=1, expand=True
    )
    is_reference = summary["model"] == "reference"
    baseline_by_model = summary.loc[
        summary.feature_variant == "raw", ["model", "pooled_oof_auc"]
    ].set_index("model")["pooled_oof_auc"]
    summary["delta_vs_same_model_raw"] = np.nan
    for index, row in summary.loc[~is_reference].iterrows():
        summary.loc[index, "delta_vs_same_model_raw"] = (
            row.pooled_oof_auc - baseline_by_model[row.model]
        )
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame[["model", "feature_variant"]] = fold_frame["model_variant"].str.split(
        "__", n=1, expand=True
    )
    raw_auc = fold_frame.loc[
        fold_frame.feature_variant == "raw",
        ["model", "fold", "fold_auc"],
    ].rename(columns={"fold_auc": "raw_fold_auc"})
    fold_frame = fold_frame.merge(raw_auc, on=["model", "fold"], how="left")
    fold_frame["delta_vs_same_model_raw"] = fold_frame["fold_auc"] - fold_frame["raw_fold_auc"]
    return summary, fold_frame


def make_error_outputs(train, folds, oof_predictions):
    out = train[[ID, TARGET]].copy()
    out["fold"] = folds
    selected_raw = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    for column in selected_raw:
        if column in train:
            out[column] = train[column].to_numpy()
    out["RAW_MISSING_COUNT"] = train.drop(columns=[ID, TARGET]).isna().sum(axis=1).to_numpy()
    prediction_details = {}
    for name, prediction in oof_predictions.items():
        prediction_details[f"PRED_{name}"] = prediction
        prediction_details[f"RANK_PCT_{name}"] = (
            pd.Series(prediction)
            .groupby(folds, sort=False)
            .rank(pct=True, method="average")
            .to_numpy()
        )
        prediction_details[f"ABS_PROB_ERROR_{name}"] = np.abs(
            train[TARGET].to_numpy() - prediction
        )
    out = pd.concat([out, pd.DataFrame(prediction_details)], axis=1)
    product_oof = pd.read_csv(
        PHASE3B / "PHASE3B_PRODUCT_FEATURES_OOF_DIAGNOSTIC.csv"
    )
    if len(product_oof) != len(train):
        raise ValueError("Phase 3B product diagnostic row count is not aligned.")
    out = pd.concat(
        [
            out,
            product_oof.loc[:, phase3b.ALL_PRODUCT].reset_index(drop=True),
        ],
        axis=1,
    )
    for name, prediction in oof_predictions.items():
        rank_pct = out[f"RANK_PCT_{name}"].to_numpy()
        out[f"TOP_5PCT_{name}"] = rank_pct >= 0.95
        out[f"BOTTOM_HALF_POSITIVE_{name}"] = (
            (out[TARGET].to_numpy() == 1) & (rank_pct <= 0.5)
        )
    out.to_csv(OUT / "OOF_ERROR_ANALYSIS.csv", index=False)

    reference_names = [
        name
        for name in oof_predictions
        if name in (
            "lightgbm__raw_plus_product",
            "catboost__raw_plus_product",
            "xgboost__raw_plus_product",
        )
    ]
    disagreement = out[[ID, TARGET, "fold"]].copy()
    rank_cols = []
    for name in reference_names:
        rank_name = f"RANK_PCT_{name}"
        disagreement[rank_name] = out[rank_name].to_numpy()
        rank_cols.append(rank_name)
    if len(reference_names) >= 2:
        disagreement["rank_pct_std"] = disagreement[rank_cols].std(axis=1)
        disagreement["rank_pct_range"] = (
            disagreement[rank_cols].max(axis=1) - disagreement[rank_cols].min(axis=1)
        )
        disagreement["model_probability_std"] = out[
            [f"PRED_{name}" for name in reference_names]
        ].std(axis=1)
        disagreement["high_disagreement"] = (
            disagreement["rank_pct_range"] >= disagreement["rank_pct_range"].quantile(0.95)
        )
    disagreement.to_csv(OUT / "MODEL_DISAGREEMENT_ANALYSIS.csv", index=False)

    ranking_rows = []
    for name, prediction in oof_predictions.items():
        rank_pct = (
            pd.Series(prediction)
            .groupby(folds, sort=False)
            .rank(pct=True, method="average")
            .to_numpy()
        )
        for fold in sorted(np.unique(folds)):
            fold_mask = folds == fold
            bins = pd.qcut(
                rank_pct[fold_mask],
                q=10,
                labels=False,
                duplicates="drop",
            )
            for bin_id in sorted(pd.Series(bins).dropna().unique()):
                mask = np.zeros(len(train), dtype=bool)
                mask[np.flatnonzero(fold_mask)] = bins == bin_id
                ranking_rows.append(
                    {
                        "model_variant": name,
                        "fold": int(fold),
                        "rank_decile_low_to_high": int(bin_id) + 1,
                        "n": int(mask.sum()),
                        "defaults": int(train[TARGET].to_numpy()[mask].sum()),
                        "default_rate": float(train[TARGET].to_numpy()[mask].mean()),
                        "mean_predicted_probability": float(prediction[mask].mean()),
                        "mean_within_fold_rank_percentile": float(rank_pct[mask].mean()),
                    }
                )
    pd.DataFrame(ranking_rows).to_csv(OUT / "RANKING_FAILURE_ANALYSIS.csv", index=False)

    diagnostic_rows = []
    y = train[TARGET].to_numpy()
    for name, prediction in oof_predictions.items():
        rank_pct = (
            pd.Series(prediction)
            .groupby(folds, sort=False)
            .rank(pct=True, method="average")
            .to_numpy()
        )
        diagnostic_rows.append(
            {
                "model_variant": name,
                "pooled_oof_auc": roc_auc_score(y, prediction),
                "brier_score": float(np.mean((prediction - y) ** 2)),
                "mean_probability_positive": float(prediction[y == 1].mean()),
                "mean_probability_negative": float(prediction[y == 0].mean()),
                "positive_in_bottom_rank_half_pct": float(
                    np.mean(rank_pct[y == 1] <= 0.5) * 100
                ),
                "positive_in_bottom_rank_decile_pct": float(
                    np.mean(rank_pct[y == 1] <= 0.1) * 100
                ),
                "negative_in_top_rank_5pct_pct": float(
                    np.mean(rank_pct[y == 0] >= 0.95) * 100
                ),
                "top_rank_5pct_default_rate": float(y[rank_pct >= 0.95].mean()),
                "top_rank_1pct_default_rate": float(y[rank_pct >= 0.99].mean()),
                "threshold_0_5_diagnostic_only_fp": int(
                    np.sum((prediction >= 0.5) & (y == 0))
                ),
                "threshold_0_5_diagnostic_only_fn": int(
                    np.sum((prediction < 0.5) & (y == 1))
                ),
            }
        )
    pd.DataFrame(diagnostic_rows).to_csv(OUT / "OOF_ERROR_SUMMARY.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    train_path = DATA / "train.csv"
    train = pd.read_csv(train_path)
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    fold_frame = pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv")
    folds = fold_frame.set_index(ID).loc[ids, "fold"].to_numpy(dtype=np.int8)
    if len(np.unique(folds)) != 5:
        raise ValueError("Expected the preserved five fold assignments.")
    reference_baseline = pd.read_csv(PHASE2 / "OOF_BASELINE_PREDICTIONS.csv")
    baseline_pred = validate_reference(
        reference_baseline,
        ids,
        y,
        folds,
        "OOF_RAW_BASELINE",
        "corrected_baseline",
    )
    phase3_pred = validate_reference(
        pd.read_csv(PHASE3_PRODUCT_REFERENCE),
        ids,
        y,
        folds,
        "OOF_product_decoder_only",
        "phase3_product_discovery",
    )
    phase3b_oof = pd.read_csv(PHASE3B / "PHASE3B_ABLATION_OOF.csv")
    phase3b_pred = validate_reference(
        phase3b_oof,
        ids,
        y,
        folds,
        "OOF_all_product",
        "phase3b_controlled_product",
    )
    oof_predictions: dict[str, np.ndarray] = {
        "reference__corrected_baseline": baseline_pred,
        "reference__phase3_discovery": phase3_pred,
        "reference__phase3b_controlled_product": phase3b_pred,
    }
    rounds_by_fold: dict[tuple[str, int], int] = {}
    runtime_rows: list[dict] = []
    fold_runtime_rows: list[dict] = []
    model_names = [
        "lightgbm",
        "catboost",
        "xgboost",
        "hist_gradient_boosting",
        "random_forest",
    ]
    variants = ["raw", "raw_plus_product", "raw_plus_nonredundant"]

    for fold in sorted(np.unique(folds)):
        valid_mask = folds == fold
        fit_idx = np.flatnonzero(~valid_mask)
        valid_idx = np.flatnonzero(valid_mask)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=SEED + int(fold),
        )
        inner_fit_local, inner_stop_local = next(
            inner.split(np.zeros(len(fit_idx)), y[fit_idx])
        )
        inner_fit_idx = fit_idx[inner_fit_local]
        inner_stop_idx = fit_idx[inner_stop_local]
        inner_fit_frame = baseline.build_rowwise_features(raw.iloc[inner_fit_idx])
        inner_stop_frame = baseline.build_rowwise_features(raw.iloc[inner_stop_idx])
        for model_name in model_names:
            round_count, selection_seconds = select_rounds(
                model_name,
                inner_fit_frame,
                inner_stop_frame,
                y[inner_fit_idx],
                y[inner_stop_idx],
                int(fold),
            )
            rounds_by_fold[(model_name, int(fold))] = round_count
            runtime_rows.append(
                {
                    "model": model_name,
                    "fold": int(fold),
                    "stage": "inner_round_selection",
                    "feature_variant": "raw",
                    "elapsed_seconds": selection_seconds,
                    "selected_rounds_or_trees": round_count,
                    **mem_snapshot(),
                }
            )
        product_params = shared.fit_fee_and_step(raw.iloc[fit_idx])
        for model_name in model_names:
            for variant in variants:
                fit_frame, valid_frame = model_feature_frames(
                    raw.iloc[fit_idx],
                    raw.iloc[valid_idx],
                    variant,
                    product_params=product_params,
                )
                started_fit = time.perf_counter()
                before_mem = mem_snapshot()
                prediction = fit_predict(
                    model_name,
                    fit_frame,
                    valid_frame,
                    y[fit_idx],
                    rounds_by_fold[(model_name, int(fold))],
                    int(fold),
                )
                elapsed = time.perf_counter() - started_fit
                if not np.isfinite(prediction).all():
                    raise ValueError(
                        f"Nonfinite prediction for {model_name}/{variant}/fold {fold}"
                    )
                key = f"{model_name}__{variant}"
                if key not in oof_predictions:
                    oof_predictions[key] = np.full(len(train), np.nan)
                oof_predictions[key][valid_idx] = prediction
                fold_auc = roc_auc_score(y[valid_idx], prediction)
                fold_runtime_rows.append(
                    {
                        "model_variant": key,
                        "model": model_name,
                        "stage": "outer_fold_fit_predict",
                        "feature_variant": variant,
                        "fold": int(fold),
                        "n_train": len(fit_idx),
                        "n_valid": len(valid_idx),
                        "fold_auc": fold_auc,
                        "rounds_or_trees": rounds_by_fold[(model_name, int(fold))],
                        "fit_predict_seconds": elapsed,
                        "memory_before_working_set_bytes": before_mem["working_set_bytes"],
                        **mem_snapshot(),
                    }
                )
                print(
                    f"fold={fold} model={model_name} variant={variant} "
                    f"auc={fold_auc:.6f} seconds={elapsed:.1f}",
                    flush=True,
                )

    for name, prediction in oof_predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"Incomplete OOF prediction vector: {name}")
    summary, fold_summary = metric_tables(oof_predictions, y, folds)
    summary.to_csv(OUT / "MODEL_BENCHMARK_RESULTS.csv", index=False)
    fold_summary.to_csv(OUT / "MODEL_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(runtime_rows + fold_runtime_rows).to_csv(
        OUT / "MODEL_RUNTIME_RESULTS.csv", index=False
    )
    prediction_frame = pd.DataFrame({ID: ids, TARGET: y, "fold": folds})
    for name, prediction in oof_predictions.items():
        prediction_frame[name] = prediction
    prediction_frame.to_csv(OUT / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv", index=False)
    make_error_outputs(train, folds, oof_predictions)

    report_rows = []
    for name, prediction in oof_predictions.items():
        row, _ = summarize_oof(name, prediction, y, folds)
        row["reference_type"] = (
            "historical_reference"
            if name.startswith("reference__")
            else "new_fixed_capacity_discovery"
        )
        report_rows.append(row)
    report_table = pd.DataFrame(report_rows)
    report_table.to_csv(OUT / "PHASE3C_SCORE_REPRODUCTION.csv", index=False)
    finished = datetime.now(timezone.utc)
    log = [
        f"started_utc={started.isoformat()}",
        f"finished_utc={finished.isoformat()}",
        f"runtime_seconds={time.perf_counter() - start_clock:.3f}",
        f"python={sys.version.split()[0]}",
        f"platform={platform.platform()}",
        f"pandas={pd.__version__}",
        f"numpy={np.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"lightgbm={lgb.__version__}",
        f"catboost={CatBoostClassifier.__module__}",
        f"xgboost={__import__('xgboost').__version__}",
        f"train_sha256={sha256(train_path)}",
        f"baseline_source_sha256={sha256(BASELINE_SOURCE)}",
        f"phase3b_runner_sha256={sha256(PHASE3B_RUNNER)}",
        f"train_shape={train.shape}",
        f"fold_count={len(np.unique(folds))}",
        "outer_validation_used_for_tuning_or_early_stopping=False",
        "test_set_used=False",
        "status=SUCCESS",
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\nscore_summary")
    print(report_table.to_string(index=False))
    print("\n" + "\n".join(log), flush=True)


if __name__ == "__main__":
    main()
