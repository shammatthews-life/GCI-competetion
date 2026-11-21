from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
TEST_DEFAULT = Path(r"D:\gci\inpuy\test.csv")
SAMPLE_DEFAULT = Path(r"D:\gci\inpuy\sample_submission.csv")
SEED = 20261113
FOLDS = 5
INNER_VALIDATION_SIZE = 0.12
INNER_STOPPING_ROUNDS = 120
MIN_ITERATIONS = 100
MAX_ITERATIONS = 3000
ID = "SK_ID_CURR"
TARGET = "TARGET"
MODEL_NAMES = ("catboost_onehot", "lightgbm_onehot")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _safe_log(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0)
    result[valid] = np.log1p(values[valid])
    return result


def make_rowwise_features(
    raw: pd.DataFrame,
    missing_indicator_columns: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ID not in raw.columns:
        raise ValueError(f"Input frame must include {ID}")
    if TARGET in raw.columns:
        raise ValueError("TARGET must not be present in prediction features")
    predictors = raw.drop(columns=[ID]).copy()
    numeric_columns = [
        column
        for column in predictors.columns
        if pd.api.types.is_numeric_dtype(predictors[column].dtype)
    ]
    categorical_columns = [
        column for column in predictors.columns if column not in numeric_columns
    ]

    numeric = predictors[numeric_columns].apply(pd.to_numeric, errors="raise")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    categorical = predictors[categorical_columns].copy()
    for column in categorical_columns:
        categorical[column] = (
            categorical[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    income = _numeric(predictors, "AMT_INCOME_TOTAL")
    credit = _numeric(predictors, "AMT_CREDIT")
    annuity = _numeric(predictors, "AMT_ANNUITY")
    goods = _numeric(predictors, "AMT_GOODS_PRICE")
    family = _numeric(predictors, "CNT_FAM_MEMBERS")
    age_days = np.abs(_numeric(predictors, "DAYS_BIRTH"))
    employed_days = np.abs(_numeric(predictors, "DAYS_EMPLOYED"))
    employed_days[employed_days == 365243.0] = np.nan

    ext = predictors[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].to_numpy(dtype=np.float64)
    ext_observed = np.isfinite(ext)
    ext_count = ext_observed.sum(axis=1)
    ext_mean = np.divide(
        np.nansum(ext, axis=1),
        ext_count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_min = np.where(ext_observed, ext, np.inf).min(axis=1)
    ext_max = np.where(ext_observed, ext, -np.inf).max(axis=1)
    ext_min[ext_count == 0] = np.nan
    ext_max[ext_count == 0] = np.nan
    ext_range = ext_max - ext_min
    ext_range[ext_count < 2] = np.nan
    ext_std = np.sqrt(
        np.divide(
            np.where(ext_observed, (ext - ext_mean[:, None]) ** 2, 0).sum(axis=1),
            ext_count,
            out=np.full(len(ext), np.nan, dtype=np.float64),
            where=ext_count > 0,
        )
    )
    missing_count = predictors.isna().sum(axis=1).to_numpy(dtype=np.float64)

    extra: dict[str, np.ndarray] = {
        "SCRATCH_AGE_YEARS": age_days / 365.25,
        "SCRATCH_EMPLOYMENT_YEARS": employed_days / 365.25,
        "SCRATCH_EMPLOYED_AGE_RATIO": _ratio(employed_days, age_days),
        "SCRATCH_AGE_AT_REGISTRATION_DAYS": age_days
        - np.abs(_numeric(predictors, "DAYS_REGISTRATION")),
        "SCRATCH_AGE_AT_ID_PUBLISH_DAYS": age_days
        - np.abs(_numeric(predictors, "DAYS_ID_PUBLISH")),
        "SCRATCH_AGE_AT_PHONE_CHANGE_DAYS": age_days
        - np.abs(_numeric(predictors, "DAYS_LAST_PHONE_CHANGE")),
        "SCRATCH_CREDIT_TO_INCOME": _ratio(credit, income),
        "SCRATCH_ANNUITY_TO_INCOME": _ratio(annuity, income),
        "SCRATCH_ANNUITY_TO_CREDIT": _ratio(annuity, credit),
        "SCRATCH_CREDIT_TO_GOODS": _ratio(credit, goods),
        "SCRATCH_GOODS_TO_INCOME": _ratio(goods, income),
        "SCRATCH_INCOME_PER_FAMILY_MEMBER": _ratio(income, family),
        "SCRATCH_ANNUITY_PER_FAMILY_MEMBER": _ratio(annuity, family),
        "SCRATCH_CREDIT_PER_FAMILY_MEMBER": _ratio(credit, family),
        "SCRATCH_LOG_INCOME": _safe_log(income),
        "SCRATCH_LOG_CREDIT": _safe_log(credit),
        "SCRATCH_LOG_ANNUITY": _safe_log(annuity),
        "SCRATCH_LOG_GOODS": _safe_log(goods),
        "SCRATCH_EXT_AVAILABLE_COUNT": ext_count.astype(np.float64),
        "SCRATCH_EXT_MEAN": ext_mean,
        "SCRATCH_EXT_MIN": ext_min,
        "SCRATCH_EXT_MAX": ext_max,
        "SCRATCH_EXT_RANGE": ext_range,
        "SCRATCH_EXT_STD": ext_std,
        "SCRATCH_EXT_DIFF_12": np.abs(ext[:, 0] - ext[:, 1]),
        "SCRATCH_EXT_DIFF_13": np.abs(ext[:, 0] - ext[:, 2]),
        "SCRATCH_EXT_DIFF_23": np.abs(ext[:, 1] - ext[:, 2]),
        "SCRATCH_RAW_MISSING_COUNT": missing_count,
        "SCRATCH_EXT_MISSING_COUNT": (3 - ext_count).astype(np.float64),
        "SCRATCH_EXT_RANGE_X_CREDIT_TO_INCOME": ext_range
        * _ratio(credit, income),
        "SCRATCH_EXT_MEAN_X_CREDIT_TO_INCOME": ext_mean
        * _ratio(credit, income),
        "SCRATCH_EXT_MEAN_X_ANNUITY_TO_INCOME": ext_mean
        * _ratio(annuity, income),
        "SCRATCH_EXT_MISSING_X_CREDIT_TO_INCOME": (3 - ext_count)
        * _ratio(credit, income),
    }

    if missing_indicator_columns is None:
        missing_indicator_columns = tuple(
            column for column in predictors if predictors[column].isna().any()
        )
    missing_not_in_schema = set(missing_indicator_columns).difference(
        predictors.columns
    )
    if missing_not_in_schema:
        raise ValueError(
            "Missingness indicator columns are absent from input: "
            f"{sorted(missing_not_in_schema)}"
        )
    for column in missing_indicator_columns:
        extra[f"SCRATCH_MISSING_{column}"] = (
            predictors[column].isna().to_numpy(dtype=np.float32)
        )

    engineered = pd.DataFrame(extra).replace([np.inf, -np.inf], np.nan)
    numeric = pd.concat(
        [numeric.reset_index(drop=True), engineered.astype(np.float32)],
        axis=1,
    )
    if numeric.columns.has_duplicates:
        raise ValueError("Duplicate numeric feature names")
    return numeric, categorical.reset_index(drop=True)


def make_fold_matrices(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    test_numeric, test_categorical = test_frame
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    fit_cat = encoder.fit_transform(categorical.iloc[fit_rows])
    valid_cat = encoder.transform(categorical.iloc[valid_rows])
    test_cat = encoder.transform(test_categorical)
    fit = np.concatenate(
        [numeric.iloc[fit_rows].to_numpy(dtype=np.float32), fit_cat], axis=1
    )
    valid = np.concatenate(
        [numeric.iloc[valid_rows].to_numpy(dtype=np.float32), valid_cat], axis=1
    )
    test = np.concatenate(
        [test_numeric.to_numpy(dtype=np.float32), test_cat], axis=1
    )
    if not np.isfinite(fit[~np.isnan(fit)]).all():
        raise ValueError("Unexpected infinite values in training feature matrix")
    return fit, valid, test, {
        "numeric_feature_count": int(numeric.shape[1]),
        "categorical_source_count": int(categorical.shape[1]),
        "onehot_feature_count": int(fit_cat.shape[1]),
        "total_feature_count": int(fit.shape[1]),
    }


def _inner_split(
    fit_rows: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=INNER_VALIDATION_SIZE,
        random_state=seed,
    )
    inner_train, inner_valid = next(
        splitter.split(np.zeros(len(fit_rows)), y[fit_rows])
    )
    return fit_rows[inner_train], fit_rows[inner_valid]


def _catboost_predictions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    x_inner_valid: np.ndarray,
    y_inner_valid: np.ndarray,
    x_outer_valid: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    selector = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        one_hot_max_size=2,
        max_ctr_complexity=1,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    started = time.perf_counter()
    selector.fit(
        x_inner_train,
        y_inner_train,
        eval_set=(x_inner_valid, y_inner_valid),
        early_stopping_rounds=INNER_STOPPING_ROUNDS,
        use_best_model=True,
    )
    rounds = max(MIN_ITERATIONS, int(selector.get_best_iteration()) + 1)
    selector_seconds = time.perf_counter() - started
    del selector

    model = CatBoostClassifier(
        iterations=rounds,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        one_hot_max_size=2,
        max_ctr_complexity=1,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    model.fit(x_fit, y_fit)
    outer = model.predict_proba(x_outer_valid)[:, 1].astype(np.float64)
    test = model.predict_proba(x_test)[:, 1].astype(np.float64)
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("CatBoost did not execute on GPU")
    return outer, test, rounds, selector_seconds


def _lightgbm_predictions(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_inner_train: np.ndarray,
    y_inner_train: np.ndarray,
    x_inner_valid: np.ndarray,
    y_inner_valid: np.ndarray,
    x_outer_valid: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    params: dict[str, Any] = {
        "objective": "binary",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "max_bin": 127,
        "n_jobs": 8,
        "verbosity": -1,
        "random_state": seed,
        "deterministic": True,
        "force_col_wise": True,
    }
    selector = lgb.LGBMClassifier(
        **params,
        n_estimators=MAX_ITERATIONS,
    )
    started = time.perf_counter()
    selector.fit(
        x_inner_train,
        y_inner_train,
        eval_set=[(x_inner_valid, y_inner_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                INNER_STOPPING_ROUNDS, first_metric_only=True, verbose=False
            ),
            lgb.log_evaluation(0),
        ],
    )
    rounds = max(MIN_ITERATIONS, int(selector.best_iteration_))
    selector_seconds = time.perf_counter() - started
    del selector
    model = lgb.LGBMClassifier(**params, n_estimators=rounds)
    model.fit(x_fit, y_fit)
    outer = model.predict_proba(x_outer_valid)[:, 1].astype(np.float64)
    test = model.predict_proba(x_test)[:, 1].astype(np.float64)
    return outer, test, rounds, selector_seconds


def run(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if TARGET not in train or ID not in train:
        raise ValueError("Train data must contain ID and TARGET")
    if TARGET in test or ID not in test:
        raise ValueError("Test data must contain ID and must not contain TARGET")
    if test.columns.tolist() != [
        column for column in train.columns if column != TARGET
    ]:
        raise ValueError("Train predictor and test schemas or column order differ")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("Sample submission columns must be exactly ID,TARGET")
    if not test[ID].is_unique or not train[ID].is_unique:
        raise ValueError("Input IDs must be unique")
    if not test[ID].reset_index(drop=True).equals(
        sample[ID].reset_index(drop=True)
    ):
        raise ValueError("Test IDs do not exactly match sample submission order")

    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, [0, 1]).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must be finite binary labels containing both classes")
    y = y_float.astype(np.int8)

    train_raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    missing_indicator_columns = tuple(
        column for column in train_raw if train_raw[column].isna().any()
    )
    train_numeric, train_categorical = make_rowwise_features(
        train_raw, missing_indicator_columns
    )
    test_numeric, test_categorical = make_rowwise_features(
        test_raw, missing_indicator_columns
    )
    # Feature engineering is rowwise; no train/test statistics are pooled.
    # The category encoder below is fit independently inside each training fold.

    categories = {
        column: int(train_categorical[column].nunique(dropna=False))
        for column in train_categorical.columns
    }
    outer = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(outer.split(np.zeros(len(y)), y))
    oof = {
        name: np.full(len(y), np.nan, dtype=np.float32)
        for name in MODEL_NAMES
    }
    test_fold_predictions = {
        name: [] for name in MODEL_NAMES
    }
    fold_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner_train_rows, inner_valid_rows = _inner_split(
            fit_rows, y, fold_seed
        )
        x_fit, x_valid, x_test, feature_info = make_fold_matrices(
            train_numeric,
            train_categorical,
            fit_rows,
            valid_rows,
            (test_numeric, test_categorical),
        )
        # Build the inner validation matrix with precisely the same vocabulary
        # fitted on the inner-training partition.
        inner_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        inner_cat_train = inner_encoder.fit_transform(
            train_categorical.iloc[inner_train_rows]
        )
        inner_cat_valid = inner_encoder.transform(
            train_categorical.iloc[inner_valid_rows]
        )
        x_inner_train = np.concatenate(
            [
                train_numeric.iloc[inner_train_rows].to_numpy(dtype=np.float32),
                inner_cat_train,
            ],
            axis=1,
        )
        x_inner_valid = np.concatenate(
            [
                train_numeric.iloc[inner_valid_rows].to_numpy(dtype=np.float32),
                inner_cat_valid,
            ],
            axis=1,
        )
        if x_inner_train.shape[1] != x_inner_valid.shape[1]:
            raise ValueError("Inner train/validation feature widths differ")

        print(
            f"FOLD {fold_number}/{FOLDS} features={feature_info['total_feature_count']} "
            f"fit={len(fit_rows)} valid={len(valid_rows)}",
            flush=True,
        )
        for model_name, fit_function in (
            ("catboost_onehot", _catboost_predictions),
            ("lightgbm_onehot", _lightgbm_predictions),
        ):
            prediction, test_prediction, rounds, selection_seconds = fit_function(
                x_fit,
                y[fit_rows],
                x_inner_train,
                y[inner_train_rows],
                x_inner_valid,
                y[inner_valid_rows],
                x_valid,
                x_test,
                fold_seed,
            )
            if not np.isfinite(prediction).all() or not np.isfinite(
                test_prediction
            ).all():
                raise ValueError(f"{model_name} produced non-finite scores")
            oof[model_name][valid_rows] = prediction.astype(np.float32)
            test_fold_predictions[model_name].append(test_prediction)
            fold_auc = float(roc_auc_score(y[valid_rows], prediction))
            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "auc": fold_auc,
                    "best_iterations": rounds,
                    "inner_selection_seconds": selection_seconds,
                    **feature_info,
                }
            )
            print(
                f"  {model_name}: AUC={fold_auc:.6f}, rounds={rounds}",
                flush=True,
            )
        del x_fit, x_valid, x_test, x_inner_train, x_inner_valid

    for model_name, prediction in oof.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF prediction vector incomplete: {model_name}")

    test_predictions = {
        name: np.mean(np.vstack(fold_preds), axis=0)
        for name, fold_preds in test_fold_predictions.items()
    }
    # Fixed equal-weight ensemble is predeclared; no weights are optimized on
    # the outer-fold labels.
    oof["equal_blend"] = (
        0.5 * oof["catboost_onehot"] + 0.5 * oof["lightgbm_onehot"]
    ).astype(np.float32)
    test_predictions["equal_blend"] = (
        0.5 * test_predictions["catboost_onehot"]
        + 0.5 * test_predictions["lightgbm_onehot"]
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    fold_frame = pd.DataFrame(fold_records)
    fold_frame.to_csv(output_dir / "PHASE29_FOLD_RESULTS.csv", index=False)
    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "outer_fold": np.full(len(y), -1, dtype=np.int8),
            **{f"oof_{name}": pred for name, pred in oof.items()},
        }
    )
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        oof_frame.loc[valid_rows, "outer_fold"] = fold_number
    oof_path = output_dir / "PHASE29_OOF_PREDICTIONS.csv"
    oof_frame.to_csv(oof_path, index=False)

    summary_rows = []
    for name, prediction in oof.items():
        fold_auc = fold_frame.loc[fold_frame.model == name, "auc"].to_numpy()
        summary_rows.append(
            {
                "model": name,
                "pooled_auc": float(roc_auc_score(y, prediction)),
                "mean_fold_auc": float(fold_auc.mean()) if len(fold_auc) else np.nan,
                "std_fold_auc": (
                    float(fold_auc.std(ddof=0)) if len(fold_auc) else np.nan
                ),
                "pooled_log_loss": float(log_loss(y, prediction)),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "PHASE29_SUMMARY.csv", index=False)

    submission_paths: dict[str, str] = {}
    for name, prediction in test_predictions.items():
        if not np.isfinite(prediction).all() or (
            (prediction < 0) | (prediction > 1)
        ).any():
            raise ValueError(f"Invalid test probabilities: {name}")
        submission = pd.DataFrame(
            {ID: test[ID], TARGET: prediction.astype(np.float64)}
        )
        if not submission[ID].equals(sample[ID]):
            raise ValueError(f"Submission ID order mismatch: {name}")
        path = output_dir / f"submission_{name}.csv"
        submission.to_csv(path, index=False)
        submission_paths[name] = str(path)

    timestamp = datetime.now(timezone.utc).isoformat()
    output_files = [
        output_dir / "PHASE29_FOLD_RESULTS.csv",
        oof_path,
        output_dir / "PHASE29_SUMMARY.csv",
        *[Path(path) for path in submission_paths.values()],
    ]
    source_path = Path(__file__).resolve()
    manifest = {
        "phase": "Phase 29 clean-scope scratch modeling",
        "timestamp_utc": timestamp,
        "status": "locally reproducible research; not organizer-certified",
        "rules_scope": {
            "inputs": "only supplied train.csv, test.csv, sample_submission.csv",
            "external_data": "none",
            "target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical columns are one-hot encoded",
            "test_covariate_fitting": "none; category encoder is fit within training partitions only",
            "manual_prediction_edits": "none",
            "feature_values": "deterministic rowwise transforms of provided columns",
            "identity_feature": "SK_ID_CURR excluded from predictors",
        },
        "validation": {
            "outer": f"{FOLDS}-fold stratified",
            "outer_seed": SEED,
            "early_stopping": (
                "inner stratified 12% split of each outer training fold; "
                "final fold model refit on all outer training rows"
            ),
            "ensemble": "fixed 0.5 CatBoost + 0.5 LightGBM; no weight search",
        },
        "models": {
            "catboost": {
                "iterations_max": MAX_ITERATIONS,
                "depth": 7,
                "learning_rate": 0.04,
                "GPU": True,
                "categorical_inputs": "numeric matrix only; categoricals one-hot encoded",
            },
            "lightgbm": {
                "iterations_max": MAX_ITERATIONS,
                "learning_rate": 0.025,
                "num_leaves": 31,
                "device": "CPU",
                "categorical_inputs": "numeric matrix with fold-fitted one-hot columns",
            },
        },
        "category_cardinality_train": categories,
        "oof_metrics": summary_rows,
        "submission_files": submission_paths,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (source_path, train_path, test_path, sample_path, *output_files)
        },
    }
    manifest_path = output_dir / "PHASE29_RUN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False), flush=True)
    print(f"RUN_COMPLETE {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / f"phase29_clean_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
    )
    args = parser.parse_args()
    run(args.train, args.test, args.sample, args.output_dir)


if __name__ == "__main__":
    main()
