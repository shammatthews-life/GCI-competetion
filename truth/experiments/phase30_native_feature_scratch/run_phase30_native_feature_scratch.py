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


ROOT = Path(__file__).resolve().parent
TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
TEST_DEFAULT = Path(r"D:\gci\inpuy\test.csv")
SAMPLE_DEFAULT = Path(r"D:\gci\inpuy\sample_submission.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_VALIDATION_SIZE = 0.12
MAX_ITERATIONS = 2400
STOPPING_ROUNDS = 120


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
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=output, where=valid)
    return output


def _ratio_key(numerator: np.ndarray, denominator: np.ndarray, decimals: int) -> np.ndarray:
    ratio = _ratio(numerator, denominator)
    result = np.full(len(ratio), "__MISSING__", dtype=object)
    valid = np.isfinite(ratio)
    scaled = np.rint(ratio[valid] * (10**decimals))
    limits = np.iinfo(np.int64)
    if np.any((scaled < limits.min) | (scaled > limits.max)):
        raise ValueError("Rounded ratio key is outside int64 range")
    result[valid] = scaled.astype(np.int64).astype(str)
    return result.astype(str)


def build_features(
    raw: pd.DataFrame,
    missing_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    if ID not in raw or TARGET in raw:
        raise ValueError("Feature input must include ID and exclude TARGET")
    predictors = raw.drop(columns=[ID]).copy()
    numeric_columns = [
        column
        for column in predictors
        if pd.api.types.is_numeric_dtype(predictors[column].dtype)
    ]
    categorical_columns = [
        column for column in predictors if column not in numeric_columns
    ]
    frame = predictors[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    ).replace([np.inf, -np.inf], np.nan)

    income = _numeric(predictors, "AMT_INCOME_TOTAL")
    credit = _numeric(predictors, "AMT_CREDIT")
    annuity = _numeric(predictors, "AMT_ANNUITY")
    goods = _numeric(predictors, "AMT_GOODS_PRICE")
    family = _numeric(predictors, "CNT_FAM_MEMBERS")
    age = np.abs(_numeric(predictors, "DAYS_BIRTH"))
    employed = np.abs(_numeric(predictors, "DAYS_EMPLOYED"))
    employed[employed == 365243.0] = np.nan

    ext = predictors[
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].to_numpy(dtype=np.float64)
    available = np.isfinite(ext)
    count = available.sum(axis=1)
    ext_mean = np.divide(
        np.nansum(ext, axis=1),
        count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=count > 0,
    )
    ext_min = np.where(available, ext, np.inf).min(axis=1)
    ext_max = np.where(available, ext, -np.inf).max(axis=1)
    ext_min[count == 0] = np.nan
    ext_max[count == 0] = np.nan
    ext_range = ext_max - ext_min
    ext_range[count < 2] = np.nan
    ext_std = np.sqrt(
        np.divide(
            np.where(available, (ext - ext_mean[:, None]) ** 2, 0).sum(axis=1),
            count,
            out=np.full(len(ext), np.nan, dtype=np.float64),
            where=count > 0,
        )
    )

    derived = {
        "P30_CREDIT_TO_INCOME": _ratio(credit, income),
        "P30_ANNUITY_TO_INCOME": _ratio(annuity, income),
        "P30_ANNUITY_TO_CREDIT": _ratio(annuity, credit),
        "P30_CREDIT_TO_GOODS": _ratio(credit, goods),
        "P30_GOODS_TO_INCOME": _ratio(goods, income),
        "P30_INCOME_PER_FAMILY_MEMBER": _ratio(income, family),
        "P30_CREDIT_PER_FAMILY_MEMBER": _ratio(credit, family),
        "P30_ANNUITY_PER_FAMILY_MEMBER": _ratio(annuity, family),
        "P30_EMPLOYMENT_TO_AGE": _ratio(employed, age),
        "P30_EXT_MEAN": ext_mean,
        "P30_EXT_RANGE": ext_range,
        "P30_EXT_STD": ext_std,
        "P30_EXT_AVAILABLE": count.astype(np.float64),
        "P30_EXT_DIFF_12": np.abs(ext[:, 0] - ext[:, 1]),
        "P30_EXT_DIFF_13": np.abs(ext[:, 0] - ext[:, 2]),
        "P30_EXT_DIFF_23": np.abs(ext[:, 1] - ext[:, 2]),
        "P30_EXT_RANGE_X_CREDIT_TO_INCOME": ext_range
        * _ratio(credit, income),
        "P30_EXT_MEAN_X_CREDIT_TO_INCOME": ext_mean
        * _ratio(credit, income),
        "P30_AGE_YEARS": age / 365.25,
        "P30_EMPLOYMENT_YEARS": employed / 365.25,
        "P30_AGE_AT_REGISTRATION": age
        - np.abs(_numeric(predictors, "DAYS_REGISTRATION")),
        "P30_AGE_AT_ID_PUBLISH": age
        - np.abs(_numeric(predictors, "DAYS_ID_PUBLISH")),
        "P30_AGE_AT_PHONE_CHANGE": age
        - np.abs(_numeric(predictors, "DAYS_LAST_PHONE_CHANGE")),
        "P30_RAW_MISSING_COUNT": predictors.isna().sum(axis=1).to_numpy(
            dtype=np.float64
        ),
    }
    for column in missing_columns:
        if column not in predictors:
            raise ValueError(f"Missingness source column is absent: {column}")
        derived[f"P30_MISSING_{column}"] = predictors[column].isna().to_numpy(
            dtype=np.float32
        )
    for name, values in derived.items():
        frame[name] = values

    category_frame = predictors[categorical_columns].copy()
    for column in categorical_columns:
        category_frame[column] = (
            category_frame[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    ratio_keys = pd.DataFrame(
        {
            "P30_KEY_ANNUITY_CREDIT_3": _ratio_key(annuity, credit, 3),
            "P30_KEY_ANNUITY_CREDIT_4": _ratio_key(annuity, credit, 4),
            "P30_KEY_ANNUITY_CREDIT_5": _ratio_key(annuity, credit, 5),
            "P30_KEY_ANNUITY_GOODS_3": _ratio_key(annuity, goods, 3),
        }
    )
    category_frame = pd.concat(
        [category_frame.reset_index(drop=True), ratio_keys], axis=1
    )
    categorical_columns = categorical_columns + ratio_keys.columns.tolist()
    for column in categorical_columns:
        category_frame[column] = category_frame[column].astype(str)
    frame = pd.concat(
        [frame.reset_index(drop=True), category_frame], axis=1
    )
    if frame.columns.has_duplicates:
        raise ValueError("Duplicate feature columns")
    return frame, categorical_columns


def _split(
    rows: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=INNER_VALIDATION_SIZE, random_state=seed
    )
    train_idx, valid_idx = next(splitter.split(np.zeros(len(rows)), y[rows]))
    return rows[train_idx], rows[valid_idx]


def _lgb_frame(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = frame.iloc[fit_rows].copy()
    apply = frame.iloc[apply_rows].copy()
    for column in categorical_columns:
        categories = pd.Index(fit[column].astype(str).unique())
        fit[column] = pd.Categorical(fit[column].astype(str), categories=categories)
        apply_values = apply[column].astype(str)
        apply_values = apply_values.where(apply_values.isin(categories), np.nan)
        apply[column] = pd.Categorical(
            apply_values, categories=categories
        )
    return fit, apply


def _fit_catboost(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    y: np.ndarray,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    test_frame: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    inner_train, inner_valid = _split(fit_rows, y, seed)
    cat_indices = [frame.columns.get_loc(column) for column in categorical_columns]
    selector = CatBoostClassifier(
        iterations=MAX_ITERATIONS,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    selector.fit(
        frame.iloc[inner_train],
        y[inner_train],
        cat_features=cat_indices,
        eval_set=(frame.iloc[inner_valid], y[inner_valid]),
        early_stopping_rounds=STOPPING_ROUNDS,
        use_best_model=True,
    )
    iterations = max(100, int(selector.get_best_iteration()) + 1)
    del selector
    model = CatBoostClassifier(
        iterations=iterations,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    model.fit(frame.iloc[fit_rows], y[fit_rows], cat_features=cat_indices)
    valid_prediction = model.predict_proba(frame.iloc[valid_rows])[:, 1]
    test_prediction = model.predict_proba(test_frame)[:, 1]
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("CatBoost did not run on GPU")
    return valid_prediction, test_prediction, iterations


def _fit_lightgbm(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    y: np.ndarray,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    test_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    inner_train, inner_valid = _split(fit_rows, y, seed)
    x_inner, x_stop = _lgb_frame(
        frame, categorical_columns, inner_train, inner_valid
    )
    params: dict[str, Any] = {
        "objective": "binary",
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 100,
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
    selector = lgb.LGBMClassifier(**params, n_estimators=MAX_ITERATIONS)
    selector.fit(
        x_inner,
        y[inner_train],
        categorical_feature=categorical_columns,
        eval_set=[(x_stop, y[inner_valid])],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iterations = max(100, int(selector.best_iteration_))
    del selector, x_inner, x_stop

    x_fit, x_valid = _lgb_frame(
        frame, categorical_columns, fit_rows, valid_rows
    )
    x_all, x_test = _lgb_frame(
        frame, categorical_columns, fit_rows, test_rows
    )
    model = lgb.LGBMClassifier(**params, n_estimators=iterations)
    model.fit(x_fit, y[fit_rows], categorical_feature=categorical_columns)
    valid_prediction = model.predict_proba(x_valid)[:, 1]
    test_prediction = model.predict_proba(x_test)[:, 1]
    return valid_prediction, test_prediction, iterations


def run(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    if TARGET not in train or ID not in train or TARGET in test or ID not in test:
        raise ValueError("Train/test input schema is missing ID or target semantics")
    if test.columns.tolist() != [
        column for column in train.columns if column != TARGET
    ]:
        raise ValueError("Train/test predictor schemas differ")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("sample_submission columns must be ID,TARGET")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("IDs must be unique")
    if not test[ID].reset_index(drop=True).equals(
        sample[ID].reset_index(drop=True)
    ):
        raise ValueError("Test IDs do not match sample submission order")
    y_raw = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(y_raw).all()
        or not np.isin(y_raw, [0, 1]).all()
        or set(np.unique(y_raw)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary labels and both classes")
    y = y_raw.astype(np.int8)
    train_raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    missing_columns = tuple(
        column for column in train_raw if train_raw[column].isna().any()
    )
    train_frame, categorical_columns = build_features(train_raw, missing_columns)
    test_frame, test_categorical = build_features(test_raw, missing_columns)
    if categorical_columns != test_categorical:
        raise ValueError("Train/test categorical feature schemas differ")
    if not train_frame.columns.equals(test_frame.columns):
        raise ValueError("Train/test feature columns differ")
    combined_rowwise_frame = pd.concat(
        [train_frame, test_frame], ignore_index=True
    )

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    predictions: dict[str, np.ndarray] = {
        "catboost_native": np.full(len(y), np.nan, dtype=np.float32),
        "lightgbm_native": np.full(len(y), np.nan, dtype=np.float32),
    }
    test_fold: dict[str, list[np.ndarray]] = {
        key: [] for key in predictions
    }
    rows: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        print(
            f"FOLD {fold_number}/{FOLDS} fit={len(fit_rows)} valid={len(valid_rows)}",
            flush=True,
        )
        for model_name in predictions:
            if model_name == "catboost_native":
                valid_prediction, test_prediction, iterations = _fit_catboost(
                    train_frame,
                    categorical_columns,
                    y,
                    fit_rows,
                    valid_rows,
                    test_frame,
                    fold_seed,
                )
            else:
                valid_prediction, test_prediction, iterations = _fit_lightgbm(
                    combined_rowwise_frame,
                    categorical_columns,
                    y,
                    fit_rows,
                    valid_rows,
                    np.arange(len(train_frame), len(train_frame) + len(test_frame)),
                    fold_seed,
                )
            if not np.isfinite(valid_prediction).all() or not np.isfinite(
                test_prediction
            ).all():
                raise ValueError(f"Non-finite predictions for {model_name}")
            predictions[model_name][valid_rows] = valid_prediction
            test_fold[model_name].append(test_prediction)
            fold_auc = float(roc_auc_score(y[valid_rows], valid_prediction))
            rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "auc": fold_auc,
                    "iterations": iterations,
                }
            )
            print(
                f"  {model_name}: AUC={fold_auc:.6f}, iterations={iterations}",
                flush=True,
            )

    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF predictions incomplete for {name}")
    test_predictions = {
        name: np.mean(np.vstack(fold_predictions), axis=0)
        for name, fold_predictions in test_fold.items()
    }
    predictions["equal_blend"] = (
        0.5 * predictions["catboost_native"]
        + 0.5 * predictions["lightgbm_native"]
    ).astype(np.float32)
    test_predictions["equal_blend"] = (
        0.5 * test_predictions["catboost_native"]
        + 0.5 * test_predictions["lightgbm_native"]
    )

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_frame = pd.DataFrame(rows)
    fold_path = output_dir / "PHASE30_FOLD_RESULTS.csv"
    fold_frame.to_csv(fold_path, index=False)
    oof = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": np.full(len(y), -1, dtype=np.int8),
            **{f"oof_{name}": value for name, value in predictions.items()},
        }
    )
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        oof.loc[valid_rows, "fold"] = fold_number
    oof_path = output_dir / "PHASE30_OOF_PREDICTIONS.csv"
    oof.to_csv(oof_path, index=False)

    summary = []
    for name, prediction in predictions.items():
        fold_auc = fold_frame.loc[fold_frame.model == name, "auc"].to_numpy()
        summary.append(
            {
                "model": name,
                "pooled_auc": float(roc_auc_score(y, prediction)),
                "mean_fold_auc": (
                    float(fold_auc.mean()) if len(fold_auc) else float("nan")
                ),
                "std_fold_auc": (
                    float(fold_auc.std(ddof=0))
                    if len(fold_auc)
                    else float("nan")
                ),
                "log_loss": float(log_loss(y, prediction)),
            }
        )
    summary_path = output_dir / "PHASE30_SUMMARY.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    submission_paths: list[Path] = []
    for name, prediction in test_predictions.items():
        if not np.isfinite(prediction).all() or (
            (prediction < 0) | (prediction > 1)
        ).any():
            raise ValueError(f"Invalid submission probability vector: {name}")
        submission = pd.DataFrame(
            {ID: test[ID], TARGET: prediction.astype(np.float64)}
        )
        if submission.columns.tolist() != [ID, TARGET] or not submission[ID].equals(
            sample[ID]
        ):
            raise ValueError(f"Invalid sample-submission alignment: {name}")
        path = output_dir / f"submission_{name}.csv"
        submission.to_csv(path, index=False)
        submission_paths.append(path)

    output_files = [fold_path, oof_path, summary_path, *submission_paths]
    manifest = {
        "phase": "Phase 30 native-category scratch research",
        "status": "local reproducibility record; not organizer-certified",
        "rule_controls": {
            "external_data": "none",
            "manual_or_rowwise_prediction_override": "none",
            "ID_feature": "excluded",
            "custom_target_encoding_feature": "none",
            "model_target_usage": (
                "training labels are used only by the fitted supervised models; "
                "native categorical model processing is not a custom encoding"
            ),
            "test_statistics": "no category statistics fit on test rows",
            "feature_engineering": (
                "deterministic transformations of provided row values; rounded "
                "ratio categories use only decimal-place precision, no empirical "
                "fee or tariff constants"
            ),
        },
        "validation": {
            "folds": FOLDS,
            "seed": SEED,
            "early_stopping": "inner 12% of outer training fold",
            "blend": "predeclared equal 50/50 probability average",
        },
        "categorical_features": categorical_columns,
        "metrics": summary,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                train_path,
                test_path,
                sample_path,
                *output_files,
            )
        },
        "competition_note": (
            "The local rule text does not expressly adjudicate native "
            "categorical target-statistic internals. This pipeline does not "
            "construct target-encoded features or use manual test edits; seek "
            "organizer clarification if their interpretation must be certain."
        ),
    }
    manifest_path = output_dir / "PHASE30_RUN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
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
        / f"phase30_native_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
    )
    args = parser.parse_args()
    run(args.train, args.test, args.sample, args.output_dir)


if __name__ == "__main__":
    main()
