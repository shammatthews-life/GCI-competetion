from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(r"D:\gci\inpuy")
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
INPUT_NAMES = ("train.csv", "test.csv", "sample_submission.csv")
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
MISSING_CATEGORY = "__MISSING__"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def make_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Create only row-wise features; this function never fits on other rows."""
    excluded = {ID_COLUMN, TARGET_COLUMN}
    features = frame.drop(columns=[c for c in excluded if c in frame.columns]).copy()
    if feature_set == "raw":
        return features
    if feature_set != "rowwise_v1":
        raise ValueError(f"Unknown feature set: {feature_set}")

    employed = frame["DAYS_EMPLOYED"].where(frame["DAYS_EMPLOYED"] != 365243)
    ext = frame[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    income = frame["AMT_INCOME_TOTAL"]
    credit = frame["AMT_CREDIT"]
    annuity = frame["AMT_ANNUITY"]
    goods = frame["AMT_GOODS_PRICE"]
    age_days = -frame["DAYS_BIRTH"]

    derived: dict[str, Any] = {
        "DAYS_EMPLOYED_SENTINEL": (frame["DAYS_EMPLOYED"] == 365243).astype(np.int8),
        "DAYS_EMPLOYED_CLEAN": employed,
        "MISSING_COUNT": frame.drop(columns=[TARGET_COLUMN], errors="ignore").isna().sum(axis=1),
        "EXT_MEAN": ext.mean(axis=1),
        "EXT_MIN": ext.min(axis=1),
        "EXT_MAX": ext.max(axis=1),
        "EXT_STD": ext.std(axis=1),
        "EXT_RANGE": ext.max(axis=1) - ext.min(axis=1),
        "EXT_NOBS": ext.notna().sum(axis=1),
        "EXT_PRODUCT": ext.prod(axis=1, min_count=1),
        "EXT_12_PRODUCT": ext["EXT_SOURCE_1"] * ext["EXT_SOURCE_2"],
        "EXT_13_PRODUCT": ext["EXT_SOURCE_1"] * ext["EXT_SOURCE_3"],
        "EXT_23_PRODUCT": ext["EXT_SOURCE_2"] * ext["EXT_SOURCE_3"],
        "EXT_12_DIFF": ext["EXT_SOURCE_1"] - ext["EXT_SOURCE_2"],
        "EXT_13_DIFF": ext["EXT_SOURCE_1"] - ext["EXT_SOURCE_3"],
        "EXT_23_DIFF": ext["EXT_SOURCE_2"] - ext["EXT_SOURCE_3"],
        "CREDIT_INCOME_RATIO": _safe_divide(credit, income),
        "ANNUITY_INCOME_RATIO": _safe_divide(annuity, income),
        "ANNUITY_CREDIT_RATIO": _safe_divide(annuity, credit),
        "CREDIT_TERM_RATIO": _safe_divide(credit, annuity),
        "GOODS_CREDIT_RATIO": _safe_divide(goods, credit),
        "CREDIT_GOODS_GAP": credit - goods,
        "GOODS_INCOME_RATIO": _safe_divide(goods, income),
        "INCOME_PER_FAMILY_MEMBER": _safe_divide(income, frame["CNT_FAM_MEMBERS"]),
        "CREDIT_PER_FAMILY_MEMBER": _safe_divide(credit, frame["CNT_FAM_MEMBERS"]),
        "ANNUITY_PER_FAMILY_MEMBER": _safe_divide(annuity, frame["CNT_FAM_MEMBERS"]),
        "LOG_INCOME": np.log1p(income.clip(lower=0)),
        "LOG_CREDIT": np.log1p(credit.clip(lower=0)),
        "LOG_ANNUITY": np.log1p(annuity.clip(lower=0)),
        "LOG_GOODS_PRICE": np.log1p(goods.clip(lower=0)),
        "AGE_YEARS": age_days / 365.25,
        "EMPLOYMENT_YEARS": -employed / 365.25,
        "EMPLOYMENT_AGE_RATIO": _safe_divide(employed, frame["DAYS_BIRTH"]),
        "REGISTRATION_AGE_RATIO": _safe_divide(frame["DAYS_REGISTRATION"], frame["DAYS_BIRTH"]),
        "ID_PUBLISH_AGE_RATIO": _safe_divide(frame["DAYS_ID_PUBLISH"], frame["DAYS_BIRTH"]),
        "PHONE_CHANGE_AGE_RATIO": _safe_divide(frame["DAYS_LAST_PHONE_CHANGE"], frame["DAYS_BIRTH"]),
        "BUREAU_QUERY_SUM": frame.filter(regex=r"^AMT_REQ_CREDIT_BUREAU_").sum(axis=1, min_count=1),
        "BUREAU_RECENT_SUM": frame[
            ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]
        ].sum(axis=1, min_count=1),
        "BUREAU_MON_YEAR_RATIO": _safe_divide(
            frame["AMT_REQ_CREDIT_BUREAU_MON"],
            frame["AMT_REQ_CREDIT_BUREAU_YEAR"] + 1.0,
        ),
    }

    engineered = pd.DataFrame(derived, index=frame.index)
    features = pd.concat([features, engineered], axis=1)
    numeric = features.select_dtypes(include=[np.number]).columns
    features.loc[:, numeric] = features.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    return features


def _clean_categories(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CAT_COLUMNS:
        result[column] = result[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
    return result


def _lgb_frame(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = train.copy()
    query = other.copy()
    for column in CAT_COLUMNS:
        values = fit[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        vocab = {value: idx for idx, value in enumerate(pd.unique(values))}
        fit_codes = values.map(vocab).astype(np.int32)
        query_codes = (
            query[column].astype("string").fillna(MISSING_CATEGORY).astype(str).map(vocab)
        )
        categories = list(range(len(vocab)))
        fit[column] = pd.Categorical(fit_codes, categories=categories)
        query[column] = pd.Categorical(query_codes, categories=categories)
    return fit, query


def _xgb_matrix(
    train: pd.DataFrame, other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    train_cat = _clean_categories(train[CAT_COLUMNS])
    other_cat = _clean_categories(other[CAT_COLUMNS])
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    train_ohe = encoder.fit_transform(train_cat).toarray()
    other_ohe = encoder.transform(other_cat).toarray()
    numeric_columns = [column for column in train.columns if column not in CAT_COLUMNS]
    train_numeric = train[numeric_columns].to_numpy(dtype=np.float32, na_value=np.nan)
    other_numeric = other[numeric_columns].to_numpy(dtype=np.float32, na_value=np.nan)
    train_matrix = np.hstack([train_numeric, train_ohe]).astype(np.float32, copy=False)
    other_matrix = np.hstack([other_numeric, other_ohe]).astype(np.float32, copy=False)
    return train_matrix, other_matrix


def _train_predict(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    if model_name == "catboost_gpu":
        fit = _clean_categories(x_train)
        valid = _clean_categories(x_valid)
        model = CatBoostClassifier(
            iterations=900,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=12.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        model.fit(fit, y_train, cat_features=CAT_COLUMNS, verbose=False)
        return model.predict_proba(valid)[:, 1]

    if model_name == "lightgbm_cpu":
        fit, valid = _lgb_frame(x_train, x_valid)
        dataset = lgb.Dataset(
            fit,
            label=y_train,
            categorical_feature=CAT_COLUMNS,
            free_raw_data=False,
        )
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.025,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 120,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.0,
            "lambda_l2": 20.0,
            "verbosity": -1,
            "num_threads": 8,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "deterministic": True,
            "force_col_wise": True,
        }
        model = lgb.train(params, dataset, num_boost_round=1100)
        return model.predict(valid, num_threads=8)

    if model_name == "xgboost_gpu":
        fit, valid = _xgb_matrix(x_train, x_valid)
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=60,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=15.0,
            max_bin=256,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            n_jobs=8,
            random_state=seed,
        )
        model.fit(fit, y_train, verbose=False)
        return model.predict_proba(valid)[:, 1]

    raise ValueError(f"Unknown model: {model_name}")


def _assert_inputs(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    if TARGET_COLUMN not in train or ID_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError("Required target or ID column is missing.")
    if not train[ID_COLUMN].is_unique:
        raise ValueError("Training IDs are not unique.")
    if not test[ID_COLUMN].is_unique or not sample[ID_COLUMN].is_unique:
        raise ValueError("Test/sample IDs are not unique.")
    if not np.array_equal(test[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()):
        raise ValueError("Test IDs do not exactly match sample-submission order.")
    if set(train[TARGET_COLUMN].dropna().unique()) != {0, 1}:
        raise ValueError("Training target must contain binary labels 0 and 1.")
    if list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("Sample submission columns are not the required schema.")


def run_candidate(
    candidate_id: str,
    feature_set: str,
    output_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 314159,
    folds: int = 5,
    models: tuple[str, ...] = ("catboost_gpu", "lightgbm_cpu", "xgboost_gpu"),
    export_submission: bool = False,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    seed_everything(seed)

    for name in INPUT_NAMES:
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required organizer input not found: {path}")
    input_hashes = {name: sha256_file(data_dir / name) for name in INPUT_NAMES}
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    _assert_inputs(train, test, sample)

    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    x_all = make_features(train, feature_set)
    x_test = make_features(test, feature_set)
    if list(x_all.columns) != list(x_test.columns):
        raise AssertionError("Training and test feature columns differ.")

    splits = list(
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )
    oof = {name: np.full(len(y), np.nan, dtype=np.float64) for name in models}
    test_predictions = {name: [] for name in models}
    fold_results: list[dict[str, Any]] = []
    started = time.time()
    print(
        f"[{candidate_id}] rows={len(train)} folds={folds} feature_set={feature_set} "
        f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}",
        flush=True,
    )

    for fold_number, (fit_idx, valid_idx) in enumerate(splits, start=1):
        if np.intersect1d(fit_idx, valid_idx).size:
            raise AssertionError("Train/validation fold indices overlap.")
        x_fit = x_all.iloc[fit_idx].reset_index(drop=True)
        x_valid = x_all.iloc[valid_idx].reset_index(drop=True)
        y_fit = y[fit_idx]
        fold_record: dict[str, Any] = {"fold": fold_number, "n_fit": len(fit_idx), "n_valid": len(valid_idx)}
        print(f"[{candidate_id}] fold {fold_number}/{folds}", flush=True)

        for model_index, model_name in enumerate(models):
            model_seed = seed + fold_number * 100 + model_index
            model_started = time.time()
            valid_pred = _train_predict(
                model_name, x_fit, y_fit, x_valid, model_seed
            )
            if valid_pred.shape != (len(valid_idx),) or not np.isfinite(valid_pred).all():
                raise AssertionError(f"Invalid validation predictions from {model_name}.")
            oof[model_name][valid_idx] = valid_pred
            fold_auc = roc_auc_score(y[valid_idx], valid_pred)
            fold_record[f"{model_name}_auc"] = float(fold_auc)
            fold_record[f"{model_name}_seconds"] = round(time.time() - model_started, 3)

            if export_submission:
                test_pred = _train_predict(
                    model_name,
                    x_all.iloc[fit_idx].reset_index(drop=True),
                    y_fit,
                    x_test.reset_index(drop=True),
                    model_seed,
                )
                if test_pred.shape != (len(test),) or not np.isfinite(test_pred).all():
                    raise AssertionError(f"Invalid test predictions from {model_name}.")
                test_predictions[model_name].append(test_pred)
            print(
                f"  {model_name}: fold_auc={fold_auc:.6f} "
                f"seconds={fold_record[f'{model_name}_seconds']:.1f}",
                flush=True,
            )
        fold_results.append(fold_record)

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise AssertionError("Every training row must have exactly one finite OOF prediction.")
    summary: dict[str, Any] = {
        "candidate_id": candidate_id,
        "feature_set": feature_set,
        "seed": seed,
        "folds": folds,
        "models": list(models),
        "fold_results": fold_results,
        "model_summary": {},
        "input_sha256": input_hashes,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "catboost": __import__("catboost").__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "seconds": None,
    }

    oof_frame = pd.DataFrame({ID_COLUMN: train[ID_COLUMN].to_numpy()})
    for model_name, values in oof.items():
        model_fold_auc = [record[f"{model_name}_auc"] for record in fold_results]
        model_summary = {
            "pooled_oof_auc": float(roc_auc_score(y, values)),
            "mean_fold_auc": float(np.mean(model_fold_auc)),
            "std_fold_auc": float(np.std(model_fold_auc, ddof=1)),
            "fold_auc": model_fold_auc,
        }
        summary["model_summary"][model_name] = model_summary
        oof_frame[f"OOF_{model_name}"] = values

    summary["seconds"] = round(time.time() - started, 3)
    oof_path = output_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    summary["oof_sha256"] = sha256_file(oof_path)

    if export_submission:
        submissions = []
        for model_name in models:
            predictions = np.mean(test_predictions[model_name], axis=0)
            if not np.isfinite(predictions).all():
                raise AssertionError(f"Non-finite test predictions in {model_name}.")
            sub = pd.DataFrame({ID_COLUMN: sample[ID_COLUMN], TARGET_COLUMN: predictions})
            if list(sub.columns) != [ID_COLUMN, TARGET_COLUMN]:
                raise AssertionError("Submission columns do not match the official schema.")
            if not np.array_equal(sub[ID_COLUMN].to_numpy(), sample[ID_COLUMN].to_numpy()):
                raise AssertionError("Submission IDs/order do not match the sample file.")
            sub_path = output_dir / f"submission_{model_name}.csv"
            sub.to_csv(sub_path, index=False)
            submissions.append(sub_path.name)
        summary["submission_files"] = submissions

    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["model_summary"], indent=2), flush=True)
    print(f"[{candidate_id}] run_dir={output_dir} seconds={summary['seconds']}", flush=True)


def main(candidate_id: str, feature_set: str, models: tuple[str, ...]) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--export-submission", action="store_true")
    args = parser.parse_args()
    run_candidate(
        candidate_id=candidate_id,
        feature_set=feature_set,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        seed=args.seed,
        folds=args.folds,
        models=models,
        export_submission=args.export_submission,
    )
