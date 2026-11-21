from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE37_OOF = Path(
    r"D:\gci\truth\experiments\phase37_multiseed_xgboost_rules_first"
    r"\runs\phase37_multiseed_20261107T173127Z\PHASE37_OOF_PREDICTIONS.csv"
)
RUN_DIR = ROOT / "runs" / "clean_catboost_constant_free_v1"
SEED = 20261113
FOLDS = 5
ID = "SK_ID_CURR"
TARGET = "TARGET"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    n = pd.to_numeric(numerator, errors="raise").to_numpy(dtype=np.float64)
    d = pd.to_numeric(denominator, errors="raise").to_numpy(dtype=np.float64)
    result = np.full(len(n), np.nan, dtype=np.float64)
    valid = np.isfinite(n) & np.isfinite(d) & (d != 0)
    np.divide(n, d, out=result, where=valid)
    return result


def rounded_key(values: np.ndarray, decimals: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.full(len(values), "__MISSING__", dtype=object)
    finite = np.isfinite(values)
    result[finite] = np.rint(values[finite] * (10**decimals)).astype(np.int64).astype(str)
    return result.astype(str)


def make_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = raw.drop(columns=[ID, TARGET], errors="ignore").copy()
    categorical = list(features.select_dtypes(include=["object", "category", "string"]).columns)
    for column in categorical:
        features[column] = (
            features[column].astype("string").fillna("__MISSING__").astype(str)
        )

    employment = raw["DAYS_EMPLOYED"]
    employment_clean = employment.where(employment != 365243)
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    ext_count = ext.notna().sum(axis=1)
    ext_mean = ext.mean(axis=1)
    ext_min = ext.min(axis=1)
    ext_max = ext.max(axis=1)
    credit = raw["AMT_CREDIT"]
    annuity = raw["AMT_ANNUITY"]
    goods = raw["AMT_GOODS_PRICE"]
    income = raw["AMT_INCOME_TOTAL"]
    age_years = -raw["DAYS_BIRTH"] / 365.25

    numeric_features: dict[str, np.ndarray | pd.Series] = {
        "DAYS_EMPLOYED_CLEAN": employment_clean,
        "DAYS_EMPLOYED_SENTINEL": (employment == 365243).astype(np.int8),
        "EXT_MISSING_COUNT": 3 - ext_count,
        "EXT_MEAN": ext_mean,
        "EXT_MIN": ext_min,
        "EXT_MAX": ext_max,
        "EXT_RANGE": ext_max - ext_min,
        "EXT_STD": ext.std(axis=1),
        "EXT_PRODUCT": ext.prod(axis=1, min_count=1),
        "EXT_WEIGHTED_MEAN": (
            2 * raw["EXT_SOURCE_1"] + 3 * raw["EXT_SOURCE_2"] + 4 * raw["EXT_SOURCE_3"]
        )
        / 9,
        "EXT_12_PRODUCT": raw["EXT_SOURCE_1"] * raw["EXT_SOURCE_2"],
        "EXT_13_PRODUCT": raw["EXT_SOURCE_1"] * raw["EXT_SOURCE_3"],
        "EXT_23_PRODUCT": raw["EXT_SOURCE_2"] * raw["EXT_SOURCE_3"],
        "EXT_12_DIFF": raw["EXT_SOURCE_1"] - raw["EXT_SOURCE_2"],
        "EXT_13_DIFF": raw["EXT_SOURCE_1"] - raw["EXT_SOURCE_3"],
        "EXT_23_DIFF": raw["EXT_SOURCE_2"] - raw["EXT_SOURCE_3"],
        "CREDIT_TO_INCOME": safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "ANNUITY_TO_CREDIT": safe_ratio(annuity, credit),
        "CREDIT_TO_ANNUITY": safe_ratio(credit, annuity),
        "GOODS_TO_CREDIT": safe_ratio(goods, credit),
        "CREDIT_MINUS_GOODS": (credit - goods).to_numpy(),
        "GOODS_TO_INCOME": safe_ratio(goods, income),
        "INCOME_PER_FAMILY_MEMBER": safe_ratio(income, raw["CNT_FAM_MEMBERS"]),
        "CREDIT_PER_FAMILY_MEMBER": safe_ratio(credit, raw["CNT_FAM_MEMBERS"]),
        "ANNUITY_PER_FAMILY_MEMBER": safe_ratio(annuity, raw["CNT_FAM_MEMBERS"]),
        "AGE_YEARS": age_years,
        "EMPLOYMENT_YEARS": -employment_clean / 365.25,
        "EMPLOYMENT_AGE_RATIO": safe_ratio(employment_clean, raw["DAYS_BIRTH"]),
        "REGISTRATION_AGE_RATIO": safe_ratio(raw["DAYS_REGISTRATION"], raw["DAYS_BIRTH"]),
        "ID_PUBLISH_AGE_RATIO": safe_ratio(raw["DAYS_ID_PUBLISH"], raw["DAYS_BIRTH"]),
        "PHONE_CHANGE_AGE_RATIO": safe_ratio(
            raw["DAYS_LAST_PHONE_CHANGE"], raw["DAYS_BIRTH"]
        ),
        "CREDIT_PER_AGE": safe_ratio(credit, age_years),
        "INCOME_PER_AGE": safe_ratio(income, age_years),
        "MISSING_VALUE_COUNT": raw.drop(columns=[ID, TARGET], errors="ignore")
        .isna()
        .sum(axis=1),
    }

    bureau_columns = [
        column
        for column in raw.columns
        if column.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if bureau_columns:
        bureau = raw[bureau_columns]
        numeric_features["BUREAU_REQUEST_SUM"] = bureau.sum(axis=1, min_count=1)
        for column in bureau_columns:
            numeric_features[f"{column}_MISSING"] = raw[column].isna().astype(np.int8)

    extra = pd.DataFrame(numeric_features).replace([np.inf, -np.inf], np.nan)
    for column in extra.columns:
        features[column] = extra[column].to_numpy()

    ann_credit = safe_ratio(annuity, credit)
    ann_goods = safe_ratio(annuity, goods)
    goods_credit = safe_ratio(goods, credit)
    key_frame = pd.DataFrame(
        {
            "CF_KEY_ANNUITY_CREDIT_3": rounded_key(ann_credit, 3),
            "CF_KEY_ANNUITY_CREDIT_4": rounded_key(ann_credit, 4),
            "CF_KEY_ANNUITY_CREDIT_5": rounded_key(ann_credit, 5),
            "CF_KEY_ANNUITY_GOODS_3": rounded_key(ann_goods, 3),
            "CF_KEY_GOODS_CREDIT_3": rounded_key(goods_credit, 3),
        }
    )
    for column in key_frame.columns:
        features[column] = key_frame[column]
        categorical.append(column)

    if features.columns.duplicated().any():
        duplicates = features.columns[features.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns: {duplicates}")
    return features, categorical


def main() -> None:
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {RUN_DIR}")
    if not TRAIN.is_file() or not PHASE37_OOF.is_file():
        raise FileNotFoundError("Required official training or reference OOF file is missing")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    reference = pd.read_csv(PHASE37_OOF)
    if (
        len(train) != len(reference)
        or not train[ID].equals(reference[ID])
        or not train[TARGET].equals(reference[TARGET])
    ):
        raise ValueError("Phase 37 reference rows, IDs, or labels do not align")
    y = train[TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    x, categorical = make_features(raw)
    categories = [x.columns.get_loc(column) for column in categorical]
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    expected_fold_ids = reference["fold_seed_20261113"].to_numpy(dtype=np.int8)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    predictions = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, float | int]] = []

    from catboost.utils import get_gpu_device_count

    gpu_count = int(get_gpu_device_count())
    if gpu_count < 1:
        raise RuntimeError("CatBoost cannot see a CUDA GPU; refusing CPU fallback")

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.72,
            border_count=128,
            one_hot_max_size=10,
            max_ctr_complexity=1,
            random_seed=SEED + fold_number,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(x.iloc[fit_rows], y[fit_rows], cat_features=categories)
        fold_prediction = model.predict_proba(x.iloc[valid_rows])[:, 1]
        if not np.isfinite(fold_prediction).all():
            raise ValueError(f"Nonfinite prediction in fold {fold_number}")
        predictions[valid_rows] = fold_prediction.astype(np.float32)
        auc = float(roc_auc_score(y[valid_rows], fold_prediction))
        records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "auc": auc,
                "iterations": 700,
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} AUC={auc:.6f} "
            f"features={x.shape[1]} gpu={gpu_count}",
            flush=True,
        )
        del model

    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Recreated folds do not match Phase 37 reference folds")
    if not np.isfinite(predictions).all():
        raise ValueError("OOF vector is incomplete")

    baseline = reference["oof_equal_blend_lgbm_xgb_three_seeds"].to_numpy(
        dtype=np.float64
    )
    blend = np.full(len(y), np.nan, dtype=np.float32)
    for fold_number in range(1, FOLDS + 1):
        mask = fold_ids == fold_number
        baseline_rank = (rankdata(baseline[mask], method="average") - 1) / (
            mask.sum() - 1
        )
        candidate_rank = (rankdata(predictions[mask], method="average") - 1) / (
            mask.sum() - 1
        )
        blend[mask] = (0.5 * baseline_rank + 0.5 * candidate_rank).astype(np.float32)

    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=False, exist_ok=False)
    oof = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_clean_catboost": predictions,
            "oof_phase37_reference": baseline.astype(np.float32),
            "oof_fixed_50_50_within_fold_rank_blend": blend,
        }
    )
    oof_path = RUN_DIR / "OOF_PREDICTIONS.csv"
    folds_path = RUN_DIR / "FOLD_RESULTS.csv"
    summary_path = RUN_DIR / "SUMMARY.json"
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(folds_path, index=False)

    fold_metrics = {}
    for name, vector in (
        ("clean_catboost", predictions),
        ("phase37_reference", baseline),
        ("fixed_50_50_within_fold_rank_blend", blend),
    ):
        per_fold = [
            float(roc_auc_score(y[fold_ids == fold], vector[fold_ids == fold]))
            for fold in range(1, FOLDS + 1)
        ]
        fold_metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, vector)),
            "mean_fold_auc": float(np.mean(per_fold)),
            "fold_auc": per_fold,
        }
    summary = {
        "experiment": "Clean V17-style product-key CatBoost expert",
        "status": "exploratory OOF; repeated-label development; not organizer-certified",
        "input_scope": "official train.csv only; no test, external data, or historical prediction CSV used as features",
        "feature_scope": "row-wise formulas and constant-free rounded ratios; no fixed fee/step constants, peer-label transfer, or pooled statistics",
        "target_statistics": "CatBoost native categorical learning, trained separately inside each outer fold",
        "seed": SEED,
        "outer_folds": FOLDS,
        "gpu": {"device": "CatBoost CUDA", "visible_device_count": gpu_count},
        "model": {
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "one_hot_max_size": 10,
            "max_ctr_complexity": 1,
        },
        "features": int(x.shape[1]),
        "categorical_features": int(len(categorical)),
        "metrics": fold_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
        },
        "sha256": {
            str(TRAIN): sha256_file(TRAIN),
            str(PHASE37_OOF): sha256_file(PHASE37_OOF),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(oof_path): sha256_file(oof_path),
            str(folds_path): sha256_file(folds_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
