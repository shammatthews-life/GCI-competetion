from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
EMPLOYMENT_SENTINEL = 365243.0
FEE_OFFSET_HYPOTHESIS = 130.5
FINANCING_STEP_HYPOTHESIS = 0.0066
PRODUCT_KEYS = ("Q3", "Q4", "Q5", "KQ3", "KQ4", "P4", "P5", "AG3")


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _integerized(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (10**decimals))
    result = np.full(len(scaled), "MISSING", dtype=object)
    finite = np.isfinite(scaled)
    result[finite] = scaled[finite].astype(np.int64).astype(str)
    return result.astype(str)


def build_rowwise_features(raw_predictors: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw_predictors.columns:
        raise ValueError("TARGET must be separated before feature construction")
    if ID_COLUMN not in raw_predictors.columns:
        raise ValueError(f"Required identifier column is missing: {ID_COLUMN}")

    predictors = raw_predictors.drop(columns=[ID_COLUMN]).copy()
    required = {
        "DAYS_EMPLOYED",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    }
    missing = required.difference(predictors.columns)
    if missing:
        raise ValueError(f"Required dictionary fields are missing: {sorted(missing)}")

    features = predictors.copy()
    original_missing_count = predictors.isna().sum(axis=1).to_numpy(dtype=np.int32)
    for column in predictors.columns:
        if (
            pd.api.types.is_object_dtype(predictors[column].dtype)
            or pd.api.types.is_string_dtype(predictors[column].dtype)
            or isinstance(predictors[column].dtype, pd.CategoricalDtype)
        ):
            features[column] = predictors[column].astype("string").fillna("MISSING").astype(str)

    employed = pd.to_numeric(predictors["DAYS_EMPLOYED"], errors="raise").to_numpy(
        dtype=np.float64
    )
    clean_employed = employed.copy()
    employed_sentinel = employed == EMPLOYMENT_SENTINEL
    clean_employed[employed_sentinel] = np.nan
    features["DAYS_EMPLOYED_IS_365243"] = employed_sentinel.astype(np.int8)
    features["DAYS_EMPLOYED_CLEAN"] = clean_employed
    features["RAW_MISSING_COUNT"] = original_missing_count
    features["EXT_SOURCE_MISSING_COUNT"] = (
        predictors[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        .isna()
        .sum(axis=1)
        .to_numpy(dtype=np.int8)
    )

    credit = pd.to_numeric(predictors["AMT_CREDIT"], errors="raise").to_numpy(
        dtype=np.float64
    )
    annuity = pd.to_numeric(predictors["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64
    )
    goods = pd.to_numeric(predictors["AMT_GOODS_PRICE"], errors="raise").to_numpy(
        dtype=np.float64
    )
    income = pd.to_numeric(predictors["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
        dtype=np.float64
    )
    age_days = pd.to_numeric(predictors["DAYS_BIRTH"], errors="raise").to_numpy(
        dtype=np.float64
    )

    features["AGE_YEARS_FROM_DAYS_BIRTH"] = -age_days / 365.25
    features["CREDIT_TO_INCOME"] = _safe_ratio(credit, income)
    features["ANNUITY_TO_INCOME"] = _safe_ratio(annuity, income)
    features["CREDIT_TO_ANNUITY"] = _safe_ratio(credit, annuity)
    features["ANNUITY_TO_CREDIT"] = _safe_ratio(annuity, credit)
    features["GOODS_TO_CREDIT"] = _safe_ratio(goods, credit)
    features["CREDIT_TO_GOODS"] = _safe_ratio(credit, goods)
    return features.reset_index(drop=True)


def build_product_keys(raw_predictors: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw_predictors.columns:
        raise ValueError("TARGET must not be used to construct product keys")
    required = {
        "NAME_CONTRACT_TYPE",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    }
    missing = required.difference(raw_predictors.columns)
    if missing:
        raise ValueError(f"Required dictionary fields are missing: {sorted(missing)}")

    credit = pd.to_numeric(raw_predictors["AMT_CREDIT"], errors="raise").to_numpy(
        dtype=np.float64
    )
    annuity = pd.to_numeric(raw_predictors["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64
    )
    goods = pd.to_numeric(raw_predictors["AMT_GOODS_PRICE"], errors="raise").to_numpy(
        dtype=np.float64
    )
    contract = (
        raw_predictors["NAME_CONTRACT_TYPE"]
        .astype("string")
        .fillna("MISSING")
        .to_numpy(dtype=str)
    )

    ratio_ac = _safe_ratio(annuity, credit)
    ratio_ag = _safe_ratio(annuity, goods)
    q3, q4, q5 = (_integerized(ratio_ac, digits) for digits in (3, 4, 5))
    p4 = _integerized(
        _safe_ratio(annuity - FEE_OFFSET_HYPOTHESIS, credit), 4
    )
    p5 = _integerized(
        _safe_ratio(annuity - FEE_OFFSET_HYPOTHESIS, credit), 5
    )
    ag3 = _integerized(ratio_ag, 3)

    financing_coordinate = _safe_ratio(credit, goods) - 1.0
    financing_coordinate /= FINANCING_STEP_HYPOTHESIS
    eligible = (
        (contract == "Cash loans")
        & np.isfinite(financing_coordinate)
        & (credit > goods)
        & (goods > 0)
    )
    financing_key = np.full(len(credit), "NONCASH_OR_INVALID", dtype=object)
    financing_key[eligible] = (
        np.rint(financing_coordinate[eligible]).astype(np.int64).astype(str)
    )

    def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray(
            [f"{a}|{b}" for a, b in zip(left, right)],
            dtype=str,
        )

    kq3 = cross(financing_key, q3)
    kq4 = cross(financing_key, q4)
    kq3[~eligible | (q3 == "MISSING")] = "NONCASH_OR_INVALID"
    kq4[~eligible | (q4 == "MISSING")] = "NONCASH_OR_INVALID"

    return pd.DataFrame(
        {
            "PRODUCT_KEY_Q3": q3,
            "PRODUCT_KEY_Q4": q4,
            "PRODUCT_KEY_Q5": q5,
            "PRODUCT_KEY_KQ3": kq3,
            "PRODUCT_KEY_KQ4": kq4,
            "PRODUCT_KEY_P4": p4,
            "PRODUCT_KEY_P5": p5,
            "PRODUCT_KEY_AG3": ag3,
        }
    )


def categorical_columns(features: pd.DataFrame) -> list[str]:
    return [
        column
        for column in features.columns
        if (
            pd.api.types.is_object_dtype(features[column].dtype)
            or pd.api.types.is_string_dtype(features[column].dtype)
            or isinstance(features[column].dtype, pd.CategoricalDtype)
        )
    ]
