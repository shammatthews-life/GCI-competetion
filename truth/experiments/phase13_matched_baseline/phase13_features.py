from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
EMPLOYMENT_SENTINEL = 365243.0
FEE_OFFSET_HYPOTHESIS = 130.5
FINANCING_STEP_HYPOTHESIS = 0.0066
PRODUCT_KEYS = (
    "PRODUCT_KEY_Q3",
    "PRODUCT_KEY_Q4",
    "PRODUCT_KEY_Q5",
    "PRODUCT_KEY_KQ3",
    "PRODUCT_KEY_KQ4",
    "PRODUCT_KEY_P4",
    "PRODUCT_KEY_P5",
    "PRODUCT_KEY_AG3",
)


def _require_no_target(frame: pd.DataFrame) -> None:
    if TARGET_COLUMN in frame.columns:
        raise ValueError("TARGET must be separated before feature construction")


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")


def _numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    return pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    values = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=values, where=valid)
    return values


def _integerized(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (10**decimals))
    result = np.full(len(scaled), "MISSING", dtype=object)
    finite = np.isfinite(scaled)
    limits = np.iinfo(np.int64)
    if np.any(finite & ((scaled < limits.min) | (scaled > limits.max))):
        raise ValueError("A product-key integer exceeds the int64 range")
    result[finite] = scaled[finite].astype(np.int64).astype(str)
    return result.astype(str)


def build_raw_feature_blocks(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_no_target(raw)
    _require_columns(raw, {ID_COLUMN})
    predictors = raw.drop(columns=[ID_COLUMN]).copy()
    numeric_columns = [
        column
        for column in predictors
        if pd.api.types.is_numeric_dtype(predictors[column].dtype)
    ]
    categorical_columns = [
        column for column in predictors if column not in numeric_columns
    ]
    numeric = predictors[numeric_columns].apply(pd.to_numeric, errors="raise")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    categorical = predictors[categorical_columns].copy()
    for column in categorical_columns:
        categorical[column] = (
            categorical[column].astype("string").fillna("MISSING").astype(str)
        )
    categorical = categorical.reset_index(drop=True)

    required = {
        "DAYS_EMPLOYED",
        "DAYS_BIRTH",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "AMT_INCOME_TOTAL",
    }
    _require_columns(predictors, required)
    employed = _numeric(predictors, "DAYS_EMPLOYED")
    employed_sentinel = employed == EMPLOYMENT_SENTINEL
    clean_employed = employed.copy()
    clean_employed[employed_sentinel] = np.nan
    ext_columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    credit = _numeric(predictors, "AMT_CREDIT")
    annuity = _numeric(predictors, "AMT_ANNUITY")
    goods = _numeric(predictors, "AMT_GOODS_PRICE")
    income = _numeric(predictors, "AMT_INCOME_TOTAL")
    age_days = _numeric(predictors, "DAYS_BIRTH")
    rowwise = pd.DataFrame(
        {
            "DAYS_EMPLOYED_IS_365243": employed_sentinel.astype(np.int8),
            "DAYS_EMPLOYED_CLEAN": clean_employed,
            "RAW_MISSING_COUNT": predictors.isna().sum(axis=1).to_numpy(
                dtype=np.int16
            ),
            "EXT_SOURCE_MISSING_COUNT": predictors[ext_columns]
            .isna()
            .sum(axis=1)
            .to_numpy(dtype=np.int8),
            "AGE_YEARS_FROM_DAYS_BIRTH": -age_days / 365.25,
            "CREDIT_TO_INCOME": _safe_ratio(credit, income),
            "ANNUITY_TO_INCOME": _safe_ratio(annuity, income),
            "CREDIT_TO_ANNUITY": _safe_ratio(credit, annuity),
            "ANNUITY_TO_CREDIT": _safe_ratio(annuity, credit),
            "GOODS_TO_CREDIT": _safe_ratio(goods, credit),
            "CREDIT_TO_GOODS": _safe_ratio(credit, goods),
        }
    )
    return numeric, categorical, rowwise


def build_product_keys(raw: pd.DataFrame) -> pd.DataFrame:
    _require_no_target(raw)
    _require_columns(
        raw,
        {"NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"},
    )
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(
        dtype=str
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
    coordinate = _safe_ratio(credit, goods) - 1.0
    coordinate /= FINANCING_STEP_HYPOTHESIS
    eligible = (
        (contract == "Cash loans")
        & np.isfinite(coordinate)
        & (credit > goods)
        & (goods > 0)
    )
    financing_key = np.full(len(credit), "NONCASH_OR_INVALID", dtype=object)
    financing_key[eligible] = np.rint(coordinate[eligible]).astype(np.int64).astype(str)

    def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([f"{a}|{b}" for a, b in zip(left, right)], dtype=str)

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


def build_label_free_decompositions(raw: pd.DataFrame) -> pd.DataFrame:
    _require_no_target(raw)
    _require_columns(
        raw,
        {"AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"},
    )
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    return pd.DataFrame(
        {
            "KEY_COMPONENT_ANNUITY_TO_GOODS": _safe_ratio(annuity, goods),
            "KEY_COMPONENT_CREDIT_MINUS_GOODS": credit - goods,
            "KEY_COMPONENT_ANNUITY_MINUS_CREDIT": annuity - credit,
        }
    ).replace([np.inf, -np.inf], np.nan)


def category_support(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    columns: tuple[str, ...],
    fold: str,
    minimum_count: int,
    representation: str,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for column in columns:
        counts = fit[column].astype(str).value_counts()
        fit_values = set(counts.index)
        apply_values = set(apply[column].astype(str))
        rows.append(
            {
                "representation": representation,
                "fold": fold,
                "column": column,
                "fit_rows": len(fit),
                "apply_rows": len(apply),
                "fit_category_count": len(counts),
                "fit_singleton_categories": int((counts == 1).sum()),
                "fit_categories_below_minimum": int((counts < minimum_count).sum()),
                "apply_unseen_categories": len(apply_values - fit_values),
                "minimum_fit_count": minimum_count,
            }
        )
    return rows
