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


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _integerized(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (10**decimals))
    result = np.full(len(scaled), "MISSING", dtype=object)
    finite = np.isfinite(scaled)
    limits = np.iinfo(np.int64)
    if np.any(finite & ((scaled < limits.min) | (scaled > limits.max))):
        raise ValueError("A product-key value exceeds the historical int64 range")
    result[finite] = scaled[finite].astype(np.int64).astype(str)
    return result.astype(str)


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")


def _number(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)


def build_product_keys(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    _require_columns(
        raw,
        {"NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"},
    )
    credit = _number(raw, "AMT_CREDIT")
    annuity = _number(raw, "AMT_ANNUITY")
    goods = _number(raw, "AMT_GOODS_PRICE")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(
        dtype=str
    )

    ratio_ac = _safe_ratio(annuity, credit)
    ratio_ag = _safe_ratio(annuity, goods)
    q3, q4, q5 = (_integerized(ratio_ac, digits) for digits in (3, 4, 5))
    p4 = _integerized(_safe_ratio(annuity - FEE_OFFSET_HYPOTHESIS, credit), 4)
    p5 = _integerized(_safe_ratio(annuity - FEE_OFFSET_HYPOTHESIS, credit), 5)
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


def build_numeric_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    predictors = raw.drop(columns=[ID_COLUMN], errors="ignore")
    columns = [
        column
        for column in predictors.columns
        if pd.api.types.is_numeric_dtype(predictors[column].dtype)
    ]
    if not columns:
        raise ValueError("No numeric predictor columns were found")
    result = predictors[columns].apply(pd.to_numeric, errors="raise").copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)


def build_decomposed_features(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    _require_columns(
        raw,
        {
            "NAME_CONTRACT_TYPE",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
        },
    )
    credit = _number(raw, "AMT_CREDIT")
    annuity = _number(raw, "AMT_ANNUITY")
    goods = _number(raw, "AMT_GOODS_PRICE")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(
        dtype=str
    )
    financing_coordinate = _safe_ratio(credit, goods) - 1.0
    financing_coordinate /= FINANCING_STEP_HYPOTHESIS
    eligible = (
        (contract == "Cash loans")
        & np.isfinite(financing_coordinate)
        & (credit > goods)
        & (goods > 0)
    )
    financing_coordinate[~eligible] = np.nan
    return pd.DataFrame(
        {
            "KEY_COMPONENT_ANNUITY_TO_CREDIT": _safe_ratio(annuity, credit),
            "KEY_COMPONENT_ANNUITY_TO_GOODS": _safe_ratio(annuity, goods),
            "KEY_COMPONENT_FEE_ADJUSTED_ANNUITY_TO_CREDIT": _safe_ratio(
                annuity - FEE_OFFSET_HYPOTHESIS, credit
            ),
            "KEY_COMPONENT_FINANCING_COORDINATE": financing_coordinate,
            "KEY_COMPONENT_FINANCING_ELIGIBLE": eligible.astype(np.int8),
        }
    )


def build_selected_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    _require_columns(
        raw,
        {
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "REGION_POPULATION_RELATIVE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        },
    )
    credit = _number(raw, "AMT_CREDIT")
    annuity = _number(raw, "AMT_ANNUITY")
    population = _number(raw, "REGION_POPULATION_RELATIVE")
    ext1 = _number(raw, "EXT_SOURCE_1")
    ext2 = _number(raw, "EXT_SOURCE_2")
    ext3 = _number(raw, "EXT_SOURCE_3")
    result = pd.DataFrame(
        {
            "INTERACTION_CREDIT_X_REGION_POPULATION": credit * population,
            "INTERACTION_ANNUITY_X_REGION_POPULATION": annuity * population,
            "INTERACTION_EXT_SOURCE_1_X_2": ext1 * ext2,
            "INTERACTION_EXT_SOURCE_2_X_3": ext2 * ext3,
        }
    )
    return result.replace([np.inf, -np.inf], np.nan)


def category_support(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    columns: tuple[str, ...],
    minimum_count: int,
    fold: int,
    variant: str,
) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for column in columns:
        fit_counts = fit[column].value_counts(dropna=False)
        apply_values = set(apply[column].astype(str))
        fit_values = set(fit_counts.index.astype(str))
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "key": column,
                "fit_rows": int(len(fit)),
                "apply_rows": int(len(apply)),
                "fit_unique_categories": int(len(fit_counts)),
                "fit_singleton_categories": int((fit_counts == 1).sum()),
                "fit_rare_categories_below_minimum": int(
                    (fit_counts < minimum_count).sum()
                ),
                "fit_categories_at_or_above_minimum": int(
                    (fit_counts >= minimum_count).sum()
                ),
                "apply_unseen_categories": int(len(apply_values - fit_values)),
                "minimum_fit_count": int(minimum_count),
            }
        )
    return rows
