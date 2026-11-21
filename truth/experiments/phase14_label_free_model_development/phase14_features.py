from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET"
EMPLOYMENT_SENTINEL = 365243.0


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0)
    result[valid] = np.log1p(values[valid])
    return result


def build_numeric_enhancements(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")

    required = {
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_FAM_MEMBERS",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")

    income = _numeric(raw, "AMT_INCOME_TOTAL")
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    family = _numeric(raw, "CNT_FAM_MEMBERS")
    age_days = -_numeric(raw, "DAYS_BIRTH")
    employed_days = -_numeric(raw, "DAYS_EMPLOYED")
    employed_days[employed_days == -EMPLOYMENT_SENTINEL] = np.nan

    external = np.column_stack(
        [_numeric(raw, f"EXT_SOURCE_{index}") for index in (1, 2, 3)]
    )
    observed = np.isfinite(external)
    count = observed.sum(axis=1)
    mean = np.divide(
        np.nansum(external, axis=1),
        count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=count > 0,
    )
    minimum = np.where(observed, external, np.inf).min(axis=1)
    maximum = np.where(observed, external, -np.inf).max(axis=1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    centered = np.where(observed, external - mean[:, None], 0.0)
    spread = np.sqrt(
        np.divide(
            np.square(centered).sum(axis=1),
            count,
            out=np.full(len(raw), np.nan, dtype=np.float64),
            where=count > 0,
        )
    )

    data: dict[str, np.ndarray] = {
        "LOG1P_AMT_INCOME_TOTAL": _safe_log1p(income),
        "LOG1P_AMT_CREDIT": _safe_log1p(credit),
        "LOG1P_AMT_ANNUITY": _safe_log1p(annuity),
        "LOG1P_AMT_GOODS_PRICE": _safe_log1p(goods),
        "LOG1P_AGE_DAYS": _safe_log1p(age_days),
        "LOG1P_EMPLOYED_DAYS_CLEAN": _safe_log1p(employed_days),
        "INCOME_PER_FAMILY_MEMBER": _safe_ratio(income, family),
        "CREDIT_PER_FAMILY_MEMBER": _safe_ratio(credit, family),
        "ANNUITY_PER_FAMILY_MEMBER": _safe_ratio(annuity, family),
        "EMPLOYMENT_TO_AGE_RATIO": _safe_ratio(employed_days, age_days),
        "EXT_SOURCE_OBSERVED_COUNT": count.astype(np.float64),
        "EXT_SOURCE_MEAN": mean,
        "EXT_SOURCE_MIN": minimum,
        "EXT_SOURCE_MAX": maximum,
        "EXT_SOURCE_RANGE": maximum - minimum,
        "EXT_SOURCE_POPULATION_SD": spread,
        "EXT_SOURCE_1_X_2": external[:, 0] * external[:, 1],
        "EXT_SOURCE_1_X_3": external[:, 0] * external[:, 2],
        "EXT_SOURCE_2_X_3": external[:, 1] * external[:, 2],
        "EXT_SOURCE_1_MINUS_2": external[:, 0] - external[:, 1],
        "EXT_SOURCE_1_MINUS_3": external[:, 0] - external[:, 2],
        "EXT_SOURCE_2_MINUS_3": external[:, 1] - external[:, 2],
        "EXT_MEAN_X_CREDIT_TO_INCOME": mean
        * _safe_ratio(credit, income),
        "EXT_MEAN_X_ANNUITY_TO_INCOME": mean
        * _safe_ratio(annuity, income),
        "EXT_RANGE_X_CREDIT_TO_GOODS": (maximum - minimum)
        * _safe_ratio(credit, goods),
        "AGE_X_EMPLOYMENT_YEARS": (age_days / 365.25)
        * (employed_days / 365.25),
        "RAW_FINANCIAL_MISSING_COUNT": raw[
            ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
        ]
        .isna()
        .sum(axis=1)
        .to_numpy(dtype=np.float64),
        "PH14_EXT_SOURCE_MISSING_COUNT": 3 - count,
    }

    document_columns = [
        column for column in raw.columns if column.startswith("FLAG_DOCUMENT_")
    ]
    if document_columns:
        document_values = raw[document_columns].apply(
            pd.to_numeric, errors="raise"
        )
        data["FLAG_DOCUMENT_COUNT"] = document_values.sum(
            axis=1, min_count=1
        ).to_numpy(dtype=np.float64)
        data["FLAG_DOCUMENT_MISSING_COUNT"] = document_values.isna().sum(
            axis=1
        ).to_numpy(dtype=np.float64)

    bureau_columns = [
        column
        for column in (
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        )
        if column in raw.columns
    ]
    if bureau_columns:
        bureau = raw[bureau_columns].apply(pd.to_numeric, errors="raise")
        data["BUREAU_REQUEST_SUM"] = bureau.sum(
            axis=1, min_count=1
        ).to_numpy(dtype=np.float64)
        data["BUREAU_REQUEST_MISSING_COUNT"] = bureau.isna().sum(
            axis=1
        ).to_numpy(dtype=np.float64)

    result = pd.DataFrame(data, index=raw.index)
    return result.replace([np.inf, -np.inf], np.nan)
