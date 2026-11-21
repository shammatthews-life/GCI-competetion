"""Domain-derived feature formulas for the competition dataset.

This file intentionally uses only the provided competition CSVs and derives
row-wise formulas that are explainable in credit-risk terms. The design goal
is to keep features grounded in domain logic rather than opaque model tuning.

Important constraint:
- No external data
- No manual test-score edits
- No test-label leakage
- No pooled train/test fitting for model-selection decisions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio while guarding against division by zero and missing data."""
    denom = denominator.replace(0, np.nan)
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create raw-data financial and risk features from the official columns only."""
    out = df.copy()

    # Family size and income capacity.
    out["income_per_member"] = safe_ratio(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["credit_per_member"] = safe_ratio(out["AMT_CREDIT"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["child_burden_index"] = safe_ratio(out["CNT_CHILDREN"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))

    # Debt burden and payment capacity.
    out["loan_to_income"] = safe_ratio(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_to_income"] = safe_ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["monthly_payment_to_income"] = safe_ratio(12.0 * out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["debt_service_gap"] = out["monthly_payment_to_income"] - 0.25
    out["annual_payment_per_loan"] = safe_ratio(12.0 * out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["loan_to_goods_price"] = safe_ratio(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    out["goods_to_credit_ratio"] = safe_ratio(out["AMT_GOODS_PRICE"], out["AMT_CREDIT"])
    out["credit_to_income_per_member"] = safe_ratio(out["AMT_CREDIT"], out["income_per_member"])
    out["annuity_to_income_per_member"] = safe_ratio(out["AMT_ANNUITY"], out["income_per_member"])

    # Employment and age stability.
    out["age_years"] = np.abs(out["DAYS_BIRTH"]) / 365.25
    out["employment_years"] = np.where(
        out["DAYS_EMPLOYED"] < 0,
        np.abs(out["DAYS_EMPLOYED"]) / 365.25,
        np.nan,
    )
    out["employment_to_age"] = safe_ratio(out["employment_years"], out["age_years"].replace(0, np.nan))
    out["employment_sentinel"] = (out["DAYS_EMPLOYED"] == 365243).astype(float)
    out["has_work_history"] = (out["DAYS_EMPLOYED"] < 0).astype(float)
    out["employment_gap_from_age"] = np.where(
        out["age_years"].notna(), out["employment_years"] - out["age_years"], np.nan
    )

    # External-score quality and disagreement.
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_values = out[ext_cols].copy()
    out["ext_score_mean"] = ext_values.mean(axis=1, skipna=True)
    out["ext_score_std"] = ext_values.std(axis=1, skipna=True)
    out["ext_score_gap"] = ext_values.max(axis=1, skipna=True) - ext_values.min(axis=1, skipna=True)
    out["ext_score_low"] = ext_values.min(axis=1, skipna=True)
    out["ext_score_dispersion_ratio"] = safe_ratio(out["ext_score_std"], out["ext_score_mean"].replace(0, np.nan))

    # Bureau inquiry load and request pressure.
    bureau_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    out["bureau_inquiry_total"] = out[bureau_cols].sum(axis=1)
    out["bureau_inquiry_recent"] = out["AMT_REQ_CREDIT_BUREAU_HOUR"] + out["AMT_REQ_CREDIT_BUREAU_MON"]
    out["bureau_inquiry_older"] = out["AMT_REQ_CREDIT_BUREAU_QRT"] + out["AMT_REQ_CREDIT_BUREAU_YEAR"]
    out["bureau_inquiry_per_income"] = safe_ratio(out["bureau_inquiry_total"], out["AMT_INCOME_TOTAL"])
    out["bureau_recent_ratio"] = safe_ratio(out["bureau_inquiry_recent"], out["bureau_inquiry_total"].replace(0, np.nan))

    # Region and household risk exposure.
    out["regional_risk_pressure"] = out["REGION_RATING_CLIENT"] / out["REGION_POPULATION_RELATIVE"]
    out["family_wage_capacity"] = safe_ratio(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["family_burden_total"] = out["AMT_ANNUITY"] * out["CNT_FAM_MEMBERS"]
    out["family_payment_to_income"] = safe_ratio(out["family_burden_total"], out["AMT_INCOME_TOTAL"])

    return out


def summarise_features(train_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Return a compact AUC summary to compare how each engineered signal relates to TARGET."""
    rows = []
    for feature in feature_names:
        series = train_df[feature].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature].median())
        auc = roc_auc_score(train_df["TARGET"], series)
        rows.append({"feature": feature, "auc": float(auc)})
    return pd.DataFrame(rows).sort_values("auc").reset_index(drop=True)


if __name__ == "__main__":
    train = pd.read_csv(r"D:\gci\inpuy\train.csv")
    test = pd.read_csv(r"D:\gci\inpuy\test.csv")

    train_engineered = add_domain_features(train)
    test_engineered = add_domain_features(test)

    feature_names = [
        "loan_to_income",
        "annuity_to_income",
        "monthly_payment_to_income",
        "debt_service_gap",
        "annual_payment_per_loan",
        "loan_to_goods_price",
        "goods_to_credit_ratio",
        "credit_to_income_per_member",
        "annuity_to_income_per_member",
        "income_per_member",
        "credit_per_member",
        "child_burden_index",
        "employment_to_age",
        "employment_sentinel",
        "ext_score_mean",
        "ext_score_std",
        "ext_score_gap",
        "ext_score_dispersion_ratio",
        "bureau_inquiry_total",
        "bureau_inquiry_recent",
        "bureau_inquiry_older",
        "bureau_inquiry_per_income",
        "bureau_recent_ratio",
        "regional_risk_pressure",
        "family_payment_to_income",
    ]

    summary = summarise_features(train_engineered, feature_names)
    print(summary.to_string(index=False))
    print("\nTrain columns added:", [c for c in feature_names if c in train_engineered.columns])
