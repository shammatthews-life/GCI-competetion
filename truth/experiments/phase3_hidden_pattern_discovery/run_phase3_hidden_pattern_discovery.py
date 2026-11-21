#!/usr/bin/env python3
"""Phase 3 milestone 1: verify references and probe loan-product structure."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(r"D:\gci\truth")
DATA_DIR = Path(r"D:\gci\inpuy")
OUT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DICT_PATH = (
    Path(r"D:\gci\rules and info")
    / "04. Competition (due Nov 20)"
    / "Competition"
    / "HomeCredit_columns_description.xlsx"
)
BASELINE_DIR = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE2B_DIR = ROOT / "experiments" / "phase2b_feature_intelligence"
REUSE_REGISTER = ROOT / "reports" / "colab_v17_audit" / "V17_FEATURE_REUSE_REGISTER.csv"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 42
FEE_GRID = np.arange(0.0, 250.0001, 0.5)
STEP_COARSE_GRID = np.arange(0.005, 0.0300001, 0.00005)
STEP_FINE_WIDTH = 0.00020
STEP_FINE_SIZE = 401
RATIO_KEY_DECIMALS = 5

spec = importlib.util.spec_from_file_location("phase3_baseline", BASELINE_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load preserved baseline: {BASELINE_SOURCE}")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    good = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[good] = a[good] / b[good]
    return out


def finite_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df:
        return np.full(len(df), np.nan, dtype=np.float64)
    return df[name].to_numpy(dtype=np.float64, na_value=np.nan)


def fit_fee_and_step(raw_fit: pd.DataFrame) -> dict[str, float]:
    """Estimate unsupervised decoder constants on fit rows only."""
    credit = finite_column(raw_fit, "AMT_CREDIT")
    annuity = finite_column(raw_fit, "AMT_ANNUITY")
    goods = finite_column(raw_fit, "AMT_GOODS_PRICE")
    cash = raw_fit["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()

    fee_mask = cash & np.isfinite(credit) & np.isfinite(annuity) & (credit > 0)
    fee_credit = credit[fee_mask]
    fee_annuity = annuity[fee_mask]
    if len(fee_credit) < 100:
        fee = 0.0
        fee_support = 0.0
    else:
        if len(fee_credit) > 30000:
            sample_positions = np.linspace(0, len(fee_credit) - 1, 30000, dtype=np.int64)
            fee_credit = fee_credit[sample_positions]
            fee_annuity = fee_annuity[sample_positions]
        best_fee = 0.0
        best_support = -1.0
        for candidate in FEE_GRID:
            keys = np.round((fee_annuity - candidate) / fee_credit, RATIO_KEY_DECIMALS)
            _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
            support = float(np.mean(counts[inverse] >= 2))
            if support > best_support:
                best_fee, best_support = float(candidate), support
        fee, fee_support = best_fee, best_support

    step_mask = (
        cash
        & np.isfinite(credit)
        & np.isfinite(goods)
        & (credit > goods)
        & (goods > 0)
    )
    x = credit[step_mask] / goods[step_mask] - 1.0
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 100:
        step, step_support, step_mae = np.nan, np.nan, np.nan
    else:
        sample = x
        if len(sample) > 40000:
            positions = np.linspace(0, len(sample) - 1, 40000, dtype=np.int64)
            sample = sample[positions]
        best_delta, best_support, best_mae = np.nan, -1.0, np.inf
        for delta in STEP_COARSE_GRID:
            lattice = sample / delta
            residual = np.abs(lattice - np.rint(lattice))
            support = float(np.mean(residual < 0.1))
            mae = float(np.mean(residual))
            if (support > best_support) or (support == best_support and mae < best_mae):
                best_delta, best_support, best_mae = float(delta), support, mae
        low = max(0.004, best_delta - STEP_FINE_WIDTH)
        high = min(0.031, best_delta + STEP_FINE_WIDTH)
        for delta in np.linspace(low, high, STEP_FINE_SIZE):
            lattice = sample / delta
            residual = np.abs(lattice - np.rint(lattice))
            support = float(np.mean(residual < 0.1))
            mae = float(np.mean(residual))
            if (support > best_support) or (support == best_support and mae < best_mae):
                best_delta, best_support, best_mae = float(delta), support, mae
        step, step_support, step_mae = best_delta, best_support, best_mae
    return {
        "fee_estimate_fit_only": fee,
        "fee_fit_repeat_support": fee_support,
        "step_estimate_fit_only": step,
        "step_fit_lattice_support": step_support,
        "step_fit_lattice_mae": step_mae,
        "n_cash_fee_fit": int(fee_mask.sum()),
        "n_cash_step_fit": int(step_mask.sum()),
    }


def fit_product_maps(raw_fit: pd.DataFrame, params: dict[str, float]) -> dict[str, dict]:
    credit = finite_column(raw_fit, "AMT_CREDIT")
    annuity = finite_column(raw_fit, "AMT_ANNUITY")
    q = safe_ratio(annuity, credit)
    p = safe_ratio(annuity - params["fee_estimate_fit_only"], credit)
    cash = raw_fit["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    step = params["step_estimate_fit_only"]
    goods = finite_column(raw_fit, "AMT_GOODS_PRICE")
    k = safe_ratio(credit / goods - 1.0, np.full(len(raw_fit), step)) if np.isfinite(step) else np.full(len(raw_fit), np.nan)
    return {
        "Q3": pd.Series(np.round(q, 3)[cash]).value_counts(dropna=True).to_dict(),
        "Q4": pd.Series(np.round(q, 4)[cash]).value_counts(dropna=True).to_dict(),
        "Q5": pd.Series(np.round(q, 5)[cash]).value_counts(dropna=True).to_dict(),
        "P3": pd.Series(np.round(p, 3)[cash]).value_counts(dropna=True).to_dict(),
        "P4": pd.Series(np.round(p, 4)[cash]).value_counts(dropna=True).to_dict(),
        "P5": pd.Series(np.round(p, 5)[cash]).value_counts(dropna=True).to_dict(),
        "K_INT": pd.Series(np.rint(k)[cash]).value_counts(dropna=True).to_dict(),
    }


def make_product_features(
    raw: pd.DataFrame,
    params: dict[str, float],
    maps: dict[str, dict],
) -> pd.DataFrame:
    credit = finite_column(raw, "AMT_CREDIT")
    annuity = finite_column(raw, "AMT_ANNUITY")
    goods = finite_column(raw, "AMT_GOODS_PRICE")
    income = finite_column(raw, "AMT_INCOME_TOTAL")
    family = finite_column(raw, "CNT_FAM_MEMBERS")
    age_days = finite_column(raw, "DAYS_BIRTH")
    cash = raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    q = safe_ratio(annuity, credit)
    fee = params["fee_estimate_fit_only"]
    p = safe_ratio(annuity - fee, credit)
    delta = params["step_estimate_fit_only"]
    raw_k = safe_ratio(credit / goods - 1.0, np.full(len(raw), delta)) if np.isfinite(delta) else np.full(len(raw), np.nan)
    raw_k = np.where(cash, raw_k, np.nan)
    k_resid = np.abs(raw_k - np.rint(raw_k))
    k_valid = cash & np.isfinite(raw_k) & (credit > goods) & (k_resid < 0.1)
    ext = np.column_stack(
        [finite_column(raw, f"EXT_SOURCE_{i}") for i in (1, 2, 3)]
    )
    ext_count = np.isfinite(ext).sum(axis=1)
    ext_mean = np.divide(
        np.where(np.isfinite(ext), ext, 0.0).sum(axis=1),
        ext_count,
        out=np.full(len(raw), np.nan),
        where=ext_count > 0,
    )
    ext_variance = np.divide(
        np.where(np.isfinite(ext), (ext - ext_mean[:, None]) ** 2, 0.0).sum(axis=1),
        ext_count - 1,
        out=np.full(len(raw), np.nan),
        where=ext_count > 1,
    )
    ext_std = np.sqrt(ext_variance)
    gap = credit - goods
    finance_share = safe_ratio(gap, credit)
    gap_goods = safe_ratio(gap, goods)
    q3, q4, q5 = np.round(q, 3), np.round(q, 4), np.round(q, 5)
    p3, p4, p5 = np.round(p, 3), np.round(p, 4), np.round(p, 5)

    def mapped_frequency(values: np.ndarray, key: str) -> np.ndarray:
        mapping = maps[key]
        return np.fromiter(
            (float(mapping.get(value, 0)) if np.isfinite(value) else np.nan for value in values),
            dtype=np.float64,
            count=len(values),
        )

    fq5 = mapped_frequency(q5, "Q5")
    fp5 = mapped_frequency(p5, "P5")
    fq5 = np.where(cash, fq5, np.nan)
    fp5 = np.where(cash, fp5, np.nan)
    fee_flag = cash & np.isfinite(fq5) & np.isfinite(fp5) & (fp5 > fq5)
    k_rounded = np.rint(raw_k)
    k_support = mapped_frequency(k_rounded, "K_INT")
    annuity_fee_resid = annuity - (fee + credit * np.round(p, RATIO_KEY_DECIMALS))

    features = {
        "FIN_CREDIT_MINUS_GOODS": gap,
        "FIN_GOODS_MINUS_CREDIT": -gap,
        "FIN_EXCESS_OVER_GOODS": gap_goods,
        "FIN_EXCESS_SHARE_OF_CREDIT": finance_share,
        "FIN_SYMMETRIC_GAP": safe_ratio(gap, np.abs(credit) + np.abs(goods)),
        "FIN_CREDIT_GOODS_LOG_RATIO": np.log1p(np.maximum(credit, 0)) - np.log1p(np.maximum(goods, 0)),
        "FIN_GOODS_TO_CREDIT": safe_ratio(goods, credit),
        "FIN_ANNUITY_TO_GOODS": safe_ratio(annuity, goods),
        "FIN_ANNUITY_TO_FINANCING_GAP": safe_ratio(annuity, gap),
        "FIN_GAP_TO_ANNUITY": safe_ratio(gap, annuity),
        "FIN_GAP_TO_INCOME": safe_ratio(gap, income),
        "FIN_GAP_TO_CREDIT": finance_share,
        "FIN_CREDIT_TO_INCOME": safe_ratio(credit, income),
        "FIN_GOODS_TO_INCOME": safe_ratio(goods, income),
        "FIN_ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "FIN_ANNUITY_TO_CREDIT": q,
        "FIN_CREDIT_TO_ANNUITY": safe_ratio(credit, annuity),
        "FIN_ANNUITY_CREDIT_LOG_RATIO": np.log1p(np.maximum(annuity, 0)) - np.log1p(np.maximum(credit, 0)),
        "FIN_CREDIT_PER_FAMILY": safe_ratio(credit, family),
        "FIN_GAP_PER_FAMILY": safe_ratio(gap, family),
        "FIN_ANNUITY_PER_FAMILY": safe_ratio(annuity, family),
        "FIN_CREDIT_PER_AGE_YEAR": safe_ratio(credit, -age_days / 365.25),
        "FIN_ANNUITY_PER_AGE_YEAR": safe_ratio(annuity, -age_days / 365.25),
        "PROD_Q3": q3,
        "PROD_Q4": q4,
        "PROD_Q5": q5,
        "PROD_P3": p3,
        "PROD_P4": p4,
        "PROD_P5": p5,
        "PROD_PF4_FLAG": fee_flag.astype(np.float64),
        "PROD_FEE_ADJUSTED_ANNUITY": annuity - fee,
        "PROD_FEE_RECONSTRUCTION_RESIDUAL": annuity_fee_resid,
        "PROD_Q5_FIT_FREQUENCY_LOG": np.log1p(fq5),
        "PROD_P5_FIT_FREQUENCY_LOG": np.log1p(fp5),
        "PROD_FEE_FLAG": fee_flag.astype(np.float64),
        "PROD_K_DECIMAL": raw_k,
        "PROD_K_NEAREST_INTEGER": np.where(k_valid, k_rounded, np.nan),
        "PROD_K_LATTICE_RESIDUAL": k_resid,
        "PROD_K_VALID": k_valid.astype(np.float64),
        "PROD_K_FIT_FREQUENCY_LOG": np.log1p(k_support),
        "PROD_EXT_MEAN_X_FINANCE_SHARE": ext_mean * finance_share,
        "PROD_EXT_DISPERSION_X_FINANCE_SHARE": ext_std * finance_share,
        "PROD_EXT_MEAN_X_K": ext_mean * raw_k,
        "PROD_EXT_DISPERSION_X_K": ext_std * raw_k,
        "PROD_CREDIT_INCOME_X_K": safe_ratio(credit, income) * raw_k,
        "PROD_ANNUITY_INCOME_X_K": safe_ratio(annuity, income) * raw_k,
        "PROD_CREDIT_GOODS_X_EXT_MEAN": safe_ratio(credit, goods) * ext_mean,
        "PROD_ANNUITY_CREDIT_X_EXT_MEAN": q * ext_mean,
        "PROD_FINANCE_SHARE_X_AGE": finance_share * (-age_days / 365.25),
        "PROD_FINANCE_SHARE_X_FAMILY": finance_share * family,
        "PROD_ANNUITY_INCOME_X_FAMILY": safe_ratio(annuity, income) * family,
    }
    frame = pd.DataFrame(features, index=raw.index)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.reset_index(drop=True)


FEATURE_DEFINITIONS = [
    ("FIN_CREDIT_MINUS_GOODS", "AMT_CREDIT - AMT_GOODS_PRICE", "Financing amount above stated goods price; signed difference."),
    ("FIN_GOODS_MINUS_CREDIT", "AMT_GOODS_PRICE - AMT_CREDIT", "Reverse signed difference; exact sign duplicate."),
    ("FIN_EXCESS_OVER_GOODS", "(AMT_CREDIT - AMT_GOODS_PRICE) / AMT_GOODS_PRICE", "Credit premium relative to goods price."),
    ("FIN_EXCESS_SHARE_OF_CREDIT", "(AMT_CREDIT - AMT_GOODS_PRICE) / AMT_CREDIT", "Share of credit exceeding stated goods price."),
    ("FIN_SYMMETRIC_GAP", "(CREDIT - GOODS) / (abs(CREDIT) + abs(GOODS))", "Bounded scale-free financing difference."),
    ("FIN_CREDIT_GOODS_LOG_RATIO", "log1p(CREDIT) - log1p(GOODS)", "Log-scale financing gap; guarded to nonnegative values."),
    ("FIN_GOODS_TO_CREDIT", "GOODS / CREDIT", "Goods coverage relative to credit; reciprocal relation to credit/goods."),
    ("FIN_ANNUITY_TO_GOODS", "ANNUITY / GOODS", "Payment intensity relative to asset price."),
    ("FIN_ANNUITY_TO_FINANCING_GAP", "ANNUITY / (CREDIT - GOODS)", "Payment amount relative to financed excess; undefined for nonpositive gap."),
    ("FIN_GAP_TO_ANNUITY", "(CREDIT - GOODS) / ANNUITY", "Financing excess per unit annuity."),
    ("FIN_GAP_TO_INCOME", "(CREDIT - GOODS) / INCOME", "Financing excess relative to income."),
    ("FIN_GAP_TO_CREDIT", "(CREDIT - GOODS) / CREDIT", "Same as excess share; duplicate alias."),
    ("FIN_CREDIT_TO_INCOME", "CREDIT / INCOME", "Loan size relative to income; existing baseline duplicate."),
    ("FIN_GOODS_TO_INCOME", "GOODS / INCOME", "Asset price relative to income."),
    ("FIN_ANNUITY_TO_INCOME", "ANNUITY / INCOME", "Annuity relative to income; existing baseline duplicate."),
    ("FIN_ANNUITY_TO_CREDIT", "ANNUITY / CREDIT", "Annuity intensity; existing baseline duplicate."),
    ("FIN_CREDIT_TO_ANNUITY", "CREDIT / ANNUITY", "Payment-multiple proxy; existing baseline duplicate and not a stated term."),
    ("FIN_ANNUITY_CREDIT_LOG_RATIO", "log1p(ANNUITY) - log1p(CREDIT)", "Log-scale payment/credit relation."),
    ("FIN_CREDIT_PER_FAMILY", "CREDIT / CNT_FAM_MEMBERS", "Mechanical per-member allocation; assumption not economically literal."),
    ("FIN_GAP_PER_FAMILY", "(CREDIT - GOODS) / CNT_FAM_MEMBERS", "Mechanical per-member allocation of financing gap."),
    ("FIN_ANNUITY_PER_FAMILY", "ANNUITY / CNT_FAM_MEMBERS", "Mechanical equal-responsibility assumption; exploratory only."),
    ("FIN_CREDIT_PER_AGE_YEAR", "CREDIT / (-DAYS_BIRTH/365.25)", "Age-scaled loan amount; unit has no direct lending interpretation."),
    ("FIN_ANNUITY_PER_AGE_YEAR", "ANNUITY / (-DAYS_BIRTH/365.25)", "Age-scaled annuity; unit has no direct lending interpretation."),
    ("PROD_Q3", "round(ANNUITY/CREDIT, 3)", "v17 Q3-style rounded repayment ratio."),
    ("PROD_Q4", "round(ANNUITY/CREDIT, 4)", "v17 Q4-style rounded repayment ratio."),
    ("PROD_Q5", "round(ANNUITY/CREDIT, 5)", "v17 Q5-style rounded repayment ratio."),
    ("PROD_P3", "round((ANNUITY - F_fit)/CREDIT, 3)", "Fee-adjusted ratio; F_fit estimated on training fold only."),
    ("PROD_P4", "round((ANNUITY - F_fit)/CREDIT, 4)", "Fee-adjusted ratio; F_fit estimated on training fold only."),
    ("PROD_P5", "round((ANNUITY - F_fit)/CREDIT, 5)", "Fee-adjusted ratio; F_fit estimated on training fold only."),
    ("PROD_PF4_FLAG", "fit-only repeat-frequency(P4) > fit-only repeat-frequency(Q5)", "v17-style relative support flag, recomputed from fit rows only."),
    ("PROD_FEE_ADJUSTED_ANNUITY", "ANNUITY - F_fit", "Annuity after fold-estimated candidate fixed offset."),
    ("PROD_FEE_RECONSTRUCTION_RESIDUAL", "A - (F_fit + CREDIT*round((A-F_fit)/CREDIT,5))", "Absolute reconstruction residual from rounded fee-adjusted ratio."),
    ("PROD_Q5_FIT_FREQUENCY_LOG", "log1p(count_fit(round(A/C,5)))", "Training-fold-only product-key frequency."),
    ("PROD_P5_FIT_FREQUENCY_LOG", "log1p(count_fit(round((A-F_fit)/C,5)))", "Training-fold-only fee-adjusted product-key frequency."),
    ("PROD_FEE_FLAG", "fit-only repeat-frequency(P5) > fit-only repeat-frequency(Q5)", "Candidate fixed-fee family indicator."),
    ("PROD_K_DECIMAL", "(CREDIT/GOODS - 1)/delta_fit", "v17 K_DEC-style financing lattice coordinate; delta estimated on fit only."),
    ("PROD_K_NEAREST_INTEGER", "round(PROD_K_DECIMAL) if cash and lattice residual < 0.1", "Candidate integer financing tier; validity gate applied."),
    ("PROD_K_LATTICE_RESIDUAL", "abs(PROD_K_DECIMAL - round(PROD_K_DECIMAL))", "Distance to inferred financing lattice."),
    ("PROD_K_VALID", "cash & CREDIT>GOODS & lattice residual<0.1", "Indicator for rows consistent with inferred financing lattice."),
    ("PROD_K_FIT_FREQUENCY_LOG", "log1p(count_fit(round(PROD_K_DECIMAL)))", "Training-fold-only integer-tier frequency."),
    ("PROD_EXT_MEAN_X_FINANCE_SHARE", "EXT_MEAN * (CREDIT-GOODS)/CREDIT", "External risk score conditional on financing intensity."),
    ("PROD_EXT_DISPERSION_X_FINANCE_SHARE", "EXT_STD * (CREDIT-GOODS)/CREDIT", "External-score disagreement conditional on financing intensity."),
    ("PROD_EXT_MEAN_X_K", "EXT_MEAN * PROD_K_DECIMAL", "External risk score conditional on inferred financing tier."),
    ("PROD_EXT_DISPERSION_X_K", "EXT_STD * PROD_K_DECIMAL", "Score disagreement conditional on inferred financing tier."),
    ("PROD_CREDIT_INCOME_X_K", "(CREDIT/INCOME) * PROD_K_DECIMAL", "Income burden conditional on inferred financing tier."),
    ("PROD_ANNUITY_INCOME_X_K", "(ANNUITY/INCOME) * PROD_K_DECIMAL", "Payment burden conditional on inferred financing tier."),
    ("PROD_CREDIT_GOODS_X_EXT_MEAN", "(CREDIT/GOODS) * EXT_MEAN", "External score conditional on credit/goods ratio."),
    ("PROD_ANNUITY_CREDIT_X_EXT_MEAN", "(ANNUITY/CREDIT) * EXT_MEAN", "External score conditional on annuity/credit ratio."),
    ("PROD_FINANCE_SHARE_X_AGE", "finance share * age years", "Financing intensity by age."),
    ("PROD_FINANCE_SHARE_X_FAMILY", "finance share * CNT_FAM_MEMBERS", "Financing intensity by household size; not per-member income."),
    ("PROD_ANNUITY_INCOME_X_FAMILY", "(ANNUITY/INCOME) * CNT_FAM_MEMBERS", "Repayment burden interaction; not an equal-share assumption."),
]

MODEL_FEATURES = [
    "FIN_EXCESS_OVER_GOODS",
    "FIN_EXCESS_SHARE_OF_CREDIT",
    "FIN_SYMMETRIC_GAP",
    "FIN_CREDIT_GOODS_LOG_RATIO",
    "FIN_ANNUITY_TO_GOODS",
    "FIN_ANNUITY_TO_FINANCING_GAP",
    "FIN_GAP_TO_ANNUITY",
    "FIN_GAP_TO_INCOME",
    "FIN_GOODS_TO_INCOME",
    "FIN_ANNUITY_CREDIT_LOG_RATIO",
    "FIN_GAP_PER_FAMILY",
    "PROD_Q3",
    "PROD_Q4",
    "PROD_Q5",
    "PROD_P3",
    "PROD_P4",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_ADJUSTED_ANNUITY",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_Q5_FIT_FREQUENCY_LOG",
    "PROD_P5_FIT_FREQUENCY_LOG",
    "PROD_K_DECIMAL",
    "PROD_K_NEAREST_INTEGER",
    "PROD_K_LATTICE_RESIDUAL",
    "PROD_K_VALID",
    "PROD_K_FIT_FREQUENCY_LOG",
    "PROD_EXT_MEAN_X_FINANCE_SHARE",
    "PROD_EXT_DISPERSION_X_FINANCE_SHARE",
    "PROD_EXT_MEAN_X_K",
    "PROD_EXT_DISPERSION_X_K",
    "PROD_CREDIT_INCOME_X_K",
    "PROD_ANNUITY_INCOME_X_K",
    "PROD_CREDIT_GOODS_X_EXT_MEAN",
    "PROD_ANNUITY_CREDIT_X_EXT_MEAN",
    "PROD_FINANCE_SHARE_X_AGE",
    "PROD_FINANCE_SHARE_X_FAMILY",
    "PROD_ANNUITY_INCOME_X_FAMILY",
]


def write_csv(name: str, rows) -> None:
    pd.DataFrame(rows).to_csv(OUT / name, index=False)


def metric_row(variant: str, target: np.ndarray, predictions: np.ndarray, fold_ids: np.ndarray) -> dict:
    fold_auc = [
        roc_auc_score(target[fold_ids == fold], predictions[fold_ids == fold])
        for fold in sorted(np.unique(fold_ids))
    ]
    return {
        "variant": variant,
        "pooled_oof_auc": roc_auc_score(target, predictions),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "std_fold_auc": float(np.std(fold_auc, ddof=1)),
        **{f"fold_{i+1}_auc": value for i, value in enumerate(fold_auc)},
    }


def inspect_data(train: pd.DataFrame, test: pd.DataFrame, dictionary: pd.DataFrame):
    desc = dict(zip(dictionary["Row"].astype(str), dictionary["Description"].astype(str)))
    rows = []
    category_associations = []
    numeric = train.select_dtypes(include=[np.number])
    for column in train.columns:
        s = train[column]
        numeric_col = pd.api.types.is_numeric_dtype(s)
        if column != TARGET and not numeric_col:
            grouped = (
                train.groupby(column, dropna=False, observed=True)[TARGET]
                .agg(["size", "mean"])
                .sort_values("size", ascending=False)
            )
            for value, stat in grouped.iterrows():
                category_associations.append(
                    {
                        "column": column,
                        "category": value,
                        "count": int(stat["size"]),
                        "default_rate": float(stat["mean"]),
                        "analysis_only_not_a_feature": True,
                    }
                )
        class0 = train.loc[train[TARGET] == 0, column] if column != TARGET else None
        class1 = train.loc[train[TARGET] == 1, column] if column != TARGET else None
        row = {
            "column": column,
            "official_meaning": desc.get(column, "Official dictionary entry not found"),
            "dtype": str(s.dtype),
            "train_rows": len(s),
            "test_rows": len(test),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean()),
            "unique_count": int(s.nunique(dropna=True)),
            "cardinality_pct": float(s.nunique(dropna=True) / max(len(s), 1)),
            "zero_pct": float((s == 0).mean()) if numeric_col else np.nan,
            "min": float(s.min()) if numeric_col and s.notna().any() else np.nan,
            "q01": float(s.quantile(0.01)) if numeric_col and s.notna().any() else np.nan,
            "q05": float(s.quantile(0.05)) if numeric_col and s.notna().any() else np.nan,
            "q25": float(s.quantile(0.25)) if numeric_col and s.notna().any() else np.nan,
            "median": float(s.median()) if numeric_col and s.notna().any() else np.nan,
            "mean": float(s.mean()) if numeric_col and s.notna().any() else np.nan,
            "std": float(s.std()) if numeric_col and s.notna().any() else np.nan,
            "q75": float(s.quantile(0.75)) if numeric_col and s.notna().any() else np.nan,
            "q95": float(s.quantile(0.95)) if numeric_col and s.notna().any() else np.nan,
            "q99": float(s.quantile(0.99)) if numeric_col and s.notna().any() else np.nan,
            "max": float(s.max()) if numeric_col and s.notna().any() else np.nan,
            "skew": float(s.skew()) if numeric_col and s.notna().sum() > 2 else np.nan,
            "outlier_pct_iqr": np.nan,
            "target_default_rate": float(train.loc[s.notna(), TARGET].mean()) if column != TARGET and TARGET in train else np.nan,
            "target_default_rate_when_missing": float(train.loc[s.isna(), TARGET].mean()) if column != TARGET and TARGET in train and s.isna().any() else np.nan,
            "target_default_rate_when_present": float(train.loc[s.notna(), TARGET].mean()) if column != TARGET and TARGET in train and s.notna().any() else np.nan,
            "sentinel_365243_count": int((s == 365243).sum()) if numeric_col else 0,
            "negative_count": int((s < 0).sum()) if numeric_col else 0,
            "target0_mean": float(class0.mean()) if numeric_col and class0 is not None and class0.notna().any() else np.nan,
            "target1_mean": float(class1.mean()) if numeric_col and class1 is not None and class1.notna().any() else np.nan,
            "target0_median": float(class0.median()) if numeric_col and class0 is not None and class0.notna().any() else np.nan,
            "target1_median": float(class1.median()) if numeric_col and class1 is not None and class1.notna().any() else np.nan,
        }
        if numeric_col and s.notna().any():
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                row["outlier_pct_iqr"] = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "ALL_COLUMN_STATISTICS.csv", index=False)
    pd.DataFrame(category_associations).to_csv(OUT / "COLUMN_TARGET_ASSOCIATIONS.csv", index=False)

    corr = numeric.drop(columns=[TARGET], errors="ignore").corr(method="spearman")
    deps = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if np.isfinite(value) and abs(value) >= 0.80:
                deps.append(
                    {
                        "left": left,
                        "right": right,
                        "spearman_correlation": value,
                        "absolute_correlation": abs(value),
                        "interpretation": "Strong monotone association; not proof of mathematical dependency.",
                    }
                )
    pd.DataFrame(deps).sort_values("absolute_correlation", ascending=False).to_csv(
        OUT / "COLUMN_DEPENDENCY_REPORT.csv", index=False
    )
    return rows, deps


def create_reference_verification(train: pd.DataFrame):
    targets = train[[ID, TARGET]].sort_values(ID).reset_index(drop=True)
    predictions = {
        "raw_baseline": BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv",
        "existing_domain_formulas": BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv",
        "peer_relative_features": PHASE2B_DIR / "OOF_PREDICTIONS.csv",
        "external_score_interactions": PHASE2B_DIR / "OOF_PREDICTIONS.csv",
        "combined_phase2b": PHASE2B_DIR / "OOF_PREDICTIONS.csv",
    }
    columns = {
        "raw_baseline": "OOF_RAW_BASELINE",
        "existing_domain_formulas": "OOF_PLUS_FORMULAS",
        "peer_relative_features": "OOF_peer_deviations",
        "external_score_interactions": "OOF_external_interactions",
        "combined_phase2b": "OOF_combined_peer_and_interactions",
    }
    expected = {
        "raw_baseline": 0.7606584505947516,
        "existing_domain_formulas": 0.7603717800983119,
        "peer_relative_features": 0.7606927027435864,
        "external_score_interactions": 0.7600584384947837,
        "combined_phase2b": 0.7599088653226742,
    }
    fold_path = BASELINE_DIR / "FOLD_ASSIGNMENTS.csv"
    fold_df = pd.read_csv(fold_path).sort_values(ID).reset_index(drop=True)
    if not fold_df[ID].equals(targets[ID]):
        raise ValueError("Saved baseline fold IDs do not align with training IDs.")
    fold_ids = fold_df["fold"].to_numpy()
    rows = []
    for variant, path in predictions.items():
        frame = pd.read_csv(path).sort_values(ID).reset_index(drop=True)
        if not frame[ID].equals(targets[ID]):
            raise ValueError(f"{variant}: OOF IDs do not align with train IDs.")
        if not frame[TARGET].equals(targets[TARGET]):
            raise ValueError(f"{variant}: saved OOF labels do not match train TARGET.")
        score = roc_auc_score(targets[TARGET], frame[columns[variant]])
        row = metric_row(variant, targets[TARGET].to_numpy(), frame[columns[variant]].to_numpy(), fold_ids)
        row["claimed_historical_auc"] = expected[variant]
        row["absolute_difference_from_claim"] = abs(score - expected[variant])
        row["reproduced"] = bool(abs(score - expected[variant]) < 1e-10)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "REFERENCE_METRICS.csv", index=False)
    return rows, fold_ids


def formula_rows():
    duplicate = {
        "FIN_GOODS_MINUS_CREDIT": "Exact sign reversal of FIN_CREDIT_MINUS_GOODS.",
        "FIN_GAP_TO_CREDIT": "Exact duplicate of FIN_EXCESS_SHARE_OF_CREDIT.",
        "FIN_CREDIT_TO_INCOME": "Existing baseline CREDIT_TO_INCOME.",
        "FIN_ANNUITY_TO_INCOME": "Existing baseline ANNUITY_TO_INCOME.",
        "FIN_ANNUITY_TO_CREDIT": "Existing baseline ANNUITY_TO_CREDIT.",
        "FIN_CREDIT_TO_ANNUITY": "Existing baseline CREDIT_TERM; not a contractual term.",
        "FIN_GOODS_TO_CREDIT": "Reciprocal of baseline CREDIT_TO_GOODS where positive.",
    }
    return [
        {
            "feature_id": f"P3_{i:02d}",
            "feature_name": name,
            "formula": formula,
            "economic_or_statistical_meaning": meaning,
            "source_columns": "Supplied train predictors only",
            "assumptions_and_risks": (
                "Fit-only decoder parameters where applicable; safe division returns missing for zero/missing denominators. "
                "Annuity period is not specified; no annual/monthly claim is made."
            ),
            "v17_related": name.startswith("PROD_"),
            "duplicate_or_overlap": duplicate.get(name, ""),
            "implementation_status": "Implemented in candidate builder",
            "validation_status": "Fold-safe diagnostics and one family CV run; not independently confirmed",
        }
        for i, (name, formula, meaning) in enumerate(FEATURE_DEFINITIONS, 1)
    ]


def main():
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    dictionary = pd.read_excel(DICT_PATH)
    if TARGET not in train or TARGET in test:
        raise ValueError("Train/test target schema is not valid.")
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train and test predictor schemas differ.")
    if train[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError("Duplicate IDs found in supplied data.")

    ref_rows, fold_ids = create_reference_verification(train)
    if not all(row["reproduced"] for row in ref_rows):
        raise ValueError("One or more historical OOF references did not reproduce.")
    with REUSE_REGISTER.open("r", encoding="utf-8-sig", newline="") as stream:
        reuse_entries = sum(1 for _ in stream) - 1
    if reuse_entries != 23:
        raise ValueError(
            f"Expected 23 physical data rows in the v17 reuse register, got {reuse_entries}."
        )

    stats_rows, deps = inspect_data(train, test, dictionary)
    train_predictors = train.drop(columns=[TARGET]).reset_index(drop=True)
    labels = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    baseline_oof = pd.read_csv(BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv").set_index(ID)
    baseline_oof = baseline_oof.loc[ids, "OOF_RAW_BASELINE"].to_numpy()
    candidate_oof = np.full(len(train), np.nan, dtype=np.float64)
    candidate_features_oof = None
    fold_model_rows = []
    product_stability_rows = []
    product_key_distribution_rows = []
    reconstruction_rows = []
    fold_params = []

    lgb_config = baseline.MODEL_CONFIG["lightgbm"]
    model_params = {
        key: value
        for key, value in lgb_config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    base_fold_map = pd.read_csv(BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(ID)["fold"]
    current_fold_ids = base_fold_map.loc[ids].to_numpy()
    if not np.array_equal(current_fold_ids, fold_ids):
        raise ValueError("Fold mapping mismatch after ordering by input IDs.")

    for fold in sorted(np.unique(fold_ids)):
        outer_valid = fold_ids == fold
        outer_fit = ~outer_valid
        fit_positions = np.flatnonzero(outer_fit)
        valid_positions = np.flatnonzero(outer_valid)
        y_fit = labels[outer_fit]
        inner_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=20261102 + int(fold),
        )
        inner_fit_local, inner_stop_local = next(
            inner_split.split(np.zeros(len(y_fit)), y_fit)
        )
        inner_fit_pos = fit_positions[inner_fit_local]
        inner_stop_pos = fit_positions[inner_stop_local]
        inner_params = fit_fee_and_step(train_predictors.iloc[inner_fit_pos])
        inner_maps = fit_product_maps(train_predictors.iloc[inner_fit_pos], inner_params)
        inner_fit_base = baseline.build_rowwise_features(train_predictors.iloc[inner_fit_pos])
        inner_stop_base = baseline.build_rowwise_features(train_predictors.iloc[inner_stop_pos])
        inner_fit_extra = make_product_features(train_predictors.iloc[inner_fit_pos], inner_params, inner_maps)
        inner_stop_extra = make_product_features(train_predictors.iloc[inner_stop_pos], inner_params, inner_maps)
        inner_fit_x = pd.concat([inner_fit_base, inner_fit_extra[MODEL_FEATURES]], axis=1)
        inner_stop_x = pd.concat([inner_stop_base, inner_stop_extra[MODEL_FEATURES]], axis=1)
        fit_matrix, stop_matrix = baseline.make_lightgbm_frames(inner_fit_x, inner_stop_x)
        train_set = lgb.Dataset(fit_matrix, label=labels[inner_fit_pos])
        stop_set = lgb.Dataset(stop_matrix, label=labels[inner_stop_pos], reference=train_set)
        selection_model = lgb.train(
            {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
            train_set,
            num_boost_round=lgb_config["num_boost_round"],
            valid_sets=[stop_set],
            callbacks=[
                lgb.early_stopping(lgb_config["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iteration = int(selection_model.best_iteration or lgb_config["num_boost_round"])

        outer_params = fit_fee_and_step(train_predictors.iloc[fit_positions])
        outer_maps = fit_product_maps(train_predictors.iloc[fit_positions], outer_params)
        fit_raw_fold = train_predictors.iloc[fit_positions]
        fit_labels_fold = labels[fit_positions]
        fit_cash = fit_raw_fold["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        fit_credit_fold = finite_column(fit_raw_fold, "AMT_CREDIT")
        fit_annuity_fold = finite_column(fit_raw_fold, "AMT_ANNUITY")
        fit_goods_fold = finite_column(fit_raw_fold, "AMT_GOODS_PRICE")
        q_fold = safe_ratio(fit_annuity_fold, fit_credit_fold)
        p_fold = safe_ratio(fit_annuity_fold - outer_params["fee_estimate_fit_only"], fit_credit_fold)
        k_fold = safe_ratio(
            fit_credit_fold / fit_goods_fold - 1.0,
            np.full(len(fit_raw_fold), outer_params["step_estimate_fit_only"]),
        ) if np.isfinite(outer_params["step_estimate_fit_only"]) else np.full(len(fit_raw_fold), np.nan)
        fold_key_values = {
            "Q3": np.round(q_fold, 3),
            "Q4": np.round(q_fold, 4),
            "Q5": np.round(q_fold, 5),
            "P3": np.round(p_fold, 3),
            "P4": np.round(p_fold, 4),
            "P5": np.round(p_fold, 5),
            "K_INT": np.rint(k_fold),
        }
        for key_name, key_values in fold_key_values.items():
            use = fit_cash & np.isfinite(key_values)
            if not np.any(use):
                continue
            key_frame = pd.DataFrame(
                {"key": key_values[use], "target": fit_labels_fold[use]}
            )
            group_stats = (
                key_frame.groupby("key", observed=True)["target"]
                .agg(["size", "mean"])
                .sort_values("size", ascending=False)
                .head(25)
            )
            for key_value, stat in group_stats.iterrows():
                product_key_distribution_rows.append(
                    {
                        "fold": int(fold),
                        "fit_only": True,
                        "key_family": key_name,
                        "key_value": key_value,
                        "fit_count": int(stat["size"]),
                        "fit_share_of_cash_rows": float(stat["size"] / use.sum()),
                        "descriptive_default_rate_not_used_as_feature": float(stat["mean"]),
                    }
                )
        fit_base = baseline.build_rowwise_features(train_predictors.iloc[fit_positions])
        valid_base = baseline.build_rowwise_features(train_predictors.iloc[valid_positions])
        fit_extra = make_product_features(train_predictors.iloc[fit_positions], outer_params, outer_maps)
        valid_extra = make_product_features(train_predictors.iloc[valid_positions], outer_params, outer_maps)
        if candidate_features_oof is None:
            candidate_features_oof = pd.DataFrame(
                np.nan,
                index=np.arange(len(train)),
                columns=valid_extra.columns,
                dtype=np.float64,
            )
        candidate_features_oof.iloc[valid_positions, :] = valid_extra.to_numpy()
        fit_x = pd.concat([fit_base, fit_extra[MODEL_FEATURES]], axis=1)
        valid_x = pd.concat([valid_base, valid_extra[MODEL_FEATURES]], axis=1)
        fit_matrix, valid_matrix = baseline.make_lightgbm_frames(fit_x, valid_x)
        final_model = lgb.train(
            {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
            lgb.Dataset(fit_matrix, label=y_fit),
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(0)],
        )
        prediction = final_model.predict(valid_matrix, num_iteration=best_iteration)
        candidate_oof[valid_positions] = prediction
        fold_auc = roc_auc_score(labels[valid_positions], prediction)
        fold_model_rows.append(
            {
                "fold": int(fold),
                "n_fit": int(outer_fit.sum()),
                "n_validation": int(outer_valid.sum()),
                "best_iteration_inner_only": best_iteration,
                "fold_auc_product_family": fold_auc,
                "fold_auc_saved_baseline": roc_auc_score(labels[valid_positions], baseline_oof[valid_positions]),
                "fold_delta": fold_auc - roc_auc_score(labels[valid_positions], baseline_oof[valid_positions]),
                **outer_params,
            }
        )
        fold_params.append({"fold": int(fold), **outer_params})

        val_cash = train_predictors.iloc[valid_positions]["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        val_credit = finite_column(train_predictors.iloc[valid_positions], "AMT_CREDIT")
        val_goods = finite_column(train_predictors.iloc[valid_positions], "AMT_GOODS_PRICE")
        val_annuity = finite_column(train_predictors.iloc[valid_positions], "AMT_ANNUITY")
        val_q = safe_ratio(val_annuity, val_credit)
        val_p = safe_ratio(val_annuity - outer_params["fee_estimate_fit_only"], val_credit)
        val_k = safe_ratio(
            val_credit / val_goods - 1.0,
            np.full(len(valid_positions), outer_params["step_estimate_fit_only"]),
        ) if np.isfinite(outer_params["step_estimate_fit_only"]) else np.full(len(valid_positions), np.nan)
        fit_q5 = np.round(
            safe_ratio(
                finite_column(train_predictors.iloc[fit_positions], "AMT_ANNUITY"),
                finite_column(train_predictors.iloc[fit_positions], "AMT_CREDIT"),
            ),
            5,
        )
        fit_p5 = np.round(
            safe_ratio(
                finite_column(train_predictors.iloc[fit_positions], "AMT_ANNUITY") - outer_params["fee_estimate_fit_only"],
                finite_column(train_predictors.iloc[fit_positions], "AMT_CREDIT"),
            ),
            5,
        )
        qkeys = set(pd.Series(fit_q5).dropna().unique())
        pkeys = set(pd.Series(fit_p5).dropna().unique())
        val_q_keys = np.round(val_q, 5)
        val_p_keys = np.round(val_p, 5)
        q_coverage = float(np.mean([k in qkeys for k, keep in zip(val_q_keys, val_cash) if keep and np.isfinite(k)])) if np.any(val_cash & np.isfinite(val_q_keys)) else np.nan
        p_coverage = float(np.mean([k in pkeys for k, keep in zip(val_p_keys, val_cash) if keep and np.isfinite(k)])) if np.any(val_cash & np.isfinite(val_p_keys)) else np.nan
        k_resid = np.abs(val_k - np.rint(val_k))
        val_lattice_mask = val_cash & (val_credit > val_goods) & np.isfinite(val_goods)
        val_v17_lattice_support = (
            float(
                np.mean(
                    np.abs(
                        (val_credit[val_lattice_mask] / val_goods[val_lattice_mask] - 1.0)
                        / 0.0132
                        - np.rint(
                            (val_credit[val_lattice_mask] / val_goods[val_lattice_mask] - 1.0)
                            / 0.0132
                        )
                    )
                    < 0.1
                )
            )
            if val_lattice_mask.any()
            else np.nan
        )
        product_stability_rows.append(
            {
                "fold": int(fold),
                "fee_estimate_fit_only": outer_params["fee_estimate_fit_only"],
                "fee_fit_repeat_support": outer_params["fee_fit_repeat_support"],
                "fee_constant_v17": 129.5,
                "fixed_fee_adjusted_q5_seen_in_train_pct": p_coverage,
                "unadjusted_q5_seen_in_train_pct": q_coverage,
                "step_estimate_fit_only": outer_params["step_estimate_fit_only"],
                "step_fit_lattice_support": outer_params["step_fit_lattice_support"],
                "step_constant_v17": 0.0132,
                "validation_v17_step_lattice_support": val_v17_lattice_support,
                "validation_cash_credit_gt_goods_n": int((val_cash & (val_credit > val_goods) & np.isfinite(val_k)).sum()),
                "validation_k_within_0_1_of_integer_pct": float(np.mean(k_resid[val_cash & (val_credit > val_goods) & np.isfinite(k_resid)] < 0.1)) if np.any(val_cash & (val_credit > val_goods) & np.isfinite(k_resid)) else np.nan,
            }
        )
        recon_base = np.abs(val_annuity - val_credit * np.round(val_q, 5))
        recon_fee = np.abs(val_annuity - (outer_params["fee_estimate_fit_only"] + val_credit * np.round(val_p, 5)))
        for method, errors in (("unadjusted_q5", recon_base), ("fold_estimated_fee_p5", recon_fee)):
            good = np.isfinite(errors)
            reconstruction_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "n_valid": int(good.sum()),
                    "mean_abs_error": float(np.mean(errors[good])) if good.any() else np.nan,
                    "median_abs_error": float(np.median(errors[good])) if good.any() else np.nan,
                    "p95_abs_error": float(np.quantile(errors[good], 0.95)) if good.any() else np.nan,
                    "max_abs_error": float(np.max(errors[good])) if good.any() else np.nan,
                }
            )

    if not np.isfinite(candidate_oof).all():
        raise ValueError("Candidate OOF predictions are incomplete.")
    product_oof = pd.DataFrame({ID: ids, TARGET: labels, "fold": fold_ids, "OOF_PRODUCT_FAMILY": candidate_oof})
    product_oof.to_csv(OUT / "PHASE3_PRODUCT_OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(fold_model_rows).to_csv(OUT / "FOLD_LEVEL_CV_RESULTS.csv", index=False)
    pd.DataFrame(fold_params).to_csv(OUT / "PRODUCT_KEY_FOLD_STABILITY.csv", index=False)
    pd.DataFrame(product_stability_rows).to_csv(OUT / "PRODUCT_DECODING_RESULTS.csv", index=False)
    pd.DataFrame(product_key_distribution_rows).to_csv(OUT / "PRODUCT_KEY_DISTRIBUTIONS.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(OUT / "PRODUCT_FORMULA_RECONSTRUCTION_ERRORS.csv", index=False)

    # Fold-wise OOF candidate values are kept as diagnostics, not used to fit feature statistics.
    if candidate_features_oof is None:
        raise ValueError("No candidate feature folds were generated.")
    formula_results = []
    distribution_rows = []
    for name in candidate_features_oof.columns:
        values = candidate_features_oof[name].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        auc = roc_auc_score(labels[valid], values[valid]) if valid.sum() and len(np.unique(labels[valid])) == 2 else np.nan
        formula_results.append(
            {
                "feature_name": name,
                "valid_count": int(valid.sum()),
                "missing_count": int((~valid).sum()),
                "infinite_count": 0,
                "raw_univariate_oof_auc": auc,
                "direction_adjusted_oof_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                "interpretation": "Exploratory held-out univariate diagnostic only; not a model ablation.",
            }
        )
        v = values[valid]
        distribution_rows.append(
            {
                "feature_name": name,
                "valid_count": int(valid.sum()),
                "missing_count": int((~valid).sum()),
                "min": float(np.min(v)) if len(v) else np.nan,
                "q01": float(np.quantile(v, 0.01)) if len(v) else np.nan,
                "q05": float(np.quantile(v, 0.05)) if len(v) else np.nan,
                "median": float(np.median(v)) if len(v) else np.nan,
                "mean": float(np.mean(v)) if len(v) else np.nan,
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                "q95": float(np.quantile(v, 0.95)) if len(v) else np.nan,
                "q99": float(np.quantile(v, 0.99)) if len(v) else np.nan,
                "max": float(np.max(v)) if len(v) else np.nan,
                "target_default_rate": float(labels.mean()),
            }
        )
    pd.DataFrame(formula_results).to_csv(OUT / "FORMULA_VALIDATION_RESULTS.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(OUT / "PRODUCT_FEATURE_DISTRIBUTIONS.csv", index=False)
    pd.DataFrame(formula_rows()).to_csv(OUT / "DOMAIN_FORMULA_CATALOG_V4.csv", index=False)
    pd.DataFrame(formula_rows()).to_csv(OUT / "FEATURE_REGISTRY.csv", index=False)

    aucs = pd.DataFrame(fold_model_rows)
    ref_auc = roc_auc_score(labels, baseline_oof)
    family_auc = roc_auc_score(labels, candidate_oof)
    ablation = [
        {
            "experiment": "verified_raw_baseline_saved_oof",
            "pooled_oof_auc": ref_auc,
            "delta_vs_raw_baseline": 0.0,
            "mean_fold_auc": float(np.mean([roc_auc_score(labels[fold_ids == f], baseline_oof[fold_ids == f]) for f in np.unique(fold_ids)])),
            "feature_count_added": 0,
            "status": "Historical reference reproduced",
        },
        {
            "experiment": "batch1_loan_product_and_repayment_features",
            "pooled_oof_auc": family_auc,
            "delta_vs_raw_baseline": family_auc - ref_auc,
            "mean_fold_auc": float(aucs["fold_auc_product_family"].mean()),
            "feature_count_added": len(MODEL_FEATURES),
            "status": "Discovery CV; confirmation still required",
        },
    ]
    pd.DataFrame(ablation).to_csv(OUT / "FEATURE_ABLATION_RESULTS.csv", index=False)
    pd.DataFrame(ablation).to_csv(OUT / "FEATURE_FAMILY_RESULTS.csv", index=False)
    shortlist = [
        {
            "feature_name": name,
            "decision": "HOLD_FOR_CONFIRMATION",
            "reason": "Included only as part of the predeclared loan-product family; no individual ablation or untouched confirmation yet.",
        }
        for name in MODEL_FEATURES
    ]
    pd.DataFrame(shortlist).to_csv(OUT / "FINAL_FEATURE_SHORTLIST.csv", index=False)

    relations = [
        ("ANNUITY/CREDIT", "Existing baseline ratio; v17 Q3/Q4/Q5 are rounded encodings of same scalar."),
        ("CREDIT/ANNUITY", "Reciprocal of annuity/credit for positive values; not an actual term without payment-frequency/contract information."),
        ("CREDIT/GOODS", "Existing baseline ratio; v17 K_DEC hypothesis says cash-loan excess over 1 lies near a 0.0132 integer lattice."),
        ("ANNUIY_FEE_ADJUSTED/CREDIT", "v17 P keys subtract a proposed fixed 129.5 amount before normalization; fee constant requires fold-only estimation and held-out verification."),
        ("CREDIT_MINUS_GOODS", "Signed difference is exactly related to credit/goods ratio: gap = goods*(ratio-1) when goods is nonzero."),
        ("ANNUITY/GOODS", "Algebraically equals (annuity/credit)*(credit/goods) where denominators are nonzero."),
        ("CREDIT/INCOME_PER_MEMBER", "Equals (credit/income)*family_count when family count and income are positive; equal financial responsibility is not implied."),
    ]
    pd.DataFrame(
        [{"relationship": r[0], "derivation_or_observation": r[1], "status": "Data-derived algebra; not causal"} for r in relations]
    ).to_csv(OUT / "MATHEMATICAL_RELATIONSHIP_CATALOG.csv", index=False)
    pd.DataFrame(
        [
            {"candidate": "Fixed fee in annuity ratio", "evidence_to_check": "fit-only fee estimate; held-out repeated-key coverage; reconstruction error", "status": "Measured in this batch"},
            {"candidate": "Cash-loan financing increment lattice", "evidence_to_check": "fit-only delta estimate; held-out integer proximity; fold stability", "status": "Measured in this batch"},
            {"candidate": "Rounded annuity/credit ratio keys Q3-Q5", "evidence_to_check": "key concentration and fit-to-validation key coverage", "status": "Measured in this batch"},
            {"candidate": "Fee-adjusted P3-P5 keys", "evidence_to_check": "same measures with fold-fitted fee", "status": "Measured in this batch"},
            {"candidate": "Financial ratios conditional on external score", "evidence_to_check": "incremental pooled OOF family ablation", "status": "Measured in this batch"},
            {"candidate": "Frequency-based applicant identity linkage", "evidence_to_check": "not considered; disabled by rule gate"},
            {"candidate": "Target encoding of product keys", "evidence_to_check": "not considered; disabled by rule gate"},
        ]
    ).to_csv(OUT / "HIDDEN_STRUCTURE_CANDIDATES.csv", index=False)
    pd.DataFrame(
        [
            {"formula_family": "Loan product", "formula": formula, "permission": "ALLOWED_WITH_RESTRICTIONS", "restriction": "Use only supplied fields; any unknown constants and key frequencies are fitted on outer training rows only; no pooled test covariates."}
            for formula in [
                "Q3-Q5 = round(AMT_ANNUITY / AMT_CREDIT, 3/4/5)",
                "P3-P5 = round((AMT_ANNUITY - fee_fit) / AMT_CREDIT, 3/4/5)",
                "K_DEC = (AMT_CREDIT / AMT_GOODS_PRICE - 1) / step_fit",
                "fit-only product-key frequency counts",
            ]
        ]
    ).to_csv(OUT / "LOAN_STRUCTURE_FORMULAS.csv", index=False)

    train_bytes = [
        f"train.csv sha256={sha256(TRAIN_PATH)} size_bytes={TRAIN_PATH.stat().st_size}",
        f"test.csv sha256={sha256(TEST_PATH)} size_bytes={TEST_PATH.stat().st_size}",
        f"official_dictionary sha256={sha256(DICT_PATH)} size_bytes={DICT_PATH.stat().st_size}",
        f"baseline_source sha256={sha256(BASELINE_SOURCE)}",
        f"saved_baseline_oof sha256={sha256(BASELINE_DIR / 'OOF_BASELINE_PREDICTIONS.csv')}",
        f"saved_baseline_folds sha256={sha256(BASELINE_DIR / 'FOLD_ASSIGNMENTS.csv')}",
        f"phase2b_oof sha256={sha256(PHASE2B_DIR / 'OOF_PREDICTIONS.csv')}",
        f"v17_reuse_register sha256={sha256(REUSE_REGISTER)}",
    ]
    (OUT / "INPUT_DATA_HASHES.txt").write_text("\n".join(train_bytes) + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "cpu_threads": baseline.CPU_THREADS,
        "execution_started_utc": started_utc,
        "execution_finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\nlightgbm=={lgb.__version__}\nopenpyxl=={__import__('openpyxl').__version__}\n",
        encoding="utf-8",
    )
    code_hash = sha256(Path(__file__).resolve())

    ref_md = [
        "# Historical OOF reference verification",
        "",
        f"Executed: {environment['execution_finished_utc']}",
        "Method: recomputed pooled ROC-AUC and all five fold AUCs from saved predictions; aligned rows by SK_ID_CURR; checked saved TARGET against supplied train.csv TARGET; checked fold IDs.",
        "",
        pd.DataFrame(ref_rows).to_string(index=False),
        "",
        "All five references reproduced within 1e-10. This verifies saved prediction metrics and row/label alignment, not an independent rerun of historical model training.",
    ]
    (OUT / "REFERENCE_RESULTS_VERIFICATION.md").write_text("\n".join(ref_md) + "\n", encoding="utf-8")

    rules_md = """# Phase 3 rule compliance gate

Authority reviewed: supplied course README and official data dictionary, summarized in the pre-existing local official rules audit. No live organizer portal or later amendment was checked. Lack of a specific ban is not treated as approval.

| Method | Status | Phase 3 control |
|---|---|---|
| Row-wise transformations of supplied columns | ALLOWED | Use only official input columns; no external facts/data. |
| Statistics fit on each training fold | ALLOWED_WITH_RESTRICTIONS | Fit only on the current fit split; apply unchanged to validation rows. |
| Pooled train/test unsupervised statistics | UNRESOLVED | Disabled. |
| Frequency encoding | ALLOWED_WITH_RESTRICTIONS | Training-fold-only map; unseen keys map to zero; no test rows in fit. |
| Target encoding | UNRESOLVED | Disabled. |
| Repeat-applicant inference | UNRESOLVED | Disabled. |
| Derived applicant fingerprints | UNRESOLVED | Disabled. |
| Cross-validation target statistics | UNRESOLVED | Disabled with target encoding; no labels in formula construction. |
| Clustering | ALLOWED_WITH_RESTRICTIONS | Not used in Batch 1; if tried, fit clusters within each training fold only. |
| External information/data | PROHIBITED | Not used. |

Batch 1 uses only row-wise supplied columns and unsupervised fee/step/key-frequency estimates fitted on outer training partitions. No target encoding, pooled train/test stats, inferred person links, external data, or hand-created labels are used.
"""
    (OUT / "PHASE3_RULE_COMPLIANCE.md").write_text(rules_md, encoding="utf-8")
    write_csv(
        "FEATURE_METHOD_PERMISSION_MATRIX.csv",
        [
            {"method": "Row-wise formula", "status": "ALLOWED", "enabled": True, "control": "Supplied columns only."},
            {"method": "Training-fold statistic", "status": "ALLOWED_WITH_RESTRICTIONS", "enabled": True, "control": "Fit only on current training split."},
            {"method": "Pooled train/test unsupervised statistic", "status": "UNRESOLVED", "enabled": False, "control": "Disabled pending organizer clarification."},
            {"method": "Frequency encoding", "status": "ALLOWED_WITH_RESTRICTIONS", "enabled": True, "control": "Training fold only; unseen levels map to zero."},
            {"method": "Target encoding", "status": "UNRESOLVED", "enabled": False, "control": "Disabled."},
            {"method": "Repeat-applicant inference", "status": "UNRESOLVED", "enabled": False, "control": "Disabled."},
            {"method": "Derived applicant fingerprints", "status": "UNRESOLVED", "enabled": False, "control": "Disabled."},
            {"method": "CV target statistics", "status": "UNRESOLVED", "enabled": False, "control": "No target-derived features."},
            {"method": "Clustering", "status": "ALLOWED_WITH_RESTRICTIONS", "enabled": False, "control": "Not used; if enabled, fold-local unsupervised fit only."},
            {"method": "External information", "status": "PROHIBITED", "enabled": False, "control": "No external data."},
        ],
    )

    feature_desc = pd.DataFrame(FEATURE_DEFINITIONS, columns=["feature_name", "formula", "meaning"])
    model_summary = pd.DataFrame(ablation).to_string(index=False)
    key_summary = pd.DataFrame(product_stability_rows).to_string(index=False)
    product_md = f"""# Hidden loan-product and repayment structure — Batch 1

## Scope and guardrails

Only the supplied training predictors and training TARGET were used. TARGET is used solely for held-out ROC-AUC. The official data dictionary defines AMT_ANNUITY only as “Loan annuity”; it does not specify payment periodicity, so this report does not call it monthly or annual and does not infer a contractual term from CREDIT/ANNUITY. Test covariates were schema-checked but excluded from discovery, fitting, and validation.

The v17 source defines `K_DEC = round((CREDIT/GOODS - 1)/0.0132)` with a 0.1 lattice tolerance, and product ratios `Q3/Q4/Q5 = round(ANNUITY/CREDIT, 3/4/5)`. It estimates no documented derivation for constants 129.5 and 0.0132. This experiment re-estimates the candidate fee offset and lattice step using each training partition only; the fixed v17 constants are reported solely as hypotheses.

## Mathematical hypotheses

{feature_desc.to_string(index=False)}

## CV family result

{model_summary}

The model family contains {len(MODEL_FEATURES)} distinct numeric columns after excluding exact aliases and baseline-duplicate ratios. It is one family-level ablation, not 39 individually tested ablations. The result uses the historical fixed five folds and is a discovery measurement, not an untouched confirmation.

## Fold-local decoder estimates and held-out key coverage

{key_summary}

Fee estimation maximized within-fit duplicate support for rounded 5-decimal `(ANNUITY-F)/CREDIT` keys over F in [0, 250] at 0.5 increments. The financing increment maximized the within-fit proportion of cash-loan `(CREDIT/GOODS-1)/delta` values within 0.1 of an integer over a predeclared grid, with a local refinement. These are exploratory unsupervised fit objectives, not proof of an actual fee/tariff or financing mechanism.

## Interpretation limits

- A repeated rounded ratio may be caused by amount rounding, common standardized product terms, or finite precision; it is not itself evidence of a separate financial product.
- Annuity differences and ratios are not labeled with a time period in the official dictionary. No annualization or repayment-term claim is supported.
- `K_DEC` is only a lattice coordinate; no borrower-facing interpretation is established.
- A fixed fee/step estimate is accepted only if it is stable across folds and predicts repeated structure on held-out rows. See `PRODUCT_KEY_FOLD_STABILITY.csv`, `PRODUCT_DECODING_RESULTS.csv`, and `PRODUCT_FORMULA_RECONSTRUCTION_ERRORS.csv`.
- The family AUC is not a final claim of improvement; repeated discovery on these folds can overfit. All features remain on hold until a separately frozen confirmation.
"""
    (OUT / "HIDDEN_LOAN_PRODUCT_RESEARCH.md").write_text(product_md, encoding="utf-8")

    ref_table = pd.DataFrame(ref_rows).to_string(index=False)
    anatomy = f"""# Dataset anatomy — Phase 3 Batch 1

Training shape: {train.shape}; test shape: {test.shape}. Training predictors: {train.shape[1]-1}; test predictors: {test.shape[1]}. Positive rate: {train[TARGET].mean():.8f}. Dictionary descriptions and numeric/categorical summary statistics for every training column are in `ALL_COLUMN_STATISTICS.csv`.

The dictionary describes credit, annuity, goods price, income, family size, external scores, and bureau inquiries. Importantly, AMT_ANNUITY has no declared frequency, and the bureau windows have distinct intervals; they are not summed into a rate in this batch.

Numeric pairwise Spearman correlations with |rho| >= 0.80 are listed in `COLUMN_DEPENDENCY_REPORT.csv`. Such correlations are screening signals, not proof that one field is mathematically derived from another. Sentinel, missingness, zeros, quantiles, IQR outlier fraction, and TARGET-default rate are recorded per feature in the statistics file.

## Structural themes examined

- Loan structure: annuity/credit, credit/goods, goods/credit, difference, financing share, and their rounded/repeated forms.
- Candidate fixed-fee adjustment: `(ANNUITY - F_fit) / CREDIT`, with fee chosen on outer-fit rows.
- Candidate financing increment: `(CREDIT/GOODS - 1) / delta_fit`, with delta chosen on outer-fit cash loans.
- Repayment / affordability: annuity-to-income, annuity-to-goods, financing gap relative to income/annuity, and household-size interactions.
- Product × creditworthiness: external score summary and dispersion crossed with financing gap or candidate lattice coordinate.

No causal interpretation or inferred applicant identity is attempted. `MATHEMATICAL_RELATIONSHIP_CATALOG.csv` records exact algebraic dependencies and caveats.
"""
    (OUT / "DEEP_DATASET_ANATOMY.md").write_text(anatomy, encoding="utf-8")

    reference_table = pd.DataFrame(ref_rows).to_string(index=False)
    cv_table = pd.DataFrame(ablation).to_string(index=False)
    report = f"""# Phase 3 execution report — first milestone

Status: **completed for reference verification and Batch 1 discovery; not a complete Phase 3 program.**

Execution: {environment['execution_started_utc']} to {environment['execution_finished_utc']} UTC; runtime {environment['runtime_seconds']:.1f} seconds. Code SHA-256: `{code_hash}`. Inputs and environment are recorded in `INPUT_DATA_HASHES.txt` and `environment.json`.

## Historical baseline verification

{reference_table}

Saved OOF IDs and labels matched the supplied training file; fold IDs matched the corrected five-fold assignment. This reproduces the saved metrics, not an independent rerun of the historical training script. Complete 23-row v17 reuse register was read and confirmed before running.

## Batch 1 result

{cv_table}

The added family uses {len(MODEL_FEATURES)} engineered numeric features, with unknown fee and financing-step parameters estimated independently on each applicable training split. Outer validation rows were not used for feature fitting, product-key counts, early stopping, or parameter estimation. Iterations were selected on an inner stratified split of each outer training fold. OOF predictions are saved in `PHASE3_PRODUCT_OOF_PREDICTIONS.csv`.

## Scientific interpretation

Batch 1 tests the v17 fixed-fee and financing-lattice ideas rather than copying the constants. The output files show decoder stability, held-out repeated-key coverage, and reconstruction errors. Even a high lattice support is not proof of the business meaning; evidence must be stable and out-of-sample. The exact product formula result is not promoted if its OOF family delta is negligible, unstable, or negative.

All 51 listed candidate formulas were implemented for diagnostics; duplicates and unsuitable aliases are marked in `DOMAIN_FORMULA_CATALOG_V4.csv`. Only the predeclared nonduplicate feature family was evaluated in the model. Individual formula AUCs are exploratory held-out univariate diagnostics, not individual model improvements. No individual feature is promoted.

## Compliance and limitations

No external datasets, hand-created labels, target encoding, pooled train/test statistics, repeat-applicant inference, or test labels were used. Frequency maps and decoder parameters were fitted only on the current training partition. See `PHASE3_RULE_COMPLIANCE.md`.

This is not an untouched confirmation because the historical folds have been used by prior discovery work. A frozen confirmation split or nested validation is required before promotion. No 0.79+ result is claimed, no test inference/submission was produced, and future batches are not run or marked complete.

## Produced outputs

See the Phase 3 folder for reference verification, rule matrix, full predictor statistics, relationship/catalog files, decoder diagnostics, formula registry, fold-level results, and reproducible runner. Runtime log is `EXPERIMENT_LOG.md`.
"""
    (OUT / "PHASE3_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")

    reproducibility = f"""# Experiment reproducibility audit

- Exact train/test input paths are pinned and SHA-256 hashes are in `INPUT_DATA_HASHES.txt`.
- Train schema is 34 columns including TARGET; test schema is 33 predictors; predictor columns and IDs are validated.
- All saved historical OOF predictions were aligned to train rows by SK_ID_CURR; target arrays and five fixed fold assignments were verified.
- Historical pooled AUCs and five fold AUCs were recalculated from saved predictions. Historical model training itself was not rerun in this verification step.
- The complete v17 reuse register has {reuse_entries} physical data rows and was read directly. Its CSV quoting is malformed, so generic CSV parsing fails; it was not used as a model input.
- Feature transforms use supplied row-level values plus fold-fit fee, lattice, and frequency parameters; no feature parameter was fit on outer validation rows or test rows.
- Inner split early stopping does not use outer validation labels. Model uses the preserved raw baseline LightGBM configuration and helper.
- Random seeds: outer folds inherited from saved seed-42 assignments; inner split seed is 20261102 + fold; LightGBM seed is baseline seed + fold.
- Python {sys.version.split()[0]}, pandas {pd.__version__}, NumPy {np.__version__}, scikit-learn {sklearn.__version__}, LightGBM {lgb.__version__}; exact versions in environment.json.
- Reproducing discovery metrics still requires running this script against the hashed files. CPU timing and LightGBM nondeterminism may cause tiny numerical differences.
- The fixed 5 folds are reused for comparability and are not an untouched confirmation set.
"""
    (OUT / "EXPERIMENT_REPRODUCIBILITY_AUDIT.md").write_text(reproducibility, encoding="utf-8")
    log = [
        f"execution_started_utc={environment['execution_started_utc']}",
        f"execution_finished_utc={environment['execution_finished_utc']}",
        f"runtime_seconds={environment['runtime_seconds']:.3f}",
        f"train_shape={train.shape}",
        f"test_shape={test.shape}",
        f"v17_reuse_register_rows={reuse_entries}",
        f"formulas_registered={len(FEATURE_DEFINITIONS)}",
        f"model_features_added={len(MODEL_FEATURES)}",
        f"baseline_pooled_auc={ref_auc:.12f}",
        f"product_family_pooled_auc={family_auc:.12f}",
        f"delta_vs_baseline={family_auc-ref_auc:.12f}",
        f"code_sha256={code_hash}",
        "warnings=RuntimeWarning ignored for vectorized missing-value arithmetic; no other warnings recorded",
        "status=SUCCESS",
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print("outputs:")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(f"{path} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
