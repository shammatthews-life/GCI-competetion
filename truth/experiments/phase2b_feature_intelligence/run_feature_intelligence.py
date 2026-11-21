#!/usr/bin/env python3
"""Dataset anatomy and fold-safe feature-family experiment for Phase 2B."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(r"D:\gci\truth")
DATA_DIR = Path(r"D:\gci\inpuy")
OUT = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
DICT_PATH = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition"
    r"\HomeCredit_columns_description.xlsx"
)
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
VALIDATION_SOURCE = (
    ROOT
    / "experiments"
    / "phase2_advanced_feature_engineering"
    / "run_corrected_validation.py"
)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_module("preserved_raw_baseline", BASELINE_SOURCE)
validation = load_module("corrected_validation", VALIDATION_SOURCE)

PEER_FEATURES = [
    "PEER_LOG_INCOME_BY_INCOME_TYPE",
    "PEER_LOG_INCOME_BY_OCCUPATION",
    "PEER_CREDIT_INCOME_BY_INCOME_OCCUPATION",
    "PEER_ANNUITY_CREDIT_BY_CONTRACT",
    "PEER_EXT2_BY_EDUCATION",
    "PEER_EXT3_BY_EDUCATION",
    "PEER_EXT2_BY_INCOME_TYPE",
    "PEER_EXT3_BY_INCOME_TYPE",
]
INTERACTION_FEATURES = [
    "EXT1_X_CREDIT_INCOME",
    "EXT2_X_CREDIT_INCOME",
    "EXT3_X_CREDIT_INCOME",
    "EXT1_X_CREDIT_GOODS",
    "EXT2_X_CREDIT_GOODS",
    "EXT3_X_CREDIT_GOODS",
    "EXT1_X_ANNUITY_INCOME",
    "EXT2_X_ANNUITY_INCOME",
    "EXT3_X_ANNUITY_INCOME",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    out[valid] = numerator[valid] / denominator[valid]
    return out


def source_values(raw: pd.DataFrame) -> pd.DataFrame:
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    income = raw["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "LOG_INCOME": np.log1p(np.maximum(income, 0)),
            "CREDIT_INCOME": safe_ratio(credit, income),
            "ANNUITY_CREDIT": safe_ratio(annuity, credit),
            "EXT2": raw["EXT_SOURCE_2"].to_numpy(dtype=np.float64),
            "EXT3": raw["EXT_SOURCE_3"].to_numpy(dtype=np.float64),
        },
        index=raw.index,
    )


def peer_deviations(
    fit_raw: pd.DataFrame, apply_raw: pd.DataFrame, min_group_size: int = 30
) -> pd.DataFrame:
    """Fit robust group medians/IQRs on fit_raw only and apply to apply_raw."""
    fit_raw = fit_raw.reset_index(drop=True)
    apply_raw = apply_raw.reset_index(drop=True)
    fit_values = source_values(fit_raw)
    apply_values = source_values(apply_raw)
    definitions = [
        ("LOG_INCOME", ["NAME_INCOME_TYPE"], "PEER_LOG_INCOME_BY_INCOME_TYPE"),
        ("LOG_INCOME", ["OCCUPATION_TYPE"], "PEER_LOG_INCOME_BY_OCCUPATION"),
        (
            "CREDIT_INCOME",
            ["NAME_INCOME_TYPE", "OCCUPATION_TYPE"],
            "PEER_CREDIT_INCOME_BY_INCOME_OCCUPATION",
        ),
        (
            "ANNUITY_CREDIT",
            ["NAME_CONTRACT_TYPE"],
            "PEER_ANNUITY_CREDIT_BY_CONTRACT",
        ),
        ("EXT2", ["NAME_EDUCATION_TYPE"], "PEER_EXT2_BY_EDUCATION"),
        ("EXT3", ["NAME_EDUCATION_TYPE"], "PEER_EXT3_BY_EDUCATION"),
        ("EXT2", ["NAME_INCOME_TYPE"], "PEER_EXT2_BY_INCOME_TYPE"),
        ("EXT3", ["NAME_INCOME_TYPE"], "PEER_EXT3_BY_INCOME_TYPE"),
    ]
    output: dict[str, np.ndarray] = {}
    for value_name, group_columns, feature_name in definitions:
        fit_groups = fit_raw[group_columns].fillna("__MISSING__").astype(str)
        apply_groups = apply_raw[group_columns].fillna("__MISSING__").astype(str)
        fit_keys = fit_groups.agg("\x1f".join, axis=1)
        apply_keys = apply_groups.agg("\x1f".join, axis=1)
        fit_temp = pd.DataFrame(
            {"_group": fit_keys, "_value": fit_values[value_name].to_numpy()}
        )
        grouped = fit_temp.groupby("_group", dropna=False)["_value"]
        stats = pd.DataFrame(
            {
                "median": grouped.median(),
                "q25": grouped.quantile(0.25),
                "q75": grouped.quantile(0.75),
                "count": grouped.count(),
            }
        )
        global_values = fit_values[value_name].dropna()
        global_median = float(global_values.median()) if len(global_values) else 0.0
        global_iqr = (
            float(global_values.quantile(0.75) - global_values.quantile(0.25))
            if len(global_values)
            else 1.0
        )
        if not np.isfinite(global_iqr) or global_iqr <= 0:
            global_iqr = 1.0

        aligned = stats.reindex(apply_keys.to_numpy())
        group_count = aligned["count"].to_numpy(dtype=np.float64)
        eligible = np.isfinite(group_count) & (group_count >= min_group_size)
        medians = aligned["median"].to_numpy(dtype=np.float64)
        q25 = aligned["q25"].to_numpy(dtype=np.float64)
        q75 = aligned["q75"].to_numpy(dtype=np.float64)
        medians = np.where(eligible, medians, global_median)
        scales = np.where(eligible, q75 - q25, global_iqr)
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, global_iqr)
        values = apply_values[value_name].to_numpy(dtype=np.float64)
        output[feature_name] = (values - medians) / scales
    return pd.DataFrame(output)


def interaction_features(raw: pd.DataFrame) -> pd.DataFrame:
    credit_income = safe_ratio(
        raw["AMT_CREDIT"].to_numpy(dtype=np.float64),
        raw["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64),
    )
    credit_goods = safe_ratio(
        raw["AMT_CREDIT"].to_numpy(dtype=np.float64),
        raw["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64),
    )
    annuity_income = safe_ratio(
        raw["AMT_ANNUITY"].to_numpy(dtype=np.float64),
        raw["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64),
    )
    output = {}
    for i, ext in enumerate(("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"), 1):
        score = raw[ext].to_numpy(dtype=np.float64)
        output[f"EXT{i}_X_CREDIT_INCOME"] = score * credit_income
        output[f"EXT{i}_X_CREDIT_GOODS"] = score * credit_goods
        output[f"EXT{i}_X_ANNUITY_INCOME"] = score * annuity_income
    return pd.DataFrame(output, index=raw.index)


def make_variant_features(
    raw: pd.DataFrame,
    variant: str,
    peer_fit_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    target_free = raw.drop(columns=["TARGET"], errors="ignore").reset_index(drop=True)
    result = baseline.build_rowwise_features(target_free)
    if variant in ("external_interactions", "combined"):
        result = pd.concat(
            [result, interaction_features(target_free).reset_index(drop=True)],
            axis=1,
        )
    if variant in ("peer_deviations", "combined"):
        if peer_fit_raw is None:
            raise ValueError("Peer features require explicit fit-only reference rows")
        peer = peer_deviations(
            peer_fit_raw.drop(columns=["TARGET"], errors="ignore"),
            target_free,
        )
        result = pd.concat([result, peer.reset_index(drop=True)], axis=1)
    if result.columns.duplicated().any():
        raise ValueError(f"{variant} produced duplicate feature names")
    values = result.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isinf(values).any():
        raise ValueError(f"{variant} produced infinite values")
    return result


def params_for_fold(fold: int) -> dict[str, Any]:
    config = baseline.MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = baseline.FOLD_SEED + fold
    return params


def select_rounds(
    outer_fit: pd.DataFrame, fold: int, variant: str
) -> int:
    y = outer_fit["TARGET"].to_numpy(dtype=np.int8)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=20261102 + fold,
    )
    fit_idx, stop_idx = next(splitter.split(np.zeros(len(y)), y))
    inner_fit = outer_fit.iloc[fit_idx]
    inner_stop = outer_fit.iloc[stop_idx]
    fit_features = make_variant_features(inner_fit, variant, inner_fit)
    stop_features = make_variant_features(inner_stop, variant, inner_fit)
    fit_matrix, stop_matrix = baseline.make_lightgbm_frames(
        fit_features, stop_features
    )
    fit_set = lgb.Dataset(fit_matrix, label=y[fit_idx])
    stop_set = lgb.Dataset(stop_matrix, label=y[stop_idx], reference=fit_set)
    settings = baseline.MODEL_CONFIG["lightgbm"]
    model = lgb.train(
        params_for_fold(fold),
        fit_set,
        num_boost_round=settings["num_boost_round"],
        valid_sets=[stop_set],
        callbacks=[
            lgb.early_stopping(
                settings["early_stopping_rounds"], verbose=False
            )
        ],
    )
    rounds = int(model.best_iteration or model.current_iteration())
    if rounds <= 0:
        raise RuntimeError(f"No stopping iteration selected for fold {fold}")
    return rounds


def run_cv(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    y = train["TARGET"].to_numpy(dtype=np.int8)
    oof = np.full(len(train), np.nan)
    rows = []
    for fold, (fit_idx, valid_idx) in enumerate(folds, 1):
        fit_raw = train.iloc[fit_idx]
        valid_raw = train.iloc[valid_idx]
        rounds = select_rounds(fit_raw, fold, variant)
        fit_features = make_variant_features(fit_raw, variant, fit_raw)
        valid_features = make_variant_features(valid_raw, variant, fit_raw)
        fit_matrix, valid_matrix = baseline.make_lightgbm_frames(
            fit_features, valid_features
        )
        fit_set = lgb.Dataset(fit_matrix, label=y[fit_idx])
        model = lgb.train(
            params_for_fold(fold), fit_set, num_boost_round=rounds
        )
        prediction = model.predict(valid_matrix, num_iteration=rounds)
        oof[valid_idx] = prediction
        auc = float(roc_auc_score(y[valid_idx], prediction))
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "auc": auc,
                "best_iteration": rounds,
                "n_fit": len(fit_idx),
                "n_valid": len(valid_idx),
            }
        )
        print(
            f"{variant} fold={fold} auc={auc:.8f} rounds={rounds}",
            flush=True,
        )
        del model, fit_set
    if not np.isfinite(oof).all():
        raise AssertionError(f"OOF coverage incomplete for {variant}")
    return pd.DataFrame(rows), oof


def domain_for_column(column: str, dtype: str) -> str:
    if column == "TARGET":
        return "Outcome"
    if column == "SK_ID_CURR":
        return "Identifier"
    if column.startswith("AMT_REQ_CREDIT_BUREAU_"):
        return "Credit bureau inquiries"
    if column.startswith("EXT_SOURCE_"):
        return "External risk scores"
    if column in {"AMT_CREDIT", "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE"}:
        return "Loan and contract structure"
    if column in {"AMT_ANNUITY"}:
        return "Repayment / annuity"
    if column in {"AMT_INCOME_TOTAL", "NAME_INCOME_TYPE"}:
        return "Income and financial capacity"
    if column in {"DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"}:
        return "Age and time since event"
    if column in {"CNT_CHILDREN", "CNT_FAM_MEMBERS", "NAME_FAMILY_STATUS"}:
        return "Household and family"
    if column in {"OWN_CAR_AGE", "NAME_HOUSING_TYPE"}:
        return "Housing / property / assets"
    if column in {"REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT"}:
        return "Regional"
    if column.startswith("FLAG_"):
        return "Application contact / data indicator"
    if column in {"CODE_GENDER", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"}:
        return "Applicant category"
    return "Other application attribute" if dtype == "object" else "Other numeric application attribute"


def write_dataset_anatomy(train: pd.DataFrame, test: pd.DataFrame) -> None:
    dictionary = pd.read_excel(DICT_PATH)
    descriptions = {
        str(row["Row"]): str(row["Description"])
        for _, row in dictionary.iterrows()
        if pd.notna(row.get("Row")) and pd.notna(row.get("Description"))
    }
    rows = []
    quality = []
    numeric = train.select_dtypes(include=[np.number])
    for column in train.columns:
        series = train[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        missing = int(series.isna().sum())
        nonmissing = series.dropna()
        record: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "domain": domain_for_column(column, str(series.dtype)),
            "dictionary_description": descriptions.get(column, "No matching dictionary row"),
            "n_unique": int(series.nunique(dropna=True)),
            "missing_count": missing,
            "missing_pct": float(series.isna().mean()),
            "zero_pct": float((series == 0).mean()) if is_numeric else np.nan,
        }
        if is_numeric and len(nonmissing):
            record.update(
                {
                    "min": float(nonmissing.min()),
                    "max": float(nonmissing.max()),
                    "mean": float(nonmissing.mean()),
                    "std": float(nonmissing.std()),
                    "median": float(nonmissing.median()),
                    "q01": float(nonmissing.quantile(0.01)),
                    "q05": float(nonmissing.quantile(0.05)),
                    "q25": float(nonmissing.quantile(0.25)),
                    "q75": float(nonmissing.quantile(0.75)),
                    "q95": float(nonmissing.quantile(0.95)),
                    "q99": float(nonmissing.quantile(0.99)),
                    "skew": float(nonmissing.skew()),
                    "target_pearson": (
                        float(series.corr(train["TARGET"]))
                        if column != "TARGET"
                        else 1.0
                    ),
                }
            )
        else:
            record.update(
                {
                    name: np.nan
                    for name in (
                        "min", "max", "mean", "std", "median", "q01",
                        "q05", "q25", "q75", "q95", "q99", "skew",
                        "target_pearson",
                    )
                }
            )
        rows.append(record)
        quality.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_pct": float(series.isna().mean()),
                "zero_count": int((series == 0).sum()) if is_numeric else np.nan,
                "zero_pct": float((series == 0).mean()) if is_numeric else np.nan,
                "unique_nonmissing": int(series.nunique(dropna=True)),
                "sentinel_365243_count": int((series == 365243).sum())
                if column == "DAYS_EMPLOYED"
                else 0,
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / "COLUMN_STATISTICS.csv", index=False)
    pd.DataFrame(quality).to_csv(
        OUT / "DATA_QUALITY_AND_MISSINGNESS.csv", index=False
    )
    numeric_corr = numeric.corr(method="pearson")
    numeric_corr.to_csv(OUT / "COLUMN_RELATIONSHIP_MATRIX.csv")
    mapping = stats[
        ["column", "domain", "dtype", "dictionary_description"]
    ].copy()
    mapping["train_rows"] = len(train)
    mapping["test_rows"] = len(test)
    mapping.to_csv(OUT / "COLUMN_DOMAIN_MAPPING.csv", index=False)
    anatomy = [
        "# Dataset anatomy — deep dive (initial evidence pass)",
        "",
        f"- Training rows/columns: {train.shape}",
        f"- Test rows/columns: {test.shape}",
        f"- Predictors excluding ID: {len(train.columns) - 2}",
        f"- Positive target rate: {train['TARGET'].mean():.6f}",
        "- Source is limited to the supplied competition train/test CSVs and local official data dictionary.",
        "- Descriptive target correlations are diagnostics only; they were not used for feature selection or to tune the outer-fold experiment.",
        "",
        "## Domain structure",
        "",
        mapping[["domain", "column"]].groupby("domain")["column"].apply(
            lambda x: ", ".join(x)
        ).to_string(),
        "",
        "## Notable data-quality observations",
        "",
        f"- DAYS_EMPLOYED=365243 occurs in {(train['DAYS_EMPLOYED'] == 365243).sum():,} training rows and is treated as a sentinel, never as employment tenure.",
        f"- Bureau inquiry fields share missingness in {train['AMT_REQ_CREDIT_BUREAU_HOUR'].isna().sum():,} rows.",
        f"- External score missing counts: {train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].isna().sum().to_dict()}",
        "- See COLUMN_STATISTICS.csv and DATA_QUALITY_AND_MISSINGNESS.csv for every field's distribution and missingness.",
        "- Numeric Pearson relationships, including with TARGET for descriptive review, are in COLUMN_RELATIONSHIP_MATRIX.csv.",
        "",
        "## Interpretation boundary",
        "",
        "Statistics and target associations describe this training sample; they do not establish causal effects or justify hand-written prediction rules. The value 365243 is handled as an explicit sentinel. No test row was used for profiling or feature fitting in this file.",
    ]
    (OUT / "DATASET_ANATOMY_DEEP_DIVE.md").write_text(
        "\n".join(anatomy), encoding="utf-8"
    )


def write_hypothesis_registry() -> None:
    hypotheses = [
        ("PEER01", PEER_FEATURES[0], "Peer-group deviation", "robust z-score of log income versus NAME_INCOME_TYPE median/IQR", "AMT_INCOME_TOTAL;NAME_INCOME_TYPE", "Income atypicality may encode capacity beyond absolute income.", "positive for unusually high debt/capacity mismatch", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER02", PEER_FEATURES[1], "Peer-group deviation", "robust z-score of log income versus OCCUPATION_TYPE median/IQR", "AMT_INCOME_TOTAL;OCCUPATION_TYPE", "Compares income with applicants in the same recorded occupation.", "risk may rise with low relative income", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER03", PEER_FEATURES[2], "Peer-group deviation", "robust z-score of AMT_CREDIT/AMT_INCOME_TOTAL versus (NAME_INCOME_TYPE,OCCUPATION_TYPE) median/IQR", "AMT_CREDIT;AMT_INCOME_TOTAL;NAME_INCOME_TYPE;OCCUPATION_TYPE", "Debt burden relative to a narrow peer group may reveal unusual borrowing.", "positive for high relative burden", "fit-only group stats; rare groups back off global fit median/IQR", "Candidate; inductive fit-only statistics"),
        ("PEER04", PEER_FEATURES[3], "Peer-group deviation", "robust z-score of AMT_ANNUITY/AMT_CREDIT versus NAME_CONTRACT_TYPE median/IQR", "AMT_ANNUITY;AMT_CREDIT;NAME_CONTRACT_TYPE", "Compares payment structure within contract type.", "unknown", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER05", PEER_FEATURES[4], "Peer-group deviation", "robust z-score of EXT_SOURCE_2 versus NAME_EDUCATION_TYPE median/IQR", "EXT_SOURCE_2;NAME_EDUCATION_TYPE", "Relative score position may complement raw score.", "negative for lower relative score", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER06", PEER_FEATURES[5], "Peer-group deviation", "robust z-score of EXT_SOURCE_3 versus NAME_EDUCATION_TYPE median/IQR", "EXT_SOURCE_3;NAME_EDUCATION_TYPE", "Relative score position may complement raw score.", "negative for lower relative score", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER07", PEER_FEATURES[6], "Peer-group deviation", "robust z-score of EXT_SOURCE_2 versus NAME_INCOME_TYPE median/IQR", "EXT_SOURCE_2;NAME_INCOME_TYPE", "Tests whether credit-score rank differs by income class.", "negative for lower relative score", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("PEER08", PEER_FEATURES[7], "Peer-group deviation", "robust z-score of EXT_SOURCE_3 versus NAME_INCOME_TYPE median/IQR", "EXT_SOURCE_3;NAME_INCOME_TYPE", "Tests whether credit-score rank differs by income class.", "negative for lower relative score", "fit-only group stats", "Candidate; inductive fit-only statistics"),
        ("INT01", INTERACTION_FEATURES[0], "External score × affordability", "EXT_SOURCE_1 * (AMT_CREDIT/AMT_INCOME_TOTAL)", "EXT_SOURCE_1;AMT_CREDIT;AMT_INCOME_TOTAL", "Risk score may modify the meaning of borrowing burden.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT02", INTERACTION_FEATURES[1], "External score × affordability", "EXT_SOURCE_2 * (AMT_CREDIT/AMT_INCOME_TOTAL)", "EXT_SOURCE_2;AMT_CREDIT;AMT_INCOME_TOTAL", "Risk score may modify the meaning of borrowing burden.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT03", INTERACTION_FEATURES[2], "External score × affordability", "EXT_SOURCE_3 * (AMT_CREDIT/AMT_INCOME_TOTAL)", "EXT_SOURCE_3;AMT_CREDIT;AMT_INCOME_TOTAL", "Risk score may modify the meaning of borrowing burden.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT04", INTERACTION_FEATURES[3], "External score × financing", "EXT_SOURCE_1 * (AMT_CREDIT/AMT_GOODS_PRICE)", "EXT_SOURCE_1;AMT_CREDIT;AMT_GOODS_PRICE", "Risk score may modify the meaning of financed share.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT05", INTERACTION_FEATURES[4], "External score × financing", "EXT_SOURCE_2 * (AMT_CREDIT/AMT_GOODS_PRICE)", "EXT_SOURCE_2;AMT_CREDIT;AMT_GOODS_PRICE", "Risk score may modify the meaning of financed share.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT06", INTERACTION_FEATURES[5], "External score × financing", "EXT_SOURCE_3 * (AMT_CREDIT/AMT_GOODS_PRICE)", "EXT_SOURCE_3;AMT_CREDIT;AMT_GOODS_PRICE", "Risk score may modify the meaning of financed share.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT07", INTERACTION_FEATURES[6], "External score × payment burden", "EXT_SOURCE_1 * (AMT_ANNUITY/AMT_INCOME_TOTAL)", "EXT_SOURCE_1;AMT_ANNUITY;AMT_INCOME_TOTAL", "Tests score-dependent payment affordability.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT08", INTERACTION_FEATURES[7], "External score × payment burden", "EXT_SOURCE_2 * (AMT_ANNUITY/AMT_INCOME_TOTAL)", "EXT_SOURCE_2;AMT_ANNUITY;AMT_INCOME_TOTAL", "Tests score-dependent payment affordability.", "conditional", "row-wise", "Allowed raw-column formula"),
        ("INT09", INTERACTION_FEATURES[8], "External score × payment burden", "EXT_SOURCE_3 * (AMT_ANNUITY/AMT_INCOME_TOTAL)", "EXT_SOURCE_3;AMT_ANNUITY;AMT_INCOME_TOTAL", "Tests score-dependent payment affordability.", "conditional", "row-wise", "Allowed raw-column formula"),
    ]
    frame = pd.DataFrame(
        hypotheses,
        columns=[
            "feature_id", "feature_name", "feature_family", "mathematical_formula",
            "source_columns", "economic_reasoning", "expected_relationship",
            "transformation_scope", "rule_compliance_status",
        ],
    )
    frame["missing_value_risk"] = frame["feature_name"].str.contains("EXT|PEER_EXT")
    frame["zero_denominator_risk"] = frame["mathematical_formula"].str.contains("/")
    frame["outlier_risk"] = frame["feature_family"].str.contains("Peer")
    frame["duplicate_check"] = "Not an exact duplicate of baseline by formula; assess model redundancy."
    frame["implementation_status"] = "Implemented and family-tested"
    frame["validation_status"] = "Measured only as family ablation; not individually selected"
    frame.to_csv(OUT / "FEATURE_HYPOTHESES.csv", index=False)
    frame.to_csv(OUT / "FEATURE_REGISTRY.csv", index=False)


def write_formula_catalog() -> None:
    rows = [
        ("F01", "credit_to_goods", "AMT_CREDIT / AMT_GOODS_PRICE", "Loan / contract", "Financing intensity relative to goods price.", "Existing baseline has inverse CREDIT_TO_GOODS; this is a monotone re-expression.", "Retain only if model ablation confirms useful numerical representation."),
        ("F02", "annuity_to_credit", "AMT_ANNUITY / AMT_CREDIT", "Repayment structure", "Payment size relative to credit principal.", "Already represented in baseline.", "Do not annualize unless period is documented."),
        ("F03", "annuity_to_income", "AMT_ANNUITY / AMT_INCOME_TOTAL", "Affordability", "Payment relative to reported income.", "Already represented in baseline.", "No universal threshold assumed."),
        ("F04", "income_per_family_member", "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS", "Household", "Nominal income divided by household size.", "Already represented in baseline.", "Descriptive capacity proxy; not equal contribution claim."),
        ("F05", "employment_clean", "DAYS_EMPLOYED with 365243 mapped to missing plus sentinel flag", "Employment", "Avoid treating sentinel as duration.", "Already represented in baseline.", "Use sentinel explicitly."),
        ("F06", "external_score_relative_dispersion", "sample_std(available EXT_SOURCE_1..3) / mean(available scores); require >=2 scores", "External scores", "Cross-source disagreement scaled by average level.", "Existing baseline includes mean, range, std and pairwise interactions but not this robust relative statistic.", "Mean-near-zero stability and missingness require validation."),
        ("F07", "bureau_windows_separate", "Each supplied HOUR, MON, QRT, YEAR count separately", "Bureau inquiries", "Preserve distinct windows without assuming comparable durations.", "Raw columns already included.", "No sum-as-rate interpretation."),
        ("F08", "peer_relative_income", "robust z-score of log income against fit-only category median/IQR", "Peer deviation", "Measures unusual income relative to a recorded peer category.", "New in this batch.", "Fit-only statistics; rare groups back off."),
        ("F09", "peer_relative_burden", "robust z-score of credit/income against fit-only income-type x occupation median/IQR", "Peer deviation", "Tests debt burden relative to similar work/income categories.", "New in this batch.", "Fit-only statistics; no group target mean."),
        ("F10", "external_score_x_affordability", "EXT_SOURCE_i * (AMT_CREDIT / AMT_INCOME_TOTAL)", "Interaction", "Allows external risk evidence to modify borrowing burden.", "New in this batch.", "Rowwise transform; missing score remains missing."),
        ("F11", "external_score_x_financing", "EXT_SOURCE_i * (AMT_CREDIT / AMT_GOODS_PRICE)", "Interaction", "Allows risk evidence to modify financed share.", "New in this batch.", "Rowwise transform."),
        ("F12", "external_score_x_payment_burden", "EXT_SOURCE_i * (AMT_ANNUITY / AMT_INCOME_TOTAL)", "Interaction", "Allows risk evidence to modify payment burden.", "New in this batch.", "Rowwise transform."),
        ("F13", "family_payment_scaling", "AMT_ANNUITY * CNT_FAM_MEMBERS / AMT_INCOME_TOTAL", "Household", "Tests family-size adjustment.", "Previously tested in formula set.", "Questionable equal-responsibility assumption; not endorsed."),
        ("F14", "regional_pressure_ratio", "REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE", "Regional", "Ratio of a region rating and normalized population.", "Previously tested in formula set.", "Semantics uncertain; exploratory only."),
    ]
    catalog = pd.DataFrame(
        rows,
        columns=[
            "formula_id", "feature_name", "formula", "domain_family",
            "reasoning", "baseline_overlap_or_status", "validation_caution",
        ],
    )
    catalog["rule_status"] = "Supplied-column-derived; no external data"
    catalog["experiment_status"] = "See corrected family-level OOF results; not individually selected"
    catalog.to_csv(OUT / "DOMAIN_FORMULA_CATALOG_V3.csv", index=False)

    previous = pd.read_csv(
        ROOT
        / "experiments"
        / "phase2_advanced_feature_engineering"
        / "BASELINE_VALIDATION_RESULTS.csv"
    )
    previous["measurement_status"] = "corrected_inner_stopping_5fold"
    previous["source"] = str(
        ROOT
        / "experiments"
        / "phase2_advanced_feature_engineering"
        / "BASELINE_VALIDATION_RESULTS.csv"
    )
    previous.to_csv(OUT / "FORMULA_OPTIMIZATION_RESULTS.csv", index=False)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if set(train.columns) - {"TARGET"} != set(test.columns):
        raise ValueError("Train and test schemas do not match")
    if "TARGET" in test.columns:
        raise ValueError("Test dataset must not contain TARGET")

    write_dataset_anatomy(train, test)
    write_hypothesis_registry()
    write_formula_catalog()

    labels = train["TARGET"].to_numpy(dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(train, labels))
    pd.DataFrame(
        [
            {"SK_ID_CURR": int(train.iloc[i]["SK_ID_CURR"]), "fold": fold}
            for fold, (_, valid_idx) in enumerate(folds, 1)
            for i in valid_idx
        ]
    ).to_csv(OUT / "FOLD_ASSIGNMENTS.csv", index=False)

    fold_frames = []
    oof_columns: dict[str, np.ndarray] = {}
    variants = [
        ("raw_baseline", "baseline"),
        ("external_interactions", "external_interactions"),
        ("peer_deviations", "peer_deviations"),
        ("combined_peer_and_interactions", "combined"),
    ]
    for name, variant in variants:
        result, oof = run_cv(train, folds, variant)
        result["experiment"] = name
        fold_frames.append(result)
        oof_columns[name] = oof
    fold_results = pd.concat(fold_frames, ignore_index=True)
    fold_results.to_csv(OUT / "FOLD_LEVEL_CV_RESULTS.csv", index=False)

    oof_table = pd.DataFrame(
        {
            "SK_ID_CURR": train["SK_ID_CURR"],
            "TARGET": labels,
            **{f"OOF_{name}": pred for name, pred in oof_columns.items()},
        }
    )
    oof_table.to_csv(OUT / "OOF_PREDICTIONS.csv", index=False)
    summary_rows = []
    base_pooled = roc_auc_score(labels, oof_columns["raw_baseline"])
    for name, _ in variants:
        subset = fold_results[fold_results["experiment"] == name]
        aucs = subset["auc"].to_numpy(dtype=np.float64)
        pooled = float(roc_auc_score(labels, oof_columns[name]))
        summary_rows.append(
            {
                "experiment": name,
                "mean_fold_auc": float(aucs.mean()),
                "std_fold_auc": float(aucs.std(ddof=1)),
                "pooled_oof_auc": pooled,
                "delta_pooled_vs_baseline": pooled - base_pooled,
                "folds_improved_vs_baseline": int(
                    (
                        subset["auc"].to_numpy()
                        > fold_results[
                            fold_results["experiment"] == "raw_baseline"
                        ]["auc"].to_numpy()
                    ).sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "FEATURE_FAMILY_ABLATION.csv", index=False)
    summary.to_csv(OUT / "FEATURE_COMBINATION_RESULTS.csv", index=False)
    summary[summary["experiment"].str.contains("peer")].to_csv(
        OUT / "PEER_GROUP_FEATURE_RESULTS.csv", index=False
    )
    summary[summary["experiment"].str.contains("external_interactions")].to_csv(
        OUT / "INTERACTION_DISCOVERY_RESULTS.csv", index=False
    )
    summary[summary["experiment"].str.contains("combined")].to_csv(
        OUT / "GROUP_DEVIATION_FEATURE_RESULTS.csv", index=False
    )

    # Descriptive missingness intersections are label-free.
    missing_cols = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    miss = train[missing_cols].isna()
    missing_patterns = (
        miss.astype("uint8")
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .rename_axis("missingness_pattern")
        .reset_index(name="row_count")
    )
    missing_patterns["row_fraction"] = missing_patterns["row_count"] / len(train)
    missing_patterns.to_csv(OUT / "MISSINGNESS_PATTERN_RESULTS.csv", index=False)

    pd.DataFrame(
        [
            {
                "family": "Peer group deviations",
                "candidate_count": len(PEER_FEATURES),
                "tested": True,
                "result": summary.loc[
                    summary["experiment"] == "peer_deviations",
                    "delta_pooled_vs_baseline",
                ].iloc[0],
            },
            {
                "family": "External score × financial interactions",
                "candidate_count": len(INTERACTION_FEATURES),
                "tested": True,
                "result": summary.loc[
                    summary["experiment"] == "external_interactions",
                    "delta_pooled_vs_baseline",
                ].iloc[0],
            },
            {
                "family": "Combined peer + external interactions",
                "candidate_count": len(PEER_FEATURES) + len(INTERACTION_FEATURES),
                "tested": True,
                "result": summary.loc[
                    summary["experiment"] == "combined_peer_and_interactions",
                    "delta_pooled_vs_baseline",
                ].iloc[0],
            },
        ]
    ).to_csv(OUT / "FEATURE_SCREENING_RESULTS.csv", index=False)

    # Report candidate distribution health from the full training rows. These
    # are diagnostics, not label-driven candidate selection.
    diagnostics_rows = []
    for name, variant in variants[1:]:
        for _, (fit_idx, valid_idx) in enumerate(folds, 1):
            fit_raw = train.iloc[fit_idx]
            valid_raw = train.iloc[valid_idx]
            if variant in ("external_interactions", "combined"):
                extras = interaction_features(
                    valid_raw.drop(columns=["TARGET"], errors="ignore")
                )
            else:
                extras = peer_deviations(
                    fit_raw.drop(columns=["TARGET"], errors="ignore"),
                    valid_raw.drop(columns=["TARGET"], errors="ignore"),
                )
            for column in extras:
                arr = extras[column].to_numpy(dtype=np.float64)
                finite = arr[np.isfinite(arr)]
                diagnostics_rows.append(
                    {
                        "feature": column,
                        "fold": _ + 1,
                        "n_valid": int(np.isfinite(arr).sum()),
                        "n_missing": int((~np.isfinite(arr)).sum()),
                        "min": float(finite.min()) if finite.size else np.nan,
                        "p01": float(np.quantile(finite, 0.01)) if finite.size else np.nan,
                        "median": float(np.median(finite)) if finite.size else np.nan,
                        "p99": float(np.quantile(finite, 0.99)) if finite.size else np.nan,
                        "max": float(finite.max()) if finite.size else np.nan,
                    }
                )
    pd.DataFrame(diagnostics_rows).drop_duplicates(
        ["feature", "fold"]
    ).to_csv(OUT / "FEATURE_DISTRIBUTION_DIAGNOSTICS.csv", index=False)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = {
        "timestamp_utc": timestamp,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_sha256": sha256(TRAIN_PATH),
        "test_sha256": sha256(TEST_PATH),
        "baseline_source": str(BASELINE_SOURCE),
        "baseline_source_sha256": sha256(BASELINE_SOURCE),
        "validation_source": str(VALIDATION_SOURCE),
        "folds": 5,
        "outer_seed": 42,
        "early_stopping": "inner 10% stratified split of outer-training partition",
        "peer_group_statistics": "fit on inner-fit or outer-fit rows only; min group size 30; global fit-only median/IQR fallback",
        "rules": "provided train.csv/test.csv only; no test-label or external-data use",
        "python": sys.version.replace("\n", " "),
        "lightgbm": lgb.__version__,
        "runtime_seconds": round(time.time() - started, 2),
    }
    (OUT / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        "numpy\npandas\nscikit-learn\nlightgbm\nopenpyxl\n",
        encoding="utf-8",
    )
    log = [
        f"Execution timestamp UTC: {timestamp}",
        f"Train shape: {train.shape}; Test shape: {test.shape}",
        f"Command: python {Path(__file__).resolve()}",
        f"Runtime seconds: {metadata['runtime_seconds']}",
        "Status: successful",
        "Outer fold stopping leakage: none; iteration selection used inner training split.",
        "Peer group fit scope: fit partition only.",
        "External data: none.",
        "",
        summary.to_string(index=False),
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log), encoding="utf-8")

    (OUT / "DATASET_ANATOMY_DEEP_DIVE.md").write_text(
        (OUT / "DATASET_ANATOMY_DEEP_DIVE.md").read_text(encoding="utf-8")
        + "\n\n## Phase 2B experiment\n\n"
        + summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    (OUT / "FEATURE_REDUNDANCY_REPORT.md").write_text(
        "# Feature redundancy report\n\n"
        "- The previous formula experiment established that many ratios duplicate baseline features or differ only by a monotone scaling.\n"
        "- This batch tests peer-normalized deviations and external-score interactions, not duplicate ratios.\n"
        "- Peer features use fit-only medians and IQRs, with rare groups falling back to fit-only global statistics.\n"
        "- Each included interaction combines an official external score with an existing loan-burden ratio; LightGBM may already learn some interactions, so the ablation tests incremental value.\n"
        "- No univariate target ranking was used to select these features.\n",
        encoding="utf-8",
    )
    (OUT / "EXPERIMENT_LOG.md").write_text(
        "# Phase 2B experiment log\n\n"
        f"- {timestamp}: completed dataset profile and predeclared batch of {len(PEER_FEATURES)} peer deviations plus {len(INTERACTION_FEATURES)} external-score interactions.\n"
        "- Fixed five stratified folds (seed 42); identical fold IDs across variants.\n"
        "- Inner-fold early stopping; no outer-fold labels used for stopping or selecting features.\n"
        "- Full 300–500 candidate generation, clustering, categorical cross encodings, model comparison, and ensemble optimization are not performed in this batch.\n"
        f"- Runtime seconds: {metadata['runtime_seconds']}.\n",
        encoding="utf-8",
    )
    (OUT / "RULE_COMPLIANCE_REPORT.md").write_text(
        "# Rule compliance report — this research batch\n\n"
        "- Data inputs: only the supplied competition train and test CSVs; test used for schema validation only.\n"
        "- No external datasets, outside labels, test labels, hand labeling, manual prediction overrides, or target encoding.\n"
        "- Candidate features use supplied columns only. Peer-group statistics are fitted within each training partition and applied to held-out rows.\n"
        "- No pooled train/test covariate fitting.\n"
        "- This report reflects the local rule materials previously reviewed; organizer clarification is still needed for any rule ambiguity outside this batch.\n"
        "- No submission was generated.\n",
        encoding="utf-8",
    )
    (OUT / "PHASE2B_FINAL_REPORT.md").write_text(
        "# Phase 2B — first feature-intelligence results\n\n"
        f"Execution timestamp: {timestamp}\n\n"
        "## Results\n\n"
        + summary.to_string(index=False)
        + "\n\n## Decision\n\n"
        "Retain only families with positive pooled OOF deltas for a separate, predeclared confirmation run. A single five-fold development run is not final proof because candidate families were compared on the same OOF folds. A negative result rejects that family for the current model/configuration; it does not prove the underlying domain idea has no value.\n\n"
        "## Scope not yet executed\n\n"
        "Clustering, categorical frequency/cross features, comprehensive per-column hypothesis generation, broad nonlinear screening, residual-driven discovery, model comparison, and ensemble optimization remain pending. No claim is made that Phase 2B is complete or that the 0.80 target is reached.\n",
        encoding="utf-8",
    )

    print(f"Execution complete: {timestamp}")
    print(f"Runtime: {metadata['runtime_seconds']} seconds")
    print(summary.to_string(index=False))
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
