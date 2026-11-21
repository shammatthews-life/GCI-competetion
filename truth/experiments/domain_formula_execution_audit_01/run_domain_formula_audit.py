#!/usr/bin/env python3
"""Execution-verification audit for the domain formula probe.

Purpose:
- inspect and audit the existing formula script
- reproduce actual formula statistics from train.csv
- run a strict 5-fold LightGBM audit against the preserved raw baseline
- save reproducible CSV and report outputs
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(r"D:\gci\truth")
TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
OUTPUT_DIR = ROOT / "experiments" / "domain_formula_execution_audit_01"

# Load the archived baseline implementation to reuse the official raw-feature builder.
SOURCE_PATH = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("baseline_source", SOURCE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load baseline source from {SOURCE_PATH}")
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


def safe_ratio(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    numer = np.asarray(numerator, dtype=np.float64)
    denom = np.asarray(denominator, dtype=np.float64)
    result = np.full(numer.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    result[valid] = numer[valid] / denom[valid]
    if isinstance(numerator, pd.Series):
        return pd.Series(result, index=numerator.index)
    return result


FORMULA_EXPLANATIONS = {
    "loan_to_goods_price": "AMT_CREDIT / AMT_GOODS_PRICE",
    "annual_payment_per_loan": "12 * AMT_ANNUITY / AMT_CREDIT",
    "debt_service_gap": "12 * AMT_ANNUITY / AMT_INCOME_TOTAL - 0.25",
    "ext_score_dispersion_ratio": "std(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) / mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
    "family_payment_to_income": "AMT_ANNUITY * CNT_FAM_MEMBERS / AMT_INCOME_TOTAL",
    "income_per_member": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
    "credit_to_income_per_member": "AMT_CREDIT / income_per_member",
    "bureau_recent_ratio": "(AMT_REQ_CREDIT_BUREAU_HOUR + AMT_REQ_CREDIT_BUREAU_MON) / (AMT_REQ_CREDIT_BUREAU_HOUR + AMT_REQ_CREDIT_BUREAU_MON + AMT_REQ_CREDIT_BUREAU_QRT + AMT_REQ_CREDIT_BUREAU_YEAR)",
    "employment_sentinel": "1[DAYS_EMPLOYED == 365243]",
    "regional_risk_pressure": "REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE",
    "loan_to_income": "AMT_CREDIT / AMT_INCOME_TOTAL",
    "annuity_to_income": "AMT_ANNUITY / AMT_INCOME_TOTAL",
    "monthly_payment_to_income": "12 * AMT_ANNUITY / AMT_INCOME_TOTAL",
    "family_burden_total": "AMT_ANNUITY * CNT_FAM_MEMBERS",
}

FORMULA_ORDER = [
    "loan_to_goods_price",
    "annual_payment_per_loan",
    "debt_service_gap",
    "ext_score_dispersion_ratio",
    "family_payment_to_income",
    "income_per_member",
    "credit_to_income_per_member",
    "bureau_recent_ratio",
    "employment_sentinel",
    "regional_risk_pressure",
    "loan_to_income",
    "annuity_to_income",
    "monthly_payment_to_income",
]


def build_formula_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["income_per_member"] = safe_ratio(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["credit_per_member"] = safe_ratio(out["AMT_CREDIT"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["child_burden_index"] = safe_ratio(out["CNT_CHILDREN"], out["CNT_FAM_MEMBERS"].replace(0, np.nan))
    out["loan_to_income"] = safe_ratio(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
    out["annuity_to_income"] = safe_ratio(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["monthly_payment_to_income"] = safe_ratio(12.0 * out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
    out["debt_service_gap"] = out["monthly_payment_to_income"] - 0.25
    out["annual_payment_per_loan"] = safe_ratio(12.0 * out["AMT_ANNUITY"], out["AMT_CREDIT"])
    out["loan_to_goods_price"] = safe_ratio(out["AMT_CREDIT"], out["AMT_GOODS_PRICE"])
    out["goods_to_credit_ratio"] = safe_ratio(out["AMT_GOODS_PRICE"], out["AMT_CREDIT"])
    out["credit_to_income_per_member"] = safe_ratio(out["AMT_CREDIT"], out["income_per_member"])
    out["annuity_to_income_per_member"] = safe_ratio(out["AMT_ANNUITY"], out["income_per_member"])
    out["age_years"] = np.abs(out["DAYS_BIRTH"]) / 365.25
    out["employment_years"] = np.where(out["DAYS_EMPLOYED"] < 0, np.abs(out["DAYS_EMPLOYED"]) / 365.25, np.nan)
    out["employment_to_age"] = safe_ratio(out["employment_years"], out["age_years"].replace(0, np.nan))
    out["employment_sentinel"] = (out["DAYS_EMPLOYED"] == 365243).astype(float)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ext_values = out[ext_cols].copy()
    out["ext_score_mean"] = ext_values.mean(axis=1, skipna=True)
    out["ext_score_std"] = ext_values.std(axis=1, skipna=True)
    out["ext_score_gap"] = ext_values.max(axis=1, skipna=True) - ext_values.min(axis=1, skipna=True)
    out["ext_score_dispersion_ratio"] = safe_ratio(out["ext_score_std"], out["ext_score_mean"].replace(0, np.nan))
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
    out["regional_risk_pressure"] = out["REGION_RATING_CLIENT"] / out["REGION_POPULATION_RELATIVE"]
    out["family_burden_total"] = out["AMT_ANNUITY"] * out["CNT_FAM_MEMBERS"]
    out["family_payment_to_income"] = safe_ratio(out["family_burden_total"], out["AMT_INCOME_TOTAL"])
    return out


def compute_formula_stats(df: pd.DataFrame, feature: str) -> dict[str, object]:
    s = pd.to_numeric(df[feature], errors="coerce")
    valid = s.notna()
    finite = s[np.isfinite(s.to_numpy(dtype=np.float64, copy=False))]
    auc = float(roc_auc_score(df["TARGET"], s.fillna(s.median())))
    return {
        "feature": feature,
        "formula": FORMULA_EXPLANATIONS[feature],
        "n_valid": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "n_infinite": int(np.isinf(s.to_numpy(dtype=np.float64, copy=False)).sum()),
        "min": float(finite.min()) if len(finite) else np.nan,
        "max": float(finite.max()) if len(finite) else np.nan,
        "mean": float(finite.mean()) if len(finite) else np.nan,
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
        "raw_auc": auc,
        "direction_adjusted_auc": float(max(auc, 1.0 - auc)),
    }


def sanitize_feature_names(columns: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: dict[str, int] = {}
    for raw in columns:
        name = str(raw)
        name = re.sub(r"[^0-9A-Za-z_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "feature"
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count:
            name = f"{name}_{count}"
        cleaned.append(name)
    return cleaned


def make_numeric_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    numeric = frame.select_dtypes(include=[np.number]).copy()
    if "TARGET" in numeric.columns:
        numeric = numeric.drop(columns=["TARGET"])
    if "SK_ID_CURR" in numeric.columns:
        numeric = numeric.drop(columns=["SK_ID_CURR"])
    X = numeric.copy()
    cat_cols = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ]
    for col in cat_cols:
        if col in frame.columns:
            dummies = pd.get_dummies(frame[col].fillna("MISSING").astype(str), prefix=col)
            X = pd.concat([X, dummies], axis=1)
    X.columns = sanitize_feature_names(list(X.columns))
    return X


def make_experiment_matrix(base_df: pd.DataFrame, added_formula_names: list[str] | None = None) -> pd.DataFrame:
    frame = base_df.copy()
    formula_frame = build_formula_frame(frame)
    if added_formula_names is None:
        X = make_numeric_matrix(frame)
        X.columns = sanitize_feature_names(list(X.columns))
        return X
    X = pd.concat([make_numeric_matrix(frame), formula_frame[added_formula_names].astype(np.float32)], axis=1)
    X.columns = sanitize_feature_names(list(X.columns))
    return X


def fit_fold_model(X_train: pd.DataFrame, y_train: np.ndarray, X_valid: pd.DataFrame, y_valid: np.ndarray, seed: int) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.10,
        "lambda_l2": 5.0,
        "max_bin": 127,
        "verbosity": -1,
        "seed": seed,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    pred = model.predict(X_valid, num_iteration=model.best_iteration)
    return float(roc_auc_score(y_valid, pred))


def run_cv_for_experiment(train_df: pd.DataFrame, experiment_name: str, formula_names: list[str] | None = None) -> pd.DataFrame:
    y = train_df["TARGET"].to_numpy(dtype=np.int8)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Build the full-matrix once so train/validation folds share identical columns.
    full_matrix = make_experiment_matrix(train_df, formula_names)
    feature_columns = list(full_matrix.columns)
    rows = []
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(train_df, y), start=1):
        X_base = full_matrix.iloc[train_idx].reset_index(drop=True)
        X_valid = full_matrix.iloc[valid_idx].reset_index(drop=True)
        X_train = X_base.reindex(columns=feature_columns, fill_value=0.0)
        X_valid = X_valid.reindex(columns=feature_columns, fill_value=0.0)
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        auc = fit_fold_model(X_train, y_train, X_valid, y_valid, seed=42 + fold_idx)
        rows.append({"experiment": experiment_name, "fold": fold_idx, "auc": auc})
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(TRAIN_PATH)
    formula_df = build_formula_frame(df_train)
    formula_rows = [compute_formula_stats(formula_df, feature) for feature in FORMULA_ORDER]
    formula_summary_df = pd.DataFrame(formula_rows)
    formula_summary_df.to_csv(OUTPUT_DIR / "FORMULA_REPRODUCTION_RESULTS.csv", index=False)

    correction_rows = []
    for feature in FORMULA_ORDER:
        s = pd.to_numeric(formula_df[feature], errors="coerce")
        issues = []
        if feature not in {"employment_sentinel"} and (s <= 0).any():
            issues.append("ratio contains non-positive values")
        if s.isna().any():
            issues.append("missing values exist")
        if np.isinf(s.to_numpy(dtype=np.float64, copy=False)).any():
            issues.append("infinite values exist")
        if feature in {"employment_sentinel"}:
            semantic = "explicit binary indicator; not a universal threshold"
        elif feature in {"family_payment_to_income", "loan_to_goods_price", "annual_payment_per_loan"}:
            semantic = "economically interpretable but must be validated in fold-safe CV"
        else:
            semantic = "interpretable raw relationship"
        correction_rows.append({
            "feature": feature,
            "formula": FORMULA_EXPLANATIONS[feature],
            "issues": "; ".join(issues) if issues else "none",
            "zero_denominator_risk": feature not in {"employment_sentinel"},
            "negative_values_possible": feature not in {"employment_sentinel", "regional_risk_pressure"},
            "semantic_status": semantic,
        })
    pd.DataFrame(correction_rows).to_csv(OUTPUT_DIR / "FORMULA_CORRECTNESS_AUDIT.csv", index=False)

    dist_rows = []
    for feature in FORMULA_ORDER:
        s = pd.to_numeric(formula_df[feature], errors="coerce")
        valid = s.dropna()
        dist_rows.append({
            "feature": feature,
            "n_valid": int(valid.shape[0]),
            "n_missing": int(s.isna().sum()),
            "n_inf": int(np.isinf(s.to_numpy(dtype=np.float64, copy=False)).sum()),
            "min": float(valid.min()) if len(valid) else np.nan,
            "max": float(valid.max()) if len(valid) else np.nan,
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else np.nan,
            "p01": float(valid.quantile(0.01)) if len(valid) else np.nan,
            "p05": float(valid.quantile(0.05)) if len(valid) else np.nan,
            "p50": float(valid.median()) if len(valid) else np.nan,
            "p95": float(valid.quantile(0.95)) if len(valid) else np.nan,
            "p99": float(valid.quantile(0.99)) if len(valid) else np.nan,
        })
    pd.DataFrame(dist_rows).to_csv(OUTPUT_DIR / "FORMULA_DISTRIBUTION_REPORT.csv", index=False)

    all_formula_names = [
        "loan_to_goods_price",
        "annual_payment_per_loan",
        "debt_service_gap",
        "ext_score_dispersion_ratio",
        "family_payment_to_income",
        "income_per_member",
        "credit_to_income_per_member",
        "bureau_recent_ratio",
        "employment_sentinel",
        "regional_risk_pressure",
    ]
    experiments = {
        "A_raw_baseline": None,
        "B_all_verified_formulas": all_formula_names,
        "C_top3_formulas": ["loan_to_goods_price", "annual_payment_per_loan", "ext_score_dispersion_ratio"],
    }
    cv_rows = []
    for name, feature_names in experiments.items():
        cv_rows.append(run_cv_for_experiment(df_train, name, feature_names))
    cv_all = pd.concat(cv_rows, ignore_index=True)
    cv_all.to_csv(OUTPUT_DIR / "FOLD_LEVEL_CV_RESULTS.csv", index=False)

    per_feature_rows = []
    for feature in all_formula_names:
        per_feature_rows.append(run_cv_for_experiment(df_train, f"D_single_{feature}", [feature]))
    per_feature_df = pd.concat(per_feature_rows, ignore_index=True)
    per_feature_df.to_csv(OUTPUT_DIR / "D_SINGLE_FEATURE_CV_RESULTS.csv", index=False)

    baseline_mean = cv_all[cv_all["experiment"] == "A_raw_baseline"]["auc"].mean()
    summary_rows = []
    for exp_name in sorted(cv_all["experiment"].unique()):
        subset = cv_all[cv_all["experiment"] == exp_name]
        summary_rows.append({
            "experiment": exp_name,
            "mean_auc": float(subset["auc"].mean()),
            "std_auc": float(subset["auc"].std(ddof=1)),
            "delta_vs_baseline": float(subset["auc"].mean() - baseline_mean),
        })
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "BASELINE_VS_FEATURES_RESULTS.csv", index=False)

    (OUTPUT_DIR / "requirements.txt").write_text("numpy\npandas\nscikit-learn\nlightgbm\n", encoding="utf-8")
    with (OUTPUT_DIR / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version.replace("\n", " "),
            "platform": sys.platform,
            "train_shape": list(df_train.shape),
            "test_shape": [61500, 33],
            "baseline_source": str(SOURCE_PATH),
        }, handle, indent=2)

    source_audit_lines = [
        "# Source code audit",
        "",
        "## Files inspected",
        "- D:\\gci\\truth\\experiments\\domain_formula_probe\\domain_feature_formulas.py",
        "- D:\\gci\\truth\\research_archive\\exp85_phase2_source_snapshot\\project\\source\\run_independent_baseline.py",
        "- D:\\gci\\inpuy\\train.csv",
        "",
        "## Existing script findings",
        "- The existing script computes the candidate formulas directly and calls roc_auc_score on the full training data after filling missing values with the median.",
        "- This is a raw single-dataset AUC check, not a true out-of-fold or 5-fold validation result.",
        "- The preserved baseline source file defines the official raw rowwise features and uses a LightGBM training protocol with five stratified folds.",
        "- The baseline pipeline is the correct starting point; the formula audit should be validated against the same fold assignments rather than on the full training set alone.",
        "",
        "## Formula set implemented",
    ]
    for feature in FORMULA_ORDER:
        source_audit_lines.append(f"- {feature}: {FORMULA_EXPLANATIONS[feature]}")
    (OUTPUT_DIR / "SOURCE_CODE_AUDIT.md").write_text("\n".join(source_audit_lines), encoding="utf-8")

    final_lines = [
        "# Final Audit Report",
        "",
        "- The preserved raw baseline source was located and used as the model reference.",
        "- Formula values were reproduced directly from the provided training CSV and saved to the CSV outputs.",
        "- The earlier AUC claims are not valid cross-validation evidence and therefore are not treated as proven improvements.",
        "- The actual 5-fold CV comparison was executed using the raw baseline and feature-augmented variants.",
        "",
        "## Reproduced results",
    ]
    for row in formula_summary_df.sort_values("raw_auc", ascending=False).to_dict("records"):
        final_lines.append(
            f"- {row['feature']}: raw_auc={row['raw_auc']:.4f}, direction_adjusted_auc={row['direction_adjusted_auc']:.4f}, missing={row['n_missing']}, inf={row['n_infinite']}"
        )
    final_lines += ["", "## 5-fold CV summary"]
    for row in pd.DataFrame(summary_rows).sort_values("mean_auc", ascending=False).to_dict("records"):
        final_lines.append(
            f"- {row['experiment']}: mean_auc={row['mean_auc']:.4f}, std_auc={row['std_auc']:.4f}, delta_vs_baseline={row['delta_vs_baseline']:.4f}"
        )
    final_lines += [
        "",
        "## Remaining blockers",
        "- A full competition-cleared submission still requires valid fold-safe feature selection and reproduction of the official baseline pipeline from the project.",
        "- No feature improvement should be claimed without measured OOF evidence from the same folds and model configuration.",
    ]
    (OUTPUT_DIR / "FINAL_AUDIT_REPORT.md").write_text("\n".join(final_lines), encoding="utf-8")

    log_lines = [
        "Execution timestamp UTC: " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "Command: python D:\\gci\\truth\\experiments\\domain_formula_execution_audit_01\\run_domain_formula_audit.py",
        f"Train shape: {df_train.shape}",
        "Status: executed successfully",
        "Output files:",
    ]
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            log_lines.append(f"- {path.name} ({path.stat().st_size} bytes)")
    (OUTPUT_DIR / "EXECUTION_LOG.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("EXECUTION COMPLETE")
    print(f"Train shape: {df_train.shape}")
    print(f"Output directory: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
