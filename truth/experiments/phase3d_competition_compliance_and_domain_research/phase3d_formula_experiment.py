from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE3B = ROOT / "experiments" / "phase3b_deep_validation"
PHASE3C = ROOT / "experiments" / "phase3c_model_intelligence"
NOTEBOOK = Path(r"C:\Users\SHARON\Downloads\GCI_v5_dropPF4_colab.ipynb")
OUT = ROOT / "experiments" / "phase3d_competition_compliance_and_domain_research"
ID = "SK_ID_CURR"
TARGET = "TARGET"
EPSILON = 1e-6


FORMULA_INFO = [
    {
        "formula": "RISK_ADJUSTED_LTV",
        "definition": "(AMT_CREDIT / AMT_GOODS_PRICE) / (mean(observed EXT_SOURCE_1..3) + 0.05)",
        "required_inputs": "AMT_CREDIT; AMT_GOODS_PRICE; EXT_SOURCE_1..3",
        "decision": "TESTED",
        "audit_note": "LTV exists in baseline. Fixed 0.05 score-scale stabilizer limits near-zero denominator extremes; external-score direction is not explicit in the dictionary.",
    },
    {
        "formula": "EXTERNAL_SCORE_CV",
        "definition": "population_std(observed EXT_SOURCE_1..3) / mean(observed EXT_SOURCE_1..3)",
        "required_inputs": "EXT_SOURCE_1..3",
        "decision": "TESTED",
        "audit_note": "Baseline has EXT_STD and EXT_MEAN separately; CV is a normalized dispersion candidate. Requires >=2 scores and mean > 1e-6.",
    },
    {
        "formula": "LTV_EXTERNAL_UNCERTAINTY",
        "definition": "(AMT_CREDIT / AMT_GOODS_PRICE) * EXTERNAL_SCORE_CV",
        "required_inputs": "AMT_CREDIT; AMT_GOODS_PRICE; EXT_SOURCE_1..3",
        "decision": "TESTED",
        "audit_note": "Risk-stress interaction. No use of labels, pooled statistics, or external data.",
    },
    {
        "formula": "FINANCIAL_STRESS_PROXY",
        "definition": "RISK_ADJUSTED_LTV * (AMT_ANNUITY / AMT_INCOME_TOTAL)",
        "required_inputs": "AMT_CREDIT; AMT_GOODS_PRICE; EXT_SOURCE_1..3; AMT_ANNUITY; AMT_INCOME_TOTAL",
        "decision": "TESTED_WITH_PERIOD_CAVEAT",
        "audit_note": "Adds a nonlinear interaction of existing baseline inputs. Interpret only as a unitless proxy; AMT_ANNUITY periodicity is unspecified.",
    },
    {
        "formula": "BUREAU_MONTH_TO_OLDER_WINDOWS",
        "definition": "AMT_REQ_CREDIT_BUREAU_MON / (AMT_REQ_CREDIT_BUREAU_QRT + AMT_REQ_CREDIT_BUREAU_YEAR)",
        "required_inputs": "AMT_REQ_CREDIT_BUREAU_MON; AMT_REQ_CREDIT_BUREAU_QRT; AMT_REQ_CREDIT_BUREAU_YEAR",
        "decision": "TESTED",
        "audit_note": "Descriptive window-count ratio only; not called acceleration. Missing/zero denominator yields missing.",
    },
    {
        "formula": "BUREAU_MONTH_SHARE",
        "definition": "MON / (MON + QRT + YEAR)",
        "required_inputs": "AMT_REQ_CREDIT_BUREAU_MON; AMT_REQ_CREDIT_BUREAU_QRT; AMT_REQ_CREDIT_BUREAU_YEAR",
        "decision": "NOT_TESTED_REDUNDANT",
        "audit_note": "Closely related to BUREAU_MONTH_TO_OLDER_WINDOWS; not added as a second near-duplicate candidate.",
    },
    {
        "formula": "COLLATERAL_GAP",
        "definition": "AMT_GOODS_PRICE - AMT_CREDIT",
        "required_inputs": "AMT_GOODS_PRICE; AMT_CREDIT",
        "decision": "NOT_TESTED_REDUNDANT",
        "audit_note": "Not necessarily a down payment; may be negative and is a one-to-one transform of goods price and credit. Baseline already includes both and LTV.",
    },
    {
        "formula": "DOWN_PAYMENT_SHARE_PROXY",
        "definition": "(AMT_GOODS_PRICE - AMT_CREDIT) / AMT_GOODS_PRICE",
        "required_inputs": "AMT_GOODS_PRICE; AMT_CREDIT",
        "decision": "NOT_TESTED_REDUNDANT",
        "audit_note": "Exactly 1 - LTV when goods price is nonzero; not independent evidence and not proven to be an actual down payment.",
    },
    {
        "formula": "ANNUITY_TO_INCOME",
        "definition": "AMT_ANNUITY / AMT_INCOME_TOTAL",
        "required_inputs": "AMT_ANNUITY; AMT_INCOME_TOTAL",
        "decision": "ALREADY_IN_BASELINE_NOT_RETESTED",
        "audit_note": "Already present in the preserved baseline. Dictionary says 'Loan annuity' but does not state its periodicity; do not call annualized PTI without confirmation.",
    },
    {
        "formula": "MONTHLY_PAYMENT_TO_INCOME",
        "definition": "12 * AMT_ANNUITY / AMT_INCOME_TOTAL",
        "required_inputs": "AMT_ANNUITY; AMT_INCOME_TOTAL",
        "decision": "NOT_TESTED_PERIOD_UNVERIFIED",
        "audit_note": "Annualization assumes annuity is monthly; official local dictionary does not establish this period.",
    },
    {
        "formula": "INCOME_PER_FAMILY_MEMBER",
        "definition": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
        "required_inputs": "AMT_INCOME_TOTAL; CNT_FAM_MEMBERS",
        "decision": "ALREADY_IN_BASELINE_NOT_RETESTED",
        "audit_note": "Already present; equal household responsibility is not implied by the definition.",
    },
    {
        "formula": "ANNUITY_PER_FAMILY_MEMBER",
        "definition": "AMT_ANNUITY / CNT_FAM_MEMBERS",
        "required_inputs": "AMT_ANNUITY; CNT_FAM_MEMBERS",
        "decision": "NOT_TESTED_UNJUSTIFIED_ALLOCATION",
        "audit_note": "Equal payment allocation across family members is not supported by the dictionary.",
    },
    {
        "formula": "ANNUITY_TO_GOODS_PRICE",
        "definition": "AMT_ANNUITY / AMT_GOODS_PRICE",
        "required_inputs": "AMT_ANNUITY; AMT_GOODS_PRICE",
        "decision": "NOT_TESTED_PERIOD_UNVERIFIED",
        "audit_note": "Existing units are ambiguous and no repayment period is supplied; baseline already has payment/collateral inputs.",
    },
    {
        "formula": "AMORTIZING_LOAN_IMPLIED_RATE",
        "definition": "Invert annuity/credit = r*(1+r)^n / ((1+r)^n - 1)",
        "required_inputs": "AMT_ANNUITY; AMT_CREDIT; repayment count n; payment interval",
        "decision": "NOT_IDENTIFIABLE_FROM_AVAILABLE_COLUMNS",
        "audit_note": "No observed repayment term or payment interval exists. For fixed assumed n, inversion is a monotone transform of annuity/credit and adds no rank information; not run as an experiment.",
    },
    {
        "formula": "DSCR",
        "definition": "(AMT_INCOME_TOTAL / 12) / AMT_ANNUITY",
        "required_inputs": "AMT_INCOME_TOTAL; AMT_ANNUITY; income and annuity periods",
        "decision": "ALREADY_IN_BASELINE_PERIOD_UNVERIFIED",
        "audit_note": "Baseline has this exact 12x conversion; income/annuity periods are not specified by the local dictionary.",
    },
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray, positive_denominator=False) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator > 0 if positive_denominator else denominator != 0
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    result[valid] = numerator[valid] / denominator[valid]
    result[~np.isfinite(result)] = np.nan
    return result


def make_formula_features(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET in raw.columns:
        raise ValueError("Formula builder must not receive TARGET.")
    get = lambda column: raw[column].to_numpy(dtype=np.float64)
    credit = get("AMT_CREDIT")
    goods = get("AMT_GOODS_PRICE")
    ext = np.column_stack([get(f"EXT_SOURCE_{index}") for index in (1, 2, 3)])
    ext_count = np.isfinite(ext).sum(axis=1)
    ext_sum = np.where(np.isfinite(ext), ext, 0.0).sum(axis=1)
    ext_mean = np.divide(
        ext_sum,
        ext_count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_var = np.divide(
        np.where(np.isfinite(ext), (ext - ext_mean[:, None]) ** 2, 0.0).sum(axis=1),
        ext_count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=ext_count > 0,
    )
    ext_std = np.sqrt(ext_var)
    ext_cv = np.full(len(raw), np.nan, dtype=np.float64)
    cv_valid = (ext_count >= 2) & np.isfinite(ext_mean) & (ext_mean > EPSILON)
    ext_cv[cv_valid] = ext_std[cv_valid] / ext_mean[cv_valid]

    ltv = safe_ratio(credit, goods, positive_denominator=True)
    risk_adjusted_ltv = safe_ratio(ltv, ext_mean + 0.05, positive_denominator=True)
    ltv_uncertainty = ltv * ext_cv
    ltv_uncertainty[~np.isfinite(ltv_uncertainty)] = np.nan
    payment_to_income = safe_ratio(
        get("AMT_ANNUITY"), get("AMT_INCOME_TOTAL"), positive_denominator=True
    )
    financial_stress = risk_adjusted_ltv * payment_to_income
    financial_stress[~np.isfinite(financial_stress)] = np.nan

    month = get("AMT_REQ_CREDIT_BUREAU_MON")
    older = get("AMT_REQ_CREDIT_BUREAU_QRT") + get("AMT_REQ_CREDIT_BUREAU_YEAR")
    bureau_ratio = safe_ratio(month, older, positive_denominator=True)

    result = pd.DataFrame(
        {
            "RISK_ADJUSTED_LTV": risk_adjusted_ltv,
            "EXTERNAL_SCORE_CV": ext_cv,
            "LTV_EXTERNAL_UNCERTAINTY": ltv_uncertainty,
            "FINANCIAL_STRESS_PROXY": financial_stress,
            "BUREAU_MONTH_TO_OLDER_WINDOWS": bureau_ratio,
        },
        index=raw.index,
    )
    result = result.replace([np.inf, -np.inf], np.nan)
    if result.columns.duplicated().any():
        raise ValueError("Duplicate formula names.")
    return result


def audit_formula_distributions(features: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for column in features.columns:
        values = features[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        valid = values[finite]
        if valid.size:
            auc = float(roc_auc_score(y[finite], values[finite])) if np.unique(y[finite]).size == 2 else np.nan
            rows.append(
                {
                    "formula": column,
                    "n_total": len(values),
                    "n_valid": int(finite.sum()),
                    "n_missing": int(np.isnan(values).sum()),
                    "n_infinite": int(np.isinf(values).sum()),
                    "minimum": float(valid.min()),
                    "maximum": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std(ddof=0)),
                    "raw_auc_diagnostic": auc,
                    "direction_adjusted_auc_diagnostic": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                    "auc_interpretation": "full-sample univariate diagnostic only; not a CV estimate",
                }
            )
        else:
            rows.append(
                {
                    "formula": column,
                    "n_total": len(values),
                    "n_valid": 0,
                    "n_missing": int(np.isnan(values).sum()),
                    "n_infinite": int(np.isinf(values).sum()),
                    "minimum": np.nan,
                    "maximum": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                    "raw_auc_diagnostic": np.nan,
                    "direction_adjusted_auc_diagnostic": np.nan,
                    "auc_interpretation": "full-sample univariate diagnostic only; not a CV estimate",
                }
            )
    return pd.DataFrame(rows)


def validate_reference(reference: pd.DataFrame, ids: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    if reference[ID].duplicated().any():
        raise ValueError("Phase 3C baseline OOF has duplicate IDs.")
    aligned = reference.set_index(ID).reindex(ids)
    if aligned[TARGET].isna().any() or not np.array_equal(aligned[TARGET].to_numpy(dtype=np.int8), y):
        raise ValueError("Phase 3C baseline OOF target alignment failed.")
    if "fold" in aligned and not np.array_equal(aligned["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("Phase 3C baseline OOF fold alignment failed.")
    prediction = aligned["lightgbm__raw_plus_product"].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Phase 3C baseline OOF contains nonfinite predictions.")
    return prediction


def main() -> None:
    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_module(
        "phase3d_preserved_baseline",
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
    )
    phase3b = load_module("phase3d_phase3b", PHASE3B / "phase3b_ablation.py")

    train_path = DATA / "train.csv"
    official_train = Path(r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition\input\train.csv")
    if sha256(train_path) != sha256(official_train):
        raise ValueError("Working train.csv hash does not match the supplied competition train.csv.")
    train = pd.read_csv(train_path)
    if TARGET not in train or train[TARGET].isna().any():
        raise ValueError("Training data have a missing TARGET or TARGET column.")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)

    fold_frame = pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv")
    folds = fold_frame.set_index(ID).loc[ids, "fold"].to_numpy(dtype=np.int8)
    if len(np.unique(folds)) != 5:
        raise ValueError("Expected five preserved stratified folds.")
    reference = pd.read_csv(PHASE3C / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv")
    baseline_oof = validate_reference(reference, ids, y, folds)

    formulas = make_formula_features(raw)
    distributions = audit_formula_distributions(formulas, y)
    distributions.to_csv(OUT / "FORMULA_DISTRIBUTIONS.csv", index=False)
    pd.DataFrame(FORMULA_INFO).to_csv(OUT / "FORMULA_CORRECTNESS_AUDIT.csv", index=False)

    fold_runtime = pd.read_csv(PHASE3C / "MODEL_RUNTIME_RESULTS.csv")
    rounds_by_fold = (
        fold_runtime.loc[
            (fold_runtime.model == "lightgbm") & (fold_runtime.stage == "inner_round_selection"),
            ["fold", "selected_rounds_or_trees"],
        ]
        .set_index("fold")["selected_rounds_or_trees"]
        .to_dict()
    )
    if set(map(int, rounds_by_fold)) != set(map(int, np.unique(folds))):
        raise ValueError("Missing preserved inner-fold round selections.")

    candidates = {
        "formula__risk_adjusted_ltv": ["RISK_ADJUSTED_LTV"],
        "formula__external_score_cv": ["EXTERNAL_SCORE_CV"],
        "formula__ltv_external_uncertainty": ["LTV_EXTERNAL_UNCERTAINTY"],
        "formula__financial_stress_proxy": ["FINANCIAL_STRESS_PROXY"],
        "formula__bureau_month_to_older_windows": ["BUREAU_MONTH_TO_OLDER_WINDOWS"],
        "formula__all_screened": list(formulas.columns),
    }
    predictions: dict[str, np.ndarray] = {"baseline__lightgbm_raw_plus_product": baseline_oof}
    fold_rows: list[dict] = []
    for fold in sorted(np.unique(folds)):
        valid_idx = np.flatnonzero(folds == fold)
        fit_idx = np.flatnonzero(folds != fold)
        raw_fit, raw_valid = raw.iloc[fit_idx], raw.iloc[valid_idx]
        fit_base = baseline.build_rowwise_features(raw_fit)
        valid_base = baseline.build_rowwise_features(raw_valid)
        product_params = phase3b.shared.fit_fee_and_step(raw_fit)
        fit_product, valid_product, _ = phase3b.make_family(
            raw_fit,
            raw_valid,
            feature_params=product_params,
            family_name="all_product",
        )
        fit_base = pd.concat([fit_base, fit_product.reset_index(drop=True)], axis=1)
        valid_base = pd.concat([valid_base, valid_product.reset_index(drop=True)], axis=1)
        rounds = int(rounds_by_fold[int(fold)])
        print(f"fold={fold} rounds={rounds} train={len(fit_idx)} valid={len(valid_idx)}", flush=True)

        for candidate, columns in candidates.items():
            fit_frame = pd.concat(
                [fit_base.reset_index(drop=True), formulas.iloc[fit_idx][columns].reset_index(drop=True)],
                axis=1,
            )
            valid_frame = pd.concat(
                [valid_base.reset_index(drop=True), formulas.iloc[valid_idx][columns].reset_index(drop=True)],
                axis=1,
            )
            fit_x, valid_x = baseline.make_lightgbm_frames(fit_frame, valid_frame)
            config = baseline.MODEL_CONFIG["lightgbm"]
            params = {
                key: value
                for key, value in config.items()
                if key not in ("device", "num_boost_round", "early_stopping_rounds")
            }
            model = lgb.train(
                {**params, "seed": baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_x, label=y[fit_idx]),
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            prediction = model.predict(valid_x, num_iteration=rounds)
            if not np.isfinite(prediction).all():
                raise ValueError(f"{candidate} fold {fold} produced nonfinite predictions.")
            if candidate not in predictions:
                predictions[candidate] = np.full(len(train), np.nan, dtype=np.float64)
            predictions[candidate][valid_idx] = prediction
            baseline_fold_auc = float(roc_auc_score(y[valid_idx], baseline_oof[valid_idx]))
            candidate_auc = float(roc_auc_score(y[valid_idx], prediction))
            fold_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "n_train": int(len(fit_idx)),
                    "n_valid": int(len(valid_idx)),
                    "fold_auc": candidate_auc,
                    "baseline_fold_auc": baseline_fold_auc,
                    "paired_fold_delta": candidate_auc - baseline_fold_auc,
                    "boosting_rounds_selected_inside_outer_training_only": rounds,
                    "outer_validation_used_for_early_stopping": False,
                }
            )
            print(f"  {candidate}: AUC={candidate_auc:.6f} delta={candidate_auc-baseline_fold_auc:+.6f}", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(OUT / "FOLD_LEVEL_CV_RESULTS.csv", index=False)
    oof_frame = pd.DataFrame({ID: ids, TARGET: y, "fold": folds})
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF predictions incomplete for {name}.")
        oof_frame[name] = prediction
    oof_frame.to_csv(OUT / "FORMULA_OOF_PREDICTIONS.csv", index=False)

    summary_rows = []
    for name, prediction in predictions.items():
        subset = fold_results.loc[fold_results.candidate == name]
        if name == "baseline__lightgbm_raw_plus_product":
            fold_auc = [roc_auc_score(y[folds == fold], prediction[folds == fold]) for fold in sorted(np.unique(folds))]
            deltas = np.zeros(len(fold_auc))
        else:
            fold_auc = subset.sort_values("fold")["fold_auc"].to_numpy(dtype=np.float64)
            deltas = subset.sort_values("fold")["paired_fold_delta"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "candidate": name,
                "pooled_oof_auc": float(roc_auc_score(y, prediction)),
                "mean_fold_auc": float(np.mean(fold_auc)),
                "std_fold_auc": float(np.std(fold_auc, ddof=1)),
                "mean_paired_fold_delta_vs_baseline": float(np.mean(deltas)),
                "std_paired_fold_delta_vs_baseline": float(np.std(deltas, ddof=1)),
                "folds_improved": int(np.sum(deltas > 0)),
                "validation_status": "discovery on previously examined fixed folds",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "BASELINE_VS_FORMULAS_RESULTS.csv", index=False)

    notebook_data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    inventory = []
    saved_text = []
    for index, cell in enumerate(notebook_data.get("cells", [])):
        output_count = 0
        output_chars = 0
        for output in cell.get("outputs", []):
            output_count += 1
            text = "".join(output.get("text", []))
            output_chars += len(text)
            if text:
                saved_text.append(f"--- cell {index} saved stdout ---\n{text}")
        inventory.append(
            {
                "cell": index,
                "cell_type": cell.get("cell_type"),
                "execution_count": cell.get("execution_count"),
                "source_chars": len("".join(cell.get("source", []))),
                "output_count": output_count,
                "saved_text_output_chars": output_chars,
            }
        )
    pd.DataFrame(inventory).to_csv(OUT / "V5_NOTEBOOK_CELL_INVENTORY.csv", index=False)
    saved_output_text = "\n".join(saved_text)
    (OUT / "V5_NOTEBOOK_SAVED_OUTPUT.txt").write_text(saved_output_text, encoding="utf-8")
    score_claims = [
        {
            "claim": "old_blend_pooled_held_out_auc",
            "value": "0.77202",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Referenced old cache and applicant-level OOF vectors not found locally.",
        },
        {
            "claim": "new_blend_pooled_held_out_auc",
            "value": "0.77198",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Applicant-level OOF vectors and fold labels not stored in the notebook.",
        },
        {
            "claim": "old_new_50_50_mix_pooled_held_out_auc",
            "value": "0.77213",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Applicant-level OOF vectors and fold labels not stored in the notebook.",
        },
        {
            "claim": "new_minus_old_mean_fold_delta",
            "value": "-0.00005",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Saved fold summaries only; raw predictions unavailable.",
        },
        {
            "claim": "new_minus_old_positive_folds",
            "value": "2/5",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Saved fold summaries only; raw predictions unavailable.",
        },
        {
            "claim": "mix_minus_old_mean_fold_delta",
            "value": "+0.00010",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Saved fold summaries only; raw predictions unavailable.",
        },
        {
            "claim": "mix_minus_old_positive_folds",
            "value": "4/5",
            "source": "saved stdout in notebook cell 9",
            "independently_reproduced": False,
            "reason": "Saved fold summaries only; raw predictions unavailable.",
        },
        {
            "claim": "v3_A_reference_auc",
            "value": "0.77746",
            "source": "notebook cell 0/9 text",
            "independently_reproduced": False,
            "reason": "No matching v3_A OOF vector/cache found in the inspected experiment locations.",
        },
    ]
    pd.DataFrame(score_claims).to_csv(OUT / "V5_SCORE_CLAIMS.csv", index=False)

    input_files = [
        train_path,
        official_train,
        PHASE2 / "FOLD_ASSIGNMENTS.csv",
        PHASE3C / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv",
        PHASE3C / "MODEL_RUNTIME_RESULTS.csv",
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
        PHASE3B / "phase3b_ablation.py",
        NOTEBOOK,
    ]
    input_files.extend(
        Path(r"C:\Users\SHARON\Downloads") / name
        for name in [
            "WhatsApp Image 2026-11-03 at 10.58.26 PM (1).jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.26 PM.jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.27 PM (1).jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.27 PM (2).jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.27 PM.jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.28 PM (1).jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.28 PM.jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.29 PM (1).jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.29 PM.jpeg",
            "WhatsApp Image 2026-11-03 at 10.58.30 PM.jpeg",
        ]
    )
    pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in input_files]
    ).to_csv(OUT / "INPUT_SHA256_MANIFEST.csv", index=False)
    environment = {
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "train_rows": int(len(train)),
        "train_columns_including_target": int(train.shape[1]),
        "target_rate": float(y.mean()),
        "folds": 5,
        "feature_generation": "row-wise formulas only; target excluded",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    best = summary.loc[summary.candidate != "baseline__lightgbm_raw_plus_product"].sort_values(
        ["pooled_oof_auc", "mean_paired_fold_delta_vs_baseline"], ascending=False
    ).iloc[0]
    final_report = f"""# Phase 3D formula experiment report

## Status and scope

Completed a fold-safe, row-wise experiment on the supplied 171,202-row training set. No test labels, external data, pooled train/test statistics, label-derived formula values, or edits to the champion/earlier experiments were used. All results reuse the existing five fixed folds and are discovery evidence only.

## Prespecified model and comparison

- Reference: preserved LightGBM `raw_plus_product` OOF predictions, aligned by applicant ID, TARGET, and the saved five-fold assignment.
- Each candidate reuses the same Phase 3C fold-local selected boosting rounds: `{json.dumps({int(k): int(v) for k, v in rounds_by_fold.items()}, sort_keys=True)}`.
- Product features and model categorical encodings are fitted inside each outer training fold. Added formulas are deterministic per-row operations.
- No outer validation labels were used for feature selection, early stopping, or parameter fitting.
- Candidate scores: see `BASELINE_VS_FORMULAS_RESULTS.csv` and `FOLD_LEVEL_CV_RESULTS.csv`.
- Best screened formula configuration: `{best['candidate']}`, pooled OOF AUC `{best['pooled_oof_auc']:.6f}`, paired fold delta `{best['mean_paired_fold_delta_vs_baseline']:+.6f}`. This is not a promotion decision.

## Formula interpretation

Five nonduplicate candidates were tested: risk-adjusted LTV, normalized external-score dispersion, LTV weighted by external-score dispersion, a financial-stress interaction, and the month-to-older bureau-window count ratio. See `FORMULA_CORRECTNESS_AUDIT.csv` for guarded definitions and excluded hypotheses. In particular:

- The baseline already contains LTV, annuity/income, family-normalized income, external-score mean/standard deviation, and an income/annuity coverage transformation. Re-testing those as novel inputs would be redundant.
- The inverse external-score ratio is stabilized with a fixed 0.05 on the normalized score scale because the observed source values include near-zero values; the score direction/meaning is not further defined in the local dictionary. Treat the result only as a hypothesis.
- The dictionary calls `AMT_ANNUITY` “Loan annuity” but does not state its time period. Annualizing it, labeling it monthly, or treating income/annuity as a calibrated DSCR is unsupported by this local dictionary.
- `AMT_GOODS_PRICE - AMT_CREDIT` is not established as an actual down payment; its normalized version is exactly `1 - LTV`.
- The implied-rate inversion cannot identify a contractual rate without repayment count and payment interval. For fixed assumed term, it is only a monotonic transformation of annuity/credit.
- Bureau ratio is descriptive, not an “acceleration” measure. The local dictionary’s timing descriptions do not establish equal-length, non-overlapping rates for every window.

## V5 notebook audit

The supplied v5 notebook has {len(inventory)} cells; all saved code cells have `execution_count=None`, despite containing saved stdout. The saved output says old/new/mix pooled held-out AUCs were 0.77202/0.77198/0.77213; these values cannot be independently recomputed because the referenced old cache (`gci_cache_repro`) and OOF vectors are not present locally. Its own fold table has the all-new blend below the old blend on 3 of 5 folds, with mean paired fold delta -0.00005. The 50/50 mix is positive on 4/5 folds but mean gain is only +0.00010, below the notebook’s own +0.00020 gate. These are notebook claims, not verified new estimates. The notebook itself says to keep v3_A (claimed 0.77746) when the gate fails; that v3_A score/vector was not found for independent reproduction.

## Outputs and limitations

See `SOURCE_CODE_AUDIT.md`, `FORMULA_CORRECTNESS_AUDIT.csv`, `FORMULA_DISTRIBUTIONS.csv`, `FORMULA_OOF_PREDICTIONS.csv`, `FOLD_LEVEL_CV_RESULTS.csv`, `BASELINE_VS_FORMULAS_RESULTS.csv`, `V5_NOTEBOOK_CELL_INVENTORY.csv`, `V5_NOTEBOOK_SAVED_OUTPUT.txt`, `V5_SCORE_CLAIMS.csv`, `INPUT_SHA256_MANIFEST.csv`, `OUTPUT_FILE_MANIFEST.csv`, `environment.json`, and `EXECUTION_LOG.txt`.

No candidate is promoted. The folds have been examined in earlier phases, and selecting a result on them adds researcher degrees of freedom. Confirm any frozen candidate on genuinely unseen labels if available; a new random split of the same exposed labels is not independent confirmation.
"""
    (OUT / "FINAL_AUDIT_REPORT.md").write_text(final_report, encoding="utf-8")
    expected_names = [
        "SOURCE_CODE_AUDIT.md",
        "phase3d_formula_experiment.py",
        "FORMULA_CORRECTNESS_AUDIT.csv",
        "FORMULA_DISTRIBUTIONS.csv",
        "FOLD_LEVEL_CV_RESULTS.csv",
        "FORMULA_OOF_PREDICTIONS.csv",
        "BASELINE_VS_FORMULAS_RESULTS.csv",
        "V5_NOTEBOOK_CELL_INVENTORY.csv",
        "V5_NOTEBOOK_SAVED_OUTPUT.txt",
        "V5_SCORE_CLAIMS.csv",
        "INPUT_SHA256_MANIFEST.csv",
        "environment.json",
        "FINAL_AUDIT_REPORT.md",
    ]
    expected_paths = [OUT / name for name in expected_names]
    missing_or_empty = [str(path) for path in expected_paths if not path.is_file() or path.stat().st_size == 0]
    if missing_or_empty:
        raise RuntimeError(f"Expected output files missing or empty: {missing_or_empty}")
    output_manifest = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in expected_paths
    ]
    manifest_path = OUT / "OUTPUT_FILE_MANIFEST.csv"
    pd.DataFrame(output_manifest).to_csv(manifest_path, index=False)
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise RuntimeError("OUTPUT_FILE_MANIFEST.csv is empty.")
    log_lines = [
        f"command: python {Path(__file__)}",
        f"started_utc: {started_utc.isoformat()}",
        f"finished_utc: {environment['finished_utc']}",
        f"train_shape: {train.shape}",
        f"target_rate: {y.mean():.8f}",
        f"official_train_sha256: {sha256(official_train)}",
        f"working_train_sha256: {sha256(train_path)}",
        f"fold_sizes: {json.dumps({int(f): int((folds == f).sum()) for f in np.unique(folds)}, sort_keys=True)}",
        f"elapsed_seconds: {environment['elapsed_seconds']:.2f}",
        "status: SUCCESS",
        "errors: none",
        "warnings: v5 saved scores are notebook output only and were not reconstructed from raw caches; fixed folds are previously examined discovery folds.",
        "verified_outputs: " + json.dumps([str(path) for path in expected_paths + [manifest_path]], indent=2),
    ]
    log_path = OUT / "EXECUTION_LOG.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if log_path.stat().st_size == 0:
        raise RuntimeError("EXECUTION_LOG.txt is empty.")
    print(summary.to_string(index=False), flush=True)
    print(f"best_candidate={best['candidate']} pooled_oof_auc={best['pooled_oof_auc']:.6f}", flush=True)
    print(f"output_directory={OUT}", flush=True)
    print(f"elapsed_seconds={environment['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
