from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
BASELINE_DIR = ROOT / "experiments" / "phase2_advanced_feature_engineering"
OUT = ROOT / "experiments" / "phase3b_deep_validation"
SHARED_PATH = PHASE3 / "run_phase3_hidden_pattern_discovery.py"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FEE_GRID = np.arange(0.0, 250.0001, 0.5)
LATTICE_GRID = np.arange(0.0058, 0.0146001, 0.0002)

spec = importlib.util.spec_from_file_location("phase3b_shared", SHARED_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared Phase 3 feature code: {SHARED_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(raw: pd.DataFrame, name: str) -> np.ndarray:
    if name not in raw:
        return np.full(len(raw), np.nan, dtype=np.float64)
    return raw[name].to_numpy(dtype=np.float64, na_value=np.nan)


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(a), np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[valid] = a[valid] / b[valid]
    return out


def summarize_error(error: np.ndarray) -> dict[str, float | int]:
    error = error[np.isfinite(error)]
    if not len(error):
        return {
            "n": 0,
            "exact_pct": np.nan,
            "mae": np.nan,
            "median_abs_error": np.nan,
            "p90_abs_error": np.nan,
            "p95_abs_error": np.nan,
            "max_abs_error": np.nan,
            "pct_abs_error_le_0_01": np.nan,
        }
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "exact_pct": float(np.mean(absolute < 1e-8) * 100),
        "mae": float(np.mean(absolute)),
        "median_abs_error": float(np.median(absolute)),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "max_abs_error": float(np.max(absolute)),
        "pct_abs_error_le_0_01": float(np.mean(absolute <= 0.01) * 100),
    }


def key_coverage(fit_keys: np.ndarray, valid_keys: np.ndarray) -> tuple[float, float, float]:
    fit = fit_keys[np.isfinite(fit_keys)]
    valid = valid_keys[np.isfinite(valid_keys)]
    if not len(fit) or not len(valid):
        return np.nan, np.nan, np.nan
    counts = pd.Series(fit).value_counts()
    mapped = np.array([counts.get(key, 0) for key in valid], dtype=np.int64)
    return (
        float(np.mean(mapped > 0)),
        float(np.mean(mapped >= 2)),
        float(np.mean(counts.reindex(fit).to_numpy() >= 2)),
    )


def fit_fee(raw: pd.DataFrame) -> tuple[float, float]:
    credit = finite(raw, "AMT_CREDIT")
    annuity = finite(raw, "AMT_ANNUITY")
    cash = raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    valid = cash & np.isfinite(credit) & np.isfinite(annuity) & (credit > 0)
    x, y = credit[valid], annuity[valid]
    if len(x) < 100:
        return 0.0, 0.0
    if len(x) > 30000:
        ix = np.linspace(0, len(x) - 1, 30000, dtype=np.int64)
        x, y = x[ix], y[ix]
    best_fee, best_support = 0.0, -1.0
    for candidate in FEE_GRID:
        keys = np.round((y - candidate) / x, 5)
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        support = float(np.mean(counts[inverse] >= 2))
        if support > best_support:
            best_fee, best_support = float(candidate), support
    return best_fee, best_support


def fit_proportional_fee(raw: pd.DataFrame) -> tuple[float, float]:
    credit = finite(raw, "AMT_CREDIT")
    annuity = finite(raw, "AMT_ANNUITY")
    cash = raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    valid = cash & np.isfinite(credit) & np.isfinite(annuity) & (credit > 0)
    q = annuity[valid] / credit[valid]
    if len(q) > 16000:
        ix = np.linspace(0, len(q) - 1, 16000, dtype=np.int64)
        q = q[ix]
    alpha_grid = np.arange(-0.05, 0.0500001, 0.00025)
    best_alpha, best_support = 0.0, -1.0
    for alpha in alpha_grid:
        keys = np.round(q - alpha, 5)
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        support = float(np.mean(counts[inverse] >= 2))
        if support > best_support:
            best_alpha, best_support = float(alpha), support
    return best_alpha, best_support


def fold_feature_data(
    raw: pd.DataFrame, folds: np.ndarray
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    feature_names = shared.MODEL_FEATURES
    oof_values = {name: np.full(len(raw), np.nan, dtype=np.float64) for name in feature_names}
    fee_rows: list[dict[str, object]] = []
    lattice_rows: list[dict[str, object]] = []
    reconstruction_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    for fold in sorted(np.unique(folds)):
        fit_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)
        fit_raw, val_raw = raw.iloc[fit_idx], raw.iloc[val_idx]
        fit_params = shared.fit_fee_and_step(fit_raw)
        maps = shared.fit_product_maps(fit_raw, fit_params)
        fit_features = shared.make_product_features(fit_raw, fit_params, maps)
        val_features = shared.make_product_features(val_raw, fit_params, maps)
        for name in feature_names:
            oof_values[name][val_idx] = val_features[name].to_numpy(dtype=np.float64)

        fit_credit, val_credit = finite(fit_raw, "AMT_CREDIT"), finite(val_raw, "AMT_CREDIT")
        fit_goods, val_goods = finite(fit_raw, "AMT_GOODS_PRICE"), finite(val_raw, "AMT_GOODS_PRICE")
        fit_ann, val_ann = finite(fit_raw, "AMT_ANNUITY"), finite(val_raw, "AMT_ANNUITY")
        fit_cash = fit_raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        val_cash = val_raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        q_fit, q_val = safe_ratio(fit_ann, fit_credit), safe_ratio(val_ann, val_credit)
        k_fit = safe_ratio(safe_ratio(fit_credit, fit_goods) - 1, np.full(len(fit_raw), fit_params["step_estimate_fit_only"]))
        k_val = safe_ratio(safe_ratio(val_credit, val_goods) - 1, np.full(len(val_raw), fit_params["step_estimate_fit_only"]))
        p_fit = safe_ratio(fit_ann - fit_params["fee_estimate_fit_only"], fit_credit)
        p_val = safe_ratio(val_ann - fit_params["fee_estimate_fit_only"], val_credit)

        # Q/P/K reconstruction uses only cash-loan rows and keeps missing rows excluded.
        for component, fit_raw_key, val_raw_key, observed, reconstructed in [
            (
                "Q5_unadjusted_annuity",
                np.round(q_fit[fit_cash], 5),
                np.round(q_val[val_cash], 5),
                val_ann[val_cash],
                np.round(q_val[val_cash], 5) * val_credit[val_cash],
            ),
            (
                "P5_fit_fee_annuity",
                np.round(p_fit[fit_cash], 5),
                np.round(p_val[val_cash], 5),
                val_ann[val_cash],
                fit_params["fee_estimate_fit_only"] + np.round(p_val[val_cash], 5) * val_credit[val_cash],
            ),
            (
                "K_fit_step_credit_over_goods",
                np.rint(k_fit[fit_cash]),
                np.rint(k_val[val_cash]),
                val_credit[val_cash],
                val_goods[val_cash] * (1 + np.rint(k_val[val_cash]) * fit_params["step_estimate_fit_only"]),
            ),
        ]:
            coverage, repeated, fit_repeat = key_coverage(fit_raw_key, val_raw_key)
            err = summarize_error(observed - reconstructed)
            reconstruction_rows.append(
                {
                    "fold": int(fold),
                    "component": component,
                    "fee_fit_only": fit_params["fee_estimate_fit_only"] if component.startswith("P") else np.nan,
                    "step_fit_only": fit_params["step_estimate_fit_only"] if component.startswith("K") else np.nan,
                    "fit_key_unique_count": int(pd.Series(fit_raw_key).nunique()),
                    "validation_key_unique_count": int(pd.Series(val_raw_key).nunique()),
                    "validation_seen_key_pct": coverage * 100,
                    "validation_key_repeated_in_fit_pct": repeated * 100,
                    "fit_repeated_key_pct": fit_repeat * 100,
                    **err,
                }
            )

        # Predeclared nearby financing increments; all fit/validation diagnostics are target-free.
        fit_x = (fit_credit / fit_goods - 1)
        val_x = (val_credit / val_goods - 1)
        fit_mask = fit_cash & np.isfinite(fit_x) & (fit_credit > fit_goods) & (fit_goods > 0)
        val_mask = val_cash & np.isfinite(val_x) & (val_credit > val_goods) & (val_goods > 0)
        for step in np.unique(np.r_[LATTICE_GRID, 0.0066, 0.0132, fit_params["step_estimate_fit_only"]]):
            fit_scaled = fit_x[fit_mask] / step
            val_scaled = val_x[val_mask] / step
            fit_res = np.abs(fit_scaled - np.rint(fit_scaled))
            val_res = np.abs(val_scaled - np.rint(val_scaled))
            val_supported = val_res < 0.1
            val_integers = np.rint(val_scaled[val_supported]).astype(np.int64)
            odd_share = (
                float(np.mean(np.abs(val_integers) % 2 == 1) * 100)
                if len(val_integers)
                else np.nan
            )
            recon = val_goods[val_mask] * (1 + np.rint(val_scaled) * step)
            err = summarize_error(val_credit[val_mask] - recon)
            lattice_rows.append(
                {
                    "fold": int(fold),
                    "candidate_step": float(step),
                    "fit_n": int(len(fit_res)),
                    "fit_within_0_1_pct": float(np.mean(fit_res < 0.1) * 100) if len(fit_res) else np.nan,
                    "fit_residual_mean": float(np.mean(fit_res)) if len(fit_res) else np.nan,
                    "validation_n": int(len(val_res)),
                    "validation_within_0_1_pct": float(np.mean(val_res < 0.1) * 100) if len(val_res) else np.nan,
                    "validation_residual_mean": float(np.mean(val_res)) if len(val_res) else np.nan,
                    "validation_odd_integer_share_among_supported_pct": odd_share,
                    **{f"credit_reconstruction_{k}": v for k, v in err.items()},
                }
            )

        # Fee candidates are selected without labels; report held-out ratio-key coverage and reconstruction.
        fee_fit, fee_support = fit_fee(fit_raw)
        rng = np.random.default_rng(20261103 + int(fold))
        subsample_fees = []
        cash_positions = np.flatnonzero(fit_cash & np.isfinite(fit_credit) & np.isfinite(fit_ann))
        subsample_fee_by_fraction = {}
        for fraction in (0.25, 0.50, 0.75):
            size = max(1000, int(len(cash_positions) * fraction))
            chosen = rng.choice(cash_positions, size=size, replace=False)
            subsample_fee_by_fraction[str(fraction)] = fit_fee(fit_raw.iloc[chosen])[0]
        for rep in range(5):
            chosen = rng.choice(cash_positions, size=max(1000, int(len(cash_positions) * 0.6)), replace=False)
            sub_fee, _ = fit_fee(fit_raw.iloc[chosen])
            subsample_fees.append(sub_fee)
        robust_fee = float(np.median(subsample_fees))
        cash_credit = fit_credit[cash_positions]
        cash_annuity = fit_ann[cash_positions]
        credit_bounds = np.quantile(cash_credit, [0.01, 0.99])
        annuity_bounds = np.quantile(cash_annuity, [0.01, 0.99])
        trim_mask = (
            (cash_credit >= credit_bounds[0])
            & (cash_credit <= credit_bounds[1])
            & (cash_annuity >= annuity_bounds[0])
            & (cash_annuity <= annuity_bounds[1])
        )
        trimmed_fee, trimmed_fee_support = fit_fee(
            fit_raw.iloc[cash_positions[trim_mask]]
        )
        alpha, alpha_support = fit_proportional_fee(fit_raw)
        credit_cut = np.unique(np.quantile(fit_credit[np.isfinite(fit_credit)], [0.2, 0.4, 0.6, 0.8]))
        fit_credit_bin = np.digitize(fit_credit, credit_cut, right=True)
        val_credit_bin = np.digitize(val_credit, credit_cut, right=True)
        regime_fees: dict[tuple[str, int], float] = {}
        contract_fit = fit_raw["NAME_CONTRACT_TYPE"].fillna("MISSING").astype(str).to_numpy()
        contract_val = val_raw["NAME_CONTRACT_TYPE"].fillna("MISSING").astype(str).to_numpy()
        for contract in np.unique(contract_fit):
            for bin_id in range(5):
                group_mask = (contract_fit == contract) & (fit_credit_bin == bin_id)
                if int(group_mask.sum()) >= 500:
                    group_frame = fit_raw.iloc[np.flatnonzero(group_mask)]
                    regime_fees[(contract, bin_id)], _ = fit_fee(group_frame)
        regime_fee_details = {
            f"{contract}|credit_bin_{bin_id}": value
            for (contract, bin_id), value in regime_fees.items()
        }
        fee_methods = [
            ("no_fee", np.zeros(len(val_raw)), 0.0),
            ("fixed_129_5", np.full(len(val_raw), 129.5), 129.5),
            ("fit_mode_fee", np.full(len(val_raw), fee_fit), fee_fit),
            ("subsample_median_fee", np.full(len(val_raw), robust_fee), robust_fee),
            ("outlier_trimmed_fee", np.full(len(val_raw), trimmed_fee), trimmed_fee),
            ("multiple_contract_credit_regime", np.array([
                regime_fees.get((c, int(b)), fee_fit) for c, b in zip(contract_val, val_credit_bin)
            ]), np.nan),
            ("proportional_fee_alpha_times_credit", alpha * val_credit, alpha),
        ]
        for method, fee_val, reported_parameter in fee_methods:
            key_fit = np.round(safe_ratio(fit_ann - fee_fit, fit_credit), 5) if method == "fit_mode_fee" else None
            # Each method gets its own corresponding fit/validation keys.
            if method == "proportional_fee_alpha_times_credit":
                fit_fee_value = alpha * fit_credit
            elif method == "multiple_contract_credit_regime":
                fit_fee_value = np.array([
                    regime_fees.get((c, int(b)), fee_fit) for c, b in zip(contract_fit, fit_credit_bin)
                ])
            else:
                constant = float(fee_val[0]) if len(fee_val) else 0.0
                fit_fee_value = np.full(len(fit_raw), constant)
            p_fit_candidate = safe_ratio(fit_ann - fit_fee_value, fit_credit)
            p_val_candidate = safe_ratio(val_ann - fee_val, val_credit)
            qfk = np.round(p_fit_candidate[fit_cash], 5)
            qvk = np.round(p_val_candidate[val_cash], 5)
            seen, repeated, fit_repeat = key_coverage(qfk, qvk)
            err = summarize_error(
                val_ann[val_cash] - (fee_val[val_cash] + qvk * val_credit[val_cash])
            )
            fee_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "parameter_estimate": reported_parameter,
                    "global_fit_mode_fee": fee_fit,
                    "fit_mode_repeat_support": fee_support,
                    "subsample_fees": json.dumps(subsample_fees),
                    "subsample_fee_by_training_fraction": json.dumps(subsample_fee_by_fraction, sort_keys=True),
                    "robust_fee_median": robust_fee,
                    "trimmed_fee": trimmed_fee,
                    "trimmed_fee_repeat_support": trimmed_fee_support,
                    "proportional_alpha": alpha,
                    "proportional_fit_repeat_support": alpha_support,
                    "regime_count": len(regime_fees),
                    "regime_fee_estimates": json.dumps(regime_fee_details, sort_keys=True),
                    "validation_seen_key_pct": seen * 100,
                    "validation_key_repeated_in_fit_pct": repeated * 100,
                    "fit_repeated_key_pct": fit_repeat * 100,
                    **err,
                }
            )

        # Group stability is computed with cut points and category vocabularies learned on fit rows.
        raw_groups: list[tuple[str, np.ndarray, np.ndarray]] = []
        numeric_specs = [
            ("income_band", "AMT_INCOME_TOTAL"),
            ("credit_band", "AMT_CREDIT"),
            ("goods_band", "AMT_GOODS_PRICE"),
            ("annuity_band", "AMT_ANNUITY"),
            ("employment_duration_band", "DAYS_EMPLOYED"),
            ("age_band", "DAYS_BIRTH"),
            ("family_size_band", "CNT_FAM_MEMBERS"),
        ]
        for label, column in numeric_specs:
            f = finite(fit_raw, column)
            v = finite(val_raw, column)
            finite_f = f[np.isfinite(f)]
            edges = np.unique(np.quantile(finite_f, [0.2, 0.4, 0.6, 0.8])) if len(finite_f) else np.array([])
            if len(edges):
                raw_groups.append((label, np.digitize(f, edges), np.digitize(v, edges)))
        ext = np.column_stack([finite(val_raw, f"EXT_SOURCE_{i}") for i in (1, 2, 3)])
        ext_fit = np.column_stack([finite(fit_raw, f"EXT_SOURCE_{i}") for i in (1, 2, 3)])
        fit_score_count = np.isfinite(ext_fit).sum(axis=1)
        val_score_count = np.isfinite(ext).sum(axis=1)
        fscore = np.divide(
            np.nansum(ext_fit, axis=1),
            fit_score_count,
            out=np.full(len(fit_raw), np.nan),
            where=fit_score_count > 0,
        )
        vscore = np.divide(
            np.nansum(ext, axis=1),
            val_score_count,
            out=np.full(len(val_raw), np.nan),
            where=val_score_count > 0,
        )
        for label, fscore, vscore in [("external_score_band", fscore, vscore)]:
            fscore = np.where(np.isfinite(fscore), fscore, np.nan)
            finite_f = fscore[np.isfinite(fscore)]
            edges = np.unique(np.quantile(finite_f, [0.2, 0.4, 0.6, 0.8]))
            raw_groups.append((label, np.digitize(fscore, edges), np.digitize(vscore, edges)))
        for label, column in [
            ("occupation", "OCCUPATION_TYPE"),
            ("contract_type", "NAME_CONTRACT_TYPE"),
            ("organization_type", "ORGANIZATION_TYPE"),
        ]:
            vocab = set(fit_raw[column].fillna("MISSING").astype(str))
            f = fit_raw[column].fillna("MISSING").astype(str).to_numpy()
            v = val_raw[column].fillna("MISSING").astype(str).to_numpy()
            raw_groups.append((label, f, np.where(np.isin(v, list(vocab)), v, "UNSEEN")))
        for group_name, fit_group, val_group in raw_groups:
            for group_value in pd.unique(val_group):
                val_group_mask = val_group == group_value
                fit_group_mask = fit_group == group_value
                if not val_group_mask.any():
                    continue
                val_q = np.round(q_val[val_cash & val_group_mask], 5)
                fit_q = np.round(q_fit[fit_cash & fit_group_mask], 5)
                val_p = np.round(p_val[val_cash & val_group_mask], 5)
                fit_p = np.round(p_fit[fit_cash & fit_group_mask], 5)
                val_k = np.rint(k_val[val_cash & val_group_mask])
                fit_k = np.rint(k_fit[fit_cash & fit_group_mask])
                q_cov, q_rep, _ = key_coverage(fit_q, val_q)
                p_cov, p_rep, _ = key_coverage(fit_p, val_p)
                k_cov, k_rep, _ = key_coverage(fit_k, val_k)
                vi = np.flatnonzero(val_cash & val_group_mask)
                q_err = summarize_error(val_ann[vi] - np.round(q_val[vi], 5) * val_credit[vi])
                p_err = summarize_error(val_ann[vi] - (fit_params["fee_estimate_fit_only"] + np.round(p_val[vi], 5) * val_credit[vi]))
                ki = np.flatnonzero(val_cash & val_group_mask)
                k_err = summarize_error(val_credit[ki] - val_goods[ki] * (1 + np.rint(k_val[ki]) * fit_params["step_estimate_fit_only"]))
                group_rows.append(
                    {
                        "fold": int(fold),
                        "grouping": group_name,
                        "group_value": str(group_value),
                        "fit_group_n": int(fit_group_mask.sum()),
                        "validation_group_n": int(val_group_mask.sum()),
                        "validation_cash_n": int((val_cash & val_group_mask).sum()),
                        "q5_seen_pct": q_cov * 100,
                        "q5_repeated_fit_pct": q_rep * 100,
                        "p5_seen_pct": p_cov * 100,
                        "p5_repeated_fit_pct": p_rep * 100,
                        "k_integer_seen_pct": k_cov * 100,
                        "k_integer_repeated_fit_pct": k_rep * 100,
                        "q5_reconstruction_median_abs_error": q_err["median_abs_error"],
                        "p5_reconstruction_median_abs_error": p_err["median_abs_error"],
                        "k_reconstruction_median_abs_error": k_err["median_abs_error"],
                        "fee_fit_only": fit_params["fee_estimate_fit_only"],
                        "step_fit_only": fit_params["step_estimate_fit_only"],
                    }
                )

    result = pd.DataFrame(oof_values)
    return result, fee_rows, lattice_rows, reconstruction_rows, group_rows


def new_formula_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    credit, goods, annuity, income = (
        finite(raw, name) for name in ("AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
    )
    family = finite(raw, "CNT_FAM_MEMBERS")
    birth = finite(raw, "DAYS_BIRTH")
    employed = finite(raw, "DAYS_EMPLOYED")
    ext = np.column_stack([finite(raw, f"EXT_SOURCE_{i}") for i in (1, 2, 3)])
    ext_n = np.isfinite(ext).sum(axis=1)
    ext_mean = np.divide(np.nansum(ext, axis=1), ext_n, out=np.full(len(raw), np.nan), where=ext_n > 0)
    ext_min = np.min(np.where(np.isfinite(ext), ext, np.inf), axis=1)
    ext_max = np.max(np.where(np.isfinite(ext), ext, -np.inf), axis=1)
    ext_min[ext_n == 0] = np.nan
    ext_max[ext_n == 0] = np.nan
    variance = np.divide(
        np.where(np.isfinite(ext), (ext - ext_mean[:, None]) ** 2, 0).sum(axis=1),
        ext_n,
        out=np.full(len(raw), np.nan),
        where=ext_n > 0,
    )
    ext_std = np.sqrt(variance)
    pair_diffs = {
        "NEW_EXT_DIFF_12": ext[:, 0] - ext[:, 1],
        "NEW_EXT_DIFF_13": ext[:, 0] - ext[:, 2],
        "NEW_EXT_DIFF_23": ext[:, 1] - ext[:, 2],
    }
    pair_values = np.column_stack(list(pair_diffs.values()))
    pair_max = np.max(np.where(np.isfinite(pair_values), np.abs(pair_values), -np.inf), axis=1)
    pair_max[ext_n < 2] = np.nan
    bureau_names = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    bureau = {name: finite(raw, name) for name in bureau_names if name in raw}
    bureau_sum = np.nansum(np.column_stack(list(bureau.values())), axis=1) if bureau else np.full(len(raw), np.nan)
    features = {
        "NEW_CREDIT_GOODS_GAP_TO_INCOME": safe_ratio(credit - goods, income),
        "NEW_GOODS_TO_INCOME": safe_ratio(goods, income),
        "NEW_ANNUITY_TO_GOODS": safe_ratio(annuity, goods),
        "NEW_CREDIT_GOODS_LOG_RATIO": np.log1p(np.maximum(credit, 0)) - np.log1p(np.maximum(goods, 0)),
        "NEW_EXT_MEAN": ext_mean,
        "NEW_EXT_MIN": ext_min,
        "NEW_EXT_MAX": ext_max,
        "NEW_EXT_RANGE": ext_max - ext_min,
        "NEW_EXT_STD": ext_std,
        "NEW_EXT_RELATIVE_DISPERSION": safe_ratio(ext_std, ext_mean),
        **pair_diffs,
        "NEW_EXT_PAIRWISE_MAX_ABS_DIFF": pair_max,
        "NEW_EXT_SCORE_COUNT": ext_n.astype(np.float64),
        "NEW_EMPLOYMENT_SENTINEL": (employed == 365243.0).astype(np.float64),
        "NEW_EMPLOYED_YEARS_CLEAN": -np.where(employed == 365243.0, np.nan, employed) / 365.25,
        "NEW_EMPLOYMENT_TO_AGE": safe_ratio(np.where(employed == 365243.0, np.nan, employed), birth + 1),
        "NEW_CREDIT_PER_FAMILY": safe_ratio(credit, family),
        "NEW_INCOME_PER_FAMILY": safe_ratio(income, family),
        "NEW_BUREAU_WINDOW_COUNT_SUM": bureau_sum,
        "NEW_BUREAU_MONTH_PER_YEAR_RATIO": safe_ratio(bureau.get("AMT_REQ_CREDIT_BUREAU_MON", np.full(len(raw), np.nan)), bureau.get("AMT_REQ_CREDIT_BUREAU_YEAR", np.full(len(raw), np.nan))),
    }
    frame = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)
    metadata = [
        ("NEW_CREDIT_GOODS_GAP_TO_INCOME", "(AMT_CREDIT-AMT_GOODS_PRICE)/AMT_INCOME_TOTAL", "Financing excess relative to income; mechanically defined when income is nonzero."),
        ("NEW_GOODS_TO_INCOME", "AMT_GOODS_PRICE/AMT_INCOME_TOTAL", "Stated goods price relative to income."),
        ("NEW_ANNUITY_TO_GOODS", "AMT_ANNUITY/AMT_GOODS_PRICE", "Annuity intensity against goods price; annuity period unspecified."),
        ("NEW_CREDIT_GOODS_LOG_RATIO", "log1p(max(CREDIT,0))-log1p(max(GOODS,0))", "Scale-compressed credit/goods relation; negatives clipped and require interpretation caution."),
        ("NEW_EXT_MEAN", "mean of observed EXT_SOURCE_1..3", "Aggregates external risk signals; count and missingness matter."),
        ("NEW_EXT_MIN", "min of observed EXT_SOURCE_1..3", "Worst observed external score; incomparable when availability differs."),
        ("NEW_EXT_MAX", "max of observed EXT_SOURCE_1..3", "Best observed external score; incomparable when availability differs."),
        ("NEW_EXT_RANGE", "max(observed)-min(observed)", "Absolute disagreement; only defined for at least one score."),
        ("NEW_EXT_STD", "population SD of observed EXT_SOURCE_1..3", "Dispersion among available scores."),
        ("NEW_EXT_RELATIVE_DISPERSION", "EXT_STD/EXT_MEAN when mean != 0", "Scale-relative disagreement; unstable near zero; interpreted only descriptively."),
        ("NEW_EXT_DIFF_12", "EXT_SOURCE_1-EXT_SOURCE_2", "Pairwise score disagreement; duplicates baseline EXT_DIFF_01."),
        ("NEW_EXT_DIFF_13", "EXT_SOURCE_1-EXT_SOURCE_3", "Pairwise score disagreement; duplicates baseline EXT_DIFF_02."),
        ("NEW_EXT_DIFF_23", "EXT_SOURCE_2-EXT_SOURCE_3", "Pairwise score disagreement; duplicates baseline EXT_DIFF_12."),
        ("NEW_EXT_PAIRWISE_MAX_ABS_DIFF", "max(abs(pairwise EXT_SOURCE differences))", "Equivalent to the baseline EXT_RANGE when at least two scores are observed."),
        ("NEW_EXT_SCORE_COUNT", "count of nonmissing EXT_SOURCE_1..3", "Availability signal; duplicates baseline EXT_COUNT."),
        ("NEW_EMPLOYMENT_SENTINEL", "1[DAYS_EMPLOYED==365243]", "Source sentinel flag; duplicates baseline feature."),
        ("NEW_EMPLOYED_YEARS_CLEAN", "-DAYS_EMPLOYED/365.25 after sentinel->missing", "Employment duration proxy; sentinel excluded."),
        ("NEW_EMPLOYMENT_TO_AGE", "clean DAYS_EMPLOYED/(DAYS_BIRTH+1)", "Relative employment-history proxy; day signs retained."),
        ("NEW_CREDIT_PER_FAMILY", "AMT_CREDIT/CNT_FAM_MEMBERS", "Mechanical allocation; not a claim that household members share liability."),
        ("NEW_INCOME_PER_FAMILY", "AMT_INCOME_TOTAL/CNT_FAM_MEMBERS", "Household-capacity proxy; not individual disposable income."),
        ("NEW_BUREAU_WINDOW_COUNT_SUM", "sum of available hour/day/week/quarter/year inquiry counts", "Duplicates baseline BUREAU_INQUIRY_SUM; overlapping windows are double-counted and do not form a unique-event total."),
        ("NEW_BUREAU_MONTH_PER_YEAR_RATIO", "AMT_REQ_CREDIT_BUREAU_MON/AMT_REQ_CREDIT_BUREAU_YEAR", "Windows overlap and denominator zero/missing; not a rate or temporal distribution."),
    ]
    definitions = pd.DataFrame(
        metadata,
        columns=["feature_name", "formula", "domain_hypothesis_and_caveat"],
    )
    decisions = {
        "NEW_CREDIT_GOODS_GAP_TO_INCOME": "RETEST_IN_FIXED_ROUND_PHASE3B; previously tested in Phase 3 financial family",
        "NEW_GOODS_TO_INCOME": "RETEST_IN_FIXED_ROUND_PHASE3B; related to baseline credit/income and credit/goods",
        "NEW_ANNUITY_TO_GOODS": "RETEST_IN_FIXED_ROUND_PHASE3B; previously tested in Phase 3 financial family",
        "NEW_CREDIT_GOODS_LOG_RATIO": "RETEST_IN_FIXED_ROUND_PHASE3B; previously tested in Phase 3 financial family",
        "NEW_EXT_RELATIVE_DISPERSION": "EXCLUDE_NEAR_ZERO_UNSTABLE_AND_OVERLAPS_BASELINE_COMPONENTS",
        "NEW_BUREAU_WINDOW_COUNT_SUM": "EXCLUDE_BASELINE_DUPLICATE_AND_OVERLAPPING_WINDOWS",
        "NEW_BUREAU_MONTH_PER_YEAR_RATIO": "EXCLUDE_SEMANTICALLY_INVALID_OVERLAPPING_WINDOWS",
    }
    definitions["modeling_decision"] = definitions["feature_name"].map(decisions).fillna(
        "EXCLUDE_EXACT_OR_NEAR_DUPLICATE_OF_BASELINE"
    )
    return frame, definitions


def safe_spearman_correlations(
    frame: pd.DataFrame, values: np.ndarray
) -> pd.Series:
    correlations: dict[str, float] = {}
    response = np.asarray(values, dtype=np.float64)
    for column in frame.columns:
        predictor = frame[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(predictor) & np.isfinite(response)
        if (
            valid.sum() < 2
            or np.unique(predictor[valid]).size < 2
            or np.unique(response[valid]).size < 2
        ):
            continue
        correlation = pd.Series(predictor[valid]).corr(
            pd.Series(response[valid]), method="spearman"
        )
        if np.isfinite(correlation):
            correlations[str(column)] = float(correlation)
    return pd.Series(correlations, dtype=np.float64)


def main() -> None:
    started = time.time()
    started_utc = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    folds = pd.read_csv(BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(ID).loc[raw[ID], "fold"].to_numpy()
    product, fee_rows, lattice_rows, reconstruction_rows, group_rows = fold_feature_data(raw, folds)

    # Descriptive feature diagnostics: target-dependent columns are explicitly diagnostic only.
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in raw]
    base_numeric = raw.select_dtypes(include=[np.number]).drop(columns=[ID], errors="ignore")
    variable_base = base_numeric.loc[:, base_numeric.nunique(dropna=True) > 1]
    diagnostics = []
    for name in product.columns:
        values = product[name].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(values)
        valid = values[finite_mask]
        auc = roc_auc_score(y[finite_mask], values[finite_mask]) if len(np.unique(y[finite_mask])) == 2 else np.nan
        x = pd.Series(values).fillna(pd.Series(values).median()).to_numpy().reshape(-1, 1)
        mi = mutual_info_classif(x, y, random_state=42, n_neighbors=5)[0]
        if pd.Series(valid).nunique() > 1:
            correlations = safe_spearman_correlations(variable_base, values)
        else:
            correlations = pd.Series(dtype=np.float64)
        high_corr = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
        top_corr_name = str(high_corr.index[0]) if len(high_corr) else ""
        top_corr = float(high_corr.iloc[0]) if len(high_corr) else np.nan
        diagnostics.append(
            {
                "feature_name": name,
                "valid_count": int(finite_mask.sum()),
                "missing_count": int((~finite_mask).sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "missing_pct": float((~finite_mask).mean() * 100),
                "unique_value_count": int(pd.Series(valid).nunique()),
                "cardinality_per_valid_row": float(pd.Series(valid).nunique() / max(1, len(valid))),
                "mode_count": int(pd.Series(valid).value_counts().iloc[0]) if len(valid) else 0,
                "mode_share_pct": float(pd.Series(valid).value_counts(normalize=True).iloc[0] * 100) if len(valid) else np.nan,
                "min": float(np.min(valid)) if len(valid) else np.nan,
                "median": float(np.median(valid)) if len(valid) else np.nan,
                "mean": float(np.mean(valid)) if len(valid) else np.nan,
                "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
                "max": float(np.max(valid)) if len(valid) else np.nan,
                "raw_univariate_auc_diagnostic_only": auc,
                "direction_adjusted_auc_diagnostic_only": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                "mutual_information_diagnostic_only": float(mi),
                "top_raw_numeric_spearman_feature": top_corr_name,
                "top_raw_numeric_spearman": top_corr,
                "fit_fold_feature_importance": "REQUIRES_PHASE3B_ABLATION_RUN",
                "incremental_cv_contribution": "SEE_PRODUCT_COMPONENT_ABLATIONS",
                "target_used_for_feature_construction": False,
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostics)
    importance_path = OUT / "PRODUCT_FEATURE_IMPORTANCE.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        product_importance = (
            importance.loc[importance.variant == "all_product"]
            .groupby("feature_name", as_index=False)
            .agg(
                mean_gain_importance=("gain_importance", "mean"),
                mean_gain_share=("gain_share", "mean"),
            )
        )
        diagnostic_frame = diagnostic_frame.drop(
            columns=["fit_fold_feature_importance"], errors="ignore"
        ).merge(product_importance, how="left", on="feature_name")
    diagnostic_frame.to_csv(OUT / "PRODUCT_FEATURE_DIAGNOSTICS.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(OUT / "FEE_HYPOTHESIS_ANALYSIS.csv", index=False)
    pd.DataFrame(lattice_rows).to_csv(OUT / "FINANCING_LATTICE_ANALYSIS.csv", index=False)
    pd.DataFrame(reconstruction_rows).to_csv(OUT / "PRODUCT_RECONSTRUCTION_STABILITY.csv", index=False)
    pd.DataFrame(group_rows).to_csv(OUT / "PRODUCT_GROUP_STABILITY.csv", index=False)
    product.to_csv(OUT / "PHASE3B_PRODUCT_FEATURES_OOF_DIAGNOSTIC.csv", index=False)

    # Descriptive redundancy across supplied raw numeric columns and OOF product features.
    combined = pd.concat([base_numeric.reset_index(drop=True), product], axis=1)
    variable_columns = combined.loc[:, combined.nunique(dropna=True) > 1]
    corr = variable_columns.corr(method="spearman", min_periods=1000)
    redundancy = []
    product_names = list(product.columns)
    for i, left in enumerate(product_names):
        for right in list(base_numeric.columns) + product_names[i + 1 :]:
            value = (
                corr.loc[left, right]
                if left in corr.index and right in corr.columns
                else np.nan
            )
            if np.isfinite(value) and abs(value) >= 0.95:
                redundancy.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "spearman_correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                        "threshold": 0.95,
                        "interpretation": "Descriptive redundancy only; no target-based selection.",
                    }
                )
    pd.DataFrame(redundancy).to_csv(OUT / "FEATURE_REDUNDANCY_REPORT.csv", index=False)

    # Target-free control diagnostics: parameter and rounding sensitivity, not model scores.
    control_rows = []
    for fold in sorted(np.unique(folds)):
        fit = raw.iloc[np.flatnonzero(folds != fold)]
        valid = raw.iloc[np.flatnonzero(folds == fold)]
        c_fit, c_val = finite(fit, "AMT_CREDIT"), finite(valid, "AMT_CREDIT")
        a_fit, a_val = finite(fit, "AMT_ANNUITY"), finite(valid, "AMT_ANNUITY")
        g_fit, g_val = finite(fit, "AMT_GOODS_PRICE"), finite(valid, "AMT_GOODS_PRICE")
        cash_fit = fit["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        cash_val = valid["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        q_fit, q_val = safe_ratio(a_fit, c_fit), safe_ratio(a_val, c_val)
        q_fit_mask = cash_fit & np.isfinite(q_fit)
        q_val_mask = cash_val & np.isfinite(q_val)
        for rounding in ("round", "floor", "ceil"):
            for precision in (3, 4, 5, 6):
                op = {"round": np.round, "floor": np.floor, "ceil": np.ceil}[rounding]
                fk = op(q_fit[q_fit_mask] * 10**precision) / 10**precision
                vk = op(q_val[q_val_mask] * 10**precision) / 10**precision
                seen, repeated, fit_repeat = key_coverage(fk, vk)
                rec = summarize_error(a_val[q_val_mask] - vk * c_val[q_val_mask])
                control_rows.append(
                    {
                        "fold": int(fold),
                        "control_type": "q_rounding",
                        "setting": f"{rounding}_{precision}_decimals",
                        "fit_key_unique": int(pd.Series(fk).nunique()),
                        "validation_key_unique": int(pd.Series(vk).nunique()),
                        "validation_seen_pct": seen * 100,
                        "validation_repeated_in_fit_pct": repeated * 100,
                        "fit_repeated_pct": fit_repeat * 100,
                        **rec,
                        "interpretation": "Unsupervised rounding/reconstruction sensitivity; not a model ablation.",
                    }
                )
        for fee in (17.3, 83.7, 177.1, 237.9):
            fee_fit_mask = cash_fit & np.isfinite(a_fit) & np.isfinite(c_fit) & (c_fit > 0)
            fee_val_mask = cash_val & np.isfinite(a_val) & np.isfinite(c_val) & (c_val > 0)
            fk = np.round((a_fit[fee_fit_mask] - fee) / c_fit[fee_fit_mask], 5)
            vk = np.round((a_val[fee_val_mask] - fee) / c_val[fee_val_mask], 5)
            seen, repeated, fit_repeat = key_coverage(fk, vk)
            rec = summarize_error(a_val[fee_val_mask] - (fee + vk * c_val[fee_val_mask]))
            control_rows.append(
                {
                    "fold": int(fold),
                    "control_type": "randomized_fee_constant",
                    "setting": str(fee),
                    "fit_key_unique": int(pd.Series(fk).nunique()),
                    "validation_key_unique": int(pd.Series(vk).nunique()),
                    "validation_seen_pct": seen * 100,
                    "validation_repeated_in_fit_pct": repeated * 100,
                    "fit_repeated_pct": fit_repeat * 100,
                    **rec,
                    "interpretation": "Predeclared arbitrary fee controls; no TARGET used.",
                }
            )
        x_fit, x_val = safe_ratio(c_fit, g_fit) - 1, safe_ratio(c_val, g_val) - 1
        valid_fit = cash_fit & np.isfinite(x_fit) & (c_fit > g_fit) & (g_fit > 0)
        valid_val = cash_val & np.isfinite(x_val) & (c_val > g_val) & (g_val > 0)
        for step in (0.0071, 0.0097, 0.0179, 0.0231):
            kf, kv = np.rint(x_fit[valid_fit] / step), np.rint(x_val[valid_val] / step)
            seen, repeated, fit_repeat = key_coverage(kf, kv)
            rec = summarize_error(c_val[valid_val] - g_val[valid_val] * (1 + kv * step))
            control_rows.append(
                {
                    "fold": int(fold),
                    "control_type": "randomized_lattice_step",
                    "setting": str(step),
                    "fit_key_unique": int(pd.Series(kf).nunique()),
                    "validation_key_unique": int(pd.Series(kv).nunique()),
                    "validation_seen_pct": seen * 100,
                    "validation_repeated_in_fit_pct": repeated * 100,
                    "fit_repeated_pct": fit_repeat * 100,
                    **rec,
                    "interpretation": "Predeclared arbitrary step controls; no TARGET used.",
                }
            )
        # Small deterministic annuity perturbations measure whether Q5 keys depend on cents/rounding noise.
        for sigma in (0.01, 0.1, 1.0):
            rng = np.random.default_rng(92000 + int(fold) + int(sigma * 100))
            noise_fit = rng.normal(0, sigma, size=len(a_fit))
            noise_val = rng.normal(0, sigma, size=len(a_val))
            perturb_mask = q_val_mask
            base = np.round(q_val[perturb_mask], 5)
            noisy_ratio = safe_ratio(a_val + noise_val, c_val)
            noisy = np.round(noisy_ratio[perturb_mask], 5)
            control_rows.append(
                {
                    "fold": int(fold),
                    "control_type": "annuity_measurement_perturbation",
                    "setting": f"normal_sd_{sigma}",
                    "fit_key_unique": np.nan,
                    "validation_key_unique": int(pd.Series(base).nunique()),
                    "validation_seen_pct": float(np.mean(base == noisy) * 100),
                    "validation_repeated_in_fit_pct": np.nan,
                    "fit_repeated_pct": np.nan,
                    **summarize_error(base - noisy),
                    "interpretation": "Validation ratio-key agreement after seeded feature perturbation.",
                }
            )
        # Frequency thresholds are assessed on fit-only Q5 frequencies and the validation share retained.
        fit_q5 = np.round(q_fit[cash_fit], 5)
        val_q5 = np.round(q_val[cash_val], 5)
        counts = pd.Series(fit_q5).value_counts()
        val_counts = pd.Series(val_q5).map(counts).fillna(0).to_numpy()
        for threshold in (2, 5, 10, 25, 50, 100):
            control_rows.append(
                {
                    "fold": int(fold),
                    "control_type": "key_frequency_threshold",
                    "setting": f"minimum_fit_count_{threshold}",
                    "fit_key_unique": int((counts >= threshold).sum()),
                    "validation_key_unique": int(pd.Series(val_q5).nunique()),
                    "validation_seen_pct": float(np.mean(val_counts >= threshold) * 100),
                    "validation_repeated_in_fit_pct": float(np.mean(val_counts >= 2) * 100),
                    "fit_repeated_pct": float(np.mean(counts.reindex(fit_q5).to_numpy() >= threshold) * 100),
                    "n": int(len(val_q5)),
                    "interpretation": "Train-only rarity threshold diagnostics; no label usage.",
                }
            )
    pd.DataFrame(control_rows).to_csv(OUT / "NEGATIVE_CONTROL_RESULTS.csv", index=False)

    # New formulas are row-wise, with explicit domain caveats; label associations are diagnostics only.
    new_frame, hypotheses = new_formula_frame(raw)
    hypotheses.to_csv(OUT / "NEW_FORMULA_HYPOTHESES.csv", index=False)
    new_rows = []
    for name in new_frame:
        values = new_frame[name].to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        auc = roc_auc_score(y[mask], values[mask]) if len(np.unique(y[mask])) == 2 else np.nan
        if pd.Series(values[mask]).nunique() > 1:
            existing = safe_spearman_correlations(variable_base, values)
        else:
            existing = pd.Series(dtype=np.float64)
        best_name = existing.abs().idxmax() if len(existing) else ""
        new_rows.append(
            {
                "feature_name": name,
                "valid_count": int(mask.sum()),
                "missing_count": int((~mask).sum()),
                "infinite_count": int(np.isinf(values).sum()),
                "min": float(np.min(values[mask])) if mask.any() else np.nan,
                "median": float(np.median(values[mask])) if mask.any() else np.nan,
                "mean": float(np.mean(values[mask])) if mask.any() else np.nan,
                "std": float(np.std(values[mask], ddof=1)) if mask.sum() > 1 else np.nan,
                "max": float(np.max(values[mask])) if mask.any() else np.nan,
                "univariate_auc_descriptive_only": auc,
                "direction_adjusted_auc_descriptive_only": max(auc, 1 - auc) if np.isfinite(auc) else np.nan,
                "most_redundant_raw_numeric_column": str(best_name),
                "max_abs_spearman_with_raw_numeric": float(existing.abs().max()) if len(existing) else np.nan,
                "incremental_cv_status": "NOT_RUN_YET",
                "target_used_for_feature_construction": False,
            }
        )
    pd.DataFrame(new_rows).to_csv(OUT / "NEW_FORMULA_VALIDATION.csv", index=False)

    started_finish = datetime.now(timezone.utc)
    log_lines = [
        f"started_utc={started_utc.isoformat()}",
        f"finished_utc={started_finish.isoformat()}",
        f"runtime_seconds={time.time()-started:.3f}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"train_sha256={sha256(DATA/'train.csv')}",
        f"baseline_folds_sha256={sha256(BASELINE_DIR/'FOLD_ASSIGNMENTS.csv')}",
        f"shared_runner_sha256={sha256(SHARED_PATH)}",
        f"analysis_source_sha256={sha256(Path(__file__).resolve())}",
        "outer_fold_validation_features_fit_on_outer_train_only=True",
        "TARGET_passed_to_feature_builder=False",
        "status=SUCCESS",
        f"product_feature_count={product.shape[1]}",
        f"lattice_rows={len(lattice_rows)}",
        f"fee_rows={len(fee_rows)}",
        f"group_rows={len(group_rows)}",
    ]
    with (OUT / "EXECUTION_LOG.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n" + "\n".join(log_lines) + "\n")
    (OUT / "PHASE3B_PRODUCT_ANALYSIS_LOG.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
