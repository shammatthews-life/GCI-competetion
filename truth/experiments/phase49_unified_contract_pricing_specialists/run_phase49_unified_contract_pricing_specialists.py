from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
MIN_EXPERT_FIT_ROWS = 5000
MIN_EXPERT_VALID_ROWS = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_contract_gate(raw: pd.DataFrame) -> np.ndarray:
    if "NAME_CONTRACT_TYPE" not in raw.columns:
        raise ValueError("NAME_CONTRACT_TYPE missing from input columns")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING")
    contract_values = contract.to_numpy(dtype=str)
    if set(np.unique(contract_values)) != {"Cash loans", "Revolving loans"}:
        raise ValueError("Unexpected contract categories; refusing implicit routing")
    return (contract_values == "Revolving loans").astype(np.int8)


def build_pricing_block(df: pd.DataFrame, m1: float, m2: float, m3: float) -> pd.DataFrame:
    """
    Constructs domain pricing, fee decoding, and external geometry features.
    Medians are strictly passed from outer training folds (zero global leakage).
    """
    e1 = df["EXT_SOURCE_1"].fillna(m1).to_numpy(dtype=np.float64)
    e2 = df["EXT_SOURCE_2"].fillna(m2).to_numpy(dtype=np.float64)
    e3 = df["EXT_SOURCE_3"].fillna(m3).to_numpy(dtype=np.float64)
    u0 = np.clip(e1, 1e-4, 1.0 - 1e-4)
    u1 = np.clip(e2, 1e-4, 1.0 - 1e-4)
    u2 = np.clip(e3, 1e-4, 1.0 - 1e-4)

    inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").to_numpy(dtype=np.float64)
    cr = pd.to_numeric(df["AMT_CREDIT"], errors="coerce").to_numpy(dtype=np.float64)
    an = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce").to_numpy(dtype=np.float64)
    gp_raw = pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce").to_numpy(dtype=np.float64)
    gp = np.where(np.isfinite(gp_raw), gp_raw, cr)
    is_cash = (df["NAME_CONTRACT_TYPE"].to_numpy(dtype=str) == "Cash loans")

    with np.errstate(divide="ignore", invalid="ignore"):
        qb = (an - 129.50) / (cr + 1e-5)
        ratio_cg = cr / (gp + 1e-5) - 1.0
        step_res = np.abs(ratio_cg - 0.0066 * np.round(ratio_cg / 0.0066))
        neg_down = np.maximum(cr - gp, 0.0) / (gp + 1.0)
        q0_fee = np.where(is_cash & (qb > 0), qb, an / (cr + 1e-5))

        monthly_inc = inc / 12.0
        solv_cons = (monthly_inc / (an + 1.0)) * (u1 * u2)
        surplus = (monthly_inc - an) / (an + 1.0)
        ext_prod = u0 * u1 * u2
        ext_harm = 3.0 / ((1.0 / u0) + (1.0 / u1) + (1.0 / u2))
        ext_min = np.minimum(np.minimum(u0, u1), u2)

    feats = {
        "Q0_FEE": np.nan_to_num(q0_fee, nan=0.0),
        "TARIFF_STEP_0066_RESIDUAL": np.nan_to_num(step_res, nan=0.0),
        "NEGATIVE_DOWNPAYMENT": np.nan_to_num(neg_down, nan=0.0),
        "EXT_PRODUCT": np.nan_to_num(ext_prod, nan=0.0),
        "EXT_HARMONIC": np.nan_to_num(ext_harm, nan=0.0),
        "EXT_MIN": np.nan_to_num(ext_min, nan=0.0),
        "MANIFOLD_SOLVENCY_CONSENSUS": np.nan_to_num(solv_cons, nan=0.0),
        "INCOME_SURPLUS_TO_ANNUITY": np.nan_to_num(surplus, nan=0.0),
    }
    return pd.DataFrame(feats, index=df.index)


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return csr_matrix(values)


def make_matrices_with_pricing(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    min_frequency: int,
) -> tuple[csr_matrix, csr_matrix]:
    numeric, categorical, rowwise, _, keys, _ = blocks

    # In-fold medians strictly computed on fit_rows
    m1 = float(raw.iloc[fit_rows]["EXT_SOURCE_1"].median())
    m2 = float(raw.iloc[fit_rows]["EXT_SOURCE_2"].median())
    m3 = float(raw.iloc[fit_rows]["EXT_SOURCE_3"].median())

    fit_pricing = build_pricing_block(raw.iloc[fit_rows], m1, m2, m3)
    apply_pricing = build_pricing_block(raw.iloc[apply_rows], m1, m2, m3)

    fit_numeric = pd.concat(
        [
            numeric.iloc[fit_rows].reset_index(drop=True),
            rowwise.iloc[fit_rows].reset_index(drop=True),
            fit_pricing.reset_index(drop=True),
        ],
        axis=1,
    )
    apply_numeric = pd.concat(
        [
            numeric.iloc[apply_rows].reset_index(drop=True),
            rowwise.iloc[apply_rows].reset_index(drop=True),
            apply_pricing.reset_index(drop=True),
        ],
        axis=1,
    )

    fit_blocks = [_numeric_matrix(fit_numeric)]
    apply_blocks = [_numeric_matrix(apply_numeric)]

    raw_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32, sparse_output=True)
    fit_raw = raw_encoder.fit_transform(categorical.iloc[fit_rows].astype(str)).tocsr()
    apply_raw = raw_encoder.transform(categorical.iloc[apply_rows].astype(str)).tocsr()
    fit_blocks.append(fit_raw)
    apply_blocks.append(apply_raw)

    fit_key_values = keys.iloc[fit_rows].astype(str)
    apply_key_values = keys.iloc[apply_rows].astype(str)
    key_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=min_frequency,
        dtype=np.float32,
        sparse_output=True,
    )
    fit_key = key_encoder.fit_transform(fit_key_values).tocsr()
    apply_key = key_encoder.transform(apply_key_values).tocsr()
    fit_blocks.append(fit_key)
    apply_blocks.append(apply_key)

    fit_counts: list[np.ndarray] = []
    apply_counts: list[np.ndarray] = []
    for column in phase33.KEY_COLUMNS:
        counts = fit_key_values[column].value_counts()
        fit_counts.append(fit_key_values[column].map(counts).to_numpy(np.float32))
        apply_counts.append(
            apply_key_values[column].map(counts).fillna(0).to_numpy(np.float32)
        )
    fit_blocks.append(csr_matrix(np.column_stack(fit_counts), dtype=np.float32))
    apply_blocks.append(csr_matrix(np.column_stack(apply_counts), dtype=np.float32))

    return (
        hstack(fit_blocks, format="csr", dtype=np.float32),
        hstack(apply_blocks, format="csr", dtype=np.float32),
    )


def equal_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a + b) / 2.0).astype(np.float32)


def _lgb_model(seed: int, iterations: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        num_leaves=31,
        min_child_samples=50,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=10.0,
        max_bin=255,
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def _xgb_model(seed: int, iterations: int, stopping: bool) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=iterations,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=40,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=8,
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
        verbosity=0,
    )


def train_predict_lgb(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    inner_fit_idx, inner_valid_idx = next(inner.split(np.zeros(len(fit_rows)), y[fit_rows]))
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]

    x_inner_fit, x_inner_valid = make_matrices_with_pricing(
        blocks, raw, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = _lgb_model(seed, MAX_ITERATIONS)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_valid, y[inner_valid])],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    iterations = max(100, int(selector.best_iteration_ or MAX_ITERATIONS))
    del selector, x_inner_fit, x_inner_valid

    x_fit, x_apply = make_matrices_with_pricing(
        blocks, raw, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    model = _lgb_model(seed, iterations)
    model.fit(x_fit, y[fit_rows])
    prediction = model.predict_proba(x_apply)[:, 1]
    return prediction.astype(np.float32), iterations


def train_predict_xgb(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
    inner_fit_idx, inner_valid_idx = next(inner.split(np.zeros(len(fit_rows)), y[fit_rows]))
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]

    x_inner_fit, x_inner_valid = make_matrices_with_pricing(
        blocks, raw, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = _xgb_model(seed, MAX_ITERATIONS, stopping=True)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_valid, y[inner_valid])],
        verbose=False,
    )
    iterations = max(100, int(selector.best_iteration) + 1)
    del selector, x_inner_fit, x_inner_valid

    x_fit, x_apply = make_matrices_with_pricing(
        blocks, raw, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    model = _xgb_model(seed, iterations, stopping=False)
    model.fit(x_fit, y[fit_rows], verbose=False)
    prediction = model.predict_proba(x_apply)[:, 1]
    return prediction.astype(np.float32), iterations


def _metric_record(
    y: np.ndarray,
    prediction: np.ndarray,
    folds_indices: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float]:
    fold_aucs = [
        float(roc_auc_score(y[valid], prediction[valid]))
        for _, valid in folds_indices
    ]
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "pooled_log_loss": float(log_loss(y, prediction)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
    }


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must contain unique IDs and TARGET")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.float64)
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    gate = make_contract_gate(raw)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)

    preds: dict[str, np.ndarray] = {
        "lgb_pricing_global": np.full(len(y), np.nan, dtype=np.float32),
        "lgb_pricing_contract_mix": np.full(len(y), np.nan, dtype=np.float32),
        "xgb_pricing_global": np.full(len(y), np.nan, dtype=np.float32),
        "xgb_pricing_contract_mix": np.full(len(y), np.nan, dtype=np.float32),
        "cross_model_pricing_global_blend": np.full(len(y), np.nan, dtype=np.float32),
        "cross_model_pricing_expert_blend": np.full(len(y), np.nan, dtype=np.float32),
    }

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        print(f"\n--- Fold {fold_number}/{FOLDS} (Fit rows: {len(fit_rows):,}, Valid rows: {len(valid_rows):,}) ---", flush=True)

        # 1. LightGBM Pricing Global
        t0 = time.perf_counter()
        lgb_glob, lgb_glob_iter = train_predict_lgb(blocks, raw, fit_rows, valid_rows, y, fold_seed)
        preds["lgb_pricing_global"][valid_rows] = lgb_glob
        print(f"  [LGBM Pricing Global] AUC: {roc_auc_score(y[valid_rows], lgb_glob):.6f} ({lgb_glob_iter} trees, {time.perf_counter()-t0:.1f}s)", flush=True)

        # 2. LightGBM Pricing Contract Experts
        t0 = time.perf_counter()
        lgb_hard = np.full(len(valid_rows), np.nan, dtype=np.float32)
        for g in (0, 1):
            g_fit = fit_rows[gate[fit_rows] == g]
            g_valid_mask = gate[valid_rows] == g
            g_valid = valid_rows[g_valid_mask]
            pred_exp, _ = train_predict_lgb(blocks, raw, g_fit, g_valid, y, fold_seed + g + 1)
            lgb_hard[g_valid_mask] = pred_exp
        lgb_mix = equal_blend(lgb_glob, lgb_hard)
        preds["lgb_pricing_contract_mix"][valid_rows] = lgb_mix
        print(f"  [LGBM Pricing Mix]    AUC: {roc_auc_score(y[valid_rows], lgb_mix):.6f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        # 3. XGBoost Pricing Global
        t0 = time.perf_counter()
        xgb_glob, xgb_glob_iter = train_predict_xgb(blocks, raw, fit_rows, valid_rows, y, fold_seed)
        preds["xgb_pricing_global"][valid_rows] = xgb_glob
        print(f"  [XGB  Pricing Global] AUC: {roc_auc_score(y[valid_rows], xgb_glob):.6f} ({xgb_glob_iter} trees, {time.perf_counter()-t0:.1f}s)", flush=True)

        # 4. XGBoost Pricing Contract Experts
        t0 = time.perf_counter()
        xgb_hard = np.full(len(valid_rows), np.nan, dtype=np.float32)
        for g in (0, 1):
            g_fit = fit_rows[gate[fit_rows] == g]
            g_valid_mask = gate[valid_rows] == g
            g_valid = valid_rows[g_valid_mask]
            pred_exp, _ = train_predict_xgb(blocks, raw, g_fit, g_valid, y, fold_seed + g + 1)
            xgb_hard[g_valid_mask] = pred_exp
        xgb_mix = equal_blend(xgb_glob, xgb_hard)
        preds["xgb_pricing_contract_mix"][valid_rows] = xgb_mix
        print(f"  [XGB  Pricing Mix]    AUC: {roc_auc_score(y[valid_rows], xgb_mix):.6f} ({time.perf_counter()-t0:.1f}s)", flush=True)

        # 5. Cross-Model Blends
        cross_glob = equal_blend(lgb_glob, xgb_glob)
        cross_exp = equal_blend(lgb_mix, xgb_mix)
        preds["cross_model_pricing_global_blend"][valid_rows] = cross_glob
        preds["cross_model_pricing_expert_blend"][valid_rows] = cross_exp
        print(f"  --> Cross Pricing Global Blend AUC: {roc_auc_score(y[valid_rows], cross_glob):.6f}", flush=True)
        print(f"  --> Cross Pricing Expert Blend AUC: {roc_auc_score(y[valid_rows], cross_exp):.6f}", flush=True)

    output_dir.mkdir(parents=False, exist_ok=False)
    oof_path = output_dir / "PHASE49_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE49_SUMMARY.json"

    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{f"oof_{name}": values for name, values in preds.items()},
        }
    )
    oof_frame.to_csv(oof_path, index=False)

    metrics = {
        name: _metric_record(y, values, folds)
        for name, values in preds.items()
    }

    summary: dict[str, Any] = {
        "phase": "Phase 49 unified contract pricing & cross-architecture specialist synthesis",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "pricing_features_added": [
            "Q0_FEE",
            "TARIFF_STEP_0066_RESIDUAL",
            "NEGATIVE_DOWNPAYMENT",
            "EXT_PRODUCT",
            "EXT_HARMONIC",
            "EXT_MIN",
            "MANIFOLD_SOLVENCY_CONSENSUS",
            "INCOME_SURPLUS_TO_ANNUITY",
        ],
        "metrics": metrics,
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + "=" * 90)
    print("  PHASE 49 COMPLETE: POOLED OOF METRICS")
    print("=" * 90)
    for name, m in metrics.items():
        delta_vs_p38 = (m["pooled_auc"] - 0.764835) * 10000.0
        print(f"  {name:<34} Pooled AUC: {m['pooled_auc']:.6f} ({delta_vs_p38:+.1f} bps vs P38) | LogLoss: {m['pooled_log_loss']:.6f} | Mean Fold: {m['mean_fold_auc']:.6f} (±{m['std_fold_auc']:.6f})")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase49_pricing_experts_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)

