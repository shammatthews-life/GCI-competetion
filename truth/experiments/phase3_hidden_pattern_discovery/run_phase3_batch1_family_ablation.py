#!/usr/bin/env python3
"""Separate product-decoding, financial, and interaction value under fixed folds."""

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
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
OUT = HERE / "batch1_family_ablation_02"
OUT.mkdir(parents=True, exist_ok=True)
RUNNER_PATH = HERE / "run_phase3_hidden_pattern_discovery.py"
spec = importlib.util.spec_from_file_location("phase3_shared", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared Phase 3 feature code: {RUNNER_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def lattice_support(values: np.ndarray, delta: float) -> tuple[int, float, float]:
    if not np.isfinite(delta) or delta <= 0:
        return 0, np.nan, np.nan
    valid = np.isfinite(values)
    scaled = values[valid] / delta
    residual = np.abs(scaled - np.rint(scaled))
    return int(valid.sum()), float(np.mean(residual < 0.1)), float(np.mean(residual))


def key_test(train_values, valid_values):
    counts = pd.Series(train_values).value_counts(dropna=True)
    valid = np.isfinite(valid_values)
    mapped = np.array([counts.get(v, 0) if ok else -1 for v, ok in zip(valid_values, valid)])
    if not valid.any():
        return {
            "fit_repeat_share": np.nan,
            "valid_seen_share": np.nan,
            "valid_repeated_share": np.nan,
            "valid_key_frequency_increased_share": np.nan,
        }
    return {
        "fit_repeat_share": float((counts.reindex(train_values).fillna(0).to_numpy() >= 2).mean()),
        "valid_seen_share": float(np.mean(mapped[valid] >= 1)),
        "valid_repeated_share": float(np.mean(mapped[valid] >= 2)),
        "valid_key_frequency_increased_share": float(np.mean(mapped[valid] > 0)),
    }


def main():
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    train_path = shared.TRAIN_PATH
    train = pd.read_csv(train_path)
    y = train[shared.TARGET].to_numpy(dtype=np.int8)
    ids = train[shared.ID].to_numpy()
    raw = train.drop(columns=[shared.TARGET]).reset_index(drop=True)
    fold_map = pd.read_csv(shared.BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(shared.ID)["fold"]
    folds = fold_map.loc[ids].to_numpy()
    baseline_oof = pd.read_csv(shared.BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv").set_index(shared.ID)
    if not baseline_oof.loc[ids, shared.TARGET].to_numpy().astype(np.int8).tolist() == y.tolist():
        raise ValueError("Saved baseline target mismatch.")
    baseline_prediction = baseline_oof.loc[ids, "OOF_RAW_BASELINE"].to_numpy()

    product_interactions = [
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
    all_features = shared.MODEL_FEATURES
    families = {
        "product_decoder_only": [
            c for c in all_features if c.startswith("PROD_") and c not in product_interactions
        ],
        "financial_formulas_only": [c for c in all_features if c.startswith("FIN_")],
        "product_financial_interactions_only": product_interactions,
        "all_phase3_batch1_features": all_features,
    }
    for name, cols in families.items():
        if not cols or len(set(cols)) != len(cols) or not set(cols).issubset(set(all_features)):
            raise ValueError(f"Invalid feature family {name}: {cols}")

    predictions = {
        name: np.full(len(train), np.nan, dtype=np.float64) for name in families
    }
    fold_rows = []
    fee_rows = []
    lattice_rows = []
    config = shared.baseline.MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }

    for fold in sorted(np.unique(folds)):
        valid_mask = folds == fold
        fit_positions = np.flatnonzero(~valid_mask)
        valid_positions = np.flatnonzero(valid_mask)
        y_fit = y[fit_positions]
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=20261102 + int(fold),
        )
        inner_fit_local, inner_stop_local = next(inner.split(np.zeros(len(y_fit)), y_fit))
        inner_fit_pos = fit_positions[inner_fit_local]
        inner_stop_pos = fit_positions[inner_stop_local]
        inner_params = shared.fit_fee_and_step(raw.iloc[inner_fit_pos])
        inner_maps = shared.fit_product_maps(raw.iloc[inner_fit_pos], inner_params)
        inner_fit_base = shared.baseline.build_rowwise_features(raw.iloc[inner_fit_pos])
        inner_stop_base = shared.baseline.build_rowwise_features(raw.iloc[inner_stop_pos])
        inner_fit_features = shared.make_product_features(raw.iloc[inner_fit_pos], inner_params, inner_maps)
        inner_stop_features = shared.make_product_features(raw.iloc[inner_stop_pos], inner_params, inner_maps)
        inner_best = {}

        for name, columns in families.items():
            fit_x = pd.concat(
                [inner_fit_base, inner_fit_features[columns]], axis=1
            )
            stop_x = pd.concat(
                [inner_stop_base, inner_stop_features[columns]], axis=1
            )
            fit_matrix, stop_matrix = shared.baseline.make_lightgbm_frames(fit_x, stop_x)
            fit_set = lgb.Dataset(fit_matrix, label=y[inner_fit_pos])
            stop_set = lgb.Dataset(stop_matrix, label=y[inner_stop_pos], reference=fit_set)
            selector = lgb.train(
                {**params, "seed": shared.baseline.FOLD_SEED + int(fold)},
                fit_set,
                num_boost_round=config["num_boost_round"],
                valid_sets=[stop_set],
                callbacks=[
                    lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            inner_best[name] = int(selector.best_iteration or config["num_boost_round"])

        outer_params = shared.fit_fee_and_step(raw.iloc[fit_positions])
        outer_maps = shared.fit_product_maps(raw.iloc[fit_positions], outer_params)
        fit_base = shared.baseline.build_rowwise_features(raw.iloc[fit_positions])
        valid_base = shared.baseline.build_rowwise_features(raw.iloc[valid_positions])
        fit_features = shared.make_product_features(raw.iloc[fit_positions], outer_params, outer_maps)
        valid_features = shared.make_product_features(raw.iloc[valid_positions], outer_params, outer_maps)
        fold_baseline_auc = roc_auc_score(y[valid_positions], baseline_prediction[valid_positions])

        for name, columns in families.items():
            fit_x = pd.concat([fit_base, fit_features[columns]], axis=1)
            valid_x = pd.concat([valid_base, valid_features[columns]], axis=1)
            fit_matrix, valid_matrix = shared.baseline.make_lightgbm_frames(fit_x, valid_x)
            model = lgb.train(
                {**params, "seed": shared.baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_matrix, label=y_fit),
                num_boost_round=inner_best[name],
                callbacks=[lgb.log_evaluation(0)],
            )
            pred = model.predict(valid_matrix, num_iteration=inner_best[name])
            predictions[name][valid_positions] = pred
            fold_rows.append(
                {
                    "fold": int(fold),
                    "family": name,
                    "features_added": len(columns),
                    "best_iteration_inner_only": inner_best[name],
                    "fold_auc": roc_auc_score(y[valid_positions], pred),
                    "fold_auc_baseline": fold_baseline_auc,
                    "fold_delta_vs_baseline": roc_auc_score(y[valid_positions], pred) - fold_baseline_auc,
                    "fee_fit_only": outer_params["fee_estimate_fit_only"],
                    "step_fit_only": outer_params["step_estimate_fit_only"],
                }
            )

        fit_cash = raw.iloc[fit_positions]["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        valid_cash = raw.iloc[valid_positions]["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        fit_credit = shared.finite_column(raw.iloc[fit_positions], "AMT_CREDIT")
        fit_goods = shared.finite_column(raw.iloc[fit_positions], "AMT_GOODS_PRICE")
        fit_annuity = shared.finite_column(raw.iloc[fit_positions], "AMT_ANNUITY")
        valid_credit = shared.finite_column(raw.iloc[valid_positions], "AMT_CREDIT")
        valid_goods = shared.finite_column(raw.iloc[valid_positions], "AMT_GOODS_PRICE")
        valid_annuity = shared.finite_column(raw.iloc[valid_positions], "AMT_ANNUITY")
        q_fit = shared.safe_ratio(fit_annuity, fit_credit)
        q_valid = shared.safe_ratio(valid_annuity, valid_credit)
        p_fit_est = shared.safe_ratio(fit_annuity - outer_params["fee_estimate_fit_only"], fit_credit)
        p_valid_est = shared.safe_ratio(valid_annuity - outer_params["fee_estimate_fit_only"], valid_credit)
        p_fit_v17 = shared.safe_ratio(fit_annuity - 129.5, fit_credit)
        p_valid_v17 = shared.safe_ratio(valid_annuity - 129.5, valid_credit)
        fee_variants = {
            "unadjusted_q5": (np.round(q_fit[fit_cash], 5), np.round(q_valid[valid_cash], 5)),
            "fold_estimated_fee_p5": (
                np.round(p_fit_est[fit_cash], 5),
                np.round(p_valid_est[valid_cash], 5),
            ),
            "v17_fixed_fee_129_5_p5": (
                np.round(p_fit_v17[fit_cash], 5),
                np.round(p_valid_v17[valid_cash], 5),
            ),
        }
        for method, (fit_keys, valid_keys) in fee_variants.items():
            measured = key_test(fit_keys, valid_keys)
            fee_rows.append(
                {
                    "fold": int(fold),
                    "method": method,
                    "fee_used": (
                        0.0 if method == "unadjusted_q5"
                        else 129.5 if method == "v17_fixed_fee_129_5_p5"
                        else outer_params["fee_estimate_fit_only"]
                    ),
                    "fit_cash_rows": int(len(fit_keys)),
                    "valid_cash_rows": int(len(valid_keys)),
                    "fit_key_duplicate_share": measured["fit_repeat_share"],
                    "valid_key_seen_in_fit_share": measured["valid_seen_share"],
                    "valid_key_repeated_in_fit_share": measured["valid_repeated_share"],
                }
            )

        fit_increment = (
            fit_credit / fit_goods - 1.0
        )
        valid_increment = (
            valid_credit / valid_goods - 1.0
        )
        fit_lattice = fit_cash & (fit_credit > fit_goods) & np.isfinite(fit_increment)
        valid_lattice = valid_cash & (valid_credit > valid_goods) & np.isfinite(valid_increment)
        step_tests = {
            "fold_estimated_step": outer_params["step_estimate_fit_only"],
            "v17_0_0132": 0.0132,
            "half_v17_0_0066": 0.0066,
        }
        for label, delta in step_tests.items():
            n_fit, fit_support, fit_mae = lattice_support(fit_increment[fit_lattice], delta)
            n_valid, valid_support, valid_mae = lattice_support(valid_increment[valid_lattice], delta)
            coords = valid_increment[valid_lattice] / delta
            rounded = np.rint(coords)
            on_lattice = np.abs(coords - rounded) < 0.1
            odd_share = float(np.mean((np.abs(rounded[on_lattice]).astype(np.int64) % 2) == 1)) if on_lattice.any() else np.nan
            lattice_rows.append(
                {
                    "fold": int(fold),
                    "step_hypothesis": label,
                    "step": delta,
                    "fit_n_cash_credit_gt_goods": n_fit,
                    "fit_lattice_support": fit_support,
                    "fit_lattice_mae": fit_mae,
                    "validation_n_cash_credit_gt_goods": n_valid,
                    "validation_lattice_support": valid_support,
                    "validation_lattice_mae": valid_mae,
                    "validation_odd_integer_share_among_supported": odd_share,
                }
            )

    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"OOF predictions incomplete for {name}.")
    oof = pd.DataFrame(
        {
            shared.ID: ids,
            shared.TARGET: y,
            "fold": folds,
            "OOF_BASELINE_REFERENCE": baseline_prediction,
            **{f"OOF_{name}": pred for name, pred in predictions.items()},
        }
    )
    oof.to_csv(OUT / "BATCH1_FAMILY_OOF_PREDICTIONS.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "BATCH1_FAMILY_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(fee_rows).to_csv(OUT / "FEE_HYPOTHESIS_FOLD_TESTS.csv", index=False)
    pd.DataFrame(lattice_rows).to_csv(OUT / "FINANCING_LATTICE_FOLD_TESTS.csv", index=False)

    summary = []
    for name, pred in predictions.items():
        subset = fold_df.loc[fold_df.family == name].sort_values("fold")
        fold_auc = subset["fold_auc"].to_numpy()
        fold_delta = subset["fold_delta_vs_baseline"].to_numpy()
        summary.append(
            {
                "family": name,
                "features_added": len(families[name]),
                "pooled_oof_auc": roc_auc_score(y, pred),
                "mean_fold_auc": float(np.mean(fold_auc)),
                "std_fold_auc": float(np.std(fold_auc, ddof=1)),
                "mean_paired_fold_delta": float(np.mean(fold_delta)),
                "std_paired_fold_delta": float(np.std(fold_delta, ddof=1)),
                "pooled_delta_vs_verified_baseline": roc_auc_score(y, pred) - roc_auc_score(y, baseline_prediction),
                "folds_improved": int((fold_delta > 0).sum()),
                **{f"fold_{i+1}_delta": float(value) for i, value in enumerate(fold_delta)},
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "BATCH1_FAMILY_ABLATION_RESULTS.csv", index=False)

    env = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "source_sha256": sha256(Path(__file__)),
        "shared_feature_runner_sha256": sha256(RUNNER_PATH),
        "train_sha256": sha256(train_path),
        "seed": 42,
        "outer_folds": 5,
        "inner_stop_fraction": 0.1,
        "outer_validation_used_for_early_stopping": False,
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\nscikit-learn=={sklearn.__version__}\nlightgbm=={lgb.__version__}\n",
        encoding="utf-8",
    )
    base_score = roc_auc_score(y, baseline_prediction)
    report = f"""# Batch 1 family ablation — discovery results

Run: {env['started_utc']} to {env['finished_utc']} UTC; {env['runtime_seconds']:.1f} seconds.

The saved corrected baseline OOF was re-aligned by ID and labels. It remains the comparator (pooled AUC {base_score:.12f}). Each of the four feature families got its own inner-split early-stopping choice, then a refit on the complete outer training fold. Product constants and key frequencies were fit on the current fit partition only. No test covariates, TARGET-derived features, or external data were used.

## Family-level comparison

{summary_df.to_string(index=False)}

`product_decoder_only` isolates Q/P rounded keys, fit-only frequency signals, fold-estimated fee candidates, and the financing lattice. `financial_formulas_only` tests domain ratios without product decoding. `product_financial_interactions_only` tests external-score/financing interactions. The combined family is included to measure complementarity.

These are development-fold results on folds already used in Phase 2, not untouched confirmation. A positive family result is only a discovery lead and is not promoted.

## Fee hypothesis

See `FEE_HYPOTHESIS_FOLD_TESTS.csv`. It compares unadjusted Q5, the fold-estimated fee P5, and v17's fixed 129.5 P5 using fit-set key duplication and validation key repetition in the fit map. A fee is not supported merely because a rounded key appears in the fit set; the adjusted key should improve held-out repeated-key coverage consistently.

## Lattice hypothesis

See `FINANCING_LATTICE_FOLD_TESTS.csv`. Fold-local searches selected the coarser-grid candidate 0.0066 in the initial run, exactly half of v17's 0.0132. Both are stable candidates but they have different integer interpretations. A strong 0.0066 score can arise because every 0.0132 step is also an even 0.0066 step; inspect held-out odd-integer support and actual error distributions before naming a primitive increment.

## Decision

No individual formula is promoted. The 0.79 objective remains unachieved/unverified. Continue only after reviewing the separate family ablations and freeze a new confirmation protocol before feature selection.
"""
    (OUT / "BATCH1_FAMILY_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    log = [
        f"started_utc={env['started_utc']}",
        f"finished_utc={env['finished_utc']}",
        f"runtime_seconds={env['runtime_seconds']:.3f}",
        f"train_shape={train.shape}",
        f"family_count={len(families)}",
        f"baseline_pooled_auc={base_score:.12f}",
        f"source_sha256={env['source_sha256']}",
        "status=SUCCESS",
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(summary_df.to_string(index=False))
    print("outputs:")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(f"{path} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
