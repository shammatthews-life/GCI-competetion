#!/usr/bin/env python3
"""Isolate the information source within the Phase 3 product decoder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = Path(__file__).resolve().parent
OUT = HERE / "batch1_product_component_03"
OUT.mkdir(parents=True, exist_ok=True)
SHARED_PATH = HERE / "run_phase3_hidden_pattern_discovery.py"
spec = importlib.util.spec_from_file_location("phase3_shared_components", SHARED_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared Phase 3 feature runner: {SHARED_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    train = pd.read_csv(shared.TRAIN_PATH)
    y = train[shared.TARGET].to_numpy(dtype=np.int8)
    ids = train[shared.ID].to_numpy()
    raw = train.drop(columns=[shared.TARGET]).reset_index(drop=True)
    fold_map = pd.read_csv(shared.BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(shared.ID)["fold"]
    folds = fold_map.loc[ids].to_numpy()
    baseline_data = pd.read_csv(shared.BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv").set_index(shared.ID)
    baseline_prediction = baseline_data.loc[ids, "OOF_RAW_BASELINE"].to_numpy()
    if not np.array_equal(baseline_data.loc[ids, shared.TARGET].to_numpy(), y):
        raise ValueError("Saved baseline labels differ from the supplied train target.")

    product_features = [
        c for c in shared.MODEL_FEATURES
        if c.startswith("PROD_")
        and not c.startswith(
            (
                "PROD_EXT_",
                "PROD_CREDIT_INCOME_X_",
                "PROD_ANNUITY_INCOME_X_",
                "PROD_CREDIT_GOODS_X_",
                "PROD_ANNUITY_CREDIT_X_",
                "PROD_FINANCE_SHARE_X_",
            )
        )
    ]
    families = {
        "k_lattice_only": [
            "PROD_K_DECIMAL",
            "PROD_K_NEAREST_INTEGER",
            "PROD_K_LATTICE_RESIDUAL",
            "PROD_K_VALID",
            "PROD_K_FIT_FREQUENCY_LOG",
        ],
        "q_rounded_ratio_keys_only": [
            "PROD_Q3",
            "PROD_Q4",
            "PROD_Q5",
            "PROD_Q5_FIT_FREQUENCY_LOG",
        ],
        "p_fee_adjusted_keys_only": [
            "PROD_P3",
            "PROD_P4",
            "PROD_P5",
            "PROD_PF4_FLAG",
            "PROD_FEE_ADJUSTED_ANNUITY",
            "PROD_FEE_RECONSTRUCTION_RESIDUAL",
            "PROD_P5_FIT_FREQUENCY_LOG",
        ],
        "all_product_decoder_features": product_features,
    }
    known = set(shared.MODEL_FEATURES)
    for name, columns in families.items():
        if not columns or not set(columns).issubset(known):
            raise ValueError(f"Invalid component family: {name}")

    predictions = {
        family: np.full(len(train), np.nan, dtype=np.float64) for family in families
    }
    rows = []
    model_config = shared.baseline.MODEL_CONFIG["lightgbm"]
    model_params = {
        key: value
        for key, value in model_config.items()
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
        inner_fit = fit_positions[inner_fit_local]
        inner_stop = fit_positions[inner_stop_local]
        inner_params = shared.fit_fee_and_step(raw.iloc[inner_fit])
        inner_maps = shared.fit_product_maps(raw.iloc[inner_fit], inner_params)
        inner_fit_base = shared.baseline.build_rowwise_features(raw.iloc[inner_fit])
        inner_stop_base = shared.baseline.build_rowwise_features(raw.iloc[inner_stop])
        inner_fit_extra = shared.make_product_features(raw.iloc[inner_fit], inner_params, inner_maps)
        inner_stop_extra = shared.make_product_features(raw.iloc[inner_stop], inner_params, inner_maps)
        best_rounds = {}

        for family, columns in families.items():
            fit_x = pd.concat([inner_fit_base, inner_fit_extra[columns]], axis=1)
            stop_x = pd.concat([inner_stop_base, inner_stop_extra[columns]], axis=1)
            fit_matrix, stop_matrix = shared.baseline.make_lightgbm_frames(fit_x, stop_x)
            fit_set = lgb.Dataset(fit_matrix, label=y[inner_fit])
            stop_set = lgb.Dataset(stop_matrix, label=y[inner_stop], reference=fit_set)
            selection = lgb.train(
                {**model_params, "seed": shared.baseline.FOLD_SEED + int(fold)},
                fit_set,
                num_boost_round=model_config["num_boost_round"],
                valid_sets=[stop_set],
                callbacks=[
                    lgb.early_stopping(model_config["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_rounds[family] = int(selection.best_iteration or model_config["num_boost_round"])

        outer_params = shared.fit_fee_and_step(raw.iloc[fit_positions])
        outer_maps = shared.fit_product_maps(raw.iloc[fit_positions], outer_params)
        fit_base = shared.baseline.build_rowwise_features(raw.iloc[fit_positions])
        valid_base = shared.baseline.build_rowwise_features(raw.iloc[valid_positions])
        fit_extra = shared.make_product_features(raw.iloc[fit_positions], outer_params, outer_maps)
        valid_extra = shared.make_product_features(raw.iloc[valid_positions], outer_params, outer_maps)
        base_auc = roc_auc_score(y[valid_positions], baseline_prediction[valid_positions])

        for family, columns in families.items():
            fit_x = pd.concat([fit_base, fit_extra[columns]], axis=1)
            valid_x = pd.concat([valid_base, valid_extra[columns]], axis=1)
            fit_matrix, valid_matrix = shared.baseline.make_lightgbm_frames(fit_x, valid_x)
            model = lgb.train(
                {**model_params, "seed": shared.baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_matrix, label=y_fit),
                num_boost_round=best_rounds[family],
                callbacks=[lgb.log_evaluation(0)],
            )
            pred = model.predict(valid_matrix, num_iteration=best_rounds[family])
            predictions[family][valid_positions] = pred
            auc = roc_auc_score(y[valid_positions], pred)
            rows.append(
                {
                    "fold": int(fold),
                    "family": family,
                    "feature_count": len(columns),
                    "best_iteration_inner_only": best_rounds[family],
                    "fold_auc": auc,
                    "baseline_fold_auc": base_auc,
                    "fold_delta": auc - base_auc,
                    "fit_fee": outer_params["fee_estimate_fit_only"],
                    "fit_step": outer_params["step_estimate_fit_only"],
                }
            )

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(OUT / "PRODUCT_COMPONENT_FOLD_RESULTS.csv", index=False)
    oof = pd.DataFrame(
        {
            shared.ID: ids,
            shared.TARGET: y,
            "fold": folds,
            **{f"OOF_{name}": pred for name, pred in predictions.items()},
        }
    )
    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"Missing OOF values in {name}")
    oof.to_csv(OUT / "PRODUCT_COMPONENT_OOF_PREDICTIONS.csv", index=False)

    summary = []
    for family, pred in predictions.items():
        component = fold_results.loc[fold_results.family == family].sort_values("fold")
        deltas = component["fold_delta"].to_numpy()
        aucs = component["fold_auc"].to_numpy()
        summary.append(
            {
                "family": family,
                "feature_count": len(families[family]),
                "pooled_oof_auc": roc_auc_score(y, pred),
                "mean_fold_auc": float(np.mean(aucs)),
                "std_fold_auc": float(np.std(aucs, ddof=1)),
                "mean_paired_fold_delta": float(np.mean(deltas)),
                "std_paired_fold_delta": float(np.std(deltas, ddof=1)),
                "pooled_delta_vs_baseline": roc_auc_score(y, pred) - roc_auc_score(y, baseline_prediction),
                "folds_improved": int((deltas > 0).sum()),
                **{f"fold_{i+1}_delta": float(v) for i, v in enumerate(deltas)},
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "PRODUCT_COMPONENT_ABLATION_RESULTS.csv", index=False)

    env = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "lightgbm": lgb.__version__,
        "source_sha256": sha256(Path(__file__)),
        "shared_feature_runner_sha256": sha256(SHARED_PATH),
        "train_sha256": sha256(shared.TRAIN_PATH),
        "outer_folds": 5,
        "inner_stop_fraction": 0.1,
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    report = f"""# Product decoder component ablation

Run: {env['started_utc']} to {env['finished_utc']} UTC; {env['runtime_seconds']:.1f} seconds.

The same fixed outer folds, preserved baseline builder/model configuration, fold-only fee/step estimation, inner-only early stopping, and untouched outer predictions were used for each feature family.

{summary_df.to_string(index=False)}

This identifies whether the Batch 1 product-decoder family derives signal from Q rounded annuity/credit ratios, P fee-adjusted ratios, or the K financing lattice. These remain discovery results on reused folds; no component is promoted without a frozen confirmation.
"""
    (OUT / "PRODUCT_COMPONENT_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    log = [
        f"started_utc={env['started_utc']}",
        f"finished_utc={env['finished_utc']}",
        f"runtime_seconds={env['runtime_seconds']:.3f}",
        f"family_count={len(families)}",
        f"source_sha256={env['source_sha256']}",
        "status=SUCCESS",
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
