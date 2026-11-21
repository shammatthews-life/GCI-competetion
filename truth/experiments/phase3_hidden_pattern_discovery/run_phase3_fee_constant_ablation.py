#!/usr/bin/env python3
"""Compare fold-estimated versus v17's fixed fee inside fee-adjusted product keys."""

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
OUT = HERE / "batch1_fee_constant_04"
OUT.mkdir(parents=True, exist_ok=True)
SHARED_PATH = HERE / "run_phase3_hidden_pattern_discovery.py"
spec = importlib.util.spec_from_file_location("phase3_shared_fee", SHARED_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared Phase 3 feature code: {SHARED_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

P_FEATURES = [
    "PROD_P3",
    "PROD_P4",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_ADJUSTED_ANNUITY",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_P5_FIT_FREQUENCY_LOG",
]
Q_FEATURES = [
    "PROD_Q3",
    "PROD_Q4",
    "PROD_Q5",
    "PROD_Q5_FIT_FREQUENCY_LOG",
]


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
    folds = (
        pd.read_csv(shared.BASELINE_DIR / "FOLD_ASSIGNMENTS.csv")
        .set_index(shared.ID)
        .loc[ids, "fold"]
        .to_numpy()
    )
    saved = pd.read_csv(shared.BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv").set_index(shared.ID)
    baseline_prediction = saved.loc[ids, "OOF_RAW_BASELINE"].to_numpy()
    if not np.array_equal(saved.loc[ids, shared.TARGET].to_numpy(), y):
        raise ValueError("Saved baseline target mismatch.")

    families = {
        "p_keys_fold_estimated_fee": P_FEATURES,
        "p_keys_fixed_v17_fee_129_5": P_FEATURES,
        "q_keys_unadjusted_control": Q_FEATURES,
    }
    predictions = {
        name: np.full(len(train), np.nan, dtype=np.float64) for name in families
    }
    fold_rows = []
    config = shared.baseline.MODEL_CONFIG["lightgbm"]
    model_params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }

    for fold in sorted(np.unique(folds)):
        valid_mask = folds == fold
        fit_positions = np.flatnonzero(~valid_mask)
        valid_positions = np.flatnonzero(valid_mask)
        y_fit = y[fit_positions]
        split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=20261102 + int(fold),
        )
        inner_fit_local, inner_stop_local = next(split.split(np.zeros(len(y_fit)), y_fit))
        inner_fit = fit_positions[inner_fit_local]
        inner_stop = fit_positions[inner_stop_local]
        inner_est = shared.fit_fee_and_step(raw.iloc[inner_fit])
        inner_fixed = {**inner_est, "fee_estimate_fit_only": 129.5}
        inner_params_by_family = {
            "p_keys_fold_estimated_fee": inner_est,
            "p_keys_fixed_v17_fee_129_5": inner_fixed,
            "q_keys_unadjusted_control": inner_est,
        }
        inner_base_fit = shared.baseline.build_rowwise_features(raw.iloc[inner_fit])
        inner_base_stop = shared.baseline.build_rowwise_features(raw.iloc[inner_stop])
        best_rounds = {}
        for family, columns in families.items():
            inner_params = inner_params_by_family[family]
            maps = shared.fit_product_maps(raw.iloc[inner_fit], inner_params)
            extra_fit = shared.make_product_features(raw.iloc[inner_fit], inner_params, maps)
            extra_stop = shared.make_product_features(raw.iloc[inner_stop], inner_params, maps)
            fit_x = pd.concat([inner_base_fit, extra_fit[columns]], axis=1)
            stop_x = pd.concat([inner_base_stop, extra_stop[columns]], axis=1)
            fit_matrix, stop_matrix = shared.baseline.make_lightgbm_frames(fit_x, stop_x)
            fit_set = lgb.Dataset(fit_matrix, label=y[inner_fit])
            stop_set = lgb.Dataset(stop_matrix, label=y[inner_stop], reference=fit_set)
            selector = lgb.train(
                {**model_params, "seed": shared.baseline.FOLD_SEED + int(fold)},
                fit_set,
                num_boost_round=config["num_boost_round"],
                valid_sets=[stop_set],
                callbacks=[
                    lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_rounds[family] = int(selector.best_iteration or config["num_boost_round"])

        outer_est = shared.fit_fee_and_step(raw.iloc[fit_positions])
        outer_params_by_family = {
            "p_keys_fold_estimated_fee": outer_est,
            "p_keys_fixed_v17_fee_129_5": {**outer_est, "fee_estimate_fit_only": 129.5},
            "q_keys_unadjusted_control": outer_est,
        }
        fit_base = shared.baseline.build_rowwise_features(raw.iloc[fit_positions])
        valid_base = shared.baseline.build_rowwise_features(raw.iloc[valid_positions])
        base_auc = roc_auc_score(y[valid_positions], baseline_prediction[valid_positions])
        for family, columns in families.items():
            feature_params = outer_params_by_family[family]
            maps = shared.fit_product_maps(raw.iloc[fit_positions], feature_params)
            extra_fit = shared.make_product_features(raw.iloc[fit_positions], feature_params, maps)
            extra_valid = shared.make_product_features(raw.iloc[valid_positions], feature_params, maps)
            fit_x = pd.concat([fit_base, extra_fit[columns]], axis=1)
            valid_x = pd.concat([valid_base, extra_valid[columns]], axis=1)
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
            fold_rows.append(
                {
                    "fold": int(fold),
                    "family": family,
                    "fee_value_used": (
                        129.5 if family == "p_keys_fixed_v17_fee_129_5"
                        else outer_est["fee_estimate_fit_only"] if family == "p_keys_fold_estimated_fee"
                        else np.nan
                    ),
                    "step_value_used": outer_est["step_estimate_fit_only"],
                    "best_iteration_inner_only": best_rounds[family],
                    "fold_auc": auc,
                    "baseline_fold_auc": base_auc,
                    "fold_delta": auc - base_auc,
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(OUT / "FEE_CONSTANT_FOLD_RESULTS.csv", index=False)
    oof = pd.DataFrame(
        {
            shared.ID: ids,
            shared.TARGET: y,
            "fold": folds,
            **{f"OOF_{name}": pred for name, pred in predictions.items()},
        }
    )
    for family, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"Incomplete predictions for {family}")
    oof.to_csv(OUT / "FEE_CONSTANT_OOF_PREDICTIONS.csv", index=False)

    summary = []
    for family, pred in predictions.items():
        df = fold_results.loc[fold_results.family == family].sort_values("fold")
        deltas = df["fold_delta"].to_numpy()
        aucs = df["fold_auc"].to_numpy()
        summary.append(
            {
                "family": family,
                "pooled_oof_auc": roc_auc_score(y, pred),
                "mean_fold_auc": float(np.mean(aucs)),
                "std_fold_auc": float(np.std(aucs, ddof=1)),
                "mean_paired_fold_delta": float(np.mean(deltas)),
                "std_paired_fold_delta": float(np.std(deltas, ddof=1)),
                "pooled_delta_vs_baseline": roc_auc_score(y, pred) - roc_auc_score(y, baseline_prediction),
                "folds_improved": int((deltas > 0).sum()),
                **{f"fold_{i+1}_delta": float(value) for i, value in enumerate(deltas)},
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "FEE_CONSTANT_ABLATION_RESULTS.csv", index=False)
    env = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "source_sha256": sha256(Path(__file__)),
        "shared_feature_runner_sha256": sha256(SHARED_PATH),
        "train_sha256": sha256(shared.TRAIN_PATH),
        "outer_folds": 5,
        "inner_stop_fraction": 0.1,
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (OUT / "FEE_CONSTANT_ABLATION_REPORT.md").write_text(
        f"""# Fixed versus fold-estimated fee ablation

Run: {env['started_utc']} to {env['finished_utc']} UTC; {env['runtime_seconds']:.1f} seconds.

The same seven P-family features were evaluated with a fee selected on the current outer training partition, with the notebook's fixed 129.5, and with unadjusted Q keys as a control. Each variant used its own inner-only early-stopping choice, the preserved LightGBM configuration, and the same outer folds.

{summary_df.to_string(index=False)}

The 129.5 candidate is not treated as an established contract fee. This is a fixed-hypothesis comparison, not independent confirmation; see the fold estimates and prior key-repeat diagnostics.
""",
        encoding="utf-8",
    )
    print(f"runtime_seconds={env['runtime_seconds']:.3f}")
    print(f"source_sha256={env['source_sha256']}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
