from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
BASELINE_DIR = ROOT / "experiments" / "phase2_advanced_feature_engineering"
OUT = ROOT / "experiments" / "phase3b_deep_validation"
SHARED_PATH = PHASE3 / "run_phase3_hidden_pattern_discovery.py"
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
RANDOM_SEED = 20261103

spec = importlib.util.spec_from_file_location("phase3b_shared_ablation", SHARED_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load shared Phase 3 feature code: {SHARED_PATH}")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

baseline_spec = importlib.util.spec_from_file_location("phase3b_baseline", BASELINE_SOURCE)
if baseline_spec is None or baseline_spec.loader is None:
    raise ImportError(f"Unable to load preserved baseline: {BASELINE_SOURCE}")
baseline = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline)

K_COLS = [
    "PROD_K_DECIMAL",
    "PROD_K_NEAREST_INTEGER",
    "PROD_K_LATTICE_RESIDUAL",
    "PROD_K_VALID",
    "PROD_K_FIT_FREQUENCY_LOG",
]
Q_COLS = [
    "PROD_Q3",
    "PROD_Q4",
    "PROD_Q5",
    "PROD_Q5_FIT_FREQUENCY_LOG",
]
P_COLS = [
    "PROD_P3",
    "PROD_P4",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_ADJUSTED_ANNUITY",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_P5_FIT_FREQUENCY_LOG",
]
ALL_PRODUCT = K_COLS + Q_COLS + P_COLS
NO_REDUNDANT = [
    "PROD_K_NEAREST_INTEGER",
    "PROD_K_LATTICE_RESIDUAL",
    "PROD_K_VALID",
    "PROD_K_FIT_FREQUENCY_LOG",
    "PROD_Q5",
    "PROD_Q5_FIT_FREQUENCY_LOG",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_P5_FIT_FREQUENCY_LOG",
]
KEY_COLUMNS = ["PROD_Q3", "PROD_Q4", "PROD_Q5", "PROD_P3", "PROD_P4", "PROD_P5", "PROD_K_NEAREST_INTEGER"]
FREQUENCY_COLUMN = {
    "PROD_Q3": None,
    "PROD_Q4": None,
    "PROD_Q5": "PROD_Q5_FIT_FREQUENCY_LOG",
    "PROD_P3": None,
    "PROD_P4": None,
    "PROD_P5": "PROD_P5_FIT_FREQUENCY_LOG",
    "PROD_K_NEAREST_INTEGER": "PROD_K_FIT_FREQUENCY_LOG",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(raw: pd.DataFrame, name: str) -> np.ndarray:
    if name not in raw:
        return np.full(len(raw), np.nan, dtype=np.float64)
    return raw[name].to_numpy(dtype=np.float64, na_value=np.nan)


def base_predictions_and_folds(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    saved = pd.read_csv(BASELINE_DIR / "OOF_BASELINE_PREDICTIONS.csv").set_index(ID)
    folds = pd.read_csv(BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(ID)["fold"]
    ids = train[ID].to_numpy()
    if saved.index.duplicated().any() or folds.index.duplicated().any():
        raise ValueError("Duplicate IDs found in saved OOF or fold assignment.")
    if set(saved.index) != set(ids) or set(folds.index) != set(ids):
        raise ValueError("Saved baseline OOF/folds do not exactly cover train IDs.")
    ordered = saved.loc[ids]
    if not np.array_equal(ordered[TARGET].to_numpy(), train[TARGET].to_numpy()):
        raise ValueError("Baseline OOF labels do not match training TARGET.")
    pred = ordered["OOF_RAW_BASELINE"].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError("Baseline OOF predictions contain missing/non-finite values.")
    return folds.loc[ids].to_numpy(), pred


def new_formula_features(raw: pd.DataFrame) -> pd.DataFrame:
    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = np.full(len(raw), np.nan, dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
        out[mask] = a[mask] / b[mask]
        return out

    credit, goods, annuity, income = (
        finite(raw, column)
        for column in ("AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL")
    )
    cols: dict[str, np.ndarray] = {
        "NEW_CREDIT_GOODS_GAP_TO_INCOME": ratio(credit - goods, income),
        "NEW_GOODS_TO_INCOME": ratio(goods, income),
        "NEW_ANNUITY_TO_GOODS": ratio(annuity, goods),
        "NEW_CREDIT_GOODS_LOG_RATIO": np.log1p(np.maximum(credit, 0)) - np.log1p(np.maximum(goods, 0)),
    }
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)


def make_family(
    raw_fit: pd.DataFrame,
    raw_valid: pd.DataFrame,
    feature_params: dict[str, float] | None = None,
    family_name: str = "all_product",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if family_name == "new_formulas":
        fit_new, valid_new = new_formula_features(raw_fit), new_formula_features(raw_valid)
        return fit_new, valid_new, list(fit_new.columns)
    params = shared.fit_fee_and_step(raw_fit) if feature_params is None else dict(feature_params)
    maps = shared.fit_product_maps(raw_fit, params)
    fit_features = shared.make_product_features(raw_fit, params, maps)
    valid_features = shared.make_product_features(raw_valid, params, maps)
    families: dict[str, list[str]] = {
        "K": K_COLS,
        "Q": Q_COLS,
        "P": P_COLS,
        "K+Q": K_COLS + Q_COLS,
        "K+P": K_COLS + P_COLS,
        "Q+P": Q_COLS + P_COLS,
        "all_product": ALL_PRODUCT,
        "no_redundant": NO_REDUNDANT,
        "stable_QK": K_COLS + Q_COLS,
        "rare_drop": ALL_PRODUCT,
        "rare_group": ALL_PRODUCT,
        "random_permuted": ALL_PRODUCT,
        "random_keys": ALL_PRODUCT,
        "random_fee_P": P_COLS,
        "random_step_K": K_COLS,
        "q_floor": Q_COLS,
        "annuity_perturbed": ALL_PRODUCT,
    }
    columns = families[family_name]
    if family_name == "random_fee_P":
        params["fee_estimate_fit_only"] = 177.1
        maps = shared.fit_product_maps(raw_fit, params)
        fit_features = shared.make_product_features(raw_fit, params, maps)
        valid_features = shared.make_product_features(raw_valid, params, maps)
    elif family_name == "random_step_K":
        params["step_estimate_fit_only"] = 0.0097
        maps = shared.fit_product_maps(raw_fit, params)
        fit_features = shared.make_product_features(raw_fit, params, maps)
        valid_features = shared.make_product_features(raw_valid, params, maps)
    elif family_name == "q_floor":
        for name, decimals in (("PROD_Q3", 3), ("PROD_Q4", 4), ("PROD_Q5", 5)):
            scale = 10**decimals
            fit_features[name] = np.floor(fit_features[name] * scale) / scale
            valid_features[name] = np.floor(valid_features[name] * scale) / scale
        fit_cash = raw_fit["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        q_counts = fit_features.loc[fit_cash, "PROD_Q5"].value_counts()
        fit_features["PROD_Q5_FIT_FREQUENCY_LOG"] = np.log1p(
            fit_features["PROD_Q5"].map(q_counts).fillna(0)
        )
        valid_features["PROD_Q5_FIT_FREQUENCY_LOG"] = np.log1p(
            valid_features["PROD_Q5"].map(q_counts).fillna(0)
        )
    elif family_name in ("rare_drop", "rare_group"):
        sentinel = np.nan if family_name == "rare_drop" else -1.0
        fit_cash = raw_fit["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        for key_name in KEY_COLUMNS:
            keys = fit_features.loc[fit_cash, key_name]
            counts = keys.value_counts()
            rare = set(counts.index[counts < 25])
            rare_count = int(counts.reindex(list(rare)).fillna(0).sum())
            fit_original = fit_features[key_name].copy()
            valid_original = valid_features[key_name].copy()
            fit_rare = fit_features[key_name].isin(rare)
            valid_rare = valid_features[key_name].isin(rare) | (
                valid_features[key_name].notna()
                & ~valid_features[key_name].isin(counts.index)
            )
            fit_features.loc[fit_rare, key_name] = sentinel
            valid_features.loc[valid_rare, key_name] = sentinel
            freq_name = FREQUENCY_COLUMN.get(key_name)
            if freq_name:
                fit_counts = fit_original.map(counts).fillna(0)
                valid_counts = valid_original.map(counts).fillna(0)
                if family_name == "rare_group":
                    fit_counts.loc[fit_rare] = rare_count
                    valid_counts.loc[valid_rare] = rare_count
                fit_features[freq_name] = np.log1p(fit_counts)
                valid_features[freq_name] = np.log1p(valid_counts)
    elif family_name == "random_permuted":
        rng = np.random.default_rng(RANDOM_SEED + len(raw_fit))
        for name in columns:
            fit_values = fit_features[name].to_numpy(copy=True)
            val_values = valid_features[name].to_numpy(copy=True)
            fit_features[name] = fit_values[rng.permutation(len(fit_values))]
            valid_features[name] = val_values[rng.permutation(len(val_values))]
    elif family_name == "random_keys":
        rng = np.random.default_rng(RANDOM_SEED + 99 + len(raw_fit))
        for name in columns:
            cardinality = max(2, int(fit_features[name].nunique(dropna=True)))
            fit_features[name] = rng.integers(0, cardinality, size=len(raw_fit)).astype(np.float64)
            valid_features[name] = rng.integers(0, cardinality, size=len(raw_valid)).astype(np.float64)
    elif family_name == "annuity_perturbed":
        rng = np.random.default_rng(RANDOM_SEED + 117 + len(raw_fit))
        fit_perturbed, valid_perturbed = raw_fit.copy(), raw_valid.copy()
        fit_perturbed["AMT_ANNUITY"] = finite(raw_fit, "AMT_ANNUITY") + rng.normal(0, 0.1, size=len(raw_fit))
        valid_perturbed["AMT_ANNUITY"] = finite(raw_valid, "AMT_ANNUITY") + rng.normal(0, 0.1, size=len(raw_valid))
        perturbed_maps = shared.fit_product_maps(fit_perturbed, params)
        fit_features = shared.make_product_features(fit_perturbed, params, perturbed_maps)
        valid_features = shared.make_product_features(valid_perturbed, params, perturbed_maps)

    return fit_features.loc[:, columns].copy(), valid_features.loc[:, columns].copy(), columns


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    started = time.time()
    started_utc = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    folds, baseline_oof = base_predictions_and_folds(train)
    if not (OUT / "CURRENT_SCORE_REPRODUCTION.csv").exists():
        raise FileNotFoundError("Run phase3b_reproduce.py successfully before ablations.")

    variants = [
        "K",
        "Q",
        "P",
        "K+Q",
        "K+P",
        "Q+P",
        "all_product",
        "no_redundant",
        "stable_QK",
        "rare_drop",
        "rare_group",
        "random_permuted",
        "random_keys",
        "random_fee_P",
        "random_step_K",
        "q_floor",
        "annuity_perturbed",
        "new_formulas",
    ]
    predictions = {name: np.full(len(train), np.nan, dtype=np.float64) for name in variants}
    baseline_refit = np.full(len(train), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    best_round_by_fold: dict[int, int] = {}
    config = baseline.MODEL_CONFIG["lightgbm"]
    model_params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }

    for fold in sorted(np.unique(folds)):
        valid_mask = folds == fold
        fit_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=20261102 + int(fold),
        )
        inner_fit_local, inner_stop_local = next(inner.split(np.zeros(len(fit_idx)), y[fit_idx]))
        inner_fit_idx, inner_stop_idx = fit_idx[inner_fit_local], fit_idx[inner_stop_local]
        inner_fit_x = baseline.build_rowwise_features(raw.iloc[inner_fit_idx])
        inner_stop_x = baseline.build_rowwise_features(raw.iloc[inner_stop_idx])
        inner_fit_matrix, inner_stop_matrix = baseline.make_lightgbm_frames(inner_fit_x, inner_stop_x)
        selector = lgb.train(
            {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
            lgb.Dataset(inner_fit_matrix, label=y[inner_fit_idx]),
            num_boost_round=config["num_boost_round"],
            valid_sets=[lgb.Dataset(inner_stop_matrix, label=y[inner_stop_idx])],
            callbacks=[
                lgb.early_stopping(config["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        rounds = int(selector.best_iteration or config["num_boost_round"])
        best_round_by_fold[int(fold)] = rounds
        fit_base = baseline.build_rowwise_features(raw.iloc[fit_idx])
        valid_base = baseline.build_rowwise_features(raw.iloc[valid_idx])
        base_auc_fold = roc_auc_score(y[valid_idx], baseline_oof[valid_idx])
        outer_product_params = shared.fit_fee_and_step(raw.iloc[fit_idx])
        fit_base_matrix, valid_base_matrix = baseline.make_lightgbm_frames(
            fit_base, valid_base
        )
        base_model = lgb.train(
            {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
            lgb.Dataset(fit_base_matrix, label=y[fit_idx]),
            num_boost_round=rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        base_prediction = base_model.predict(valid_base_matrix, num_iteration=rounds)
        if not np.isfinite(base_prediction).all():
            raise ValueError(f"Non-finite in-run baseline predictions for fold {fold}")
        baseline_refit[valid_idx] = base_prediction
        base_refit_auc_fold = roc_auc_score(y[valid_idx], base_prediction)
        fold_rows.append(
            {
                "fold": int(fold),
                "variant": "raw_baseline_refit",
                "features_added": 0,
                "boosting_rounds_from_inner_raw_baseline_only": rounds,
                "fold_auc": base_refit_auc_fold,
                "baseline_saved_fold_auc": base_auc_fold,
                "fold_delta_vs_saved_baseline": base_refit_auc_fold - base_auc_fold,
                "baseline_refit_fold_auc": base_refit_auc_fold,
                "fold_delta_vs_refit_baseline": 0.0,
                "fee_fit_only": np.nan,
                "step_fit_only": np.nan,
                "outer_validation_used_for_early_stopping": False,
            }
        )

        for variant in variants:
            fit_extra, valid_extra, columns = make_family(
                raw.iloc[fit_idx],
                raw.iloc[valid_idx],
                feature_params=outer_product_params,
                family_name=variant,
            )
            fit_x = pd.concat([fit_base, fit_extra.reset_index(drop=True)], axis=1)
            valid_x = pd.concat([valid_base, valid_extra.reset_index(drop=True)], axis=1)
            fit_matrix, valid_matrix = baseline.make_lightgbm_frames(fit_x, valid_x)
            model = lgb.train(
                {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_matrix, label=y[fit_idx]),
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            prediction = model.predict(valid_matrix, num_iteration=rounds)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite predictions for {variant}, fold {fold}")
            predictions[variant][valid_idx] = prediction
            auc = roc_auc_score(y[valid_idx], prediction)
            fold_rows.append(
                {
                    "fold": int(fold),
                    "variant": variant,
                    "features_added": len(columns),
                    "boosting_rounds_from_inner_raw_baseline_only": rounds,
                    "fold_auc": auc,
                    "baseline_saved_fold_auc": base_auc_fold,
                    "fold_delta_vs_saved_baseline": auc - base_auc_fold,
                    "baseline_refit_fold_auc": base_refit_auc_fold,
                    "fold_delta_vs_refit_baseline": auc - base_refit_auc_fold,
                    "fee_fit_only": (
                        177.1 if variant == "random_fee_P" else np.nan
                    ),
                    "step_fit_only": (
                        0.0097 if variant == "random_step_K" else np.nan
                    ),
                    "outer_validation_used_for_early_stopping": False,
                }
            )
            gains = model.feature_importance(importance_type="gain")
            model_names = model.feature_name()
            name_to_gain = dict(zip(model_names, gains))
            for name in columns:
                importance_rows.append(
                    {
                        "fold": int(fold),
                        "variant": variant,
                        "feature_name": name,
                        "gain_importance": float(name_to_gain.get(name, 0.0)),
                        "gain_share": float(name_to_gain.get(name, 0.0) / max(1.0, float(np.sum(gains)))),
                    }
                )
            print(f"fold={fold} variant={variant} auc={auc:.6f} delta={auc-base_auc_fold:+.6f} rounds={rounds}", flush=True)

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(OUT / "FOLD_LEVEL_RESULTS.csv", index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(OUT / "PRODUCT_FEATURE_IMPORTANCE.csv", index=False)
    if not np.isfinite(baseline_refit).all():
        raise ValueError("Incomplete in-run baseline OOF predictions.")
    oof = pd.DataFrame(
        {
            ID: ids,
            TARGET: y,
            "fold": folds,
            "OOF_BASELINE_REFERENCE": baseline_oof,
            "OOF_BASELINE_REFIT_FIXED_ROUNDS": baseline_refit,
        }
    )
    for variant, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Incomplete OOF predictions for {variant}")
        oof[f"OOF_{variant}"] = values
    oof.to_csv(OUT / "PHASE3B_ABLATION_OOF.csv", index=False)

    summary = []
    baseline_fold_auc = {
        int(f): float(roc_auc_score(y[folds == f], baseline_oof[folds == f]))
        for f in sorted(np.unique(folds))
    }
    base_pooled = roc_auc_score(y, baseline_oof)
    refit_pooled = roc_auc_score(y, baseline_refit)
    summary.append(
        {
            "variant": "saved_baseline_reference",
            "feature_count": 0,
            "pooled_oof_auc": base_pooled,
            "delta_vs_baseline": 0.0,
            "delta_vs_saved_baseline": 0.0,
            "mean_fold_auc": float(np.mean(list(baseline_fold_auc.values()))),
            "std_fold_auc": float(np.std(list(baseline_fold_auc.values()), ddof=1)),
            "folds_improved": 0,
            "fold_deltas": json.dumps({int(f): 0.0 for f in baseline_fold_auc}, sort_keys=True),
        }
    )
    refit_rows = fold_results.loc[fold_results.variant == "raw_baseline_refit"].sort_values("fold")
    refit_fold_auc = refit_rows["fold_auc"].to_numpy()
    refit_deltas = refit_rows["fold_delta_vs_saved_baseline"].to_numpy()
    summary.append(
        {
            "variant": "raw_baseline_refit",
            "feature_count": 0,
            "pooled_oof_auc": refit_pooled,
            "delta_vs_baseline": 0.0,
            "delta_vs_saved_baseline": refit_pooled - base_pooled,
            "mean_fold_auc": float(np.mean(refit_fold_auc)),
            "std_fold_auc": float(np.std(refit_fold_auc, ddof=1)),
            "mean_paired_fold_delta": float(np.mean(refit_deltas)),
            "std_paired_fold_delta": float(np.std(refit_deltas, ddof=1)),
            "folds_improved": int((refit_deltas > 0).sum()),
            "fold_deltas": json.dumps(
                {int(f): float(d) for f, d in zip(refit_rows["fold"], refit_deltas)},
                sort_keys=True,
            ),
            "interpretation": "In-run raw-only comparator; same fixed rounds as feature variants.",
        }
    )
    for variant in variants:
        pred = predictions[variant]
        df = fold_results.loc[fold_results["variant"] == variant].sort_values("fold")
        fold_aucs = df["fold_auc"].to_numpy()
        deltas = df["fold_delta_vs_refit_baseline"].to_numpy()
        summary.append(
            {
                "variant": variant,
                "feature_count": int(df["features_added"].iloc[0]),
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "delta_vs_baseline": float(roc_auc_score(y, pred) - refit_pooled),
                "delta_vs_saved_baseline": float(roc_auc_score(y, pred) - base_pooled),
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
                "mean_paired_fold_delta": float(np.mean(deltas)),
                "std_paired_fold_delta": float(np.std(deltas, ddof=1)),
                "folds_improved": int((deltas > 0).sum()),
                "folds_improved_vs_refit": int((deltas > 0).sum()),
                "fold_deltas": json.dumps(
                    {int(f): float(d) for f, d in zip(df["fold"], deltas)},
                    sort_keys=True,
                ),
                "interpretation": (
                    "negative control"
                    if variant in {"random_permuted", "random_keys", "random_fee_P", "random_step_K"}
                    else "reused-fold discovery result; not independent confirmation"
                ),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "PRODUCT_COMPONENT_ABLATIONS.csv", index=False)
    summary_frame.loc[summary_frame.variant.isin(["random_permuted", "random_keys", "random_fee_P", "random_step_K"])].to_csv(
        OUT / "NEGATIVE_CONTROL_MODEL_RESULTS.csv", index=False
    )

    # Update diagnostics with average gain importance from the full product-only model.
    diag_path = OUT / "PRODUCT_FEATURE_DIAGNOSTICS.csv"
    if diag_path.exists():
        diagnostics = pd.read_csv(diag_path)
        model_importance = (
            importance.loc[importance.variant == "all_product"]
            .groupby("feature_name", as_index=False)
            .agg(mean_gain_importance=("gain_importance", "mean"), mean_gain_share=("gain_share", "mean"))
        )
        diagnostics = diagnostics.drop(
            columns=["fit_fold_feature_importance"], errors="ignore"
        ).merge(model_importance, how="left", on="feature_name")
        diagnostics["incremental_cv_contribution"] = "See PRODUCT_COMPONENT_ABLATIONS.csv; individual contribution not isolated."
        diagnostics.to_csv(diag_path, index=False)

    (OUT / "FOLD_SPECIFIC_BOOSTING_ROUNDS.json").write_text(json.dumps(best_round_by_fold, indent=2) + "\n", encoding="utf-8")
    finished = datetime.now(timezone.utc)
    log_lines = [
        f"started_utc={started_utc.isoformat()}",
        f"finished_utc={finished.isoformat()}",
        f"runtime_seconds={time.time()-started:.3f}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"train_sha256={sha256(DATA/'train.csv')}",
        f"baseline_source_sha256={sha256(BASELINE_SOURCE)}",
        f"shared_feature_runner_sha256={sha256(SHARED_PATH)}",
        f"ablation_source_sha256={sha256(Path(__file__).resolve())}",
        f"outer_folds={len(np.unique(folds))}",
        f"raw_baseline_refit_pooled_oof_auc={refit_pooled}",
        f"saved_baseline_pooled_oof_auc={base_pooled}",
        f"variant_count={len(variants)}",
        f"fit_count={len(fold_rows)}",
        "status=SUCCESS",
        "outer_validation_used_for_early_stopping=False",
        "same_inner_raw_baseline_boosting_rounds_for_all_variants=True",
        "target_encoding=False",
        "pooled_train_test_statistics=False",
        "",
        summary_frame[["variant", "pooled_oof_auc", "delta_vs_baseline", "mean_fold_auc", "std_fold_auc", "folds_improved"]].to_string(index=False),
    ]
    with (OUT / "EXECUTION_LOG.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n" + "\n".join(log_lines) + "\n")
    (OUT / "PHASE3B_ABLATION_LOG.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines), flush=True)


if __name__ == "__main__":
    main()
