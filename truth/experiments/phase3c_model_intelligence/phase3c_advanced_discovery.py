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
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import RobustScaler


OUT = Path(r"D:\gci\truth\experiments\phase3c_model_intelligence")
ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE3B = ROOT / "experiments" / "phase3b_deep_validation"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261104

spec = importlib.util.spec_from_file_location(
    "phase3c_advanced_base", OUT / "phase3c_initial_benchmark.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load initial benchmark and fold-safe feature builders.")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
baseline = benchmark.baseline
phase3b = benchmark.phase3b
shared = benchmark.shared

BIN_SOURCES = {
    "EXT_SOURCE_1": lambda d: get(d, "EXT_SOURCE_1"),
    "EXT_SOURCE_2": lambda d: get(d, "EXT_SOURCE_2"),
    "EXT_SOURCE_3": lambda d: get(d, "EXT_SOURCE_3"),
    "AMT_CREDIT": lambda d: get(d, "AMT_CREDIT"),
    "AMT_ANNUITY": lambda d: get(d, "AMT_ANNUITY"),
    "AMT_INCOME_TOTAL": lambda d: get(d, "AMT_INCOME_TOTAL"),
    "AMT_GOODS_PRICE": lambda d: get(d, "AMT_GOODS_PRICE"),
    "AGE_YEARS": lambda d: -get(d, "DAYS_BIRTH") / 365.25,
    "EMPLOYED_YEARS": lambda d: np.where(
        get(d, "DAYS_EMPLOYED") == 365243.0,
        np.nan,
        -get(d, "DAYS_EMPLOYED") / 365.25,
    ),
}
BIN_COLUMNS = [f"NONLINEAR_{name}_QBIN5" for name in BIN_SOURCES]
INTERACTIONS = {
    "INTER_EXT_MEAN_X_CREDIT_TO_INCOME": (
        "mean(observed EXT_SOURCE_1..3) * AMT_CREDIT / AMT_INCOME_TOTAL",
        "Tests whether credit burden changes the ordering signal of external scores.",
    ),
    "INTER_EXT_MEAN_X_ANNUITY_TO_INCOME": (
        "mean(observed EXT_SOURCE_1..3) * AMT_ANNUITY / AMT_INCOME_TOTAL",
        "Tests external-risk signal conditional on repayment burden.",
    ),
    "INTER_EXT_RANGE_X_CREDIT_TO_GOODS": (
        "range(observed EXT_SOURCE_1..3) * AMT_CREDIT / AMT_GOODS_PRICE",
        "Tests score disagreement jointly with financing intensity.",
    ),
    "INTER_AGE_X_EMPLOYMENT_YEARS": (
        "(-DAYS_BIRTH/365.25) * cleaned(-DAYS_EMPLOYED/365.25)",
        "Tests age/employment-history interaction; sentinel employment is missing.",
    ),
    "INTER_EXT_MEAN_X_PRODUCT_K": (
        "mean(observed EXT_SOURCE_1..3) * PROD_K_NEAREST_INTEGER",
        "Tests external-score ordering conditional on the fold-local product key.",
    ),
    "INTER_EXT_DISPERSION_X_PRODUCT_P": (
        "SD(observed EXT_SOURCE_1..3) * PROD_P5",
        "Tests score disagreement conditional on the fold-local P reparameterization.",
    ),
}
MISSINGNESS_GROUPS = {
    "MISS_EXTERNAL_SCORE_COUNT": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "MISS_FINANCIAL_CORE_COUNT": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    ],
    "MISS_BUREAU_COUNT": [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
    "MISS_EMPLOYMENT_DEMOGRAPHICS_COUNT": [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CNT_FAM_MEMBERS",
    ],
    "MISS_FINANCIAL_AND_EXTERNAL_COUNT": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ],
}


def get(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return frame[column].to_numpy(dtype=np.float64, na_value=np.nan)


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[valid] = a[valid] / b[valid]
    return out


def external_stats(raw: pd.DataFrame):
    values = np.column_stack([get(raw, f"EXT_SOURCE_{i}") for i in (1, 2, 3)])
    count = np.isfinite(values).sum(axis=1)
    mean = np.divide(
        np.nansum(values, axis=1),
        count,
        out=np.full(len(raw), np.nan),
        where=count > 0,
    )
    minimum = np.where(np.isfinite(values), values, np.inf).min(axis=1)
    maximum = np.where(np.isfinite(values), values, -np.inf).max(axis=1)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    centered = np.where(np.isfinite(values), values - mean[:, None], 0.0)
    sd = np.sqrt(
        np.divide(
            np.square(centered).sum(axis=1),
            count,
            out=np.full(len(raw), np.nan),
            where=count > 0,
        )
    )
    return mean, maximum - minimum, sd


def quantile_bins(
    fit_values: np.ndarray, valid_values: np.ndarray, bins: int = 5
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    finite_fit = fit_values[np.isfinite(fit_values)]
    if len(finite_fit) < bins * 2:
        return (
            np.full(len(fit_values), np.nan),
            np.full(len(valid_values), np.nan),
            [],
        )
    cuts = np.unique(np.quantile(finite_fit, np.linspace(0, 1, bins + 1)[1:-1]))

    def apply(values):
        out = np.full(len(values), np.nan)
        mask = np.isfinite(values)
        out[mask] = np.digitize(values[mask], cuts, right=True)
        return out

    return apply(fit_values), apply(valid_values), cuts.tolist()


def fit_cluster_fold(raw_fit, raw_valid, y_fit, fold, k):
    columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    def matrix(frame):
        values = {column: get(frame, column) for column in columns}
        values["AGE_YEARS"] = -get(frame, "DAYS_BIRTH") / 365.25
        employed = get(frame, "DAYS_EMPLOYED").copy()
        employed[employed == 365243.0] = np.nan
        values["EMPLOYED_YEARS"] = -employed / 365.25
        values["DAYS_EMPLOYED"][values["DAYS_EMPLOYED"] == 365243.0] = np.nan
        values["CREDIT_TO_INCOME"] = safe_ratio(
            get(frame, "AMT_CREDIT"), get(frame, "AMT_INCOME_TOTAL")
        )
        values["ANNUITY_TO_INCOME"] = safe_ratio(
            get(frame, "AMT_ANNUITY"), get(frame, "AMT_INCOME_TOTAL")
        )
        values["CREDIT_TO_GOODS"] = safe_ratio(
            get(frame, "AMT_CREDIT"), get(frame, "AMT_GOODS_PRICE")
        )
        return pd.DataFrame(values)

    fit_raw = matrix(raw_fit)
    valid_raw = matrix(raw_valid)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = RobustScaler(quantile_range=(10, 90))
    fit_scaled = scaler.fit_transform(imputer.fit_transform(fit_raw))
    valid_scaled = scaler.transform(imputer.transform(valid_raw))
    fit_scaled = np.clip(fit_scaled, -6.0, 6.0)
    valid_scaled = np.clip(valid_scaled, -6.0, 6.0)
    model = KMeans(
        n_clusters=k, n_init=10, random_state=SEED + int(fold)
    ).fit(fit_scaled)
    valid_cluster = model.predict(valid_scaled)
    alternate = KMeans(
        n_clusters=k, n_init=10, random_state=SEED + 100 + int(fold)
    ).fit(fit_scaled)
    alternate_valid = alternate.predict(valid_scaled)
    stability = adjusted_rand_score(valid_cluster, alternate_valid)
    fit_cluster = model.labels_
    fit_profile = fit_raw.copy()
    fit_profile["cluster"] = fit_cluster
    fit_profile[TARGET] = y_fit
    return {
        "fit_raw": fit_raw,
        "valid_cluster": valid_cluster,
        "fit_cluster": fit_cluster,
        "fit_profile": fit_profile,
        "stability_ari": float(stability),
        "cluster_centers": model.cluster_centers_,
    }


def fit_lgbm_variant(fit_frame, valid_frame, y_fit, rounds, fold):
    fit_x, valid_x = baseline.make_lightgbm_frames(fit_frame, valid_frame)
    config = baseline.MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    model = lgb.train(
        {**params, "seed": baseline.FOLD_SEED + int(fold)},
        lgb.Dataset(fit_x, label=y_fit),
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(0)],
    )
    return model.predict(valid_x, num_iteration=int(rounds))


def metric_summary(name, prediction, y, folds):
    fold_auc = []
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        auc = float(roc_auc_score(y[mask], prediction[mask]))
        fold_auc.append(auc)
        fold_rows.append(
            {
                "candidate": name,
                "fold": int(fold),
                "fold_auc": auc,
            }
        )
    return (
        {
            "candidate": name,
            "pooled_oof_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(np.mean(fold_auc)),
            "std_fold_auc": float(np.std(fold_auc, ddof=1)),
        },
        fold_rows,
    )


def main():
    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    folds = (
        pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv")
        .set_index(ID)
        .loc[ids, "fold"]
        .to_numpy(dtype=np.int8)
    )
    base_oof = pd.read_csv(OUT / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv")
    base_oof = base_oof.set_index(ID).loc[ids]
    reference_product = base_oof["lightgbm__raw_plus_product"].to_numpy()
    runtime = pd.read_csv(OUT / "MODEL_RUNTIME_RESULTS.csv")
    rounds_by_fold = (
        runtime.loc[
            (runtime.model == "lightgbm")
            & (runtime.stage == "inner_round_selection"),
            ["fold", "selected_rounds_or_trees"],
        ]
        .set_index("fold")["selected_rounds_or_trees"]
        .to_dict()
    )
    candidate_names = (
        [f"bin:{name}" for name in BIN_SOURCES]
        + list(INTERACTIONS)
        + list(MISSINGNESS_GROUPS)
        + ["segment_k4", "segment_k6"]
        + [
            "family:all_nonlinear_bins",
            "family:all_interactions",
            "family:all_missingness",
            "family:nonlinear_plus_interactions",
            "family:all_screened_plus_segment_k4",
        ]
    )
    predictions = {
        name: np.full(len(train), np.nan, dtype=np.float64)
        for name in candidate_names
    }
    fold_rows = []
    profile_rows = []
    stability_rows = []
    segment_performance_rows = []
    nonlinear_pattern_rows = []
    missingness_rows = []
    redundancy_values = {name: [] for name in candidate_names}
    fit_times = []

    for fold in sorted(np.unique(folds)):
        valid_mask = folds == fold
        fit_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
        raw_fit, raw_valid = raw.iloc[fit_idx], raw.iloc[valid_idx]
        product_params = shared.fit_fee_and_step(raw_fit)
        prod_fit, prod_valid, _ = phase3b.make_family(
            raw_fit, raw_valid, feature_params=product_params, family_name="all_product"
        )
        fit_base = baseline.build_rowwise_features(raw_fit)
        valid_base = baseline.build_rowwise_features(raw_valid)
        fit_product_base = pd.concat(
            [fit_base, prod_fit.reset_index(drop=True)], axis=1
        )
        valid_product_base = pd.concat(
            [valid_base, prod_valid.reset_index(drop=True)], axis=1
        )
        rounds = int(rounds_by_fold[int(fold)])

        # Fold-local quantile cutpoints; the validation labels do not affect them.
        bin_fit, bin_valid, cut_map = {}, {}, {}
        for name, builder in BIN_SOURCES.items():
            fit_values, valid_values = builder(raw_fit), builder(raw_valid)
            bf, bv, cuts = quantile_bins(fit_values, valid_values, bins=5)
            key = f"NONLINEAR_{name}_QBIN5"
            bin_fit[key], bin_valid[key], cut_map[key] = bf, bv, cuts
            for b in sorted(np.unique(bf[np.isfinite(bf)])):
                fit_bin_mask = np.isfinite(bf) & (bf == b)
                val_bin_mask = np.isfinite(bv) & (bv == b)
                nonlinear_pattern_rows.append(
                    {
                        "feature": name,
                        "fold": int(fold),
                        "bin": int(b),
                        "fit_cut_points": json.dumps(cuts),
                        "fit_n": int(fit_bin_mask.sum()),
                        "fit_default_rate": float(y[fit_idx][fit_bin_mask].mean()),
                        "validation_n": int(val_bin_mask.sum()),
                        "validation_default_rate": (
                            float(y[valid_idx][val_bin_mask].mean())
                            if val_bin_mask.any()
                            else np.nan
                        ),
                        "validation_mean_raw_value": (
                            float(valid_values[val_bin_mask].mean())
                            if val_bin_mask.any()
                            else np.nan
                        ),
                        "cutpoints_fit_on_outer_train_only": True,
                    }
                )

        ext_mean_fit, ext_range_fit, ext_sd_fit = external_stats(raw_fit)
        ext_mean_valid, ext_range_valid, ext_sd_valid = external_stats(raw_valid)

        def interaction_frame(source, prod):
            ext_mean, ext_range, ext_sd = external_stats(source)
            credit = get(source, "AMT_CREDIT")
            income = get(source, "AMT_INCOME_TOTAL")
            annuity = get(source, "AMT_ANNUITY")
            goods = get(source, "AMT_GOODS_PRICE")
            age = -get(source, "DAYS_BIRTH") / 365.25
            employed = get(source, "DAYS_EMPLOYED")
            employed[employed == 365243.0] = np.nan
            employed = -employed / 365.25
            return pd.DataFrame(
                {
                    "INTER_EXT_MEAN_X_CREDIT_TO_INCOME": ext_mean
                    * safe_ratio(credit, income),
                    "INTER_EXT_MEAN_X_ANNUITY_TO_INCOME": ext_mean
                    * safe_ratio(annuity, income),
                    "INTER_EXT_RANGE_X_CREDIT_TO_GOODS": ext_range
                    * safe_ratio(credit, goods),
                    "INTER_AGE_X_EMPLOYMENT_YEARS": age * employed,
                    "INTER_EXT_MEAN_X_PRODUCT_K": ext_mean
                    * prod["PROD_K_NEAREST_INTEGER"].to_numpy(dtype=np.float64),
                    "INTER_EXT_DISPERSION_X_PRODUCT_P": ext_sd
                    * prod["PROD_P5"].to_numpy(dtype=np.float64),
                }
            ).replace([np.inf, -np.inf], np.nan)

        inter_fit = interaction_frame(raw_fit, prod_fit)
        inter_valid = interaction_frame(raw_valid, prod_valid)

        miss_fit, miss_valid = {}, {}
        for name, columns in MISSINGNESS_GROUPS.items():
            existing = [column for column in columns if column in raw_fit]
            miss_fit[name] = raw_fit[existing].isna().sum(axis=1).to_numpy(dtype=float)
            miss_valid[name] = raw_valid[existing].isna().sum(axis=1).to_numpy(dtype=float)
            for count in sorted(np.unique(miss_fit[name])):
                train_mask = miss_fit[name] == count
                valid_group = miss_valid[name] == count
                missingness_rows.append(
                    {
                        "feature_group": name,
                        "missing_count": int(count),
                        "fold": int(fold),
                        "train_n": int(train_mask.sum()),
                        "train_default_rate": float(y[fit_idx][train_mask].mean()),
                        "validation_n": int(valid_group.sum()),
                        "validation_default_rate": (
                            float(y[valid_idx][valid_group].mean())
                            if valid_group.any()
                            else np.nan
                        ),
                        "train_missing_fraction_mean": float(
                            raw_fit[existing].isna().to_numpy().mean()
                        ),
                    }
                )

        segment_data = {}
        for k in (4, 6, 8):
            segment = fit_cluster_fold(
                raw_fit, raw_valid, y[fit_idx], int(fold), k
            )
            segment_data[k] = segment
            fit_cluster = segment["fit_cluster"]
            val_cluster = segment["valid_cluster"]
            profile = segment["fit_profile"]
            for cluster in range(k):
                fit_cluster_mask = fit_cluster == cluster
                valid_cluster_mask = val_cluster == cluster
                profile_row = {
                    "fold": int(fold),
                    "n_clusters": k,
                    "cluster": cluster,
                    "train_n": int(fit_cluster_mask.sum()),
                    "validation_n": int(valid_cluster_mask.sum()),
                    "validation_share": float(valid_cluster_mask.mean()),
                    "train_default_rate_only": float(
                        profile.loc[profile.cluster == cluster, TARGET].mean()
                    ),
                    "validation_labels_used_for_cluster_fit": False,
                }
                for col in profile.columns:
                    if col not in ("cluster", TARGET):
                        profile_row[f"train_median_{col}"] = float(
                            profile.loc[profile.cluster == cluster, col].median()
                        )
                profile_rows.append(profile_row)
                val_mask = valid_cluster_mask
                both_classes = len(np.unique(y[valid_idx][val_mask])) == 2
                segment_performance_rows.append(
                    {
                        "fold": int(fold),
                        "n_clusters": k,
                        "cluster": cluster,
                        "validation_n": int(val_mask.sum()),
                        "validation_defaults": int(y[valid_idx][val_mask].sum()),
                        "lgbm_product_auc": (
                            float(
                                roc_auc_score(
                                    y[valid_idx][val_mask],
                                    reference_product[valid_idx][val_mask],
                                )
                            )
                            if val_mask.sum() >= 30 and both_classes
                            else np.nan
                        ),
                        "minimum_validation_group_n_for_auc": 30,
                    }
                )
                center_row = {
                    "fold": int(fold),
                    "n_clusters": k,
                    "cluster": cluster,
                    "center_seed_ari_vs_alternate": segment["stability_ari"],
                    "train_cluster_size": int(fit_cluster_mask.sum()),
                    "validation_cluster_size": int(valid_cluster_mask.sum()),
                    "cluster_labels_are_fold_local": True,
                }
                center_row.update(
                    {
                        f"scaled_center_{i}": float(value)
                        for i, value in enumerate(segment["cluster_centers"][cluster])
                    }
                )
                stability_rows.append(center_row)
            stability_rows.append(
                {
                    "fold": int(fold),
                    "n_clusters": k,
                    "cluster": -1,
                    "center_seed_ari_vs_alternate": segment["stability_ari"],
                    "train_cluster_size": len(fit_idx),
                    "validation_cluster_size": len(valid_idx),
                    "cluster_labels_are_fold_local": True,
                }
            )

        segment_fit = {}
        segment_valid = {}
        for k in (4, 6):
            segment_fit[k] = pd.get_dummies(
                pd.Series(segment_data[k]["fit_cluster"]),
                prefix=f"SEGMENT_K{k}",
                dtype=np.float32,
            ).reindex(columns=[f"SEGMENT_K{k}_{i}" for i in range(k)], fill_value=0)
            segment_valid[k] = pd.get_dummies(
                pd.Series(segment_data[k]["valid_cluster"]),
                prefix=f"SEGMENT_K{k}",
                dtype=np.float32,
            ).reindex(columns=[f"SEGMENT_K{k}_{i}" for i in range(k)], fill_value=0)

        fold_candidate_features = {}
        for name in BIN_COLUMNS:
            fold_candidate_features[f"bin:{name.removeprefix('NONLINEAR_').removesuffix('_QBIN5')}"] = (
                pd.DataFrame({name: bin_fit[name]}),
                pd.DataFrame({name: bin_valid[name]}),
            )
        for name in INTERACTIONS:
            fold_candidate_features[name] = (
                inter_fit[[name]].copy(),
                inter_valid[[name]].copy(),
            )
        for name in MISSINGNESS_GROUPS:
            fold_candidate_features[name] = (
                pd.DataFrame({name: miss_fit[name]}),
                pd.DataFrame({name: miss_valid[name]}),
            )
        for k in (4, 6):
            fold_candidate_features[f"segment_k{k}"] = (
                segment_fit[k],
                segment_valid[k],
            )

        fold_candidate_features["family:all_nonlinear_bins"] = (
            pd.DataFrame(bin_fit),
            pd.DataFrame(bin_valid),
        )
        fold_candidate_features["family:all_interactions"] = (inter_fit, inter_valid)
        fold_candidate_features["family:all_missingness"] = (
            pd.DataFrame(miss_fit),
            pd.DataFrame(miss_valid),
        )
        fold_candidate_features["family:nonlinear_plus_interactions"] = (
            pd.concat([pd.DataFrame(bin_fit), inter_fit], axis=1),
            pd.concat([pd.DataFrame(bin_valid), inter_valid], axis=1),
        )
        fold_candidate_features["family:all_screened_plus_segment_k4"] = (
            pd.concat(
                [
                    pd.DataFrame(bin_fit),
                    inter_fit,
                    pd.DataFrame(miss_fit),
                    segment_fit[4],
                ],
                axis=1,
            ),
            pd.concat(
                [
                    pd.DataFrame(bin_valid),
                    inter_valid,
                    pd.DataFrame(miss_valid),
                    segment_valid[4],
                ],
                axis=1,
            ),
        )

        for name, (fit_extra, valid_extra) in fold_candidate_features.items():
            fit_frame = pd.concat(
                [fit_product_base, fit_extra.reset_index(drop=True)], axis=1
            )
            valid_frame = pd.concat(
                [valid_product_base, valid_extra.reset_index(drop=True)], axis=1
            )
            started_fit = time.perf_counter()
            prediction = fit_lgbm_variant(
                fit_frame, valid_frame, y[fit_idx], rounds, int(fold)
            )
            fit_times.append(
                {
                    "candidate": name,
                    "fold": int(fold),
                    "fit_predict_seconds": time.perf_counter() - started_fit,
                    "rounds": rounds,
                }
            )
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite OOF for {name}, fold {fold}")
            predictions[name][valid_idx] = prediction
            auc = float(roc_auc_score(y[valid_idx], prediction))
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": int(fold),
                    "fold_auc": auc,
                    "delta_vs_lgbm_product": auc
                    - float(
                        roc_auc_score(
                            y[valid_idx], reference_product[valid_idx]
                        )
                    ),
                }
            )
            raw_order = pd.concat(
                [raw_fit.reset_index(drop=True), raw_valid.reset_index(drop=True)],
                ignore_index=True,
            )
            feature_order = pd.concat(
                [fit_extra.reset_index(drop=True), valid_extra.reset_index(drop=True)],
                ignore_index=True,
            )
            redundancy_values.setdefault(name, []).append(
                (feature_order, raw_order)
            )

    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF incomplete for candidate {name}")
    advanced_oof = pd.DataFrame(
        {
            ID: ids,
            TARGET: y,
            "fold": folds,
            "reference_lgbm_product": reference_product,
        }
    )
    for name, prediction in predictions.items():
        advanced_oof[name] = prediction
    advanced_oof.to_csv(OUT / "PHASE3C_ADVANCED_OOF_PREDICTIONS.csv", index=False)

    summary_rows = []
    detailed_fold = pd.DataFrame(fold_rows)
    for name, prediction in predictions.items():
        metrics, _ = metric_summary(name, prediction, y, folds)
        baseline_fold_auc = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            baseline_fold_auc.append(
                float(roc_auc_score(y[mask], reference_product[mask]))
            )
        candidate_fold_auc = [
            float(
                roc_auc_score(
                    y[folds == fold], prediction[folds == fold]
                )
            )
            for fold in sorted(np.unique(folds))
        ]
        deltas = np.asarray(candidate_fold_auc) - np.asarray(baseline_fold_auc)
        metrics.update(
            {
                "delta_vs_lgbm_product": metrics["pooled_oof_auc"]
                - float(roc_auc_score(y, reference_product)),
                "mean_paired_fold_delta": float(deltas.mean()),
                "std_paired_fold_delta": float(deltas.std(ddof=1)),
                "folds_improved": int((deltas > 0).sum()),
                "validation_status": "OOF discovery; historical folds are exposed",
            }
        )
        summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "PHASE3C_ADVANCED_ABLATIONS.csv", index=False)
    detailed_fold.to_csv(OUT / "PHASE3C_ADVANCED_FOLD_RESULTS.csv", index=False)
    summary.loc[summary.candidate.str.startswith(("segment_", "family:all_screened"))].to_csv(
        OUT / "SEGMENT_FEATURE_ABLATIONS.csv", index=False
    )
    summary.loc[summary.candidate.str.startswith(("bin:", "family:all_nonlinear"))].to_csv(
        OUT / "NONLINEAR_FEATURE_ABLATIONS.csv", index=False
    )
    summary.loc[summary.candidate.str.startswith(("INTER_", "family:all_interactions"))].to_csv(
        OUT / "INTERACTION_VALIDATION_RESULTS.csv", index=False
    )
    summary.loc[summary.candidate.str.startswith(("MISS_", "family:all_missingness"))].to_csv(
        OUT / "MISSINGNESS_FEATURE_ABLATIONS.csv", index=False
    )
    pd.DataFrame(profile_rows).to_csv(OUT / "APPLICANT_SEGMENT_PROFILES.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(OUT / "SEGMENT_FOLD_STABILITY.csv", index=False)
    pd.DataFrame(segment_performance_rows).to_csv(
        OUT / "SEGMENT_MODEL_PERFORMANCE.csv", index=False
    )
    pd.DataFrame(nonlinear_pattern_rows).to_csv(
        OUT / "NONLINEAR_PATTERN_DISCOVERY.csv", index=False
    )
    hypothesis_rows = [
        {
            "feature": name,
            "formula": definition[0],
            "hypothesis": definition[1],
            "required_columns": "available supplied training columns and fold-local product feature",
            "outer_validation_used_to_select_formula": False,
        }
        for name, definition in INTERACTIONS.items()
    ]
    pd.DataFrame(hypothesis_rows).to_csv(
        OUT / "INTERACTION_HYPOTHESES.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "feature": name,
                "hypothesis": (
                    "Training-fitted quantile-bin representation tests monotone "
                    "or threshold behavior without selecting cutpoints on validation."
                ),
                "n_bins": 5,
                "cutpoints_fit_on_outer_train_only": True,
            }
            for name in BIN_SOURCES
        ]
    ).to_csv(OUT / "NONLINEAR_HYPOTHESES.csv", index=False)
    pd.DataFrame(missingness_rows).to_csv(
        OUT / "MISSINGNESS_PATTERN_ANALYSIS.csv", index=False
    )

    # Descriptive redundancy: no target is read while computing these correlations.
    numeric = raw.select_dtypes(include=[np.number]).drop(columns=[ID], errors="ignore")
    redundancy_rows = []
    for candidate, fold_pairs in redundancy_values.items():
        if candidate.startswith("segment_"):
            continue
        candidate_columns = sorted(
            set().union(*(set(feature_frame.columns) for feature_frame, _ in fold_pairs))
        )
        for feature in candidate_columns:
            correlations_by_raw: dict[str, list[float]] = {}
            for feature_frame, raw_frame in fold_pairs:
                if feature not in feature_frame:
                    continue
                x = feature_frame[feature].astype("float64")
                for column in numeric:
                    y_col = raw_frame[column].astype("float64")
                    mask = x.notna() & y_col.notna()
                    if (
                        mask.sum() < 1000
                        or x[mask].nunique() < 2
                        or y_col[mask].nunique() < 2
                    ):
                        continue
                    corr = x[mask].corr(y_col[mask], method="spearman")
                    if np.isfinite(corr):
                        correlations_by_raw.setdefault(str(column), []).append(
                            float(corr)
                        )
            mean_correlations = {
                column: float(np.mean(values))
                for column, values in correlations_by_raw.items()
            }
            best_name = (
                max(mean_correlations, key=lambda key: abs(mean_correlations[key]))
                if mean_correlations
                else ""
            )
            best_corr = mean_correlations.get(best_name, np.nan)
            redundancy_rows.append(
                {
                    "candidate_family": candidate,
                    "candidate_feature": feature,
                    "most_redundant_raw_feature": best_name,
                    "max_abs_mean_fold_spearman_with_raw": abs(best_corr)
                    if np.isfinite(best_corr)
                    else np.nan,
                    "target_used": False,
                    "correlation_is_descriptive_not_incremental": True,
                }
            )
    pd.DataFrame(redundancy_rows).to_csv(
        OUT / "FEATURE_REDUNDANCY_REPORT.csv", index=False
    )

    # Equal-weight blends are predeclared descriptive probes; no weights are fitted.
    blend_sources = {
        "lightgbm_product": base_oof["lightgbm__raw_plus_product"].to_numpy(),
        "catboost_product": base_oof["catboost__raw_plus_product"].to_numpy(),
        "xgboost_product": base_oof["xgboost__raw_plus_product"].to_numpy(),
        "lightgbm_raw": base_oof["lightgbm__raw"].to_numpy(),
    }
    blend_specs = {
        "blend_lgb_cat_50_50": {
            "lightgbm_product": 0.5,
            "catboost_product": 0.5,
        },
        "blend_lgb_xgb_50_50": {
            "lightgbm_product": 0.5,
            "xgboost_product": 0.5,
        },
        "blend_cat_xgb_50_50": {
            "catboost_product": 0.5,
            "xgboost_product": 0.5,
        },
        "blend_lgb_cat_xgb_equal": {
            "lightgbm_product": 1 / 3,
            "catboost_product": 1 / 3,
            "xgboost_product": 1 / 3,
        },
        "blend_lgb_raw_product_50_50": {
            "lightgbm_raw": 0.5,
            "lightgbm_product": 0.5,
        },
    }
    blend_rows, blend_fold_rows, weight_rows = [], [], []
    product_oof = {}
    for alias, column in (
        ("lightgbm_product", "lightgbm__raw_plus_product"),
        ("catboost_product", "catboost__raw_plus_product"),
        ("xgboost_product", "xgboost__raw_plus_product"),
    ):
        product_oof[alias] = base_oof[column].to_numpy()
    model_pairs = [
        ("lightgbm_product", "catboost_product"),
        ("lightgbm_product", "xgboost_product"),
        ("catboost_product", "xgboost_product"),
    ]
    corr_rows = []
    for left, right in model_pairs:
        p_left, p_right = product_oof[left], product_oof[right]
        fold_spearman = [
            spearmanr(p_left[folds == fold], p_right[folds == fold]).statistic
            for fold in sorted(np.unique(folds))
        ]
        corr_rows.append(
            {
                "model_a": left,
                "model_b": right,
                "pearson_oof": float(np.corrcoef(p_left, p_right)[0, 1]),
                "spearman_oof": float(spearmanr(p_left, p_right).statistic),
                "mean_within_fold_spearman": float(np.nanmean(fold_spearman)),
                "interpretation": "Descriptive OOF dependence; not a fitted blend weight.",
            }
        )
    pd.DataFrame(corr_rows).to_csv(OUT / "MODEL_CORRELATION_MATRIX.csv", index=False)
    for name, weights in blend_specs.items():
        prediction = sum(product_oof.get(key, blend_sources[key]) * weight for key, weight in weights.items())
        blend_summary, _ = metric_summary(name, prediction, y, folds)
        blend_summary["method"] = "fixed equal-weight discovery; no outer-label weight fitting"
        blend_summary["weights"] = json.dumps(weights, sort_keys=True)
        blend_rows.append(blend_summary)
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            blend_fold_rows.append(
                {
                    "blend": name,
                    "fold": int(fold),
                    "fold_auc": float(roc_auc_score(y[mask], prediction[mask])),
                    "weights": json.dumps(weights, sort_keys=True),
                }
            )
            for model_name, weight in weights.items():
                weight_rows.append(
                    {
                        "blend": name,
                        "fold": int(fold),
                        "model_component": model_name,
                        "weight": weight,
                        "weight_source": "fixed equal-weight; not estimated from labels",
                    }
                )
    pd.DataFrame(blend_rows).to_csv(OUT / "ENSEMBLE_ABLATION_RESULTS.csv", index=False)
    pd.DataFrame(blend_fold_rows).to_csv(OUT / "ENSEMBLE_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(
        OUT / "ENSEMBLE_WEIGHT_STABILITY.csv", index=False
    )

    family_rows = []
    benchmark_results = pd.read_csv(OUT / "MODEL_BENCHMARK_RESULTS.csv")
    for row in benchmark_results.itertuples(index=False):
        family_rows.append(
            {
                "family": row.feature_variant,
                "model": row.model,
                "pooled_oof_auc": row.pooled_oof_auc,
                "mean_fold_auc": row.mean_fold_auc,
                "std_fold_auc": row.std_fold_auc,
                "delta_vs_same_model_raw": row.delta_vs_same_model_raw,
                "evidence_type": "fixed-fold model benchmark",
            }
        )
    for row in summary.itertuples(index=False):
        family_rows.append(
            {
                "family": row.candidate,
                "model": "lightgbm",
                "pooled_oof_auc": row.pooled_oof_auc,
                "mean_fold_auc": row.mean_fold_auc,
                "std_fold_auc": row.std_fold_auc,
                "delta_vs_same_model_raw": row.delta_vs_lgbm_product,
                "evidence_type": "fixed-fold individual/family ablation",
            }
        )
    pd.DataFrame(family_rows).to_csv(
        OUT / "FEATURE_FAMILY_ABLATIONS.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "model_variant": row.model_variant,
                "pooled_oof_auc": row.pooled_oof_auc,
                "mean_fold_auc": row.mean_fold_auc,
                "std_fold_auc": row.std_fold_auc,
                "delta_vs_same_model_raw": row.delta_vs_same_model_raw,
                "evidence_type": "model/feature benchmark; discovery only",
            }
            for row in benchmark_results.itertuples(index=False)
        ]
        + [
            {
                "model_variant": row["candidate"],
                "pooled_oof_auc": row["pooled_oof_auc"],
                "mean_fold_auc": row["mean_fold_auc"],
                "std_fold_auc": row["std_fold_auc"],
                "delta_vs_same_model_raw": row["delta_vs_lgbm_product"],
                "evidence_type": "feature-family ablation; paired OOF comparison",
            }
            for row in summary.to_dict("records")
        ]
    ).to_csv(OUT / "FEATURE_FAMILY_COMPLEMENTARITY.csv", index=False)
    pd.DataFrame(fit_times).to_csv(OUT / "PHASE3C_ADVANCED_RUNTIME.csv", index=False)
    log = [
        f"started_utc={started_utc.isoformat()}",
        f"finished_utc={datetime.now(timezone.utc).isoformat()}",
        f"runtime_seconds={time.perf_counter() - started:.3f}",
        f"train_rows={len(train)}",
        f"outer_folds={len(np.unique(folds))}",
        f"train_sha256={hashlib.sha256((DATA / 'train.csv').read_bytes()).hexdigest()}",
        f"source_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        "outer_validation_labels_used_for_clustering_or_cutpoints=False",
        "outer_validation_labels_used_for_ensemble_weights=False",
        "ensemble_weights=fixed equal weight",
        "status=SUCCESS",
    ]
    (OUT / "PHASE3C_ADVANCED_LOG.txt").write_text(
        "\n".join(log) + "\n", encoding="utf-8"
    )
    print("\nAdvanced candidate results")
    print(
        summary.sort_values("pooled_oof_auc", ascending=False)
        .head(20)
        .to_string(index=False)
    )
    print("\nFixed-weight blends")
    print(pd.DataFrame(blend_rows).to_string(index=False))
    print("\n" + "\n".join(log), flush=True)


if __name__ == "__main__":
    main()
