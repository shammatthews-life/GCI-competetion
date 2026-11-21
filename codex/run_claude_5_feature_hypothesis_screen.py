import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci")
DATA = ROOT / "inpuy"
OUT = ROOT / "codex"
SEEDS = (20261051, 20261052)
OUTER_FOLDS = 3
INNER_FOLDS = 3
BOOTSTRAPS = 300
ALPHAS = {
    "product_q3": 250.0,
    "product_q4": 250.0,
    "product_q4_contract": 250.0,
    "group_te": 300.0,
    "pair_te": 600.0,
    "risk_band_te": 300.0,
}


def _key(values):
    return pd.Series(values, dtype="object").fillna("__MISSING__").astype(str).to_numpy()


def _mapping(fit_keys, fit_y, query_keys, alpha):
    stats = (
        pd.DataFrame({"key": fit_keys, "target": fit_y})
        .groupby("key", dropna=False)["target"]
        .agg(["sum", "count"])
    )
    prior = float(np.mean(fit_y))
    means = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
    query = pd.Series(query_keys, dtype="object")
    return (
        query.map(means).fillna(prior).to_numpy(np.float32),
        np.log1p(query.map(stats["count"]).fillna(0).to_numpy(np.float32)),
    )


def _nested_target_features(keys, fit_idx, valid_idx, y, inner_seed, prefix, alpha):
    fit_y = y[fit_idx]
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed
    )
    train_out = {}
    valid_out = {}
    for name, key_values in keys.items():
        all_keys = np.asarray(key_values, dtype=object)
        fit_keys = all_keys[fit_idx]
        valid_keys = all_keys[valid_idx]
        encoded = np.zeros(len(fit_idx), dtype=np.float32)
        counts = np.zeros(len(fit_idx), dtype=np.float32)
        for inner_fit, inner_hold in inner.split(np.zeros(len(fit_idx)), fit_y):
            e, c = _mapping(
                fit_keys[inner_fit], fit_y[inner_fit], fit_keys[inner_hold], alpha
            )
            encoded[inner_hold] = e
            counts[inner_hold] = c
        valid_e, valid_c = _mapping(fit_keys, fit_y, valid_keys, alpha)
        train_out[f"{prefix}_{name}_TE"] = encoded
        train_out[f"{prefix}_{name}_LOGCOUNT"] = counts
        valid_out[f"{prefix}_{name}_TE"] = valid_e
        valid_out[f"{prefix}_{name}_LOGCOUNT"] = valid_c
    return pd.DataFrame(train_out), pd.DataFrame(valid_out)


def _paired_bootstrap_delta(y, candidate, reference, seed):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        idx = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], reference[idx]
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def _build_frames(train, test):
    n_train = len(train)
    y = train["TARGET"].to_numpy(dtype=np.int8)
    raw = pd.concat(
        [
            train.drop(columns=["TARGET", "SK_ID_CURR"]),
            test.drop(columns=["SK_ID_CURR"]),
        ],
        ignore_index=True,
    )
    base = raw.copy()
    base["DAYS_EMPLOYED_SENTINEL"] = (raw["DAYS_EMPLOYED"] == 365243).astype("int8")
    base["DAYS_EMPLOYED_CLEAN"] = raw["DAYS_EMPLOYED"].where(
        raw["DAYS_EMPLOYED"] != 365243
    )
    base["AGE_YEARS"] = -raw["DAYS_BIRTH"] / 365.25
    for numerator, denominator in (
        ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("AMT_ANNUITY", "AMT_CREDIT"),
        ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        ("AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"),
    ):
        base[f"{numerator}_DIV_{denominator}"] = raw[numerator] / (
            raw[denominator].abs() + 1.0
        )
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    base["EXT_MEAN"] = ext.mean(axis=1)
    base["EXT_MIN"] = ext.min(axis=1)
    base["EXT_MAX"] = ext.max(axis=1)
    base["EXT_STD"] = ext.std(axis=1)
    for column in base.select_dtypes(include=["object", "str"]).columns:
        values = base[column].fillna("__MISSING__").astype(str)
        base[column] = pd.Categorical(values, categories=pd.Index(values.unique()))
    for column in base.select_dtypes(include="number").columns:
        if base[column].isna().any():
            base[f"{column}_MISS"] = base[column].isna().astype("int8")

    credit = raw["AMT_CREDIT"].replace(0, np.nan)
    cash = raw["NAME_CONTRACT_TYPE"].eq("Cash loans")
    q = (raw["AMT_ANNUITY"] - np.where(cash, 129.5, 0.0)) / credit
    q = q.replace([np.inf, -np.inf], np.nan)
    q3, q4, q5 = q.round(3), q.round(4), q.round(5)
    product_keys = {
        "product_q3": _key(q3),
        "product_q4": _key(q4),
        "product_q4_contract": _key(
            q4.astype("string") + "|" + raw["NAME_CONTRACT_TYPE"].astype("string")
        ),
    }
    for column, values in (("PRODUCT_Q3", q3), ("PRODUCT_Q4", q4), ("PRODUCT_Q5", q5)):
        text = _key(values)
        base[column] = pd.Categorical(text, categories=pd.Index(pd.unique(text)))

    bureau = pd.DataFrame(index=raw.index)
    bureau_hour = raw["AMT_REQ_CREDIT_BUREAU_HOUR"].astype(float)
    bureau_month = raw["AMT_REQ_CREDIT_BUREAU_MON"].astype(float)
    bureau_quarter = raw["AMT_REQ_CREDIT_BUREAU_QRT"].astype(float)
    bureau_year = raw["AMT_REQ_CREDIT_BUREAU_YEAR"].astype(float)
    for period, values in (
        ("HOUR", bureau_hour),
        ("MONTH", bureau_month),
        ("QUARTER", bureau_quarter),
        ("YEAR", bureau_year),
    ):
        bureau[f"BUREAU_LOG_{period}"] = np.log1p(values.clip(lower=0))
    recent = bureau_month.fillna(0) + bureau_quarter.fillna(0)
    total = recent + bureau_year.fillna(0) + bureau_hour.fillna(0)
    bureau["BUREAU_RECENT_SHARE"] = recent / total.where(total > 0)
    bureau["BUREAU_MONTH_SHARE"] = bureau_month / (
        bureau_month + bureau_quarter + bureau_year
    ).where((bureau_month + bureau_quarter + bureau_year) > 0)
    bureau["BUREAU_MONTH_TO_ANNUAL_RATE"] = bureau_month / (
        bureau_year / 12.0 + 1.0
    )
    bureau["BUREAU_QUARTER_TO_ANNUAL_RATE"] = bureau_quarter / (
        bureau_year / 4.0 + 1.0
    )
    bureau["BUREAU_ZERO_TOTAL"] = (total == 0).astype("int8")

    household = pd.DataFrame(index=raw.index)
    family = raw["CNT_FAM_MEMBERS"].clip(lower=1)
    children = raw["CNT_CHILDREN"].clip(lower=0).fillna(0)
    adults_equiv = (family - 0.5 * children).clip(lower=1)
    household["INCOME_PER_ADULT_EQ"] = raw["AMT_INCOME_TOTAL"] / adults_equiv
    household["CREDIT_PER_ADULT_EQ"] = raw["AMT_CREDIT"] / adults_equiv
    household["ANNUITY_PER_ADULT_EQ"] = raw["AMT_ANNUITY"] / adults_equiv
    household["DISPOSABLE_INCOME_PER_ADULT_EQ"] = (
        raw["AMT_INCOME_TOTAL"] - raw["AMT_ANNUITY"]
    ) / adults_equiv
    household["CHILDREN_PER_ADULT_EQ"] = children / adults_equiv
    household["ADULT_EQ_FRACTION"] = adults_equiv / family

    group_keys = {
        name: _key(raw[column])
        for name, column in (
            ("organization", "ORGANIZATION_TYPE"),
            ("occupation", "OCCUPATION_TYPE"),
            ("income_type", "NAME_INCOME_TYPE"),
            ("education", "NAME_EDUCATION_TYPE"),
            ("region_rating", "REGION_RATING_CLIENT"),
            ("housing", "NAME_HOUSING_TYPE"),
        )
    }
    pair_keys = {
        "org_occupation": _key(
            raw["ORGANIZATION_TYPE"].astype("string")
            + "|"
            + raw["OCCUPATION_TYPE"].astype("string")
        ),
        "income_education": _key(
            raw["NAME_INCOME_TYPE"].astype("string")
            + "|"
            + raw["NAME_EDUCATION_TYPE"].astype("string")
        ),
    }
    continuous = {
        "ext1": raw["EXT_SOURCE_1"],
        "ext2": raw["EXT_SOURCE_2"],
        "ext3": raw["EXT_SOURCE_3"],
        "credit_income": base["AMT_CREDIT_DIV_AMT_INCOME_TOTAL"],
        "annuity_income": base["AMT_ANNUITY_DIV_AMT_INCOME_TOTAL"],
        "age": base["AGE_YEARS"],
        "employment_years": -base["DAYS_EMPLOYED_CLEAN"] / 365.25,
        "bureau_year": raw["AMT_REQ_CREDIT_BUREAU_YEAR"],
    }
    risk_band_keys = {}
    for name, values in continuous.items():
        ranked = pd.Series(values).rank(method="first", na_option="keep")
        try:
            bands = pd.qcut(ranked, q=20, labels=False, duplicates="drop")
        except ValueError:
            bands = pd.Series(np.zeros(len(ranked), dtype=np.int8))
        risk_band_keys[f"{name}_band"] = _key(bands)

    categories = {
        "bureau_profile": bureau,
        "household": household,
        "group_te": group_keys,
        "pair_te": pair_keys,
        "risk_band_te": risk_band_keys,
    }
    return base, categories, product_keys, y, n_train


def _make_variant_features(name, block, fit_idx, valid_idx, y, seed):
    if name in ("group_te", "pair_te", "risk_band_te"):
        train_features, valid_features = _nested_target_features(
            block,
            fit_idx,
            valid_idx,
            y,
            inner_seed=seed + 100,
            prefix=name.upper(),
            alpha=ALPHAS[name],
        )
        return train_features, valid_features
    return block.iloc[fit_idx].reset_index(drop=True), block.iloc[valid_idx].reset_index(
        drop=True
    )


def _run_seed(seed, base, blocks, product_keys, y, n_train):
    splitter = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=seed
    )
    names = ["baseline", *blocks.keys()]
    predictions = {name: np.zeros(n_train, dtype=np.float32) for name in names}
    fold_records = []
    for fold, (fit_idx, valid_idx) in enumerate(
        splitter.split(np.zeros(n_train), y), start=1
    ):
        train_te, valid_te = _nested_target_features(
            product_keys,
            fit_idx,
            valid_idx,
            y,
            inner_seed=seed + fold,
            prefix="PRODUCT",
            alpha=ALPHAS["product_q4"],
        )
        train_base = base.iloc[fit_idx].reset_index(drop=True)
        valid_base = base.iloc[valid_idx].reset_index(drop=True)
        common_train = pd.concat([train_base, train_te], axis=1)
        common_valid = pd.concat([valid_base, valid_te], axis=1)
        params = {
            "objective": "binary",
            "n_estimators": 450,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "min_child_samples": 120,
            "max_bin": 127,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "reg_alpha": 0.05,
            "verbosity": -1,
            "n_jobs": 3,
            "random_state": seed + fold,
        }
        fold_predictions = {}
        for name in names:
            train_frame, valid_frame = common_train, common_valid
            if name != "baseline":
                add_train, add_valid = _make_variant_features(
                    name, blocks[name], fit_idx, valid_idx, y, seed + fold
                )
                train_frame = pd.concat([train_frame, add_train], axis=1)
                valid_frame = pd.concat([valid_frame, add_valid], axis=1)
            model = lgb.LGBMClassifier(**params)
            model.fit(train_frame, y[fit_idx], categorical_feature="auto")
            pred = model.predict_proba(valid_frame)[:, 1].astype(np.float32)
            predictions[name][valid_idx] = pred
            fold_predictions[name] = pred
        base_auc = roc_auc_score(y[valid_idx], fold_predictions["baseline"])
        for name in names:
            auc = roc_auc_score(y[valid_idx], fold_predictions[name])
            fold_records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "variant": name,
                    "auc": float(auc),
                    "delta_vs_baseline": float(auc - base_auc),
                }
            )
        print(
            f"seed={seed} fold={fold}/{OUTER_FOLDS} "
            f"baseline_auc={base_auc:.6f}",
            flush=True,
        )
    return predictions, fold_records


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    base, blocks, product_keys, y, n_train = _build_frames(train, test)
    all_oof = {}
    fold_rows = []
    for seed in SEEDS:
        predictions, rows = _run_seed(seed, base, blocks, product_keys, y, n_train)
        fold_rows.extend(rows)
        for name, pred in predictions.items():
            all_oof[f"seed{seed}_{name}"] = pred

    metrics = []
    for seed in SEEDS:
        baseline = all_oof[f"seed{seed}_baseline"]
        baseline_auc = roc_auc_score(y, baseline)
        for name in [
            "bureau_profile",
            "household",
            "group_te",
            "pair_te",
            "risk_band_te",
        ]:
            pred = all_oof[f"seed{seed}_{name}"]
            auc = roc_auc_score(y, pred)
            metrics.append(
                {
                    "seed": seed,
                    "variant": name,
                    "baseline_auc": float(baseline_auc),
                    "candidate_auc": float(auc),
                    "delta_auc": float(auc - baseline_auc),
                    "fold_deltas": [
                        next(
                            row["delta_vs_baseline"]
                            for row in fold_rows
                            if row["seed"] == seed
                            and row["fold"] == fold
                            and row["variant"] == name
                        )
                        for fold in range(1, OUTER_FOLDS + 1)
                    ],
                    "paired_bootstrap_95_ci": _paired_bootstrap_delta(
                        y, pred, baseline, seed + len(name)
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(OUT / "claude_5_feature_hypothesis_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        OUT / "claude_5_feature_hypothesis_fold_metrics.csv", index=False
    )
    np.savez_compressed(OUT / "claude_5_feature_hypothesis_oof.npz", **all_oof, y=y)
    metadata = {
        "data_rows": int(n_train),
        "features_before_target_encodings": int(base.shape[1]),
        "outer_folds": OUTER_FOLDS,
        "inner_folds_for_target_encoding": INNER_FOLDS,
        "seeds": list(SEEDS),
        "variants": list(blocks),
        "target_encodings_are_nested_and_fold_safe": True,
        "runtime_seconds": round(time.time() - started, 2),
        "v15_oof_cache_available": False,
        "submission_created": False,
        "interpretation": (
            "Feature-block screen against the compact product-aware LightGBM "
            "reference, not a matched comparison against the v15 ensemble."
        ),
    }
    (OUT / "claude_5_feature_hypothesis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
