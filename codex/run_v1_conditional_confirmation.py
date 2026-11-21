import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci")
DATA = ROOT / "inpuy"
OUT = ROOT / "codex"
SEEDS = (20261041, 20261042)
INNER_FOLDS = 3
OUTER_FOLDS = 3
BOOTSTRAPS = 250


def ranked_auc_delta_ci(y, candidate, reference, seed):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        idx = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], reference[idx]
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def make_encoder_features(keys, train_idx, valid_idx, y, seed, alpha=250):
    train_features = {}
    valid_features = {}
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=seed
    )

    for name, all_keys in keys.items():
        train_keys = np.asarray(all_keys, dtype=object)[train_idx]
        valid_keys = np.asarray(all_keys, dtype=object)[valid_idx]
        encoded = np.zeros(len(train_idx), dtype=np.float32)
        log_count = np.zeros(len(train_idx), dtype=np.float32)

        for fit_pos, hold_pos in inner.split(np.zeros(len(train_idx)), y[train_idx]):
            fit_y = y[train_idx][fit_pos]
            stats = pd.DataFrame(
                {"key": train_keys[fit_pos], "target": fit_y}
            ).groupby("key", dropna=False)["target"].agg(["sum", "count"])
            prior = float(fit_y.mean())
            mapping = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
            counts = stats["count"]
            hold_keys = pd.Series(train_keys[hold_pos])
            encoded[hold_pos] = (
                hold_keys.map(mapping).fillna(prior).to_numpy(np.float32)
            )
            log_count[hold_pos] = np.log1p(
                hold_keys.map(counts).fillna(0).to_numpy(np.float32)
            )

        fit_y = y[train_idx]
        stats = pd.DataFrame(
            {"key": train_keys, "target": fit_y}
        ).groupby("key", dropna=False)["target"].agg(["sum", "count"])
        prior = float(fit_y.mean())
        mapping = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
        counts = stats["count"]
        valid_keys_series = pd.Series(valid_keys)
        train_features[f"{name}_TE"] = encoded
        train_features[f"{name}_LOGCOUNT"] = log_count
        valid_features[f"{name}_TE"] = (
            valid_keys_series.map(mapping).fillna(prior).to_numpy(np.float32)
        )
        valid_features[f"{name}_LOGCOUNT"] = np.log1p(
            valid_keys_series.map(counts).fillna(0).to_numpy(np.float32)
        )

    return pd.DataFrame(train_features), pd.DataFrame(valid_features)


def main():
    started = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    n_train = len(train)
    raw = pd.concat(
        [
            train.drop(columns=["TARGET", "SK_ID_CURR"]),
            test.drop(columns=["SK_ID_CURR"]),
        ],
        ignore_index=True,
    )

    base = raw.copy()
    base["DAYS_EMPLOYED_SENTINEL"] = (base.DAYS_EMPLOYED == 365243).astype("int8")
    base["DAYS_EMPLOYED_CLEAN"] = base.DAYS_EMPLOYED.where(
        base.DAYS_EMPLOYED != 365243
    )
    base["AGE_YEARS"] = -base.DAYS_BIRTH / 365.25
    for numerator, denominator in (
        ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("AMT_ANNUITY", "AMT_CREDIT"),
        ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        ("AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"),
    ):
        base[f"{numerator}_DIV_{denominator}"] = base[numerator] / (
            base[denominator].abs() + 1
        )
    base["INCOME_PER_FAMILY"] = base.AMT_INCOME_TOTAL / (
        base.CNT_FAM_MEMBERS.fillna(1) + 1e-3
    )
    ext = base[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    base["EXT_MEAN"] = ext.mean(axis=1)
    base["EXT_MIN"] = ext.min(axis=1)
    base["EXT_STD"] = ext.std(axis=1)
    for column in base.select_dtypes(include=["object", "str"]).columns:
        base[column] = pd.Categorical(
            base[column].fillna("__MISSING__").astype(str),
            categories=pd.Index(
                base[column].fillna("__MISSING__").astype(str).unique()
            ),
        )
    for column in base.select_dtypes(include="number").columns:
        if base[column].isna().any():
            base[f"{column}_MISS"] = base[column].isna().astype("int8")

    credit = raw.AMT_CREDIT.replace(0, np.nan)
    annuity = raw.AMT_ANNUITY
    cash = raw.NAME_CONTRACT_TYPE.eq("Cash loans")
    q_raw = annuity / credit
    q_dec = pd.Series(
        np.where(cash, (annuity - 129.5) / credit, q_raw), index=raw.index
    ).replace([np.inf, -np.inf], np.nan)
    q3 = q_dec.round(3)
    q4 = q_dec.round(4)
    q5 = q_dec.round(5)
    q_frequency = q5.value_counts()
    q_centers = np.sort(q_frequency[q_frequency >= 30].index.to_numpy(float))
    q_values = q_dec.to_numpy(float)
    nearest_q = np.full(len(raw), np.nan)
    q_distance = np.full(len(raw), np.nan)
    finite_q = np.isfinite(q_values)
    q_pos = np.searchsorted(q_centers, q_values[finite_q])
    q_low = np.clip(q_pos - 1, 0, len(q_centers) - 1)
    q_high = np.clip(q_pos, 0, len(q_centers) - 1)
    low_distance = np.abs(q_values[finite_q] - q_centers[q_low])
    high_distance = np.abs(q_values[finite_q] - q_centers[q_high])
    choose_high = high_distance < low_distance
    nearest_q[finite_q] = np.where(choose_high, q_centers[q_high], q_centers[q_low])
    q_distance[finite_q] = np.where(choose_high, high_distance, low_distance)
    fee = np.where(cash, 129.5, 0.0)
    formula_residual = annuity.to_numpy() - (
        np.where(np.isfinite(nearest_q), nearest_q, q_values)
        * credit.to_numpy()
        + fee
    )

    credit_goods = (
        raw.AMT_CREDIT / raw.AMT_GOODS_PRICE.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    kg5 = credit_goods.round(5)
    kg_frequency = kg5.value_counts()
    kg_centers = np.sort(kg_frequency[kg_frequency >= 30].index.to_numpy(float))
    kg_values = credit_goods.to_numpy(float)
    kg_nearest = np.full(len(raw), np.nan)
    kg_distance = np.full(len(raw), np.nan)
    finite_kg = np.isfinite(kg_values)
    kg_pos = np.searchsorted(kg_centers, kg_values[finite_kg])
    kg_low = np.clip(kg_pos - 1, 0, len(kg_centers) - 1)
    kg_high = np.clip(kg_pos, 0, len(kg_centers) - 1)
    low_distance = np.abs(kg_values[finite_kg] - kg_centers[kg_low])
    high_distance = np.abs(kg_values[finite_kg] - kg_centers[kg_high])
    choose_high = high_distance < low_distance
    kg_nearest[finite_kg] = np.where(
        choose_high, kg_centers[kg_high], kg_centers[kg_low]
    )
    kg_distance[finite_kg] = np.where(choose_high, high_distance, low_distance)

    formula = pd.DataFrame(
        {
            "Q_RAW": q_raw,
            "Q_DEC": q_dec,
            "Q_RES3": (q_dec - q3).abs(),
            "Q_RES4": (q_dec - q4).abs(),
            "Q_RES5": (q_dec - q5).abs(),
            "Q_NEAREST_DIST": q_distance,
            "Q_SUPPORT_LOG": np.log1p(q5.map(q_frequency).to_numpy(float)),
            "ANN_RESID": formula_residual,
            "ANN_RESID_ABS": np.abs(formula_residual),
            "Q_FEE_EFFECT": q_raw - q_dec,
            "K_NEAREST_DIST": kg_distance,
            "K_NEAREST": kg_nearest,
            "K_SUPPORT_LOG": np.log1p(kg5.map(kg_frequency).to_numpy(float)),
        }
    )
    X = pd.concat([base.reset_index(drop=True), formula], axis=1).replace(
        [np.inf, -np.inf], np.nan
    )
    for column in X.select_dtypes(include=["object", "str"]).columns:
        values = X[column].fillna("__MISSING__").astype(str)
        X[column] = pd.Categorical(values, categories=pd.Index(values.unique()))

    def quantile_key(values, bins):
        return (
            "BIN:"
            + pd.qcut(
                pd.Series(values).rank(method="first"),
                q=bins,
                labels=False,
                duplicates="drop",
            )
            .fillna(-1)
            .astype(int)
            .astype(str)
            .to_numpy()
        )

    q4_key = "Q4:" + q4.fillna(-999).astype(str).to_numpy()
    q3_key = "Q3:" + q3.fillna(-999).astype(str).to_numpy()
    contract_key = (
        "CT:" + raw.NAME_CONTRACT_TYPE.fillna("__MISSING__").astype(str).to_numpy()
    )
    baseline_keys = {
        "PRODUCT_Q4": q4_key,
        "PRODUCT_Q3": q3_key,
        "PRODUCT_Q4_CONTRACT": np.char.add(
            np.char.add(q4_key.astype(str), "|"), contract_key.astype(str)
        ),
    }
    applicant_keys = {
        "Q4_INCOME": np.char.add(
            np.char.add(q4_key.astype(str), "|"),
            ("INC:" + raw.NAME_INCOME_TYPE.fillna("__MISSING__").astype(str)).to_numpy(),
        ),
        "Q4_OCCUPATION": np.char.add(
            np.char.add(q4_key.astype(str), "|"),
            ("OCC:" + raw.OCCUPATION_TYPE.fillna("__MISSING__").astype(str)).to_numpy(),
        ),
        "Q4_EDUCATION": np.char.add(
            np.char.add(q4_key.astype(str), "|"),
            ("EDU:" + raw.NAME_EDUCATION_TYPE.fillna("__MISSING__").astype(str)).to_numpy(),
        ),
    }
    burden = raw.AMT_ANNUITY / (raw.AMT_INCOME_TOTAL + 1)
    formula_keys = {
        "Q4_FORMULA_RESID": np.char.add(
            np.char.add(q4_key.astype(str), "|"),
            quantile_key(np.abs(formula_residual), 6),
        ),
        "Q4_BURDEN": np.char.add(
            np.char.add(q4_key.astype(str), "|"), quantile_key(burden, 8)
        ),
    }
    feature_groups = {
        "product_only": baseline_keys,
        "product_applicant": {**baseline_keys, **applicant_keys},
        "product_formula_burden": {**baseline_keys, **formula_keys},
    }

    params = {
        "n_estimators": 300,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 120,
        "max_bin": 127,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 3,
        "reg_alpha": 0.05,
        "objective": "binary",
        "verbosity": -1,
        "n_jobs": 4,
    }
    results = []
    predictions = {"y": y}

    for seed in SEEDS:
        folds = list(
            StratifiedKFold(
                n_splits=OUTER_FOLDS, shuffle=True, random_state=seed
            ).split(np.zeros(n_train), y)
        )
        oof = {
            name: np.zeros(n_train, dtype=np.float32)
            for name in feature_groups
        }
        fold_auc = {name: [] for name in feature_groups}

        for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
            for name, keys in feature_groups.items():
                te_fit, te_valid = make_encoder_features(
                    keys,
                    fit_idx,
                    valid_idx,
                    y,
                    seed=seed * 10 + fold_number,
                    alpha=250,
                )
                train_frame = pd.concat(
                    [
                        X.iloc[fit_idx].reset_index(drop=True),
                        te_fit.reset_index(drop=True),
                    ],
                    axis=1,
                )
                valid_frame = pd.concat(
                    [
                        X.iloc[valid_idx].reset_index(drop=True),
                        te_valid.reset_index(drop=True),
                    ],
                    axis=1,
                )
                model = lgb.LGBMClassifier(
                    **params, random_state=seed + fold_number
                )
                model.fit(
                    train_frame,
                    y[fit_idx],
                    categorical_feature="auto",
                )
                predictions_fold = model.predict_proba(valid_frame)[:, 1]
                oof[name][valid_idx] = predictions_fold
                fold_auc[name].append(
                    float(roc_auc_score(y[valid_idx], predictions_fold))
                )
                print(
                    f"seed={seed} fold={fold_number}/{OUTER_FOLDS} "
                    f"variant={name} auc={fold_auc[name][-1]:.6f}",
                    flush=True,
                )

        for name, score in oof.items():
            predictions[f"oof_{seed}_{name}"] = score
        aucs = {
            name: float(roc_auc_score(y, score)) for name, score in oof.items()
        }
        reference = oof["product_only"]
        for name, score in oof.items():
            if name == "product_only":
                continue
            delta = aucs[name] - aucs["product_only"]
            results.append(
                {
                    "seed": seed,
                    "variant": name,
                    "pooled_auc": aucs[name],
                    "product_only_auc": aucs["product_only"],
                    "delta_vs_product_only": delta,
                    "fold_auc": fold_auc[name],
                    "fold_delta_vs_product_only": [
                        fold_auc[name][i] - fold_auc["product_only"][i]
                        for i in range(OUTER_FOLDS)
                    ],
                    "bootstrap_95pct_ci_vs_product_only": ranked_auc_delta_ci(
                        y, score, reference, seed=seed + 3
                    ),
                }
            )
        print(
            f"seed={seed} pooled: "
            + ", ".join(
                f"{name}={aucs[name]:.6f}" for name in feature_groups
            ),
            flush=True,
        )

    pd.DataFrame(results).to_csv(OUT / "v_1_conditional_confirmation.csv", index=False)
    np.savez_compressed(OUT / "v_1_conditional_confirmation_oof.npz", **predictions)
    metadata = {
        "experiment": "v_1 fresh confirmation of conditional product encodings",
        "dataset": {
            "train_rows": n_train,
            "positive_rate": float(y.mean()),
        },
        "outer_seeds": list(SEEDS),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "target_encoding_alpha": 250,
        "lightgbm_version": lgb.__version__,
        "parameters": params,
        "results_csv": str(OUT / "v_1_conditional_confirmation.csv"),
        "oof_archive": str(OUT / "v_1_conditional_confirmation_oof.npz"),
        "warning": (
            "This is a compact-model feature confirmation, not a matched "
            "comparison against the v15 ensemble or its public score."
        ),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "v_1_conditional_confirmation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
