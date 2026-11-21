import hashlib
import json
import os
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
SEEDS = tuple(
    int(seed) for seed in os.environ.get("GCI_SEEDS", "42,117").split(",")
)
FOLDS = 5
INNER_FOLDS = 3
ALPHA = 250
OUTPUT_PREFIX = os.environ.get("GCI_OUTPUT_PREFIX", "v_2_submission")


def make_te(train_keys, valid_keys, train_y):
    stats = (
        pd.DataFrame({"key": train_keys, "target": train_y})
        .groupby("key", dropna=False)["target"]
        .agg(["sum", "count"])
    )
    prior = float(train_y.mean())
    means = (stats["sum"] + ALPHA * prior) / (stats["count"] + ALPHA)
    counts = stats["count"]
    keys = pd.Series(valid_keys)
    encoded = keys.map(means).fillna(prior).to_numpy(np.float32)
    log_count = np.log1p(keys.map(counts).fillna(0).to_numpy(np.float32))
    return encoded, log_count


def build_te_frames(keys, fit_idx, valid_idx, y, inner_seed, full_test_keys):
    train_columns = {}
    valid_columns = {}
    test_columns = {}
    inner = StratifiedKFold(
        n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed
    )
    fit_y = y[fit_idx]
    for name, key_values in keys.items():
        fit_keys = np.asarray(key_values, dtype=object)[fit_idx]
        valid_keys = np.asarray(key_values, dtype=object)[valid_idx]
        test_keys = np.asarray(key_values, dtype=object)[full_test_keys]
        encoded = np.zeros(len(fit_idx), dtype=np.float32)
        log_count = np.zeros(len(fit_idx), dtype=np.float32)
        for inner_fit, inner_valid in inner.split(np.zeros(len(fit_idx)), fit_y):
            e, c = make_te(
                fit_keys[inner_fit], fit_keys[inner_valid], fit_y[inner_fit]
            )
            encoded[inner_valid] = e
            log_count[inner_valid] = c

        valid_e, valid_c = make_te(fit_keys, valid_keys, fit_y)
        test_e, test_c = make_te(fit_keys, test_keys, fit_y)
        train_columns[f"{name}_TE"] = encoded
        train_columns[f"{name}_LOGCOUNT"] = log_count
        valid_columns[f"{name}_TE"] = valid_e
        valid_columns[f"{name}_LOGCOUNT"] = valid_c
        test_columns[f"{name}_TE"] = test_e
        test_columns[f"{name}_LOGCOUNT"] = test_c

    return (
        pd.DataFrame(train_columns),
        pd.DataFrame(valid_columns),
        pd.DataFrame(test_columns),
    )


def paired_bootstrap_ci(y, candidate, reference, seed, n_bootstrap=300):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], reference[idx]
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def main():
    start = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    n_train = len(train)
    n_test = len(test)
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
    base["EXT_MAX"] = ext.max(axis=1)
    base["EXT_STD"] = ext.std(axis=1)

    for column in base.select_dtypes(include=["object", "str"]).columns:
        values = base[column].fillna("__MISSING__").astype(str)
        base[column] = pd.Categorical(values, categories=pd.Index(values.unique()))
    for column in base.select_dtypes(include="number").columns:
        if base[column].isna().any():
            base[f"{column}_MISS"] = base[column].isna().astype("int8")

    credit = raw.AMT_CREDIT.replace(0, np.nan)
    annuity = raw.AMT_ANNUITY
    cash = raw.NAME_CONTRACT_TYPE.eq("Cash loans")
    q_raw = annuity / credit
    q_decoded = pd.Series(
        np.where(cash, (annuity - 129.5) / credit, q_raw), index=raw.index
    ).replace([np.inf, -np.inf], np.nan)
    q3 = q_decoded.round(3)
    q4 = q_decoded.round(4)
    q5 = q_decoded.round(5)
    q_frequency = q5.value_counts()
    q_centers = np.sort(q_frequency[q_frequency >= 30].index.to_numpy(float))
    q_values = q_decoded.to_numpy(float)
    q_nearest = np.full(len(raw), np.nan)
    q_distance = np.full(len(raw), np.nan)
    finite_q = np.isfinite(q_values)
    q_positions = np.searchsorted(q_centers, q_values[finite_q])
    q_low = np.clip(q_positions - 1, 0, len(q_centers) - 1)
    q_high = np.clip(q_positions, 0, len(q_centers) - 1)
    q_low_distance = np.abs(q_values[finite_q] - q_centers[q_low])
    q_high_distance = np.abs(q_values[finite_q] - q_centers[q_high])
    choose_high = q_high_distance < q_low_distance
    q_nearest[finite_q] = np.where(
        choose_high, q_centers[q_high], q_centers[q_low]
    )
    q_distance[finite_q] = np.where(
        choose_high, q_high_distance, q_low_distance
    )
    fee = np.where(cash, 129.5, 0.0)
    formula_residual = annuity.to_numpy() - (
        np.where(np.isfinite(q_nearest), q_nearest, q_values)
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
    kg_positions = np.searchsorted(kg_centers, kg_values[finite_kg])
    kg_low = np.clip(kg_positions - 1, 0, len(kg_centers) - 1)
    kg_high = np.clip(kg_positions, 0, len(kg_centers) - 1)
    kg_low_distance = np.abs(kg_values[finite_kg] - kg_centers[kg_low])
    kg_high_distance = np.abs(kg_values[finite_kg] - kg_centers[kg_high])
    choose_high = kg_high_distance < kg_low_distance
    kg_nearest[finite_kg] = np.where(
        choose_high, kg_centers[kg_high], kg_centers[kg_low]
    )
    kg_distance[finite_kg] = np.where(
        choose_high, kg_high_distance, kg_low_distance
    )

    formula = pd.DataFrame(
        {
            "Q_RAW": q_raw,
            "Q_DEC": q_decoded,
            "Q_RES3": (q_decoded - q3).abs(),
            "Q_RES4": (q_decoded - q4).abs(),
            "Q_RES5": (q_decoded - q5).abs(),
            "Q_NEAREST_DIST": q_distance,
            "Q_SUPPORT_LOG": np.log1p(q5.map(q_frequency).to_numpy(float)),
            "ANN_RESID": formula_residual,
            "ANN_RESID_ABS": np.abs(formula_residual),
            "Q_FEE_EFFECT": q_raw - q_decoded,
            "K_NEAREST_DIST": kg_distance,
            "K_NEAREST": kg_nearest,
            "K_SUPPORT_LOG": np.log1p(kg5.map(kg_frequency).to_numpy(float)),
        }
    )
    X = pd.concat([base.reset_index(drop=True), formula], axis=1).replace(
        [np.inf, -np.inf], np.nan
    )

    q4_key = "Q4:" + q4.fillna(-999).astype(str).to_numpy()
    q3_key = "Q3:" + q3.fillna(-999).astype(str).to_numpy()
    contract_key = (
        "CT:" + raw.NAME_CONTRACT_TYPE.fillna("__MISSING__").astype(str).to_numpy()
    )
    def combine_keys(left, right):
        return (
            pd.Series(left).astype(str) + "|" + pd.Series(right).astype(str)
        ).to_numpy(dtype=object)

    keys = {
        "PRODUCT_Q4": q4_key,
        "PRODUCT_Q3": q3_key,
        "PRODUCT_Q4_CONTRACT": combine_keys(q4_key, contract_key),
    }
    all_indices = np.arange(n_train + n_test)

    model_params = {
        "n_estimators": 400,
        "learning_rate": 0.03,
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

    seed_oofs = {}
    test_seed_predictions = []
    fold_results = []
    for seed in SEEDS:
        folds = list(
            StratifiedKFold(
                n_splits=FOLDS, shuffle=True, random_state=seed
            ).split(np.zeros(n_train), y)
        )
        oof = np.zeros(n_train, dtype=np.float32)
        seed_test = np.zeros(n_test, dtype=np.float64)
        for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
            te_fit, te_valid, te_test = build_te_frames(
                keys,
                fit_idx,
                valid_idx,
                y,
                inner_seed=seed + fold_number * 101,
                full_test_keys=all_indices[n_train:],
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
                    te_valid[
                        [f"{name}_{suffix}" for name in keys for suffix in ("TE", "LOGCOUNT")]
                    ].reset_index(drop=True),
                ],
                axis=1,
            )
            test_frame = pd.concat(
                [
                    X.iloc[n_train:].reset_index(drop=True),
                    te_test.reset_index(drop=True),
                ],
                axis=1,
            )
            model = lgb.LGBMClassifier(
                **model_params, random_state=seed + fold_number
            )
            model.fit(train_frame, y[fit_idx], categorical_feature="auto")
            valid_prediction = model.predict_proba(valid_frame)[:, 1]
            test_prediction = model.predict_proba(test_frame)[:, 1]
            oof[valid_idx] = valid_prediction
            seed_test += test_prediction / FOLDS
            fold_auc = float(roc_auc_score(y[valid_idx], valid_prediction))
            fold_results.append(
                {
                    "seed": seed,
                    "fold": fold_number,
                    "valid_auc": fold_auc,
                    "train_rows": len(fit_idx),
                    "valid_rows": len(valid_idx),
                }
            )
            print(
                f"seed={seed} fold={fold_number}/{FOLDS} "
                f"AUC={fold_auc:.6f}",
                flush=True,
            )
        seed_oofs[seed] = oof
        test_seed_predictions.append(seed_test)
        print(
            f"seed={seed} pooled OOF AUC={roc_auc_score(y, oof):.6f}",
            flush=True,
        )

    oof_prediction = np.mean(
        np.column_stack([seed_oofs[seed] for seed in SEEDS]), axis=1
    )
    test_prediction = np.mean(np.column_stack(test_seed_predictions), axis=1)
    pooled_auc = float(roc_auc_score(y, oof_prediction))
    fold_deltas = [
        item["valid_auc"]
        for item in fold_results
    ]
    test_rank = (rankdata(test_prediction, method="average") - 1) / (
        len(test_prediction) - 1
    )
    submission = sample.copy()
    submission["TARGET"] = test_rank
    output_path = OUT / f"{OUTPUT_PREFIX}_hierarchical_product.csv"
    submission.to_csv(output_path, index=False)

    reread = pd.read_csv(output_path)
    assert list(reread.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(reread) == len(sample) == n_test
    assert reread["SK_ID_CURR"].is_unique
    assert np.array_equal(
        reread["SK_ID_CURR"].to_numpy(), sample["SK_ID_CURR"].to_numpy()
    )
    assert np.isfinite(reread["TARGET"].to_numpy()).all()

    metadata = {
        "version": "v_2",
        "status": "standalone challenger; not matched against the v15 OOF",
        "submission": str(output_path),
        "submission_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": len(reread),
        "columns": list(reread.columns),
        "pooled_oof_auc_two_seed_average": pooled_auc,
        "seed_oof_auc": {
            str(seed): float(roc_auc_score(y, seed_oofs[seed])) for seed in SEEDS
        },
        "fold_results": fold_results,
        "all_fold_auc_min": float(min(fold_deltas)),
        "all_fold_auc_max": float(max(fold_deltas)),
        "feature_count": int(X.shape[1] + len(keys) * 2),
        "outer_seeds": list(SEEDS),
        "outer_folds": FOLDS,
        "inner_folds": INNER_FOLDS,
        "target_encoding_alpha": ALPHA,
        "lightgbm_version": lgb.__version__,
        "model_params": model_params,
        "runtime_seconds": round(time.time() - start, 1),
        "important_limit": (
            "The v15 submission and aligned OOF/member cache were unavailable. "
            "This OOF score is from a standalone compact challenger and must not "
            "be interpreted as an improvement over the public v15 score 0.77891."
        ),
        "oof_archive": str(OUT / f"{OUTPUT_PREFIX}_oof.npz"),
    }
    archive = {"y": y, "oof": oof_prediction, "test_prediction": test_prediction}
    for seed in SEEDS:
        archive[f"oof_seed_{seed}"] = seed_oofs[seed]
    for seed, prediction in zip(SEEDS, test_seed_predictions):
        archive[f"test_seed_{seed}"] = prediction
    np.savez_compressed(
        OUT / f"{OUTPUT_PREFIX}_oof.npz",
        **archive,
    )
    (OUT / f"{OUTPUT_PREFIX}_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
