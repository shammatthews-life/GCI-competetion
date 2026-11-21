import hashlib
import json
import os
import time
from pathlib import Path

import catboost
import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(r"D:\gci")
DATA = ROOT / "inpuy"
OUT = ROOT / "codex"
CACHE = OUT / "v_3_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
OUTER_FOLDS = 5
THREADS = min(os.cpu_count() or 1, 8)


def logit(values):
    clipped = np.clip(values, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def rank(values):
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def build_views(x_train, x_test, q0, y):
    train = x_train.copy()
    test = x_test.copy()
    train["Q0num"] = q0[: len(train)]
    test["Q0num"] = q0[len(train) :]
    columns = list(train.columns)
    ext = [
        column
        for column in columns
        if "EXT" in column
        or column in ("E12", "E13", "E23", "D12", "D13", "D23")
    ]
    loan = [
        column
        for column in columns
        if column.startswith(
            ("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num")
        )
        or column
        in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")
    ]
    person = [
        column
        for column in columns
        if column not in set(ext) | set(loan)
        and not column.startswith(("G_", "CNT_"))
    ]
    views = {"EXT": ext, "LOAN": loan, "PERSON": person}

    def make_model():
        return HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=15,
            min_samples_leaf=150,
            l2_regularization=20,
            random_state=0,
        )

    def view_frame(values):
        frame = pd.DataFrame(values)
        frame["VD_EL"] = frame["VIEW_EXT"] - frame["VIEW_LOAN"]
        frame["VD_EP"] = frame["VIEW_EXT"] - frame["VIEW_PERSON"]
        frame["VD_LP"] = frame["VIEW_LOAN"] - frame["VIEW_PERSON"]
        return frame

    folds = list(
        StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y
        )
    )
    result = {"tr": [], "va": [], "te": []}
    started = time.time()
    for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
        train_views = {}
        valid_views = {}
        test_views = {}
        for view_name, view_columns in views.items():
            inner_oof = np.zeros(len(fit_idx), dtype=np.float64)
            valid_predictions = []
            test_predictions = []
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
            for inner_fit, inner_valid in inner.split(
                np.zeros(len(fit_idx)), y[fit_idx]
            ):
                model = make_model()
                model.fit(
                    train.iloc[fit_idx[inner_fit]][view_columns],
                    y[fit_idx[inner_fit]],
                )
                inner_oof[inner_valid] = model.predict_proba(
                    train.iloc[fit_idx[inner_valid]][view_columns]
                )[:, 1]
                valid_predictions.append(
                    model.predict_proba(train.iloc[valid_idx][view_columns])[:, 1]
                )
                test_predictions.append(
                    model.predict_proba(test[view_columns])[:, 1]
                )
            train_views[f"VIEW_{view_name}"] = logit(inner_oof)
            valid_views[f"VIEW_{view_name}"] = logit(
                np.mean(valid_predictions, axis=0)
            )
            test_views[f"VIEW_{view_name}"] = logit(
                np.mean(test_predictions, axis=0)
            )
        result["tr"].append(view_frame(train_views))
        result["va"].append(view_frame(valid_views))
        result["te"].append(view_frame(test_views))
        print(
            f"views fold {fold_number}/{OUTER_FOLDS} complete "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    train_views = {}
    test_views = {}
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
    for view_name, view_columns in views.items():
        inner_oof = np.zeros(len(y), dtype=np.float64)
        test_predictions = []
        for inner_fit, inner_valid in inner.split(np.zeros(len(y)), y):
            model = make_model()
            model.fit(train.iloc[inner_fit][view_columns], y[inner_fit])
            inner_oof[inner_valid] = model.predict_proba(
                train.iloc[inner_valid][view_columns]
            )[:, 1]
            test_predictions.append(model.predict_proba(test[view_columns])[:, 1])
        train_views[f"VIEW_{view_name}"] = logit(inner_oof)
        test_views[f"VIEW_{view_name}"] = logit(
            np.mean(test_predictions, axis=0)
        )
    result["all_tr"] = view_frame(train_views)
    result["all_te"] = view_frame(test_views)
    return result, folds


def main():
    started = time.time()
    train_csv = pd.read_csv(DATA / "train.csv")
    test_csv = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target = train_csv["TARGET"].to_numpy(dtype=np.int8)
    n_train = len(train_csv)
    n_test = len(test_csv)
    all_data = pd.concat([train_csv.drop(columns=["TARGET"]), test_csv], ignore_index=True)
    categorical_columns = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ]

    def new_columns(values):
        return pd.DataFrame(values, index=all_data.index)

    def base_features(frame):
        features = frame.drop(columns=["SK_ID_CURR"]).copy()
        for column in categorical_columns:
            features[column] = (
                features[column]
                .astype("category")
                .cat.codes.replace(-1, np.nan)
                .astype(float)
            )
        return features

    def clean_features(frame, features):
        values = {
            "DE_SENT": (frame.DAYS_EMPLOYED == 365243).astype(int),
            "DAYS_EMPLOYED_CLEAN": frame.DAYS_EMPLOYED.where(
                frame.DAYS_EMPLOYED != 365243
            ),
            "ORG_XNA": (frame.ORGANIZATION_TYPE == "XNA").astype(int),
            "CAR_AGE_OUT": (frame.OWN_CAR_AGE >= 60).astype(int),
            "MISS_CNT": frame.isna().sum(axis=1),
        }
        for column in [
            "EXT_SOURCE_1",
            "EXT_SOURCE_3",
            "OWN_CAR_AGE",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]:
            values[f"{column}_NA"] = frame[column].isna().astype(int)
        return pd.concat([features, new_columns(values)], axis=1)

    def external_features(frame, features):
        external = frame[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        maximum = external.max(axis=1)
        minimum = external.min(axis=1)
        values = {
            "EXT_MEAN": external.mean(axis=1),
            "EXT_MIN": minimum,
            "EXT_MAX": maximum,
            "EXT_STD": external.std(axis=1),
            "EXT_RANGE": maximum - minimum,
            "EXT_NOBS": external.notna().sum(axis=1),
            "EXT_PROD": external.prod(axis=1),
            "EXT_W": (
                frame.EXT_SOURCE_1 * 2
                + frame.EXT_SOURCE_2 * 3
                + frame.EXT_SOURCE_3 * 4
            )
            / 9,
            "E12": frame.EXT_SOURCE_1 * frame.EXT_SOURCE_2,
            "E13": frame.EXT_SOURCE_1 * frame.EXT_SOURCE_3,
            "E23": frame.EXT_SOURCE_2 * frame.EXT_SOURCE_3,
            "D12": frame.EXT_SOURCE_1 - frame.EXT_SOURCE_2,
            "D13": frame.EXT_SOURCE_1 - frame.EXT_SOURCE_3,
            "D23": frame.EXT_SOURCE_2 - frame.EXT_SOURCE_3,
        }
        return pd.concat([features, new_columns(values)], axis=1)

    def financial_features(frame, features):
        income = frame.AMT_INCOME_TOTAL
        credit = frame.AMT_CREDIT
        annuity = frame.AMT_ANNUITY
        goods = frame.AMT_GOODS_PRICE
        values = {
            "CREDIT_INC": credit / income,
            "ANN_INC": annuity / income,
            "ANN_CREDIT": annuity / credit,
            "TERM": credit / annuity,
            "GOODS_CREDIT": goods / credit,
            "CREDIT_MINUS_GOODS": credit - goods,
            "GOODS_INC": goods / income,
            "INC_PER_FAM": income / frame.CNT_FAM_MEMBERS,
            "CREDIT_PER_FAM": credit / frame.CNT_FAM_MEMBERS,
            "ANN_PER_FAM": annuity / frame.CNT_FAM_MEMBERS,
        }
        for column in [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
        ]:
            values[f"LOG_{column}"] = np.log1p(frame[column])
        return pd.concat([features, new_columns(values)], axis=1)

    def age_features(frame, features):
        age = -frame.DAYS_BIRTH
        employment = frame.DAYS_EMPLOYED.where(frame.DAYS_EMPLOYED != 365243)
        values = {
            "AGE": age / 365.25,
            "EMP_AGE": employment / frame.DAYS_BIRTH,
            "EMP_YEARS": -employment / 365.25,
            "REG_AGE": frame.DAYS_REGISTRATION / frame.DAYS_BIRTH,
            "ID_AGE": frame.DAYS_ID_PUBLISH / frame.DAYS_BIRTH,
            "PHONE_AGE": frame.DAYS_LAST_PHONE_CHANGE / frame.DAYS_BIRTH,
            "CAR_AGE_RATIO": frame.OWN_CAR_AGE / (age / 365.25),
            "CREDIT_PER_AGE": frame.AMT_CREDIT / (age / 365.25),
            "INC_PER_AGE": frame.AMT_INCOME_TOTAL / (age / 365.25),
        }
        return pd.concat([features, new_columns(values)], axis=1)

    def bureau_features(frame, features):
        bureau_columns = [
            column for column in frame.columns if column.startswith("AMT_REQ")
        ]
        values = {
            "BUR_SUM": frame[bureau_columns].sum(axis=1, min_count=1),
            "BUR_RECENT": frame[
                ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]
            ].sum(axis=1, min_count=1),
            "BUR_MON_YEAR": frame.AMT_REQ_CREDIT_BUREAU_MON
            / (frame.AMT_REQ_CREDIT_BUREAU_YEAR + 1),
        }
        return pd.concat([features, new_columns(values)], axis=1)

    def count_features(frame, features):
        columns = categorical_columns + [
            "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH",
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "AMT_ANNUITY",
            "AMT_INCOME_TOTAL",
        ]
        values = {
            f"CNT_{column}": frame[column].map(frame[column].value_counts())
            for column in columns
        }
        return pd.concat([features, new_columns(values)], axis=1)

    def group_features(frame, features):
        temp = frame.copy()
        temp["CREDIT_INC"] = frame.AMT_CREDIT / frame.AMT_INCOME_TOTAL
        values = {}
        for group in [
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "REGION_POPULATION_RELATIVE",
        ]:
            for column in [
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "DAYS_BIRTH",
                "CREDIT_INC",
            ]:
                values[f"G_{group[:6]}_{column[-6:]}_DIFF"] = temp[column] - temp.groupby(
                    group
                )[column].transform("mean")
        return pd.concat([features, new_columns(values)], axis=1)

    def decoded_goods_features(frame, features):
        credit = frame.AMT_CREDIT
        goods = frame.AMT_GOODS_PRICE
        cash = frame.NAME_CONTRACT_TYPE == "Cash loans"
        ratio = (credit / goods - 1) / 0.0132
        nearest = ratio.round()
        valid = cash & (credit > goods) & ((ratio - nearest).abs() < 0.1)
        decoded = pd.Series(
            np.where(valid, nearest, np.where(cash & (credit == goods), 0.0, np.nan)),
            index=frame.index,
        )
        return pd.concat(
            [
                features,
                new_columns({"K_DEC": decoded, "CG_RATIO": credit / goods}),
            ],
            axis=1,
        )

    families = {
        "kdec": decoded_goods_features,
        "clean": clean_features,
        "ext": external_features,
        "fin": financial_features,
        "age": age_features,
        "bureau": bureau_features,
        "count": count_features,
        "group": group_features,
    }

    def build_features(family_names):
        features = base_features(all_data)
        for family in family_names:
            features = families[family](all_data, features)
        features = features.replace([np.inf, -np.inf], np.nan)
        return (
            features.iloc[:n_train].reset_index(drop=True),
            features.iloc[n_train:].reset_index(drop=True),
        )

    core_train, core_test = build_features(["clean", "fin", "age", "bureau"])
    all_train, all_test = build_features(
        ["clean", "ext", "fin", "age", "bureau", "count", "group"]
    )
    allk_train, allk_test = build_features(
        ["clean", "ext", "fin", "age", "bureau", "count", "group", "kdec"]
    )
    base_frames = {
        "Xc": core_train,
        "Xct": core_test,
        "Xa": all_train,
        "Xat": all_test,
        "Xak": allk_train,
        "Xakt": allk_test,
    }

    q = all_data.AMT_ANNUITY / all_data.AMT_CREDIT
    cash = (all_data.NAME_CONTRACT_TYPE == "Cash loans").to_numpy()
    q_fee = (all_data.AMT_ANNUITY - 129.5) / all_data.AMT_CREDIT
    q5, q_fee5 = q.round(5), q_fee.round(5)
    fee = (
        (q_fee5.map(q_fee5.value_counts()).to_numpy() > q5.map(q5.value_counts()).to_numpy())
        & cash
    ).astype(int)
    q0 = pd.Series(np.where(fee, q_fee, q), index=all_data.index)
    keys = {
        "Q3": q.round(3).astype(str),
        "Q4": q.round(4).astype(str),
        "Q5": q.round(5).astype(str),
        "P3": q0.round(3).astype(str),
        "P4": q0.round(4).astype(str),
        "P5": q0.round(5).astype(str),
        "PF4": q0.round(4).astype(str) + "|" + pd.Series(fee).astype(str),
    }
    smoothing = {
        "Q3": 100.0,
        "Q4": 30.0,
        "Q5": 30.0,
        "P3": 100.0,
        "P4": 30.0,
        "P5": 30.0,
        "PF4": 30.0,
    }

    def encode(fit_idx, query_idx):
        prior = float(target[fit_idx].mean())
        encoded = {}
        for key_name, key_values in keys.items():
            stats = (
                pd.DataFrame(
                    {
                        "key": key_values.iloc[fit_idx].to_numpy(),
                        "target": target[fit_idx],
                    }
                )
                .groupby("key")["target"]
                .agg(["sum", "count"])
            )
            query = key_values.iloc[query_idx]
            count = query.map(stats["count"]).fillna(0).to_numpy(float)
            total = query.map(stats["sum"]).fillna(0).to_numpy(float)
            encoded[f"TE_{key_name}"] = (
                total + smoothing[key_name] * prior
            ) / (count + smoothing[key_name])
            encoded[f"CNT_{key_name}"] = count
        return pd.DataFrame(encoded)

    def train_encoding(fit_idx):
        columns = encode(fit_idx[:10], fit_idx[:5]).columns
        inner_values = np.zeros((len(fit_idx), len(columns)), dtype=np.float32)
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        for inner_fit, inner_hold in inner.split(
            fit_idx, target[fit_idx]
        ):
            inner_values[inner_hold] = encode(
                fit_idx[inner_fit], fit_idx[inner_hold]
            ).to_numpy()
        return pd.DataFrame(inner_values, columns=columns)

    def residual_features():
        residual_input = all_data.drop(columns=["SK_ID_CURR"]).copy()
        for column in categorical_columns:
            residual_input[column] = (
                residual_input[column]
                .astype("category")
                .cat.codes.replace(-1, np.nan)
                .astype(float)
            )
        residual_input["DAYS_EMPLOYED"] = residual_input.DAYS_EMPLOYED.where(
            residual_input.DAYS_EMPLOYED != 365243
        )
        amount_columns = [
            column
            for column in residual_input.columns
            if column.startswith("EXT_SOURCE") or column.startswith("AMT_")
        ]
        definitions = {
            "EXT1": (residual_input.EXT_SOURCE_1, ["EXT_SOURCE_1"]),
            "EXT2": (residual_input.EXT_SOURCE_2, ["EXT_SOURCE_2"]),
            "EXT3": (residual_input.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
            "LOGINC": (
                np.log(residual_input.AMT_INCOME_TOTAL),
                ["AMT_INCOME_TOTAL"],
            ),
            "LOGCR": (
                np.log(residual_input.AMT_CREDIT),
                ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"],
            ),
            "EMP": (residual_input.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]),
            "REG": (residual_input.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]),
            "AGE": (residual_input.DAYS_BIRTH, ["DAYS_BIRTH"]),
            "IDP": (residual_input.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]),
            "PHONE": (
                residual_input.DAYS_LAST_PHONE_CHANGE,
                ["DAYS_LAST_PHONE_CHANGE"],
            ),
            "REGPOP": (
                residual_input.REGION_POPULATION_RELATIVE,
                ["REGION_POPULATION_RELATIVE"],
            ),
            "FAM": (residual_input.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]),
            "CAR": (residual_input.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
            "EXT1D": (residual_input.EXT_SOURCE_1, amount_columns),
            "EXT2D": (residual_input.EXT_SOURCE_2, amount_columns),
            "EXT3D": (residual_input.EXT_SOURCE_3, amount_columns),
        }
        outputs = {}
        for name, (observed, drop_columns) in definitions.items():
            predictors = residual_input.drop(columns=drop_columns)
            predictions = np.zeros(len(residual_input), dtype=np.float32)
            observed_mask = observed.notna().to_numpy()
            folds = KFold(n_splits=5, shuffle=True, random_state=0)
            for fit_idx, predict_idx in folds.split(predictors):
                train_mask = observed_mask[fit_idx]
                model = HistGradientBoostingRegressor(
                    max_iter=120,
                    learning_rate=0.1,
                    max_leaf_nodes=31,
                    min_samples_leaf=100,
                    random_state=0,
                )
                model.fit(
                    predictors.iloc[fit_idx[train_mask]],
                    observed.iloc[fit_idx[train_mask]],
                )
                predictions[predict_idx] = model.predict(
                    predictors.iloc[predict_idx]
                )
            outputs[f"RP_{name}"] = predictions
            outputs[f"RR_{name}"] = np.where(
                observed_mask, observed.to_numpy() - predictions, np.nan
            )
            print(f"residual feature {name} complete", flush=True)
        return pd.DataFrame(outputs)

    residual_path = CACHE / "resid16.pkl"
    if residual_path.exists():
        residuals = pd.read_pickle(residual_path)
        if len(residuals) != len(all_data) or residuals.shape[1] != 32:
            raise ValueError("Cached residual feature dimensions do not match inputs")
        print("loaded residual features from cache", flush=True)
    else:
        residuals = residual_features()
        residuals.to_pickle(residual_path)

    x_all_train = pd.concat(
        [base_frames["Xa"], residuals.iloc[:n_train].reset_index(drop=True)],
        axis=1,
    )
    x_all_test = pd.concat(
        [base_frames["Xat"], residuals.iloc[n_train:].reset_index(drop=True)],
        axis=1,
    )
    views_path = CACHE / "e1A_views.pkl"
    if views_path.exists():
        views = pd.read_pickle(views_path)
        if len(views["tr"]) != OUTER_FOLDS:
            raise ValueError("Cached view folds do not match the experiment")
        print("loaded consistent views from cache", flush=True)
    else:
        views, outer_splits = build_views(
            x_all_train, x_all_test, q0.to_numpy(), target
        )
        pd.to_pickle(views, views_path)

    numeric = pd.concat(
        [base_frames["Xa"], base_frames["Xat"]], ignore_index=True
    ).drop(columns=categorical_columns)
    categorical = all_data[categorical_columns].copy().reset_index(drop=True)
    for column in categorical_columns:
        categorical[column] = (
            categorical[column].astype("string").fillna("__MISSING__").astype(str)
        )
    product = pd.concat([numeric, categorical], axis=1)
    product_names = []
    for key_name in ["Q3", "Q4", "Q5", "P3", "P4", "P5", "PF4"]:
        product[f"PROD_{key_name}"] = (
            keys[key_name]
            .reset_index(drop=True)
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
        product_names.append(f"PROD_{key_name}")
    decoded_k = pd.concat(
        [base_frames["Xak"]["K_DEC"], base_frames["Xakt"]["K_DEC"]],
        ignore_index=True,
    )
    product["PROD_K_DEC"] = (
        decoded_k.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str)
    )
    product_names.append("PROD_K_DEC")
    product["PROD_FEE"] = pd.Series(fee).reset_index(drop=True).astype(str)
    product_names.append("PROD_FEE")
    cat_features = categorical_columns + product_names
    cat_train = pd.concat(
        [
            product.iloc[:n_train].reset_index(drop=True),
            residuals.iloc[:n_train].reset_index(drop=True),
        ],
        axis=1,
    )
    cat_test = pd.concat(
        [
            product.iloc[n_train:].reset_index(drop=True),
            residuals.iloc[n_train:].reset_index(drop=True),
        ],
        axis=1,
    )

    folds = list(
        StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42).split(
            np.zeros(n_train), target
        )
    )
    oof = np.zeros(n_train, dtype=np.float64)
    test_folds = []
    fold_metrics = []
    params = {
        "iterations": 900,
        "depth": 7,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 8,
        "random_seed": 42,
        "thread_count": THREADS,
        "verbose": 100,
        "allow_writing_files": False,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
    }
    for fold_number, (fit_idx, valid_idx) in enumerate(folds, start=1):
        fit_frame = pd.concat(
            [
                cat_train.iloc[fit_idx].reset_index(drop=True),
                views["tr"][fold_number - 1].reset_index(drop=True),
            ],
            axis=1,
        )
        valid_frame = pd.concat(
            [
                cat_train.iloc[valid_idx].reset_index(drop=True),
                views["va"][fold_number - 1].reset_index(drop=True),
            ],
            axis=1,
        )
        test_frame = pd.concat(
            [
                cat_test.reset_index(drop=True),
                views["te"][fold_number - 1].reset_index(drop=True),
            ],
            axis=1,
        )
        model = CatBoostClassifier(**params)
        model.fit(fit_frame, target[fit_idx], cat_features=cat_features)
        oof[valid_idx] = model.predict_proba(valid_frame)[:, 1]
        test_folds.append(model.predict_proba(test_frame)[:, 1])
        auc = float(roc_auc_score(target[valid_idx], oof[valid_idx]))
        fold_metrics.append(
            {
                "fold": fold_number,
                "valid_auc": auc,
                "train_rows": len(fit_idx),
                "valid_rows": len(valid_idx),
            }
        )
        print(f"CatBoost-E1 fold {fold_number}/5 AUC={auc:.6f}", flush=True)

    pooled_auc = float(roc_auc_score(target, oof))
    test_probability = np.mean(test_folds, axis=0)
    submission = sample.copy()
    submission["TARGET"] = rank(test_probability)
    output_path = OUT / "v_3_standalone_v17_catboost_e1.csv"
    submission.to_csv(output_path, index=False)
    checked = pd.read_csv(output_path)
    if (
        checked.shape != (n_test, 2)
        or checked.columns.tolist() != ["SK_ID_CURR", "TARGET"]
        or not checked["SK_ID_CURR"].is_unique
        or not np.array_equal(checked["SK_ID_CURR"], sample["SK_ID_CURR"])
        or not np.isfinite(checked["TARGET"]).all()
    ):
        raise ValueError("Submission validation failed")

    metrics = {
        "version": "v_3",
        "description": "standalone CatBoost-E1 probe from v17 research",
        "submission": str(output_path),
        "submission_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "rows": len(checked),
        "columns": checked.columns.tolist(),
        "pooled_oof_auc": pooled_auc,
        "fold_metrics": fold_metrics,
        "outer_folds": OUTER_FOLDS,
        "outer_seed": 42,
        "feature_count": int(cat_train.shape[1] + views["tr"][0].shape[1]),
        "catboost_version": catboost.__version__,
        "lightgbm_version": lightgbm.__version__,
        "catboost_params": params,
        "runtime_seconds": round(time.time() - started, 1),
        "warning": (
            "Standalone branch only. The v17 matched full-ensemble comparison "
            "reported no accepted improvement; no v15 blend was possible because "
            "the v15 prediction/cache bundle is missing."
        ),
    }
    np.savez_compressed(
        OUT / "v_3_standalone_v17_oof.npz",
        y=target,
        oof=oof,
        test_probability=test_probability,
    )
    (OUT / "v_3_standalone_v17_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
