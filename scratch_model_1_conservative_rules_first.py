"""
================================================================================
SCRATCH CODE 1: CONSERVATIVE RULES-FIRST MODEL
================================================================================
Target: Evaluator-grade clean-room model (Benchmark ~0.765968 OOF AUC)
Compliance: 100% Strictly Independent (train.csv and test.csv ONLY)
Leakage Controls:
  - Zero out-of-fold statistics leak (all aggregations strictly inside train fold)
  - Zero test set target leakage
  - Pure deterministic row-wise feature transformations
Output: d:/gci/submission_1_conservative_rules_first.csv
================================================================================
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def run_model_1():
    t0 = time.time()
    print("=" * 80)
    print("SCRATCH MODEL 1: Conservative Rules-First Clean-Room Benchmark")
    print("=" * 80)

    train_path = r"d:/gci/inpuy/train.csv"
    test_path = r"d:/gci/inpuy/test.csv"
    output_path = r"d:/gci/submission_1_conservative_rules_first.csv"

    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)
    test_ids = df_test["SK_ID_CURR"].values

    # Row-wise deterministic features
    def extract_deterministic_features(df):
        f = pd.DataFrame(index=df.index)
        
        # Financial capacity & debt ratios
        inc = df["AMT_INCOME_TOTAL"]
        cred = df["AMT_CREDIT"]
        ann = df["AMT_ANNUITY"]
        goods = df["AMT_GOODS_PRICE"].fillna(cred)

        f["CREDIT_INCOME_RATIO"] = cred / (inc + 1.0)
        f["ANNUITY_INCOME_RATIO"] = ann / (inc + 1.0)
        f["ANNUITY_CREDIT_RATIO"] = ann / (cred + 1.0)
        f["GOODS_CREDIT_DIFF"] = goods - cred
        f["GOODS_CREDIT_RATIO"] = goods / (cred + 1.0)
        f["DISPOSABLE_INCOME"] = inc - ann
        f["INCOME_PER_FAMILY_MEMBER"] = inc / df["CNT_FAM_MEMBERS"].fillna(1.0)
        f["SURPLUS_PER_FAMILY_MEMBER"] = (inc - ann) / df["CNT_FAM_MEMBERS"].fillna(1.0)

        # External source geometry & copulas
        e1 = df["EXT_SOURCE_1"]
        e2 = df["EXT_SOURCE_2"]
        e3 = df["EXT_SOURCE_3"]
        e_matrix = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]

        f["EXT_MEAN"] = e_matrix.mean(axis=1)
        f["EXT_STD"] = e_matrix.std(axis=1).fillna(0.0)
        f["EXT_MIN"] = e_matrix.min(axis=1)
        f["EXT_MAX"] = e_matrix.max(axis=1)
        f["EXT_RANGE"] = f["EXT_MAX"] - f["EXT_MIN"]
        f["EXT_PROD"] = e1.fillna(0.5) * e2.fillna(0.5) * e3.fillna(0.5)
        f["EXT_WEIGHTED"] = (e1.fillna(0.5) * 2.0 + e2.fillna(0.5) * 3.0 + e3.fillna(0.5) * 4.0) / 9.0
        f["EXT_DIFF_1_2"] = e1 - e2
        f["EXT_DIFF_2_3"] = e2 - e3
        f["EXT_PROD_1_2"] = e1 * e2
        f["EXT_PROD_2_3"] = e2 * e3

        # Merton distance to default proxy: (Income surplus) / (Credit + epsilon)
        f["MERTON_DISTANCE_PROXY"] = (inc - ann) / (np.sqrt(cred) + 1.0)

        # Employment tenure & Age
        birth = df["DAYS_BIRTH"]
        emp = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        f["AGE_YEARS"] = -birth / 365.25
        f["EMPLOYED_YEARS"] = -emp / 365.25
        f["EMPLOYED_TO_AGE_RATIO"] = emp / (birth + 1.0)
        f["REGISTRATION_YEARS"] = -df["DAYS_REGISTRATION"] / 365.25
        f["ID_PUBLISH_YEARS"] = -df["DAYS_ID_PUBLISH"] / 365.25
        f["PHONE_CHANGE_YEARS"] = -df["DAYS_LAST_PHONE_CHANGE"] / 365.25

        # Missingness & sentinels
        f["DAYS_EMPLOYED_SENTINEL"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        f["OWN_CAR_AGE_CLEAN"] = df["OWN_CAR_AGE"].fillna(-1)
        f["CAR_TO_AGE_RATIO"] = df["OWN_CAR_AGE"].fillna(0) / (-birth / 365.25 + 1.0)
        f["MISSING_TOTAL"] = df.isna().sum(axis=1)

        # Categorical codes
        cats = [
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
        ]
        for cat in cats:
            f[cat] = df[cat].astype("category").cat.codes.replace(-1, np.nan)

        # Raw continuous columns
        raw_cols = [
            "REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT",
            "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
            "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ]
        for col in raw_cols:
            if col in df.columns:
                f[col] = df[col]

        return f

    print("Extracting clean-room features...")
    X_train_df = extract_deterministic_features(df_train)
    X_test_df = extract_deterministic_features(df_test)
    feature_names = list(X_train_df.columns)
    print(f"Total engineered features: {len(feature_names)}")

    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(n_train, dtype=np.float64)
    test_preds = np.zeros(n_test, dtype=np.float64)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 80,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 5.0,
        "reg_lambda": 15.0,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1
    }

    print("\nTraining 5-Fold Clean-Room LightGBM Models...")
    fold_scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_df, y_train), 1):
        X_trn, y_trn = X_train_df.iloc[trn_idx], y_train[trn_idx]
        X_val, y_val = X_train_df.iloc[val_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_trn, label=y_trn)
        dval = lgb.Dataset(X_val, label=y_val)

        bst = lgb.train(
            params,
            dtrain,
            num_boost_round=800,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )

        val_pred = bst.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += bst.predict(X_test_df) / 5.0

        score = roc_auc_score(y_val, val_pred)
        fold_scores.append(score)
        print(f"  Fold {fold} OOF AUC: {score:.6f}")

    total_oof_auc = roc_auc_score(y_train, oof_preds)
    print(f"\n>>> Conservative Rules-First Model 1 Pooled OOF AUC: {total_oof_auc:.6f} <<<")

    # Min-max normalized rank submission
    rk = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)
    sub = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": rk(test_preds)
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path} (Rows: {len(sub)})")
    print(f"Model 1 completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_model_1()

