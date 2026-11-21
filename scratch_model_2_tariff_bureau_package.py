"""
================================================================================
SCRATCH CODE 2: TARIFF & BUREAU RESEARCH PACKAGE
================================================================================
Target: Tariff & Bureau Feature Package (Seeds 20261029 & 20261030)
Features:
  - Tariff Fee-Gap and Quantized Interest Lattice: (AMT_ANNUITY - 130.5) / AMT_CREDIT
  - Quantized Ratio Keys: Q3, Q4, Q5
  - Cash-Loan Lattice step proximity: ((CREDIT / GOODS) - 1.0) / 0.0066
  - Multi-seed validation: Seed 20261029 & Seed 20261030
Output: d:/gci/submission_2_tariff_bureau_package.csv
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

def run_model_2():
    t0 = time.time()
    print("=" * 80)
    print("SCRATCH MODEL 2: Tariff & Bureau Research Package (Seeds 20261029 & 20261030)")
    print("=" * 80)

    train_path = r"d:/gci/inpuy/train.csv"
    test_path = r"d:/gci/inpuy/test.csv"
    output_path = r"d:/gci/submission_2_tariff_bureau_package.csv"

    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)
    test_ids = df_test["SK_ID_CURR"].values

    def extract_tariff_bureau_features(df):
        f = pd.DataFrame(index=df.index)

        cred = df["AMT_CREDIT"]
        ann = df["AMT_ANNUITY"]
        inc = df["AMT_INCOME_TOTAL"]
        goods = df["AMT_GOODS_PRICE"].fillna(cred)

        # 1. TARIFF RESEARCH PACKAGE: Fee gap and interest quantization
        f["FEE_ADJUSTED_ANN_CREDIT"] = (ann - 130.5) / (cred + 1e-5)
        raw_ratio = ann / (cred + 1e-5)
        f["RAW_ANN_CREDIT_RATIO"] = raw_ratio
        f["TARIFF_Q3"] = np.round(raw_ratio, 3)
        f["TARIFF_Q4"] = np.round(raw_ratio, 4)
        f["TARIFF_Q5"] = np.round(raw_ratio, 5)

        # Cash-loan lattice clustering at step = 0.0066
        lattice_val = ((cred / (goods + 1e-5)) - 1.0) / 0.0066
        f["LATTICE_STEP_RAW"] = lattice_val
        f["LATTICE_INTEGER_PROXIMITY"] = np.abs(lattice_val - np.round(lattice_val))
        f["LATTICE_IS_STEP_HIT"] = (f["LATTICE_INTEGER_PROXIMITY"] < 0.05).astype(int)

        # 2. BUREAU RATIO PACKAGE & INQUIRY URGENCY
        f["BUREAU_YEAR"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0)
        f["BUREAU_QRT"] = df["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0)
        f["BUREAU_MON"] = df["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0)
        f["BUREAU_HOUR"] = df["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0)

        f["BUREAU_QRT_TO_YEAR"] = f["BUREAU_QRT"] / (f["BUREAU_YEAR"] + 1.0)
        f["BUREAU_MON_TO_QRT"] = f["BUREAU_MON"] / (f["BUREAU_QRT"] + 1.0)
        f["BUREAU_RECENT_ACCELERATION"] = (f["BUREAU_MON"] * 3.0 + f["BUREAU_QRT"]) / (f["BUREAU_YEAR"] + 1.0)

        # 3. BASELINE COVARIATES
        f["CR_INC"] = cred / (inc + 1.0)
        f["ANN_INC"] = ann / (inc + 1.0)
        f["DISPOSABLE"] = inc - ann
        birth = df["DAYS_BIRTH"]
        emp = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        f["AGE_Y"] = -birth / 365.25
        f["EMP_Y"] = -emp / 365.25
        f["EMP_AGE_R"] = emp / (birth + 1.0)
        f["DE_SENT"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)

        # External sources
        e1 = df["EXT_SOURCE_1"]
        e2 = df["EXT_SOURCE_2"]
        e3 = df["EXT_SOURCE_3"]
        f["EXT_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        f["EXT_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1).fillna(0.0)
        f["EXT_W"] = (e1.fillna(0.5)*2.0 + e2.fillna(0.5)*3.0 + e3.fillna(0.5)*4.0) / 9.0
        f["EXT1"] = e1
        f["EXT2"] = e2
        f["EXT3"] = e3

        cats = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
        for cat in cats:
            f[cat] = df[cat].astype("category").cat.codes.replace(-1, np.nan)

        return f

    print("Extracting tariff & bureau features...")
    X_train_df = extract_tariff_bureau_features(df_train)
    X_test_df = extract_tariff_bureau_features(df_test)
    print(f"Total features in package: {X_train_df.shape[1]}")

    # Multi-seed CV: Seed 20261029 and Seed 20261030
    seeds = [20261029, 20261030]
    seed_oof_preds = []
    seed_test_preds = []

    rk = lambda s: (rankdata(s) - 1.0) / (len(s) - 1.0)

    for seed in seeds:
        print(f"\n--- Running 5-Fold Stratified CV with Seed {seed} ---")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_seed = np.zeros(n_train, dtype=np.float64)
        test_seed = np.zeros(n_test, dtype=np.float64)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.04,
            "num_leaves": 28,
            "min_child_samples": 90,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.65,
            "reg_alpha": 3.0,
            "reg_lambda": 12.0,
            "random_state": seed,
            "n_jobs": 4,
            "verbose": -1
        }

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_df, y_train), 1):
            X_trn, y_trn = X_train_df.iloc[trn_idx], y_train[trn_idx]
            X_val, y_val = X_train_df.iloc[val_idx], y_train[val_idx]

            dtrain = lgb.Dataset(X_trn, label=y_trn)
            dval = lgb.Dataset(X_val, label=y_val)

            bst = lgb.train(
                params,
                dtrain,
                num_boost_round=750,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)]
            )

            val_pred = bst.predict(X_val)
            oof_seed[val_idx] = val_pred
            test_seed += bst.predict(X_test_df) / 5.0

        seed_auc = roc_auc_score(y_train, oof_seed)
        print(f"[*] Seed {seed} Final OOF AUC: {seed_auc:.6f}")
        seed_oof_preds.append(oof_seed)
        seed_test_preds.append(test_seed)

    # Blend multi-seed results
    final_oof = 0.5 * rk(seed_oof_preds[0]) + 0.5 * rk(seed_oof_preds[1])
    final_test = 0.5 * rk(seed_test_preds[0]) + 0.5 * rk(seed_test_preds[1])
    combined_auc = roc_auc_score(y_train, final_oof)
    print(f"\n>>> Combined Tariff/Bureau Multi-Seed Pooled OOF AUC: {combined_auc:.6f} <<<")

    sub = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": rk(final_test)
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path} (Rows: {len(sub)})")
    print(f"Model 2 completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_model_2()

