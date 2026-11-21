"""
================================================================================
SCRATCH CODE 3: V17 MULTI-VIEW 5-STREAM SUITE
================================================================================
Target: v17 Multi-View Pipeline (Historical OOF AUC 0.77646, LB 0.78237)
Compliance: 100% Strictly Independent (train.csv and test.csv ONLY)
Architecture:
  - Stream 1: Base LightGBM GBDT Suite
  - Stream 2: Enhanced E1 Residual Suite (demographic expectation residuals)
  - Stream 3: ExtraTrees E1 LightGBM Suite
  - Stream 4: E1A ExtraTrees LightGBM Specialist
  - Stream 5: CatBoost / GBDT Product Specialist
  - Stream 6: Canonical Multi-View Grand Fusion:
              0.35 * e1Aet + 0.25 * e1et + 0.20 * e1 + 0.10 * old + 0.10 * cb
Output: d:/gci/submission_3_v17_multiview.csv
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
from catboost import CatBoostClassifier

def run_model_3():
    t0 = time.time()
    print("=" * 80)
    print("SCRATCH MODEL 3: V17 Multi-View 5-Stream Suite (from scratch)")
    print("=" * 80)

    train_path = r"d:/gci/inpuy/train.csv"
    test_path = r"d:/gci/inpuy/test.csv"
    output_path = r"d:/gci/submission_3_v17_multiview.csv"

    print(f"Loading data from {train_path} and {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    y_train = df_train["TARGET"].values.astype(np.int32)
    n_train = len(df_train)
    n_test = len(df_test)
    test_ids = df_test["SK_ID_CURR"].values

    CATS = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
    ]

    al = pd.concat([df_train.drop(columns=["TARGET"]), df_test], ignore_index=True)

    def new_cols(d):
        return pd.DataFrame(d, index=al.index)

    # 1. Base feature encoding
    X_base = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS:
        X_base[c] = X_base[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)

    # 2. Cleaning & anomaly flags
    d_clean = {
        "DE_SENT": (al.DAYS_EMPLOYED == 365243).astype(int),
        "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
        "ORG_XNA": (al.ORGANIZATION_TYPE == "XNA").astype(int),
        "CAR_AGE_OUT": (al.OWN_CAR_AGE >= 60).astype(int),
        "MISS_CNT": al.isna().sum(axis=1)
    }
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        d_clean[c + "_NA"] = al[c].isna().astype(int)
    f_clean = new_cols(d_clean)

    # 3. External source interactions
    e = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    d_ext = {
        "EXT_MEAN": e.mean(axis=1),
        "EXT_MIN": e.min(axis=1),
        "EXT_MAX": e.max(axis=1),
        "EXT_STD": e.std(axis=1),
        "EXT_RANGE": e.max(axis=1) - e.min(axis=1),
        "EXT_NOBS": e.notna().sum(axis=1),
        "EXT_PROD": e.prod(axis=1),
        "EXT_W": (al.EXT_SOURCE_1 * 2 + al.EXT_SOURCE_2 * 3 + al.EXT_SOURCE_3 * 4) / 9.0,
        "E12": al.EXT_SOURCE_1 * al.EXT_SOURCE_2,
        "E13": al.EXT_SOURCE_1 * al.EXT_SOURCE_3,
        "E23": al.EXT_SOURCE_2 * al.EXT_SOURCE_3,
        "D12": al.EXT_SOURCE_1 - al.EXT_SOURCE_2,
        "D13": al.EXT_SOURCE_1 - al.EXT_SOURCE_3,
        "D23": al.EXT_SOURCE_2 - al.EXT_SOURCE_3
    }
    f_ext = new_cols(d_ext)

    # 4. Financial ratios
    inc = al.AMT_INCOME_TOTAL
    cred = al.AMT_CREDIT
    ann = al.AMT_ANNUITY
    goods = al.AMT_GOODS_PRICE.fillna(cred)

    d_fin = {
        "CR_INC": cred / (inc + 1e-5),
        "ANN_INC": ann / (inc + 1e-5),
        "CR_GOODS": cred / (goods + 1e-5),
        "GOODS_DIFF": goods - cred,
        "ANN_CR": ann / (cred + 1e-5),
        "DISPOSABLE": inc - ann,
        "INC_PER_CAP": inc / al.CNT_FAM_MEMBERS.fillna(1),
        "INC_SURPLUS_PER_CAP": (inc - ann) / al.CNT_FAM_MEMBERS.fillna(1)
    }
    f_fin = new_cols(d_fin)

    # 5. Age & tenure
    birth = al.DAYS_BIRTH
    emp = f_clean.DAYS_EMPLOYED_CLEAN
    d_age = {
        "AGE_Y": -birth / 365.25,
        "EMP_Y": -emp / 365.25,
        "EMP_AGE_R": emp / (birth + 1e-5),
        "REG_Y": -al.DAYS_REGISTRATION / 365.25,
        "ID_Y": -al.DAYS_ID_PUBLISH / 365.25,
        "PHONE_Y": -al.DAYS_LAST_PHONE_CHANGE / 365.25,
        "CAR_EMP_R": al.OWN_CAR_AGE / (-emp / 365.25 + 1e-5),
        "CAR_AGE_R": al.OWN_CAR_AGE / (-birth / 365.25 + 1e-5)
    }
    f_age = new_cols(d_age)

    # 6. Cohort residuals
    groups = [
        ["NAME_EDUCATION_TYPE", "OCCUPATION_TYPE"],
        ["NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE"],
        ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
    ]
    f_grp_list = []
    for grp in groups:
        for target in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            grp_name = "_".join(grp)
            m = al.groupby(grp)[target].transform("mean")
            s = al.groupby(grp)[target].transform("std").fillna(1)
            f_grp_list.append(new_cols({
                f"{target}_MEAN_{grp_name}": m,
                f"{target}_RESID_{grp_name}": al[target] - m,
                f"{target}_Z_{grp_name}": (al[target] - m) / (s + 1e-5)
            }))
    f_grp = pd.concat(f_grp_list, axis=1)

    # Construct the Manifolds
    X_old = pd.concat([X_base, f_clean, f_ext, f_fin, f_age], axis=1)
    X_e1 = pd.concat([X_old, f_grp], axis=1)

    d_e1a = {
        "E1A_RATIO_CR": cred / (goods + 1e-5),
        "E1A_PAY_RATIO": ann / (cred + 1e-5),
        "E1A_EXT_AGG": (al.EXT_SOURCE_1.fillna(0) + al.EXT_SOURCE_2.fillna(0) + al.EXT_SOURCE_3.fillna(0)) / 3.0
    }
    f_e1a = new_cols(d_e1a)
    X_e1A = pd.concat([X_e1, f_e1a], axis=1)

    # CatBoost frame
    cb_cols = [c for c in al.columns if c != "TARGET"]
    cb_frame = al[cb_cols].copy()
    cb_cats = [c for c in CATS if c in cb_frame.columns]
    for c in cb_cats:
        cb_frame[c] = cb_frame[c].fillna("MISSING").astype(str)

    X_old_tr = X_old.iloc[:n_train].reset_index(drop=True)
    X_old_te = X_old.iloc[n_train:].reset_index(drop=True)
    X_e1_tr = X_e1.iloc[:n_train].reset_index(drop=True)
    X_e1_te = X_e1.iloc[n_train:].reset_index(drop=True)
    X_e1A_tr = X_e1A.iloc[:n_train].reset_index(drop=True)
    X_e1A_te = X_e1A.iloc[n_train:].reset_index(drop=True)
    cb_tr = cb_frame.iloc[:n_train].reset_index(drop=True)
    cb_te = cb_frame.iloc[n_train:].reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(n_train), y_train))
    rk = lambda v: (rankdata(v) - 0.5) / len(v)

    # Stream 1: Base LightGBM
    print("\n[Stream 1/5] Training Base LightGBM...")
    oof_old = np.zeros(n_train, dtype=np.float32)
    test_old = np.zeros(n_test, dtype=np.float64)
    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_old_tr.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_old_tr.iloc[val_idx], label=y_train[val_idx])
        params = {"objective": "binary", "learning_rate": 0.04, "num_leaves": 15, "min_child_samples": 150,
                  "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 20,
                  "n_jobs": 4, "verbose": -1, "seed": 42 + fold}
        bst = lgb.train(params, dtrain, num_boost_round=500, valid_sets=[dval], callbacks=[lgb.early_stopping(25, verbose=False)])
        oof_old[val_idx] = bst.predict(X_old_tr.iloc[val_idx])
        test_old += bst.predict(X_old_te) / 5.0
    print(f"  -> Stream 1 OOF AUC: {roc_auc_score(y_train, oof_old):.6f}")

    # Stream 2: Enhanced E1
    print("\n[Stream 2/5] Training Enhanced E1 Residual LightGBM...")
    oof_e1 = np.zeros(n_train, dtype=np.float32)
    test_e1 = np.zeros(n_test, dtype=np.float64)
    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_e1_tr.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_e1_tr.iloc[val_idx], label=y_train[val_idx])
        params = {"objective": "binary", "learning_rate": 0.035, "num_leaves": 23, "min_child_samples": 120,
                  "feature_fraction": 0.45, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 15,
                  "n_jobs": 4, "verbose": -1, "seed": 52 + fold}
        bst = lgb.train(params, dtrain, num_boost_round=550, valid_sets=[dval], callbacks=[lgb.early_stopping(25, verbose=False)])
        oof_e1[val_idx] = bst.predict(X_e1_tr.iloc[val_idx])
        test_e1 += bst.predict(X_e1_te) / 5.0
    print(f"  -> Stream 2 OOF AUC: {roc_auc_score(y_train, oof_e1):.6f}")

    # Stream 3: ExtraTrees E1
    print("\n[Stream 3/5] Training ExtraTrees E1 LightGBM...")
    oof_e1et = np.zeros(n_train, dtype=np.float32)
    test_e1et = np.zeros(n_test, dtype=np.float64)
    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_e1_tr.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_e1_tr.iloc[val_idx], label=y_train[val_idx])
        params = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 80,
                  "feature_fraction": 0.35, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 25,
                  "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 62 + fold}
        bst = lgb.train(params, dtrain, num_boost_round=600, valid_sets=[dval], callbacks=[lgb.early_stopping(25, verbose=False)])
        oof_e1et[val_idx] = bst.predict(X_e1_tr.iloc[val_idx])
        test_e1et += bst.predict(X_e1_te) / 5.0
    print(f"  -> Stream 3 OOF AUC: {roc_auc_score(y_train, oof_e1et):.6f}")

    # Stream 4: E1A ExtraTrees
    print("\n[Stream 4/5] Training E1A ExtraTrees Specialist...")
    oof_e1Aet = np.zeros(n_train, dtype=np.float32)
    test_e1Aet = np.zeros(n_test, dtype=np.float64)
    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        dtrain = lgb.Dataset(X_e1A_tr.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_e1A_tr.iloc[val_idx], label=y_train[val_idx])
        params = {"objective": "binary", "learning_rate": 0.025, "num_leaves": 45, "min_child_samples": 60,
                  "feature_fraction": 0.3, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 30,
                  "extra_trees": True, "n_jobs": 4, "verbose": -1, "seed": 72 + fold}
        bst = lgb.train(params, dtrain, num_boost_round=650, valid_sets=[dval], callbacks=[lgb.early_stopping(25, verbose=False)])
        oof_e1Aet[val_idx] = bst.predict(X_e1A_tr.iloc[val_idx])
        test_e1Aet += bst.predict(X_e1A_te) / 5.0
    print(f"  -> Stream 4 OOF AUC: {roc_auc_score(y_train, oof_e1Aet):.6f}")

    # Stream 5: CatBoost Product Specialist
    print("\n[Stream 5/5] Training CatBoost Product Specialist...")
    oof_cb = np.zeros(n_train, dtype=np.float32)
    test_cb = np.zeros(n_test, dtype=np.float32)
    for fold, (trn_idx, val_idx) in enumerate(folds, 1):
        cb_m = CatBoostClassifier(iterations=150, depth=6, learning_rate=0.06, loss_function="Logloss",
                                  eval_metric="AUC", l2_leaf_reg=6.0, random_seed=82 + fold, task_type="CPU",
                                  thread_count=4, verbose=False)
        cb_m.fit(cb_tr.iloc[trn_idx], y_train[trn_idx], cat_features=cb_cats,
                 eval_set=(cb_tr.iloc[val_idx], y_train[val_idx]), early_stopping_rounds=25, verbose=False)
        oof_cb[val_idx] = cb_m.predict_proba(cb_tr.iloc[val_idx])[:, 1]
        test_cb += cb_m.predict_proba(cb_te)[:, 1].astype(np.float32) / 5.0
    print(f"  -> Stream 5 OOF AUC: {roc_auc_score(y_train, oof_cb):.6f}")

    # Stream 6: Canonical Fusion
    print("\n[Stream 6] Computing Canonical Multi-View Grand Fusion...")
    final_oof = rk(0.35 * rk(oof_e1Aet) + 0.25 * rk(oof_e1et) + 0.20 * rk(oof_e1) + 0.10 * rk(oof_old) + 0.10 * rk(oof_cb))
    final_test = rk(0.35 * rk(test_e1Aet) + 0.25 * rk(test_e1et) + 0.20 * rk(test_e1) + 0.10 * rk(test_old) + 0.10 * rk(test_cb))
    fusion_auc = roc_auc_score(y_train, final_oof)
    print(f"\n>>> V17 Multi-View Grand Fusion Pooled OOF AUC: {fusion_auc:.6f} <<<")

    sub = pd.DataFrame({
        "SK_ID_CURR": test_ids,
        "TARGET": final_test
    })
    sub.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path} (Rows: {len(sub)})")
    print(f"Model 3 completed in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    run_model_3()

