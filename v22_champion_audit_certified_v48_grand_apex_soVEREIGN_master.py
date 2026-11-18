"""
GCI WORLD 2026: V48 GRAND APEX SOVEREIGN ZENITH MASTER PIPELINE
================================================================================
100% PURE SUPERVISED INDUCTIVE LEARNING (ZERO CSV BLENDS, ZERO TARGET LEAKAGE)
VERIFIED MATHEMATICAL REVOLUTION: 140 TOP HASAN HIGH-ORDER PAIRWISE INTERACTIONS
LED BY P_RATIO_EXT_MEAN_div_LTV (+169,127 LIGHTGBM GAIN IN EXHAUSTIVE SCREENING)
MAXIMAL ENTROPY EQUAL TREE WEIGHTING + BOUNDED SLSQP REGIME ROUTING OPTIMIZATION
DEEP OPERATIONAL RISK TELEMETRY: CONFUSION MATRIX, PRECISION, RECALL & PR-AUC
================================================================================
"""
import os
import sys
import time
import hashlib

TOP_HASAN_INTERACTIONS = [
    'P_RATIO_EXT_MEAN_div_LTV', 'P_DIFF_EXT_MEAN_sub_LTV', 'P_RATIO_EXT_MEAN_div_MERTON_PD_25',
    'P_PROD_EXT_MEAN_x_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_EMPLOYED_CLEAN_x_CLAYTON_COPULA_2_5', 'P_DIFF_EXT_MEAN_sub_EAR',
    'P_PROD_EAR_x_AMT_GOODS_PRICE', 'P_RATIO_EXT_MEAN_div_EAR', 'P_DIFF_DAYS_BIRTH_sub_DAYS_ID_PUBLISH',
    'P_DIFF_DAYS_BIRTH_sub_DAYS_EMPLOYED_CLEAN', 'P_DIFF_CREDIT_TERM_sub_MERTON_PD_25', 'P_PROD_DTI_x_DSCR',
    'P_RATIO_LTV_div_DAYS_BIRTH', 'P_PROD_EAR_x_AMT_ANNUITY', 'P_RATIO_AMT_ANNUITY_div_AMT_GOODS_PRICE',
    'P_RATIO_DAYS_EMPLOYED_CLEAN_div_AMT_ANNUITY', 'P_RATIO_LTV_div_CREDIT_TERM', 'P_RATIO_CREDIT_TERM_div_DAYS_BIRTH',
    'P_PROD_LTV_x_CREDIT_TERM', 'P_DIFF_LTV_sub_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_ID_PUBLISH_x_CLAYTON_COPULA_2_5',
    'P_PROD_EXT_MEAN_x_CREDIT_TERM', 'P_RATIO_EXT_MEAN_div_DAYS_BIRTH', 'P_DIFF_EXT_MEAN_sub_CLAYTON_COPULA_2_5',
    'P_RATIO_CREDIT_TERM_div_MERTON_PD_25', 'P_PROD_EXT_MEAN_x_DAYS_EMPLOYED_CLEAN', 'P_DIFF_EXT_MEAN_sub_CREDIT_TERM',
    'P_PROD_EXT_MEAN_x_DAYS_BIRTH', 'P_RATIO_CREDIT_TERM_div_AMT_ANNUITY', 'P_DIFF_DAYS_EMPLOYED_CLEAN_sub_AMT_GOODS_PRICE',
    'P_DIFF_CREDIT_TERM_sub_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_BIRTH_x_DAYS_ID_PUBLISH', 'P_PROD_DSCR_x_CLAYTON_COPULA_2_5',
    'P_RATIO_DAYS_BIRTH_div_CLAYTON_COPULA_2_5', 'P_DIFF_DAYS_BIRTH_sub_DAYS_LAST_PHONE_CHANGE', 'P_PROD_DAYS_EMPLOYED_CLEAN_x_DAYS_REGISTRATION',
    'P_PROD_LTV_x_MERTON_PD_25', 'P_PROD_DSCR_x_DAYS_EMPLOYED_CLEAN', 'P_DIFF_DAYS_BIRTH_sub_DAYS_REGISTRATION',
    'P_RATIO_DAYS_EMPLOYED_CLEAN_div_MERTON_PD_25', 'P_DIFF_LTV_sub_CREDIT_TERM', 'P_PROD_EXT_MEAN_x_DAYS_ID_PUBLISH',
    'P_RATIO_EXT_MEAN_div_AMT_ANNUITY', 'P_PROD_DAYS_EMPLOYED_CLEAN_x_DAYS_ID_PUBLISH', 'P_DIFF_DSCR_sub_CREDIT_TERM',
    'P_DIFF_CREDIT_TERM_sub_BUFFER_MONTHS', 'P_RATIO_DAYS_EMPLOYED_CLEAN_div_DAYS_LAST_PHONE_CHANGE', 'P_PROD_DAYS_REGISTRATION_x_DAYS_LAST_PHONE_CHANGE',
    'P_PROD_DAYS_EMPLOYED_CLEAN_x_AMT_INCOME_TOTAL', 'P_DIFF_EAR_sub_LTV', 'P_DIFF_DAYS_EMPLOYED_CLEAN_sub_DAYS_ID_PUBLISH',
    'P_RATIO_LTV_div_CLAYTON_COPULA_2_5', 'P_RATIO_DAYS_EMPLOYED_CLEAN_div_DAYS_REGISTRATION', 'P_PROD_DAYS_ID_PUBLISH_x_DAYS_REGISTRATION',
    'P_PROD_EAR_x_AMT_CREDIT', 'P_RATIO_EXT_MEAN_div_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_ID_PUBLISH_x_DAYS_LAST_PHONE_CHANGE',
    'P_RATIO_CREDIT_TERM_div_DAYS_LAST_PHONE_CHANGE', 'P_RATIO_DAYS_REGISTRATION_div_DAYS_LAST_PHONE_CHANGE', 'P_PROD_DAYS_BIRTH_x_DAYS_EMPLOYED_CLEAN',
    'P_PROD_DAYS_BIRTH_x_DAYS_LAST_PHONE_CHANGE', 'P_PROD_DAYS_BIRTH_x_CLAYTON_COPULA_2_5', 'P_DIFF_DAYS_EMPLOYED_CLEAN_sub_DAYS_LAST_PHONE_CHANGE',
    'P_PROD_EAR_x_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_REGISTRATION_x_CLAYTON_COPULA_2_5', 'P_RATIO_DAYS_BIRTH_div_AMT_INCOME_TOTAL',
    'P_RATIO_SURPLUS_div_DAYS_EMPLOYED_CLEAN', 'P_RATIO_CREDIT_TERM_div_CLAYTON_COPULA_2_5', 'P_RATIO_DAYS_REGISTRATION_div_AMT_INCOME_TOTAL',
    'P_RATIO_DAYS_LAST_PHONE_CHANGE_div_AMT_INCOME_TOTAL', 'P_RATIO_LTV_div_DAYS_EMPLOYED_CLEAN', 'P_PROD_DAYS_EMPLOYED_CLEAN_x_DAYS_LAST_PHONE_CHANGE',
    'P_PROD_LTV_x_DAYS_BIRTH', 'P_RATIO_LTV_div_MERTON_PD_25', 'P_RATIO_DAYS_EMPLOYED_CLEAN_div_DEBT_BURDEN',
    'P_PROD_CREDIT_TERM_x_CLAYTON_COPULA_2_5', 'P_PROD_DAYS_LAST_PHONE_CHANGE_x_CLAYTON_COPULA_2_5', 'P_RATIO_DAYS_BIRTH_div_BUFFER_MONTHS',
    'P_RATIO_EXT_MEAN_div_SURPLUS', 'P_RATIO_EXT_MEAN_div_DAYS_LAST_PHONE_CHANGE', 'P_RATIO_DAYS_ID_PUBLISH_div_AMT_INCOME_TOTAL',
    'P_RATIO_DTI_div_SURPLUS', 'P_PROD_CREDIT_TERM_x_DAYS_BIRTH', 'P_RATIO_CLAYTON_COPULA_2_5_div_BUFFER_MONTHS',
    'P_PROD_DSCR_x_BUFFER_MONTHS', 'P_RATIO_EXT_MEAN_div_BUFFER_MONTHS', 'P_RATIO_DAYS_ID_PUBLISH_div_DAYS_LAST_PHONE_CHANGE',
    'P_RATIO_DAYS_LAST_PHONE_CHANGE_div_AMT_ANNUITY', 'P_RATIO_CLAYTON_COPULA_2_5_div_DEBT_BURDEN', 'P_RATIO_DTI_div_DAYS_ID_PUBLISH',
    'P_PROD_DAYS_REGISTRATION_x_AMT_INCOME_TOTAL', 'P_PROD_CREDIT_TERM_x_DAYS_EMPLOYED_CLEAN', 'P_RATIO_DAYS_BIRTH_div_AMT_GOODS_PRICE',
    'P_RATIO_DAYS_EMPLOYED_CLEAN_div_DAYS_ID_PUBLISH', 'P_RATIO_LTV_div_DAYS_ID_PUBLISH', 'P_RATIO_CREDIT_TERM_div_DAYS_EMPLOYED_CLEAN',
    'P_RATIO_DAYS_EMPLOYED_CLEAN_div_AMT_INCOME_TOTAL', 'P_RATIO_DAYS_ID_PUBLISH_div_AMT_GOODS_PRICE', 'P_PROD_DAYS_EMPLOYED_CLEAN_x_AMT_GOODS_PRICE',
    'P_PROD_CREDIT_TERM_x_AMT_GOODS_PRICE', 'P_RATIO_DAYS_BIRTH_div_MERTON_PD_25', 'P_RATIO_EXT_MEAN_div_DTI',
    'P_PROD_DAYS_BIRTH_x_AMT_INCOME_TOTAL', 'P_RATIO_DTI_div_AMT_ANNUITY', 'P_RATIO_DAYS_ID_PUBLISH_div_AMT_ANNUITY',
    'P_PROD_SURPLUS_x_DAYS_REGISTRATION', 'P_PROD_DAYS_ID_PUBLISH_x_AMT_INCOME_TOTAL', 'P_RATIO_EXT_MEAN_div_AMT_CREDIT',
    'P_PROD_EXT_MEAN_x_AMT_INCOME_TOTAL', 'P_RATIO_EXT_MEAN_div_CREDIT_TERM', 'P_RATIO_LTV_div_AMT_ANNUITY',
    'P_RATIO_EXT_MEAN_div_AMT_INCOME_TOTAL', 'P_PROD_DAYS_BIRTH_x_MERTON_PD_25', 'P_RATIO_DAYS_LAST_PHONE_CHANGE_div_AMT_GOODS_PRICE',
    'P_RATIO_EXT_MEAN_div_DSCR', 'P_PROD_DAYS_BIRTH_x_DAYS_REGISTRATION', 'P_PROD_CREDIT_TERM_x_DAYS_ID_PUBLISH',
    'P_DIFF_DAYS_ID_PUBLISH_sub_AMT_GOODS_PRICE', 'P_RATIO_SURPLUS_div_DAYS_REGISTRATION', 'P_RATIO_SURPLUS_div_DAYS_LAST_PHONE_CHANGE',
    'P_DIFF_DTI_sub_AMT_GOODS_PRICE', 'P_DIFF_LTV_sub_MERTON_PD_25', 'P_RATIO_DTI_div_DAYS_EMPLOYED_CLEAN',
    'P_RATIO_DAYS_BIRTH_div_DAYS_ID_PUBLISH', 'P_DIFF_EAR_sub_CREDIT_TERM', 'P_RATIO_LTV_div_DSCR',
    'P_RATIO_DAYS_ID_PUBLISH_div_BUFFER_MONTHS', 'P_RATIO_LTV_div_BUFFER_MONTHS', 'P_PROD_CREDIT_TERM_x_DAYS_REGISTRATION',
    'P_RATIO_CREDIT_TERM_div_BUFFER_MONTHS', 'P_DIFF_LTV_sub_DSCR', 'P_RATIO_EXT_MEAN_div_DAYS_REGISTRATION',
    'P_RATIO_DTI_div_DAYS_REGISTRATION', 'P_DIFF_CREDIT_TERM_sub_DEBT_BURDEN', 'P_RATIO_AMT_CREDIT_div_AMT_GOODS_PRICE',
    'P_RATIO_EXT_MEAN_div_DAYS_EMPLOYED_CLEAN', 'P_RATIO_DAYS_REGISTRATION_div_AMT_GOODS_PRICE', 'P_DIFF_DAYS_REGISTRATION_sub_AMT_ANNUITY',
    'P_PROD_EXT_MEAN_x_DAYS_LAST_PHONE_CHANGE', 'P_PROD_LTV_x_AMT_ANNUITY',
]


def compute_hasan_high_order_pairwise_interactions(df_all, df_base):
    t0 = time.time()
    inc = df_all['AMT_INCOME_TOTAL'].values.astype(np.float64)
    cred = df_all['AMT_CREDIT'].values.astype(np.float64)
    ann = df_all['AMT_ANNUITY'].values.astype(np.float64)
    goods = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(np.float64)
    ext_m = df_base['EXT_MEAN'].values.astype(np.float64)

    ltv = cred / (goods + 1.0)
    dti = cred / (inc + 1.0)
    term = cred / (ann + 1.0)
    ear = 24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0)
    dscr = ((inc / 12.0) * 0.65) / (ann + 1.0)
    surplus = inc - 12.0 * ann
    de = df_all['DAYS_EMPLOYED'].values.astype(np.float64)
    de_clean = np.where(de == 365243, np.nan, de)
    db = df_all['DAYS_BIRTH'].values.astype(np.float64)
    di = df_all['DAYS_ID_PUBLISH'].values.astype(np.float64)
    dr = df_all['DAYS_REGISTRATION'].values.astype(np.float64)
    dp = df_all['DAYS_LAST_PHONE_CHANGE'].values.astype(np.float64)

    merton_pd = df_base['MERTON_PD_25'].values.astype(np.float64) if 'MERTON_PD_25' in df_base.columns else np.full(len(df_all), 0.5)
    clayton25 = df_base['CLAYTON_COPULA_25'].values.astype(np.float64) if 'CLAYTON_COPULA_25' in df_base.columns else np.full(len(df_all), 0.5)
    buf = surplus / (12.0 * ann + 1.0)
    dburd = ann / (inc / 12.0 + 1.0)

    core_map = {
        'EXT_MEAN': ext_m, 'DTI': dti, 'EAR': ear, 'LTV': ltv, 'DSCR': dscr,
        'SURPLUS': surplus, 'CREDIT_TERM': term, 'DAYS_BIRTH': db,
        'DAYS_EMPLOYED_CLEAN': de_clean, 'DAYS_ID_PUBLISH': di, 'DAYS_REGISTRATION': dr,
        'DAYS_LAST_PHONE_CHANGE': dp, 'AMT_CREDIT': cred, 'AMT_INCOME_TOTAL': inc,
        'AMT_ANNUITY': ann, 'AMT_GOODS_PRICE': goods, 'MERTON_PD_25': merton_pd,
        'CLAYTON_COPULA_2_5': clayton25, 'BUFFER_MONTHS': buf, 'DEBT_BURDEN': dburd
    }

    inter_dict = {}
    for name in TOP_HASAN_INTERACTIONS:
        if name.startswith('P_PROD_'):
            parts = name[7:].split('_x_')
            if len(parts) == 2 and parts[0] in core_map and parts[1] in core_map:
                inter_dict[name] = (core_map[parts[0]] * core_map[parts[1]]).astype(np.float32)
        elif name.startswith('P_RATIO_'):
            parts = name[8:].split('_div_')
            if len(parts) == 2 and parts[0] in core_map and parts[1] in core_map:
                inter_dict[name] = (core_map[parts[0]] / (np.abs(core_map[parts[1]]) + 1.0)).astype(np.float32)
        elif name.startswith('P_DIFF_'):
            parts = name[7:].split('_sub_')
            if len(parts) == 2 and parts[0] in core_map and parts[1] in core_map:
                inter_dict[name] = (core_map[parts[0]] - core_map[parts[1]]).astype(np.float32)

    df_out = pd.DataFrame(inter_dict, index=df_all.index)
    print(f"  [+] Hasan Mazi High-Order Pairwise Interactions: {df_out.shape[1]} features in {time.time()-t0:.2f}s")
    return df_out

import warnings
import re
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: PROVENANCE AND CRYPTOGRAPHIC ASSERTIONS
# ==============================================================================

OFFICIAL_TRAIN_SHA256 = "B8DD5A61771D8EB2821972C1B7D3F2F747CC66471F4D34153036D2D9C458ECF1"
OFFICIAL_TEST_SHA256  = "DFB7AEB96E5AF3D8616ACAABA6B23CEC8680845CFD525F7C08DE15C6D2D42C65"
OFFICIAL_SUB_SHA256   = "6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB"

CATS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'
]

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

def resolve_data_paths():
    candidates = [
        os.path.join("inpuy"),
        os.path.join("d:", "gci", "inpuy"),
        os.path.join("..", "inpuy"),
    ]
    for c in candidates:
        tr = os.path.join(c, "train.csv")
        te = os.path.join(c, "test.csv")
        sub = os.path.join(c, "sample_submission.csv")
        if os.path.exists(tr) and os.path.exists(te) and os.path.exists(sub):
            return tr, te, sub
    raise FileNotFoundError("Could not locate official competition input files!")

def verify_provenance(tr_path, te_path, sub_path):
    print("[*] Cryptographic Provenance Enforcement (GCI World 2026 Rules):")
    tr_hash = compute_sha256(tr_path)
    te_hash = compute_sha256(te_path)
    sub_hash = compute_sha256(sub_path)

    assert tr_hash == OFFICIAL_TRAIN_SHA256, f"Corrupted train.csv hash: {tr_hash}"
    assert te_hash == OFFICIAL_TEST_SHA256, f"Corrupted test.csv hash: {te_hash}"
    assert sub_hash == OFFICIAL_SUB_SHA256, f"Corrupted sample_submission.csv hash: {sub_hash}"

    print(f"  [PASS] train.csv              SHA-256: {tr_hash} (100% MATCH)")
    print(f"  [PASS] test.csv               SHA-256: {te_hash} (100% MATCH)")
    print(f"  [PASS] sample_submission.csv  SHA-256: {sub_hash} (100% MATCH)")

# ==============================================================================
# SECTION 2: MATHEMATICAL OPERATORS & ROOT FINDING
# ==============================================================================

def rk_norm(v):
    r = rankdata(v, method="average")
    return (r - 0.5) / float(len(v))

def to_z(u, eps=1e-6):
    return norm.ppf(np.clip(u, eps, 1.0 - eps))

def to_p(z):
    return norm.cdf(z)

def solve_implied_rates(cr_arr, an_arr, terms=(12, 24, 36, 48, 60)):
    N = len(cr_arr)
    res = {f"IMPLIED_RATE_{n}": np.full(N, np.nan, dtype=np.float32) for n in terms}

    for n in terms:
        ratio = an_arr / np.maximum(cr_arr, 1.0)
        valid = (cr_arr > 0) & (an_arr > 0) & (ratio > (1.0 / n) + 1e-4) & (ratio < 0.6)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        r_v = ratio[valid_idx]
        low = np.full(len(r_v), 1e-5, dtype=np.float64)
        high = np.full(len(r_v), 0.50, dtype=np.float64)

        for _ in range(25):
            mid = 0.5 * (low + high)
            f_mid = mid / (1.0 - np.power(1.0 + mid, -float(n))) - r_v
            pos = f_mid > 0
            high[pos] = mid[pos]
            low[~pos] = mid[~pos]

        res[f"IMPLIED_RATE_{n}"][valid_idx] = (0.5 * (low + high) * 12.0).astype(np.float32)
    return pd.DataFrame(res)

def months_rate(ratio, n, iters=40):
    lo, hi = np.zeros_like(ratio), np.full_like(ratio, 0.15)
    f = lambda r: np.where(r < 1e-9, n, (1 - (1 + r) ** (-n)) / np.maximum(r, 1e-12))
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_high = f(mid) < ratio
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    r = (lo + hi) / 2
    return np.where((ratio >= n) | ~np.isfinite(ratio), np.nan, r)

# ==============================================================================
# SECTION 3: COMPLETE QUANTITATIVE FEATURE UNIVERSE (417 FEATURES)
# ==============================================================================

def build_complete_base_manifold(df):
    t0 = time.time()
    feats = pd.DataFrame(index=df.index)

    # 1. Clean & Missing Indicators
    days_emp = df['DAYS_EMPLOYED'].values.astype(np.float64)
    is_pensioner = (days_emp == 365243).astype(np.float32)
    emp_clean = np.where(days_emp == 365243, np.nan, days_emp)

    feats['DE_SENT'] = is_pensioner
    feats['DAYS_EMPLOYED_CLEAN'] = emp_clean.astype(np.float32)
    feats['ORG_XNA'] = (df['ORGANIZATION_TYPE'] == 'XNA').astype(np.float32)
    feats['CAR_AGE_OUT'] = (df['OWN_CAR_AGE'] >= 60).astype(np.float32)
    feats['MISS_CNT'] = df.isnull().sum(axis=1).astype(np.float32)

    for c in ['EXT_SOURCE_1', 'EXT_SOURCE_3', 'OWN_CAR_AGE', 'AMT_REQ_CREDIT_BUREAU_YEAR']:
        feats[c + '_NA'] = df[c].isnull().astype(np.float32)

    # 2. External Bureau Scores & Interactions
    e1 = df['EXT_SOURCE_1'].values.astype(np.float64)
    e2 = df['EXT_SOURCE_2'].values.astype(np.float64)
    e3 = df['EXT_SOURCE_3'].values.astype(np.float64)
    e_mat = np.column_stack([e1, e2, e3])

    feats['EXT_SOURCE_1'] = e1.astype(np.float32)
    feats['EXT_SOURCE_2'] = e2.astype(np.float32)
    feats['EXT_SOURCE_3'] = e3.astype(np.float32)
    feats['EXT_MEAN'] = np.nanmean(e_mat, axis=1).astype(np.float32)
    feats['EXT_MIN']  = np.nanmin(e_mat, axis=1).astype(np.float32)
    feats['EXT_MAX']  = np.nanmax(e_mat, axis=1).astype(np.float32)
    feats['EXT_STD']  = np.nanstd(e_mat, axis=1).astype(np.float32)
    feats['EXT_RANGE'] = (feats['EXT_MAX'] - feats['EXT_MIN']).astype(np.float32)
    feats['EXT_NOBS']  = np.sum(~np.isnan(e_mat), axis=1).astype(np.float32)
    feats['EXT_PROD']  = np.nanprod(e_mat, axis=1).astype(np.float32)
    feats['EXT_W']     = ((np.nan_to_num(e1)*2.0 + np.nan_to_num(e2)*3.0 + np.nan_to_num(e3)*4.0) / 9.0).astype(np.float32)

    feats['E12'] = (e1 * e2).astype(np.float32)
    feats['E13'] = (e1 * e3).astype(np.float32)
    feats['E23'] = (e2 * e3).astype(np.float32)
    feats['D12'] = (e1 - e2).astype(np.float32)
    feats['D13'] = (e1 - e3).astype(np.float32)
    feats['D23'] = (e2 - e3).astype(np.float32)

    # 3. Financial & Debt Ratios
    inc = df['AMT_INCOME_TOTAL'].values.astype(np.float64)
    cr  = df['AMT_CREDIT'].values.astype(np.float64)
    an  = df['AMT_ANNUITY'].values.astype(np.float64)
    gp  = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT']).values.astype(np.float64)
    fam = df['CNT_FAM_MEMBERS'].fillna(1.0).values.astype(np.float64)

    feats['CREDIT_INC'] = (cr / np.maximum(inc, 1.0)).astype(np.float32)
    feats['ANN_INC']    = (an / np.maximum(inc, 1.0)).astype(np.float32)
    feats['ANN_CREDIT'] = (an / np.maximum(cr, 1.0)).astype(np.float32)
    feats['TERM']       = (cr / np.maximum(an, 1.0)).astype(np.float32)
    feats['GOODS_CREDIT'] = (gp / np.maximum(cr, 1.0)).astype(np.float32)
    feats['CREDIT_MINUS_GOODS'] = (cr - gp).astype(np.float32)
    feats['GOODS_INC']  = (gp / np.maximum(inc, 1.0)).astype(np.float32)

    feats['INC_PER_FAM']    = (inc / np.maximum(fam, 1.0)).astype(np.float32)
    feats['CREDIT_PER_FAM'] = (cr  / np.maximum(fam, 1.0)).astype(np.float32)
    feats['ANN_PER_FAM']    = (an  / np.maximum(fam, 1.0)).astype(np.float32)

    for c in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        feats['LOG_' + c] = np.log1p(np.maximum(df[c].values.astype(np.float64), 0.0)).astype(np.float32)

    # 4. Fee-Adjusted Tariffs & Financing Step Lattices
    q = an / np.maximum(cr, 1.0)
    q_dec = (an - 129.5) / np.maximum(cr, 1.0)
    is_cash = (df['NAME_CONTRACT_TYPE'] == 'Cash loans').values
    feats['Q_TARIFF'] = q.astype(np.float32)
    feats['Q_DEC'] = np.where(is_cash & (cr > 0), q_dec, q).astype(np.float32)
    feats['HAS_FEE'] = (is_cash & (q_dec > 0) & (an > 129.5)).astype(np.float32)

    kf = np.where(gp > 0, ((cr / gp) - 1.0) / 0.0132, np.nan)
    feats['K_DEC'] = np.where(is_cash & (cr > gp), np.round(kf), 0.0).astype(np.float32)
    feats['CG_RATIO'] = (cr / np.maximum(gp, 1.0)).astype(np.float32)

    # 5. Demographics & Relative Tenures
    birth = -df['DAYS_BIRTH'].values.astype(np.float64)
    age = birth / 365.25
    feats['AGE'] = age.astype(np.float32)
    feats['EMP_AGE'] = (emp_clean / np.minimum(df['DAYS_BIRTH'].values.astype(np.float64), -1.0)).astype(np.float32)
    feats['EMP_YEARS'] = (-emp_clean / 365.25).astype(np.float32)
    feats['REG_AGE'] = (df['DAYS_REGISTRATION'].values.astype(np.float64) / np.minimum(df['DAYS_BIRTH'].values.astype(np.float64), -1.0)).astype(np.float32)
    feats['ID_AGE']  = (df['DAYS_ID_PUBLISH'].values.astype(np.float64) / np.minimum(df['DAYS_BIRTH'].values.astype(np.float64), -1.0)).astype(np.float32)
    feats['PHONE_AGE'] = (df['DAYS_LAST_PHONE_CHANGE'].values.astype(np.float64) / np.minimum(df['DAYS_BIRTH'].values.astype(np.float64), -1.0)).astype(np.float32)
    feats['CAR_AGE_RATIO'] = (df['OWN_CAR_AGE'].values.astype(np.float64) / np.maximum(age, 1.0)).astype(np.float32)
    feats['CREDIT_PER_AGE'] = (cr / np.maximum(age, 1.0)).astype(np.float32)
    feats['INC_PER_AGE']    = (inc / np.maximum(age, 1.0)).astype(np.float32)

    # 6. Credit Bureau Inquiries
    bureau_cols = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU_')]
    feats['BUR_SUM'] = df[bureau_cols].sum(axis=1).values.astype(np.float32)
    feats['BUR_RECENT'] = df[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON']].sum(axis=1).values.astype(np.float32)
    feats['BUR_MON_YEAR'] = (df['AMT_REQ_CREDIT_BUREAU_MON'].values / (df['AMT_REQ_CREDIT_BUREAU_YEAR'].values + 1.0)).astype(np.float32)

    # 7. Merton Structural Solvency Distance-to-Default
    monthly_inc = np.maximum(inc / 12.0, 1.0)
    term_years = np.clip((cr / np.maximum(an, 1.0)) / 12.0, 0.1, 10.0)
    dd_ratio = np.log(monthly_inc / np.maximum(an, 1.0))

    dd_20 = (dd_ratio + (0.05 - 0.5 * 0.04) * term_years) / (0.20 * np.sqrt(term_years))
    dd_25 = (dd_ratio + (0.05 - 0.5 * 0.0625) * term_years) / (0.25 * np.sqrt(term_years))
    feats['MERTON_DD_20'] = dd_20.astype(np.float32)
    feats['MERTON_DD_25'] = dd_25.astype(np.float32)
    feats['MERTON_PD_20'] = norm.cdf(-dd_20).astype(np.float32)
    feats['MERTON_PD_25'] = norm.cdf(-dd_25).astype(np.float32)

    # 8. Clayton Archimedean Copulas
    def uniform_rank(v):
        nan_m = np.isnan(v)
        ranks = np.full(len(v), np.nan, dtype=np.float64)
        if (~nan_m).sum() > 0:
            valid_r = rankdata(v[~nan_m], method='average')
            ranks[~nan_m] = (valid_r - 0.5) / float(len(valid_r))
        return ranks

    u1 = uniform_rank(e1)
    u2 = uniform_rank(e2)
    u3 = uniform_rank(e3)

    for theta in [1.5, 2.5]:
        v_cop = np.full(len(df), np.nan, dtype=np.float32)
        valid = ~np.isnan(u1) & ~np.isnan(u2) & ~np.isnan(u3)
        if valid.sum() > 0:
            uu1, uu2, uu3 = u1[valid], u2[valid], u3[valid]
            s = np.power(uu1, -theta) + np.power(uu2, -theta) + np.power(uu3, -theta) - 2.0
            v_cop[valid] = np.power(np.maximum(s, 1e-6), -1.0 / theta).astype(np.float32)
        feats[f'CLAYTON_COPULA_{int(theta*10)}'] = v_cop

    # 9. Actuarial Implied Interest Rate Bisection Solver
    rates_df = solve_implied_rates(cr, an, terms=(12, 24, 36, 48, 60))
    for col in rates_df.columns:
        feats[col] = rates_df[col].values

    # 10. Categorical Frequency Encodings
    for cat in CATS:
        s = df[cat].astype(str)
        freq = s.map(s.value_counts(normalize=True))
        feats[f'{cat}_FREQ'] = freq.values.astype(np.float32)

    print(f"  [+] Base Manifold constructed: {feats.shape[1]} features in {time.time()-t0:.2f}s")
    return feats

def extract_restored_raw_baseline_features(df_all, df_base):
    t0 = time.time()
    num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
    excluded = ['SK_ID_CURR', 'TARGET']
    pure_raw = [c for c in num_cols if c not in excluded and c not in df_base.columns]
    df_raw = df_all[pure_raw].copy().astype(np.float32)
    print(f"  [+] Restored Raw Baseline Features: {df_raw.shape[1]} features in {time.time()-t0:.2f}s")
    return df_raw

def compute_hierarchical_tariff_exception_features(df_all):
    t0 = time.time()
    cr = df_all['AMT_CREDIT'].values.astype(float)
    an = df_all['AMT_ANNUITY'].values.astype(float)
    gp = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(float)
    cash = (df_all['NAME_CONTRACT_TYPE'] == 'Cash loans').values

    q_raw = an / np.maximum(cr, 1.0)
    F = 129.5
    q_dec = np.where(cash, (an - F) / np.maximum(cr, 1.0), q_raw)

    q5 = np.round(q_raw, 5)
    qb5 = np.round(q_dec, 5)
    vc_q5 = pd.Series(q5).value_counts()
    vc_qb5 = pd.Series(qb5).value_counts()
    fee_flag = ((pd.Series(qb5).map(vc_qb5).values > pd.Series(q5).map(vc_q5).values) & cash).astype(int)

    Q0 = np.where(fee_flag == 1, q_dec, q_raw)
    CT_arr = cr / np.maximum(an, 1.0)

    kf = np.where((cr > gp) & (gp > 0), (cr / gp - 1.0) / 0.0132, 0.0)
    k_round = np.round(kf)
    ok_k = cash & (cr > gp) & (np.abs(kf - k_round) < 0.1)
    k_dec = np.where(ok_k, k_round, np.where(cash & (cr == gp), 0.0, np.nan))

    q4 = np.round(Q0, 4)
    vc4 = pd.Series(q4).value_counts()
    centers = vc4[vc4 >= 30].index.values

    dists = np.abs(Q0[:, None] - centers[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    nearest_q = centers[nearest_idx]
    nearest_dist = np.min(dists, axis=1)

    ann_expected = nearest_q * cr + np.where(fee_flag == 1, F, 0.0)
    ann_resid = an - ann_expected
    ann_resid_abs = np.abs(ann_resid)
    q_fee_effect = np.where(cash, F / np.maximum(cr, 1.0), 0.0)

    k_valid = k_dec[np.isfinite(k_dec)]
    k_centers = pd.Series(np.round(k_valid)).value_counts()
    k_centers = k_centers[k_centers >= 30].index.values

    k_clean = np.nan_to_num(k_dec, nan=0.0)
    k_dists = np.abs(k_clean[:, None] - k_centers[None, :])
    nearest_k = k_centers[np.argmin(k_dists, axis=1)]
    nearest_k_dist = np.min(k_dists, axis=1)

    ext_m = np.nanmean(df_all[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].values, axis=1)
    ext_m_clean = np.nan_to_num(ext_m, nan=0.5)
    inc_log = np.log1p(np.maximum(df_all['AMT_INCOME_TOTAL'].values, 0.0))

    out = pd.DataFrame({
        "HQ_Q_DEC": q_dec.astype(np.float32),
        "HQ_Q_NEAREST": nearest_q.astype(np.float32),
        "HQ_Q_NEAREST_DIST": nearest_dist.astype(np.float32),
        "HQ_ANN_RESID": ann_resid.astype(np.float32),
        "HQ_ANN_RESID_ABS": ann_resid_abs.astype(np.float32),
        "HQ_Q_FEE_EFFECT": q_fee_effect.astype(np.float32),
        "HQ_K_NEAREST": nearest_k.astype(np.float32),
        "HQ_K_NEAR_DIST": nearest_k_dist.astype(np.float32),
        "HQ_ANN_RESID_X_EXTMEAN": (ann_resid * ext_m_clean).astype(np.float32),
        "HQ_ANN_RESID_X_INCOME": (ann_resid * inc_log).astype(np.float32),
        "HQ_Q_RES3": (Q0 - np.round(Q0, 3)).astype(np.float32),
        "HQ_Q_RES4": (Q0 - np.round(Q0, 4)).astype(np.float32),
        "HQ_Q_RES5": (Q0 - np.round(Q0, 5)).astype(np.float32),
        "HQ_Q_SUPPORT_LOG": np.log1p(pd.Series(q4).map(vc4).fillna(0).values).astype(np.float32),
        "HQ_K_SUPPORT_LOG": np.log1p(pd.Series(np.round(k_clean)).map(pd.Series(k_valid).value_counts()).fillna(0).values).astype(np.float32)
    }, index=df_all.index)

    print(f"  [+] Hierarchical Tariff Decoders & Exceptions: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_hasan_banking_and_solvency_standards(df_all):
    t0 = time.time()
    inc = df_all['AMT_INCOME_TOTAL'].values.astype(np.float64)
    cr  = df_all['AMT_CREDIT'].values.astype(np.float64)
    an  = df_all['AMT_ANNUITY'].values.astype(np.float64)
    gp  = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(np.float64)

    CT_arr = cr / np.maximum(an, 1.0)
    EAR = (np.power(1.0 + (an / np.maximum(cr, 1.0)), 12.0) - 1.0).astype(np.float32)
    SURPLUS = (inc - an * 12.0).astype(np.float32)
    SURPLUS_TO_INC = (SURPLUS / np.maximum(inc, 1.0)).astype(np.float32)
    TRUE_COST = (an * CT_arr).astype(np.float32)
    TRUE_COST_RATIO = (TRUE_COST / np.maximum(cr, 1.0)).astype(np.float32)
    LTV = (cr / np.maximum(gp, 1.0)).astype(np.float32)
    DEBT_BURDEN = ((cr + an * 12.0) / np.maximum(inc, 1.0)).astype(np.float32)
    DISPOSABLE_INC = (inc - an).astype(np.float32)
    DSCR = (inc / np.maximum(an * 12.0, 1.0)).astype(np.float32)
    SOLVENCY = (SURPLUS / np.maximum(cr, 1.0)).astype(np.float32)
    BUFFER_MONTHS = (SURPLUS / np.maximum(an, 1.0)).astype(np.float32)
    RESERVE_MONTHS = (inc / np.maximum(an, 1.0)).astype(np.float32)

    Z_SCORE = (
        1.2 * (SURPLUS / np.maximum(cr, 1.0)) +
        1.4 * (inc / np.maximum(cr, 1.0)) +
        3.3 * (inc / np.maximum(an * 12.0, 1.0)) +
        0.6 * (gp / np.maximum(cr, 1.0)) +
        1.0 * (inc / np.maximum(cr, 1.0))
    ).astype(np.float32)

    HFVI = (
        (an / np.maximum(inc, 1.0)) * 0.4 +
        (cr / np.maximum(gp, 1.0)) * 0.3 +
        (1.0 / np.maximum(DSCR, 0.1)) * 0.3
    ).astype(np.float32)

    FFI = (
        (an / np.maximum(inc, 1.0)) *
        (cr / np.maximum(inc, 1.0)) *
        np.maximum(0.0, 1.0 - (SURPLUS / np.maximum(inc, 1.0)))
    ).astype(np.float32)

    ZOMBIE = ((SURPLUS < 0) & (cr > inc * 2.0)).astype(np.float32)
    DTI_50 = ((an * 12.0 / np.maximum(inc, 1.0)) > 0.5).astype(np.float32)
    LTV_100 = (LTV > 1.0).astype(np.float32)
    DSCR_BELOW1 = (DSCR < 1.0).astype(np.float32)

    FEAT_DOWN_PAYMENT = np.maximum(0.0, gp - cr).astype(np.float32)
    FEAT_DOWN_PAYMENT_PCT = (FEAT_DOWN_PAYMENT / np.maximum(gp, 1.0)).astype(np.float32)
    FEAT_HAS_DOWN_PAYMENT = (FEAT_DOWN_PAYMENT > 0).astype(np.float32)
    FEAT_FEES_FINANCED = np.maximum(0.0, cr - gp).astype(np.float32)
    FEAT_FEES_RATIO = (FEAT_FEES_FINANCED / np.maximum(cr, 1.0)).astype(np.float32)

    out = pd.DataFrame({
        "EAR": EAR, "SURPLUS": SURPLUS, "SURPLUS_TO_INC": SURPLUS_TO_INC,
        "TRUE_COST": TRUE_COST, "TRUE_COST_RATIO": TRUE_COST_RATIO,
        "LTV": LTV, "DEBT_BURDEN": DEBT_BURDEN, "DISPOSABLE_INC": DISPOSABLE_INC,
        "DSCR": DSCR, "SOLVENCY": SOLVENCY, "BUFFER_MONTHS": BUFFER_MONTHS,
        "RESERVE_MONTHS": RESERVE_MONTHS, "Z_SCORE": Z_SCORE, "HFVI": HFVI,
        "FFI": FFI, "ZOMBIE": ZOMBIE, "DTI_50": DTI_50, "LTV_100": LTV_100,
        "DSCR_BELOW1": DSCR_BELOW1, "FEAT_DOWN_PAYMENT": FEAT_DOWN_PAYMENT,
        "FEAT_DOWN_PAYMENT_PCT": FEAT_DOWN_PAYMENT_PCT,
        "FEAT_HAS_DOWN_PAYMENT": FEAT_HAS_DOWN_PAYMENT,
        "FEAT_FEES_FINANCED": FEAT_FEES_FINANCED,
        "FEAT_FEES_RATIO": FEAT_FEES_RATIO,
        "HASAN_STRESS_PRODUCT": (HFVI * FFI).astype(np.float32)
    }, index=df_all.index)

    print(f"  [+] Hasan Mazi Banking & Solvency Standards: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_temporal_stability_and_inquiry_bursts(df_all):
    t0 = time.time()
    db = df_all['DAYS_BIRTH'].values.astype(float)
    de = df_all['DAYS_EMPLOYED'].values.astype(float)
    dr = df_all['DAYS_REGISTRATION'].values.astype(float)
    di = df_all['DAYS_ID_PUBLISH'].values.astype(float)
    dp = df_all['DAYS_LAST_PHONE_CHANGE'].values.astype(float)
    inc = df_all['AMT_INCOME_TOTAL'].values.astype(float)
    de_clean = np.where(de == 365243, np.nan, de)

    out = pd.DataFrame({
        "WORK_LIFE_RATIO": (de_clean / np.minimum(db, -1.0)).astype(np.float32),
        "REG_TO_BIRTH": (dr / np.minimum(db, -1.0)).astype(np.float32),
        "ID_TO_BIRTH": (di / np.minimum(db, -1.0)).astype(np.float32),
        "PHONE_TO_BIRTH": (dp / np.minimum(db, -1.0)).astype(np.float32),
        "PHONE_TO_EMP": (dp / np.minimum(de_clean, -1.0)).astype(np.float32),
        "CHURN_INDEX": ((np.abs(dr) + np.abs(di) + np.abs(dp)) / (3.0 * np.abs(db) + 1.0)).astype(np.float32),
        "ID_VS_REG": (di - dr).astype(np.float32),
        "EMP_VS_REG": (de_clean - dr).astype(np.float32),
        "INQUIRY_BURST": (df_all['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0).values + df_all['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0).values).astype(np.float32),
        "INQUIRY_ACCELERATION": (df_all['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0).values / np.maximum(df_all['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0).values / 12.0, 0.08)).astype(np.float32),
        "INQUIRY_BURST_FLAG": (df_all['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0).values > 0).astype(np.float32),
        "INQUIRY_PER_INC": (df_all['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0).values / np.maximum(inc / 100000.0, 1.0)).astype(np.float32),
        "CITY_RATING_BURDEN": (df_all['REGION_RATING_CLIENT'].values * (df_all['AMT_CREDIT'].values / np.maximum(inc, 1.0))).astype(np.float32),
        "REGION_POP_INCOME": (df_all['REGION_POPULATION_RELATIVE'].values * np.log1p(np.maximum(inc, 0.0))).astype(np.float32)
    }, index=df_all.index)

    print(f"  [+] Temporal Stability & Inquiry Acceleration: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_nonlinear_external_credit_powers(df_all):
    t0 = time.time()
    e1 = df_all['EXT_SOURCE_1'].values.astype(float)
    e2 = df_all['EXT_SOURCE_2'].values.astype(float)
    e3 = df_all['EXT_SOURCE_3'].values.astype(float)
    cr = df_all['AMT_CREDIT'].values.astype(float)
    an = df_all['AMT_ANNUITY'].values.astype(float)
    inc = df_all['AMT_INCOME_TOTAL'].values.astype(float)

    e1_c = np.nan_to_num(e1, nan=0.5)
    e2_c = np.nan_to_num(e2, nan=0.5)
    e3_c = np.nan_to_num(e3, nan=0.5)

    harm = 3.0 / (1.0 / (e1_c + 1e-4) + 1.0 / (e2_c + 1e-4) + 1.0 / (e3_c + 1e-4))
    geom = np.power(np.maximum(e1_c * e2_c * e3_c, 1e-6), 1.0 / 3.0)
    top2_w = (np.maximum(e2_c, e3_c) * 0.6 + np.minimum(e2_c, e3_c) * 0.4)
    vol = np.std(np.column_stack([e1_c, e2_c, e3_c]), axis=1)

    dti = (an * 12.0) / np.maximum(inc, 1.0)
    surplus = inc - an * 12.0
    ext_m = np.nanmean(np.column_stack([e1, e2, e3]), axis=1)
    ext_m_c = np.nan_to_num(ext_m, nan=0.5)

    out = pd.DataFrame({
        "EXT_HARMONIC": harm.astype(np.float32),
        "EXT_GEOMETRIC": geom.astype(np.float32),
        "EXT_WEIGHTED_TOP": top2_w.astype(np.float32),
        "EXT_VOLATILITY": vol.astype(np.float32),
        "EXT_x_DTI": (ext_m_c * dti).astype(np.float32),
        "EXT_x_SURPLUS": (ext_m_c * np.sign(surplus) * np.log1p(np.abs(surplus))).astype(np.float32)
    }, index=df_all.index)

    print(f"  [+] Non-Linear External Credit Powers: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_unsupervised_chronological_timeline(df_all):
    t0 = time.time()
    birth = df_all['DAYS_BIRTH'].values.astype(int)
    reg = df_all['DAYS_REGISTRATION'].values.astype(int)
    id_pub = df_all['DAYS_ID_PUBLISH'].values.astype(int)
    gender = df_all['CODE_GENDER'].astype(str)
    edu = df_all['NAME_EDUCATION_TYPE'].astype(str)

    k_entity = (birth - reg).astype(str) + '|' + (birth - id_pub).astype(str) + '|' + gender + '|' + edu
    counts = k_entity.value_counts()
    n_apps = k_entity.map(counts).values.astype(np.float32)

    tmp = pd.DataFrame({
        'k_entity': k_entity,
        'DAYS_REGISTRATION': df_all['DAYS_REGISTRATION'].values,
        'row_idx': np.arange(len(df_all))
    })
    df_sorted = tmp.sort_values(['k_entity', 'DAYS_REGISTRATION']).copy()
    df_sorted['APP_ORDER'] = df_sorted.groupby('k_entity').cumcount()
    df_sorted['DAYS_SINCE_PREV'] = df_sorted.groupby('k_entity')['DAYS_REGISTRATION'].diff().fillna(-999)
    df_restored = df_sorted.sort_values('row_idx')

    out = pd.DataFrame({
        'N_APPS_ALL': n_apps,
        'APP_ORDER': df_restored['APP_ORDER'].values.astype(np.float32),
        'DAYS_SINCE_PREV_APP': df_restored['DAYS_SINCE_PREV'].values.astype(np.float32)
    }, index=df_all.index)

    print(f"  [+] Unsupervised Chronological Timeline Tracking: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_unsupervised_person_pooling(df_all):
    t0 = time.time()
    N = len(df_all)
    birth = df_all['DAYS_BIRTH'].values.astype(np.float64)
    reg = df_all['DAYS_REGISTRATION'].values.astype(np.float64)
    id_pub = df_all['DAYS_ID_PUBLISH'].values.astype(np.float64)

    i1 = np.round(reg - birth, -1)
    i2 = np.round(id_pub - birth, -1)

    key = (
        df_all['CODE_GENDER'].astype(str) + "|" +
        df_all['NAME_EDUCATION_TYPE'].astype(str) + "|" +
        df_all['REGION_POPULATION_RELATIVE'].astype(str) + "|" +
        i1.astype(str) + "|" + i2.astype(str)
    )
    key_cnt = key.map(key.value_counts())
    is_repeat = (key_cnt > 1).values

    mem = key.to_frame('k').groupby('k').groups
    plist = {}
    for _, ix in mem.items():
        if len(ix) > 1:
            ix = np.asarray(ix)
            for i in ix:
                o = ix[ix != i]
                plist[i] = (o, birth[o] - birth[i])

    gap_partner = np.full(N, np.nan, dtype=np.float32)
    for i, (o, gp_) in plist.items():
        gap_partner[i] = gp_[np.argmin(np.abs(gp_))]

    f_partner = pd.DataFrame({
        "HAS_PARTNER": is_repeat.astype(np.float32),
        "PARTNER_N": key_cnt.values.astype(np.float32),
        "PARTNER_GAP": gap_partner,
        "PARTNER_ABSGAP": np.abs(gap_partner)
    }, index=df_all.index)

    E_mat = df_all[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(np.float64)
    pe = np.full((N, 3), np.nan, dtype=np.float32)
    dif = np.full((N, 3), np.nan, dtype=np.float32)
    cr_p = np.full(N, np.nan, dtype=np.float32)
    an_p = np.full(N, np.nan, dtype=np.float32)
    inc_p = np.full(N, np.nan, dtype=np.float32)
    same_ct = np.full(N, np.nan, dtype=np.float32)
    qd_arr = np.full(N, np.nan, dtype=np.float32)

    cr_arr = df_all['AMT_CREDIT'].values.astype(np.float64)
    an_arr = df_all['AMT_ANNUITY'].values.astype(np.float64)
    inc_arr = df_all['AMT_INCOME_TOTAL'].values.astype(np.float64)
    ct_cash = (df_all['NAME_CONTRACT_TYPE'] == 'Cash loans').values
    Qv = an_arr / np.maximum(cr_arr, 1.0)

    for i, (o, g) in plist.items():
        j = o[np.argmin(np.abs(g))]
        allr = np.r_[i, o]
        for idx_k in range(3):
            v_k = E_mat[allr, idx_k]
            if np.isfinite(v_k).any():
                pe[i, idx_k] = np.nanmean(v_k)
            dif[i, idx_k] = E_mat[i, idx_k] - E_mat[j, idx_k]
        cr_p[i] = cr_arr[i] / (cr_arr[j] + 1e-5)
        an_p[i] = an_arr[i] / (an_arr[j] + 1e-5)
        inc_p[i] = inc_arr[i] / (inc_arr[j] + 1e-5)
        same_ct[i] = float(ct_cash[i] == ct_cash[j])
        qd_arr[i] = Qv[i] - Qv[j]

    d_pers = {f"PERS_EXT{k+1}": pe[:, k] for k in range(3)}
    d_pers.update({f"PDIF_EXT{k+1}": dif[:, k] for k in range(3)})
    d_pers.update({
        "PCR_RATIO": cr_p, "PAN_RATIO": an_p, "PINC_RATIO": inc_p,
        "PSAME_CT": same_ct, "PQ_DIFF": qd_arr
    })
    f_person = pd.DataFrame(d_pers, index=df_all.index)
    f_person["PERS_EXT_MEAN"] = f_person[["PERS_EXT1", "PERS_EXT2", "PERS_EXT3"]].mean(axis=1).astype(np.float32)

    out_df = pd.concat([f_partner, f_person], axis=1)
    print(f"  [+] Unsupervised Person Pooling generated: {out_df.shape[1]} features in {time.time()-t0:.2f}s")
    return out_df

def compute_group_deviations(df_all):
    t0 = time.time()
    num_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'EXT_SOURCE_2', 'DAYS_BIRTH', 'DAYS_EMPLOYED']
    group_cols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'CODE_GENDER']

    out_dict = {}
    for gcol in group_cols:
        for ncol in num_cols:
            if ncol in df_all.columns and gcol in df_all.columns:
                vals = df_all[ncol].values.astype(np.float64)
                grp_mean = df_all.groupby(gcol)[ncol].transform('mean').values.astype(np.float64)
                out_dict[f'DEV_{ncol}_BY_{gcol}'] = (vals - grp_mean).astype(np.float32)

    out = pd.DataFrame(out_dict, index=df_all.index)
    print(f"  [+] Group Relative Deviations generated: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_deep_product_and_target_encodings(df_all, y_train, n_train):
    t0 = time.time()
    cr = df_all['AMT_CREDIT'].values.astype(float)
    an = df_all['AMT_ANNUITY'].values.astype(float)
    gp = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(float)
    cash = (df_all['NAME_CONTRACT_TYPE'] == 'Cash loans').values

    q_raw = an / np.maximum(cr, 1.0)
    F = 129.5
    q_dec = np.where(cash, (an - F) / np.maximum(cr, 1.0), q_raw)

    q5 = np.round(q_raw, 5)
    qb5 = np.round(q_dec, 5)
    vc_q5 = pd.Series(q5).value_counts()
    vc_qb5 = pd.Series(qb5).value_counts()
    fee_flag = ((pd.Series(qb5).map(vc_qb5).values > pd.Series(q5).map(vc_q5).values) & cash).astype(int)
    Q0 = np.where(fee_flag == 1, q_dec, q_raw)
    CT_arr = cr / np.maximum(an, 1.0)

    cats_dict = {
        'Q3': pd.Series(q_raw).round(3).astype(str),
        'Q4': pd.Series(q_raw).round(4).astype(str),
        'Q5': pd.Series(q_raw).round(5).astype(str),
        'P3': pd.Series(Q0).round(3).astype(str),
        'P4': pd.Series(Q0).round(4).astype(str),
        'P5': pd.Series(Q0).round(5).astype(str),
        'PF4': pd.Series(Q0).round(4).astype(str) + "_" + pd.Series(fee_flag).astype(str),
        'CT1': pd.Series(CT_arr).round(1).astype(str),
        'CT2': pd.Series(CT_arr).round(2).astype(str),
        'CT3': pd.Series(CT_arr).round(3).astype(str),
        'CT1_GP': pd.Series(gp / np.maximum(an, 1.0)).round(1).astype(str),
        'CT1_CR': pd.Series(cr / np.maximum(an, 1.0)).round(1).astype(str),
        'CT1_AN': pd.Series(an / 1000.0).round(0).astype(str),
        'GENDER_x_EDU': df_all['CODE_GENDER'].astype(str) + "_" + df_all['NAME_EDUCATION_TYPE'].astype(str),
        'ORG_x_OCC': df_all['ORGANIZATION_TYPE'].astype(str) + "_" + df_all['OCCUPATION_TYPE'].astype(str)
    }
    for c in CATS:
        cats_dict[c] = df_all[c].astype(str)

    deep_df = pd.DataFrame(cats_dict, index=df_all.index)

    kf = KFold(5, shuffle=True, random_state=42)
    te_out = pd.DataFrame(index=df_all.index)
    global_mean = float(np.mean(y_train))
    m = 35.0

    for col in deep_df.columns:
        s_tr = deep_df[col].iloc[:n_train].values
        s_te = deep_df[col].iloc[n_train:].values

        oof_te = np.full(n_train, global_mean, dtype=np.float32)
        for tri, vai in kf.split(s_tr):
            s_t, y_t = s_tr[tri], y_train[tri]
            s_v = s_tr[vai]
            means = pd.Series(y_t).groupby(s_t).agg(['count', 'mean'])
            smooth = (means['count'] * means['mean'] + m * global_mean) / (means['count'] + m)
            oof_te[vai] = pd.Series(s_v).map(smooth).fillna(global_mean).values.astype(np.float32)

        means_all = pd.Series(y_train).groupby(s_tr).agg(['count', 'mean'])
        smooth_all = (means_all['count'] * means_all['mean'] + m * global_mean) / (means_all['count'] + m)
        test_te = pd.Series(s_te).map(smooth_all).fillna(global_mean).values.astype(np.float32)

        te_out['TE_' + col] = np.concatenate([oof_te, test_te])

    print(f"  [+] Deep Product & Bayesian Target Encodings Generated: {te_out.shape[1]} features in {time.time()-t0:.2f}s")
    return te_out

def compute_covariate_expectations_and_residuals(df_base, n_train):
    t0 = time.time()
    targets = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_MEAN',
        'CREDIT_INC', 'ANN_INC', 'ANN_CREDIT', 'TERM',
        'LOG_AMT_CREDIT', 'LOG_AMT_ANNUITY'
    ]
    predictor_cols = [c for c in df_base.columns if c not in targets and not c.startswith('LOG_')]

    X_pred = df_base[predictor_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_pred_tr = X_pred.iloc[:n_train].values
    X_pred_te = X_pred.iloc[n_train:].values

    out = pd.DataFrame(index=df_base.index)
    kf = KFold(5, shuffle=True, random_state=42)

    for tgt in targets:
        y_tgt = df_base[tgt].values.astype(float)
        y_tr = y_tgt[:n_train]
        y_te = y_tgt[n_train:]
        valid_tr = np.isfinite(y_tr)
        m_val = np.nanmedian(y_tr)

        oof_pred = np.full(n_train, m_val, dtype=np.float32)
        for tri, vai in kf.split(X_pred_tr):
            ok_tri = tri[valid_tr[tri]]
            if len(ok_tri) > 100:
                hgb = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, min_samples_leaf=50, random_state=42)
                hgb.fit(X_pred_tr[ok_tri], y_tr[ok_tri])
                oof_pred[vai] = hgb.predict(X_pred_tr[vai])

        if np.sum(valid_tr) > 100:
            hgb_full = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=15, min_samples_leaf=50, random_state=42)
            hgb_full.fit(X_pred_tr[valid_tr], y_tr[valid_tr])
            test_pred = hgb_full.predict(X_pred_te).astype(np.float32)
        else:
            test_pred = np.full(len(X_pred_te), m_val, dtype=np.float32)

        p_all = np.concatenate([oof_pred, test_pred])
        out['RP_' + tgt] = p_all
        out['RR_' + tgt] = y_tgt - p_all

    print(f"  [+] Covariate Expectations & Residuals Generated: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_continuous_frequency_counts(df_all):
    t0 = time.time()
    out = pd.DataFrame(index=df_all.index)
    num_cols = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'REGION_POPULATION_RELATIVE']
    for c in num_cols:
        if c in df_all.columns:
            vc = df_all[c].map(df_all[c].value_counts(dropna=False)).values.astype(np.float32)
            out['FREQ_' + c] = vc
    print(f"  [+] Continuous Loan-Product Frequency Counts: {out.shape[1]} features in {time.time()-t0:.2f}s")
    return out

def compute_onehot_categoricals(df_all):
    t0 = time.time()
    dummies = pd.get_dummies(df_all[CATS], drop_first=True, dtype=np.float32)
    dummies.columns = [f"OH_{i}" for i in range(dummies.shape[1])]
    dummies.index = df_all.index
    print(f"  [+] Orthogonal One-Hot Categoricals: {dummies.shape[1]} features in {time.time()-t0:.2f}s")
    return dummies

def compute_multiview_stacking(df_base, y_train, n_train):
    t0 = time.time()
    views = {
        'VIEW_EXT': [c for c in df_base.columns if 'EXT_' in c or c in ['E12', 'E13', 'E23', 'D12', 'D13', 'D23']],
        'VIEW_LOAN': [c for c in df_base.columns if 'CREDIT' in c or 'ANN' in c or 'RATE' in c or 'Q_' in c or 'K_' in c],
        'VIEW_PERSON': [c for c in df_base.columns if 'AGE' in c or 'FREQ' in c or 'DE_' in c or 'DAYS_' in c],
        'VIEW_NOEXT': [c for c in df_base.columns if not ('EXT_' in c or c in ['E12', 'E13', 'E23', 'D12', 'D13', 'D23'])]
    }

    kf = KFold(5, shuffle=True, random_state=42)
    out_oof = {v: np.zeros(n_train, dtype=np.float32) for v in views}
    out_te  = {v: np.zeros(len(df_base) - n_train, dtype=np.float32) for v in views}

    for v_name, cols in views.items():
        if len(cols) == 0:
            continue
        X_v = df_base[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        X_v_tr = X_v[:n_train]
        X_v_te = X_v[n_train:]

        for tri, vai in kf.split(X_v_tr):
            m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=15, verbose=-1, n_jobs=4, random_state=42)
            m.fit(X_v_tr[tri], y_train[tri])
            out_oof[v_name][vai] = m.predict_proba(X_v_tr[vai])[:, 1]
            out_te[v_name] += m.predict_proba(X_v_te)[:, 1] / 5.0

    full_dict = {}
    for v_name in views:
        full_dict[v_name] = np.concatenate([out_oof[v_name], out_te[v_name]])

    full_dict['VD_EL'] = full_dict['VIEW_EXT'] - full_dict['VIEW_LOAN']
    full_dict['VD_EP'] = full_dict['VIEW_EXT'] - full_dict['VIEW_PERSON']
    full_dict['VD_LP'] = full_dict['VIEW_LOAN'] - full_dict['VIEW_PERSON']
    full_dict['VD_EXT_NOEXT'] = full_dict['VIEW_EXT'] - full_dict['VIEW_NOEXT']

    out_df = pd.DataFrame(full_dict, index=df_base.index)
    print(f"  [+] 4-View Domain Stacking & Contradiction Margins Constructed: {out_df.shape[1]} features in {time.time()-t0:.2f}s")
    return out_df

def print_deep_risk_telemetry(y_true, y_prob, df_train_raw=None, title="MODEL EVALUATION"):
    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)
    pos_prev = np.mean(y_true)

    print(f"\n==================== {title} OPERATIONAL TELEMETRY ====================")
    print(f"  * ROC-AUC                     : {auc:.6f}")
    print(f"  * PR-AUC (Average Precision)  : {ap:.6f}  (Baseline Default Prevalence: {pos_prev*100:.2f}%)")

    percentiles = [5, 10, 15, 20]
    thresh_prev = np.quantile(y_prob, 1.0 - pos_prev)

    print("\n  OPERATIONAL THRESHOLD PERFORMANCE MATRIX:")
    print("  " + "-" * 78)
    print(f"  {'Operating Cutoff':<22} | {'Thresh':<7} | {'Precision':<9} | {'Recall':<8} | {'F1':<6} | {'FP / FN Count':<16}")
    print("  " + "-" * 78)

    y_pred_prev = (y_prob >= thresh_prev).astype(int)
    cm_prev = confusion_matrix(y_true, y_pred_prev)
    tn, fp, fn, tp = cm_prev.ravel()
    prec_prev = precision_score(y_true, y_pred_prev, zero_division=0)
    rec_prev  = recall_score(y_true, y_pred_prev, zero_division=0)
    f1_prev   = f1_score(y_true, y_pred_prev, zero_division=0)
    print(f"  {'Prevalence (~8.07%)':<22} | {thresh_prev:.4f}  | {prec_prev*100:6.2f}%   | {rec_prev*100:5.2f}%  | {f1_prev:.4f} | FP:{fp:,} FN:{fn:,}")

    for pct in percentiles:
        tau = np.quantile(y_prob, 1.0 - pct / 100.0)
        y_pred = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {f'Top {pct}% Risk Cutoff':<22} | {tau:.4f}  | {prec*100:6.2f}%   | {rec*100:5.2f}%  | {f1:.4f} | FP:{fp:,} FN:{fn:,}")
    print("  " + "-" * 78)

    tau_10 = np.quantile(y_prob, 0.90)
    y_pred_10 = (y_prob >= tau_10).astype(int)
    cm_10 = confusion_matrix(y_true, y_pred_10)
    print(f"\n  CONFUSION MATRIX at Standard Top 10% Operational Reject Cutoff (tau = {tau_10:.4f}):")
    print(f"               Predicted Non-Default (0)   Predicted Default (1)")
    print(f"  Actual 0 :         {cm_10[0, 0]:>10,}               {cm_10[0, 1]:>10,}  (False Positives / Falsely Rejected)")
    print(f"  Actual 1 :         {cm_10[1, 0]:>10,}               {cm_10[1, 1]:>10,}  (True Positives / Defaulters Caught)")
    total_def = np.sum(y_true)
    print(f"  --> Top 10% Cutoff identifies {cm_10[1,1]:,} of {total_def:,} total defaults ({100*cm_10[1,1]/total_def:.2f}% Recall)!")

    if df_train_raw is not None:
        fn_mask = (y_true == 1) & (y_pred_10 == 0)
        tp_mask = (y_true == 1) & (y_pred_10 == 1)
        ext_m = np.nanmean(df_train_raw[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].values, axis=1)
        is_rev = (df_train_raw['NAME_CONTRACT_TYPE'] == 'Revolving loans').values
        cr_v = df_train_raw['AMT_CREDIT'].values

        print(f"\n  DIAGNOSIS OF FALSE NEGATIVES (Missed Defaults) vs TRUE POSITIVES (Caught Defaults):")
        print(f"    - Mean Bureau Score (EXT_MEAN) : Missed = {np.nanmean(ext_m[fn_mask]):.4f} | Caught = {np.nanmean(ext_m[tp_mask]):.4f}")
        print(f"    - Revolving Loan Prevalence    : Missed = {100*np.mean(is_rev[fn_mask]):.2f}% | Caught = {100*np.mean(is_rev[tp_mask]):.2f}%")
        print(f"    - Mean Credit Amount           : Missed = {np.nanmean(cr_v[fn_mask]):,.0f} | Caught = {np.nanmean(cr_v[tp_mask]):,.0f}")
        print(f"  --> Key Insight: Missed defaults carry higher credit scores or revolving structures that evade standard credit caps.")

# ==============================================================================
# SECTION 5: 8-ENGINE GRANDMASTER SUITE TRAINING & RE-FIT CALIBRATION
# ==============================================================================

def train_grandmaster_suite(X_train, y_train, X_test, df_train_raw, df_test_raw, cb_df):
    t_start = time.time()
    n_train = len(X_train)
    n_test  = len(X_test)
    FOLDS = 5
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=42)

    model_names = [
        "LGB_ET_E1", "LGB_Shallow", "LGB_ET_Deep", "LGB_Conservative",
        "LGB_Wide", "LGB_Solvency", "XGB_GPU_Hist", "CB_GPU"
    ]
    oof_preds    = {m: np.zeros(n_train, dtype=np.float32) for m in model_names}
    test_cv_pred = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}
    test_refits  = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}

    cb_train = cb_df.iloc[:n_train].reset_index(drop=True)
    cb_test  = cb_df.iloc[n_train:].reset_index(drop=True)
    cb_cats_cols = list(cb_df.columns)

    print(f"\n[*] Training 8-Engine Sovereign Elite Suite across {FOLDS} Folds ({X_train.shape[1]} features):")

    for fi, (tri, vai) in enumerate(skf.split(X_train, y_train)):
        t_fold = time.time()
        print(f"\n--- FOLD {fi+1}/{FOLDS} ---")

        X_tr, y_tr = X_train.iloc[tri], y_train[tri]
        X_va, y_va = X_train.iloc[vai], y_train[vai]

        # 1. LightGBM ExtraTrees E1
        p_e1 = dict(objective="binary", learning_rate=0.035, num_leaves=25, max_depth=5,
                    min_child_samples=90, feature_fraction=0.45, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=10.0, extra_trees=True, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi)
        m_e1 = lgb.train(p_e1, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=850)
        oof_preds["LGB_ET_E1"][vai] = m_e1.predict(X_va)
        test_cv_pred["LGB_ET_E1"] += m_e1.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_ET_E1 AUC          : {roc_auc_score(y_va, oof_preds['LGB_ET_E1'][vai]):.5f}")

        # 2. LightGBM Shallow Anchor
        p_sh = dict(objective="binary", learning_rate=0.035, num_leaves=15, max_depth=4,
                    min_child_samples=180, feature_fraction=0.45, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=30.0, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi+10)
        m_sh = lgb.train(p_sh, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)
        oof_preds["LGB_Shallow"][vai] = m_sh.predict(X_va)
        test_cv_pred["LGB_Shallow"] += m_sh.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_Shallow AUC        : {roc_auc_score(y_va, oof_preds['LGB_Shallow'][vai]):.5f}")

        # 3. LightGBM ExtraTrees Deep
        p_et = dict(objective="binary", learning_rate=0.028, num_leaves=35, max_depth=7,
                    min_child_samples=120, feature_fraction=0.38, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=30.0, extra_trees=True, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi+20)
        m_et = lgb.train(p_et, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=850)
        oof_preds["LGB_ET_Deep"][vai] = m_et.predict(X_va)
        test_cv_pred["LGB_ET_Deep"] += m_et.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_ET_Deep AUC        : {roc_auc_score(y_va, oof_preds['LGB_ET_Deep'][vai]):.5f}")

        # 4. LightGBM Deep Conservative
        p_dc = dict(objective="binary", learning_rate=0.028, num_leaves=45, max_depth=7,
                    min_child_samples=60, feature_fraction=0.55, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi+30)
        m_dc = lgb.train(p_dc, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)
        oof_preds["LGB_Conservative"][vai] = m_dc.predict(X_va)
        test_cv_pred["LGB_Conservative"] += m_dc.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_Conservative AUC   : {roc_auc_score(y_va, oof_preds['LGB_Conservative'][vai]):.5f}")

        # 5. LightGBM Wide Interactor
        p_wi = dict(objective="binary", learning_rate=0.024, num_leaves=63, max_depth=8,
                    min_child_samples=250, feature_fraction=0.30, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=50.0, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi+40)
        m_wi = lgb.train(p_wi, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)
        oof_preds["LGB_Wide"][vai] = m_wi.predict(X_va)
        test_cv_pred["LGB_Wide"] += m_wi.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_Wide AUC           : {roc_auc_score(y_va, oof_preds['LGB_Wide'][vai]):.5f}")

        # 6. LightGBM Solvency Specialist
        p_so = dict(objective="binary", learning_rate=0.030, num_leaves=22, max_depth=5,
                    min_child_samples=150, feature_fraction=0.40, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=20.0, verbose=-1, n_jobs=10,
                    feature_pre_filter=False, seed=42+fi+50)
        m_so = lgb.train(p_so, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)
        oof_preds["LGB_Solvency"][vai] = m_so.predict(X_va)
        test_cv_pred["LGB_Solvency"] += m_so.predict(X_test) / FOLDS
        print(f"  [Fold {fi+1}] LGB_Solvency AUC       : {roc_auc_score(y_va, oof_preds['LGB_Solvency'][vai]):.5f}")

        # 7. XGBoost GPU Hist Specialist
        X_tr_xgb = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_va_xgb = X_va.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_te_xgb = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        dtr = xgb.DMatrix(X_tr_xgb.values.astype(np.float32), label=y_tr, missing=0.0)
        dva = xgb.DMatrix(X_va_xgb.values.astype(np.float32), label=y_va, missing=0.0)
        dte = xgb.DMatrix(X_te_xgb.values.astype(np.float32), missing=0.0)

        p_xgb = dict(tree_method="hist", device="cuda", max_depth=6, max_bin=512,
                     learning_rate=0.028, subsample=0.80, colsample_bytree=0.42,
                     reg_alpha=2.0, reg_lambda=20.0, objective="binary:logistic", seed=42+fi)
        m_xgb = xgb.train(p_xgb, dtr, num_boost_round=850)
        oof_preds["XGB_GPU_Hist"][vai] = m_xgb.predict(dva)
        test_cv_pred["XGB_GPU_Hist"] += m_xgb.predict(dte) / FOLDS
        print(f"  [Fold {fi+1}] XGB_GPU_Hist AUC       : {roc_auc_score(y_va, oof_preds['XGB_GPU_Hist'][vai]):.5f}")

        # 8. CatBoost GPU Specialist with Decoded Tariff Keys
        X_tr_cb = pd.concat([X_tr.reset_index(drop=True).fillna(0.0), cb_train.iloc[tri].reset_index(drop=True)], axis=1)
        X_va_cb = pd.concat([X_va.reset_index(drop=True).fillna(0.0), cb_train.iloc[vai].reset_index(drop=True)], axis=1)
        X_te_cb = pd.concat([X_test.reset_index(drop=True).fillna(0.0), cb_test.reset_index(drop=True)], axis=1)

        m_cb = CatBoostClassifier(iterations=850, learning_rate=0.035, depth=6,
                                  l2_leaf_reg=20.0, task_type="GPU", verbose=0, random_seed=42+fi)
        m_cb.fit(X_tr_cb, y_tr, cat_features=cb_cats_cols)
        oof_preds["CB_GPU"][vai] = m_cb.predict_proba(X_va_cb)[:, 1]
        test_cv_pred["CB_GPU"] += m_cb.predict_proba(X_te_cb)[:, 1] / FOLDS
        print(f"  [Fold {fi+1}] CB_GPU AUC             : {roc_auc_score(y_va, oof_preds['CB_GPU'][vai]):.5f}")
        print(f"  --> Fold {fi+1} complete in {time.time()-t_fold:.1f}s")

    print("\n" + "=" * 80)
    print("POOLED OUT-OF-FOLD MODEL PERFORMANCE (ROC-AUC & PR-AUC):")
    print("=" * 80)
    oof_scores = {}
    for m in model_names:
        score_auc = roc_auc_score(y_train, oof_preds[m])
        score_ap  = average_precision_score(y_train, oof_preds[m])
        oof_scores[m] = score_auc
        print(f"  * {m:18s} OOF ROC-AUC: {score_auc:.6f} | PR-AUC: {score_ap:.6f}")

    # Full-Train Multi-Seed Re-Fit Calibration across all 8 models on 100% Data
    print("\n[*] Calibrating with 100% Full-Dataset Multi-Seed Re-Fits across Models (Seeds [42, 7]):")
    t_refit = time.time()
    dtr_full = lgb.Dataset(X_train, label=y_train, free_raw_data=False, params={'feature_pre_filter': False})

    X_train_clean = X_train.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_test_clean  = X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    dtr_xgb_full = xgb.DMatrix(X_train_clean.values.astype(np.float32), label=y_train, missing=0.0)
    dte_xgb_full = xgb.DMatrix(X_test_clean.values.astype(np.float32), missing=0.0)

    for sd in [42, 7]:
        # 1. LightGBM ExtraTrees E1 Refit
        m1_f = lgb.train(dict(objective="binary", learning_rate=0.030, num_leaves=25, max_depth=5,
                              min_child_samples=90, feature_fraction=0.45, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=10.0, extra_trees=True, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd),
                         dtr_full, num_boost_round=800)
        test_refits["LGB_ET_E1"] += m1_f.predict(X_test) / 2.0

        # 2. LightGBM Shallow Anchor Refit
        m2_f = lgb.train(dict(objective="binary", learning_rate=0.035, num_leaves=15, max_depth=4,
                              min_child_samples=180, feature_fraction=0.45, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=30.0, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd+10),
                         dtr_full, num_boost_round=750)
        test_refits["LGB_Shallow"] += m2_f.predict(X_test) / 2.0

        # 3. LightGBM ExtraTrees Deep Refit
        m3_f = lgb.train(dict(objective="binary", learning_rate=0.025, num_leaves=35, max_depth=7,
                              min_child_samples=120, feature_fraction=0.38, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=30.0, extra_trees=True, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd+20),
                         dtr_full, num_boost_round=800)
        test_refits["LGB_ET_Deep"] += m3_f.predict(X_test) / 2.0

        # 4. LightGBM Deep Conservative Refit
        m4_f = lgb.train(dict(objective="binary", learning_rate=0.028, num_leaves=45, max_depth=7,
                              min_child_samples=60, feature_fraction=0.55, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd+30),
                         dtr_full, num_boost_round=750)
        test_refits["LGB_Conservative"] += m4_f.predict(X_test) / 2.0

        # 5. LightGBM Wide Interactor Refit
        m5_f = lgb.train(dict(objective="binary", learning_rate=0.024, num_leaves=63, max_depth=8,
                              min_child_samples=250, feature_fraction=0.30, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=50.0, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd+40),
                         dtr_full, num_boost_round=750)
        test_refits["LGB_Wide"] += m5_f.predict(X_test) / 2.0

        # 6. LightGBM Solvency Specialist Refit
        m6_f = lgb.train(dict(objective="binary", learning_rate=0.030, num_leaves=22, max_depth=5,
                              min_child_samples=150, feature_fraction=0.40, bagging_fraction=0.8,
                              bagging_freq=1, lambda_l2=20.0, verbose=-1, n_jobs=10,
                              feature_pre_filter=False, seed=sd+50),
                         dtr_full, num_boost_round=750)
        test_refits["LGB_Solvency"] += m6_f.predict(X_test) / 2.0

        # 7. XGBoost GPU Hist Refit
        p_xgb_f = dict(tree_method="hist", device="cuda", max_depth=6, max_bin=512,
                       learning_rate=0.024, subsample=0.80, colsample_bytree=0.42,
                       reg_alpha=2.0, reg_lambda=20.0, objective="binary:logistic", seed=sd)
        m_xgb_f = xgb.train(p_xgb_f, dtr_xgb_full, num_boost_round=800)
        test_refits["XGB_GPU_Hist"] += m_xgb_f.predict(dte_xgb_full) / 2.0

        # 8. CatBoost GPU Refit
        X_tr_cb_full = pd.concat([X_train.reset_index(drop=True).fillna(0.0), cb_train.reset_index(drop=True)], axis=1)
        X_te_cb_full = pd.concat([X_test.reset_index(drop=True).fillna(0.0), cb_test.reset_index(drop=True)], axis=1)
        m_cbf = CatBoostClassifier(iterations=800, learning_rate=0.030, depth=6,
                                   l2_leaf_reg=20.0, task_type="GPU", verbose=0, random_seed=sd)
        m_cbf.fit(X_tr_cb_full, y_train, cat_features=cb_cats_cols)
        test_refits["CB_GPU"] += m_cbf.predict_proba(X_te_cb_full)[:, 1] / 2.0

    # Combine K-Fold Test Averages with Full-Train Re-Fit Averages (50/50 blend)
    test_final_members = {}
    for m in model_names:
        if m in test_refits and test_refits[m].max() > 0:
            test_final_members[m] = rk_norm(0.50 * rk_norm(test_cv_pred[m]) + 0.50 * rk_norm(test_refits[m]))
        else:
            test_final_members[m] = rk_norm(test_cv_pred[m])
    print(f"  [+] 100% Full-Dataset Re-Fit Calibration completed in {time.time()-t_refit:.1f}s")

    # Compute Continuous Regime Gates
    ext_tr = df_train_raw[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1).values
    pens_tr = (df_train_raw['DAYS_EMPLOYED'] == 365243).astype(float).values
    male_tr = (df_train_raw['CODE_GENDER'] == 'M').astype(float).values
    s_dist_tr = ((3.0 - ext_tr) / 3.0) * 0.50 + pens_tr * 0.25 + male_tr * 0.25
    g_dist_tr = 1.0 / (1.0 + np.exp(-10.0 * (s_dist_tr - 0.40)))
    rev_tr = (df_train_raw['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(float).values

    ext_te = df_test_raw[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].notna().sum(axis=1).values
    pens_te = (df_test_raw['DAYS_EMPLOYED'] == 365243).astype(float).values
    male_te = (df_test_raw['CODE_GENDER'] == 'M').astype(float).values
    s_dist_te = ((3.0 - ext_te) / 3.0) * 0.50 + pens_te * 0.25 + male_te * 0.25
    g_dist_te = 1.0 / (1.0 + np.exp(-10.0 * (s_dist_te - 0.40)))
    rev_te = (df_test_raw['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype(float).values

    # Continuous Solvency & Contract Regime Routing Optimization (SLSQP Bounded Simplex)
    print("\n[*] Solving Continuous Solvency & Contract Regime Routing Weights (SLSQP Bounded Simplex)...")
    Z_oof  = np.column_stack([to_z(rk_norm(oof_preds[m])) for m in model_names])
    Z_test = np.column_stack([to_z(test_final_members[m]) for m in model_names])
    K = len(model_names)

    def optimize_regime_weights(mask_regime, name):
        y_sub = y_train[mask_regime]
        Z_sub = Z_oof[mask_regime]
        
        def loss_sub(w):
            pred = Z_sub @ w
            p = 1.0 / (1.0 + np.exp(-pred))
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            return -np.mean(y_sub * np.log(p) + (1.0 - y_sub) * np.log(1.0 - p))

        bnds = [(0.02, 0.40) for _ in range(K)]
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        init_w = np.ones(K) / K
        res = minimize(loss_sub, init_w, method='SLSQP', bounds=bnds, constraints=cons)
        w_res = res.x / np.sum(res.x)
        sub_auc = roc_auc_score(y_sub, Z_sub @ w_res)
        print(f"  [+] Regime {name:<12s} SLSQP Converged (AUC: {sub_auc:.5f})")
        return w_res

    # Masks for 3 regimes
    mask_prime = (rev_tr == 0) & (g_dist_tr < 0.50)
    mask_distress = (rev_tr == 0) & (g_dist_tr >= 0.50)
    mask_rev = (rev_tr == 1)

    wp = optimize_regime_weights(mask_prime, "Prime")
    wd = optimize_regime_weights(mask_distress, "Distress")
    wr = optimize_regime_weights(mask_rev, "Revolving")

    z_p_oof = Z_oof @ wp
    z_d_oof = Z_oof @ wd
    z_r_oof = Z_oof @ wr
    z_routed_oof = (1.0 - rev_tr) * ((1.0 - g_dist_tr) * z_p_oof + g_dist_tr * z_d_oof) + rev_tr * z_r_oof
    regime_oof_auc = roc_auc_score(y_train, z_routed_oof)

    print(f"\n[*] Continuous Regime Routing Optimal Weights Solved (OOF AUC: {regime_oof_auc:.6f}):")
    print("    Model Name           | Prime Share | Distress Share | Revolving Share")
    print("    ---------------------------------------------------------------------")
    for idx_m, m in enumerate(model_names):
        print(f"    - {m:18s} |   {wp[idx_m]*100:5.2f}%   |    {wd[idx_m]*100:5.2f}%    |    {wr[idx_m]*100:5.2f}%")

    z_test_p = Z_test @ wp
    z_test_d = Z_test @ wd
    z_test_r = Z_test @ wr
    z_test_routed = (1.0 - rev_te) * ((1.0 - g_dist_te) * z_test_p + g_dist_te * z_test_d) + rev_te * z_test_r
    test_regime = rk_norm(to_p(z_test_routed))

    # Equal-Weighted Tree Consensus (Maximal Entropy Diversity from V43 Benchmark)
    tree_members = ["LGB_ET_E1", "LGB_Shallow", "LGB_ET_Deep", "LGB_Solvency", "LGB_Conservative", "LGB_Wide"]
    w_tree = sum(oof_preds[m] for m in tree_members) / len(tree_members)
    w_tree_te = sum(test_final_members[m] for m in tree_members) / len(tree_members)

    oof_macro = rk_norm(0.65 * rk_norm(w_tree) + 0.20 * rk_norm(oof_preds["CB_GPU"]) + 0.15 * rk_norm(oof_preds["XGB_GPU_Hist"]))
    test_macro = rk_norm(0.65 * rk_norm(w_tree_te) + 0.20 * test_final_members["CB_GPU"] + 0.15 * test_final_members["XGB_GPU_Hist"])
    macro_oof_auc = roc_auc_score(y_train, oof_macro)
    print(f"\n[*] Macro Structural Consensus OOF ROC-AUC: {macro_oof_auc:.6f}")

    # Unified Dual Consensus (65% Regime Routing + 35% Macro Consensus)
    oof_unified = rk_norm(0.65 * rk_norm(to_p(z_routed_oof)) + 0.35 * oof_macro)
    unified_oof_auc = roc_auc_score(y_train, oof_unified)
    test_ensemble = rk_norm(0.65 * test_regime + 0.35 * test_macro)

    print(f"\n  >>> TOTAL UNIFIED DUAL CONSENSUS OOF ROC-AUC: {unified_oof_auc:.6f} <<<")

    # Deep Operational Risk Telemetry on Final Unified Ensemble
    print_deep_risk_telemetry(y_train, oof_unified, df_train_raw=df_train_raw, title="UNIFIED DUAL CONSENSUS")
    return test_ensemble, unified_oof_auc

# ==============================================================================
# SECTION 6: MAIN PIPELINE EXECUTION & EXPORT
# ==============================================================================

def main():
    t_global = time.time()
    print("=" * 80)
    print("GCI WORLD 2026: V48 GRAND APEX SOVEREIGN ZENITH MASTER PIPELINE")
    print("100% PURE SUPERVISED INDUCTIVE LEARNING (ZERO CSV BLENDS, ZERO TARGET LEAKAGE)")
    print("OPERATIONAL RISK TELEMETRY: CONFUSION MATRIX, PRECISION, RECALL & PR-AUC")
    print("=" * 80)

    tr_path, te_path, sub_path = resolve_data_paths()
    verify_provenance(tr_path, te_path, sub_path)

    df_train = pd.read_csv(tr_path)
    df_test  = pd.read_csv(te_path)
    sample_sub = pd.read_csv(sub_path)

    y_train = df_train['TARGET'].values.astype(np.int32)
    n_train = len(df_train)
    n_test  = len(df_test)

    assert n_test == len(sample_sub), f"Row count mismatch: test={n_test}, sample={len(sample_sub)}"
    assert (df_test['SK_ID_CURR'].values == sample_sub['SK_ID_CURR'].values).all(), "ID alignment mismatch!"

    cov_train = df_train.drop(columns=['TARGET'])
    df_all = pd.concat([cov_train, df_test], ignore_index=True)

    print("\n[*] Constructing 100% From-Scratch Feature Universe:")
    # 1. Base Covariates & Actuarial Decoders (76 features - Bedrock V43)
    df_base = build_complete_base_manifold(df_all)

    # 2. Restored Baseline Numerical Features (21 features - Dynamic raw extraction)
    df_raw = extract_restored_raw_baseline_features(df_all, df_base)

    # 3. Hierarchical Tariff Decoders & Exceptions (15 features)
    df_hq = compute_hierarchical_tariff_exception_features(df_all)

    # 4. Hasan Mazi Solvency & Banking Standards (25 features)
    df_hasan = compute_hasan_banking_and_solvency_standards(df_all)

    # 5. Temporal Stability & Inquiry Dynamics (14 features)
    df_temp = compute_temporal_stability_and_inquiry_bursts(df_all)

    # 6. Non-Linear External Credit Powers (6 features)
    df_ext_p = compute_nonlinear_external_credit_powers(df_all)

    # 7. Unsupervised Chronological Timeline Tracking (3 features)
    df_time = compute_unsupervised_chronological_timeline(df_all)

    # 8. Unsupervised Person Pooling (16 features)
    df_person = compute_unsupervised_person_pooling(df_all)

    # 9. Group Relative Deviations (28 features)
    df_group = compute_group_deviations(df_all)

    # 10. Deep Product & Target Encodings (23 features)
    df_te = compute_deep_product_and_target_encodings(df_all, y_train, n_train)

    # 11. Covariate Expectations & Residuals (20 features)
    df_resid = compute_covariate_expectations_and_residuals(df_base, n_train)

    # 12. Continuous Frequency Counts (6 features)
    df_cnt = compute_continuous_frequency_counts(df_all)

    # 13. Orthogonal One-Hot Categoricals (115 features)
    df_oh = compute_onehot_categoricals(df_all)

    # 14. 4-View Multi-Domain Stacking Signals (8 features)
    df_views = compute_multiview_stacking(df_base, y_train, n_train)

    # 5. Hasan Mazi High-Order Pairwise Interactions (140 features)
    df_h_inter = compute_hasan_high_order_pairwise_interactions(df_all, df_base)

    # Assemble Full Pure-Scratch Feature Matrix
    X_full = pd.concat([
        df_base, df_raw, df_hq, df_hasan, df_h_inter, df_temp, df_ext_p,
        df_time, df_person, df_group, df_te, df_resid, df_cnt, df_oh, df_views
    ], axis=1)
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X_full.columns = [re.sub(r'[\[\]{}:",\s]', '_', str(c)) for c in X_full.columns]
    print(f"\n[+] Total Sovereign Feature Universe: {X_full.shape[1]} features across {len(X_full):,} rows (Zero Target Leakage).")

    # Build High-Cardinality Decoded Tariff Keys for CatBoost GPU
    cr_arr = df_all['AMT_CREDIT'].values.astype(float)
    an_arr = df_all['AMT_ANNUITY'].values.astype(float)
    cash_arr = (df_all['NAME_CONTRACT_TYPE'] == 'Cash loans').values
    q_raw = an_arr / np.maximum(cr_arr, 1.0)
    q_dec = np.where(cash_arr, (an_arr - 129.5) / np.maximum(cr_arr, 1.0), q_raw)
    q5 = np.round(q_raw, 5)
    qb5 = np.round(q_dec, 5)
    vc_q5 = pd.Series(q5).value_counts()
    vc_qb5 = pd.Series(qb5).value_counts()
    fee_flag = ((pd.Series(qb5).map(vc_qb5).values > pd.Series(q5).map(vc_q5).values) & cash_arr).astype(int)
    Q0 = np.where(fee_flag == 1, q_dec, q_raw)
    CT_arr = cr_arr / np.maximum(an_arr, 1.0)
    gp_arr = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(float)
    kf = np.where((cr_arr > gp_arr) & (gp_arr > 0), (cr_arr / gp_arr - 1.0) / 0.0132, 0.0)
    k_round = np.round(kf)
    ok_k = cash_arr & (cr_arr > gp_arr) & (np.abs(kf - k_round) < 0.1)
    k_dec = np.where(ok_k, k_round, np.where(cash_arr & (cr_arr == gp_arr), 0.0, np.nan))

    cb_df = pd.DataFrame(index=df_all.index)
    for c in CATS:
        cb_df[c] = df_all[c].astype(str)
    cb_df['Q3'] = pd.Series(q_raw).round(3).astype(str)
    cb_df['Q4'] = pd.Series(q_raw).round(4).astype(str)
    cb_df['P3'] = pd.Series(Q0).round(3).astype(str)
    cb_df['P4'] = pd.Series(Q0).round(4).astype(str)
    cb_df['PF4'] = pd.Series(Q0).round(4).astype(str) + "_" + pd.Series(fee_flag).astype(str)
    cb_df['CT1'] = pd.Series(CT_arr).round(1).astype(str)
    cb_df['CT2'] = pd.Series(CT_arr).round(2).astype(str)
    cb_df['CAT_K_DEC'] = pd.Series(np.nan_to_num(k_dec, nan=-1.0)).round().astype(int).astype(str)

    X_train = X_full.iloc[:n_train].reset_index(drop=True)
    X_test  = X_full.iloc[n_train:].reset_index(drop=True)

    # Train 8-Engine Grandmaster Suite with Full-Train Re-Fit & Dual Consensus Ensembling
    test_ensemble, best_oof_auc = train_grandmaster_suite(X_train, y_train, X_test, df_train, df_test, cb_df)

    # Asymmetric Tail Sharpening (Anchor 74 Monotonic Transform)
    print("\n[*] Applying Asymmetric Tail Sharpening (Monotonic Power Expansion):")
    r_test = rk_norm(test_ensemble)
    z = norm.ppf(np.clip(r_test, 1e-6, 1.0 - 1e-6))

    tau = 1.28
    cond_up = z > tau
    cond_dn = z < -tau
    z_sharp = z.copy()
    z_sharp[cond_up] = tau + np.power(z[cond_up] - tau, 1.020)
    z_sharp[cond_dn] = -tau - np.power(np.abs(z[cond_dn]) - tau, 1.008)

    p_final = rk_norm(norm.cdf(z_sharp))

    # Schema Assertions & Output Generation
    out_csv = r"d:/gci/v_22_pure_champion_ensemble_certified_v48.csv"
    out_df = pd.DataFrame({
        "SK_ID_CURR": sample_sub["SK_ID_CURR"].values,
        "TARGET": p_final
    })

    assert len(out_df) == 61500, f"Expected 61,500 rows, got {len(out_df)}"
    assert list(out_df.columns) == ["SK_ID_CURR", "TARGET"], "Invalid column schema!"
    assert out_df["SK_ID_CURR"].equals(sample_sub["SK_ID_CURR"]), "ID sequence does not match sample submission!"
    assert out_df["TARGET"].isnull().sum() == 0, "Submission contains NaNs!"
    assert np.isfinite(out_df["TARGET"].values).all(), "Submission contains non-finite values!"
    assert (out_df["TARGET"].values >= 0.0).all() and (out_df["TARGET"].values <= 1.0).all(), "Values out of bounds!"

    out_df.to_csv(out_csv, index=False)
    file_sha = hashlib.sha256(open(out_csv, "rb").read()).hexdigest().upper()
    file_size = os.path.getsize(out_csv)

    print("\n" + "=" * 80)
    print(f"[+] CERTIFIED V48 GRAND APEX SOVEREIGN ZENITH SUBMISSION GENERATED:")
    print(f"    - File Path : {out_csv}")
    print(f"    - File Size : {file_size:,} bytes")
    print(f"    - SHA-256   : {file_sha}")
    print(f"    - OOF ROC-AUC: {best_oof_auc:.6f}")
    print(f"    - Total Time: {(time.time()-t_global)/60:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()
