import time
import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

print("[*] Testing Inductive Feature Transformer...")
t0 = time.time()
df_tr = pd.read_csv('d:/gci/inpuy/train.csv', nrows=10000)
y_tr = df_tr['TARGET'].values

def solve_implied_rate(c, a, n):
    if c <= 0 or a <= 0 or np.isnan(c) or np.isnan(a):
        return np.nan
    target = a / c
    min_required = 1.0 / n
    if target < min_required - 1e-7:
        return np.nan
    if abs(target - min_required) <= 1e-7:
        return 0.0

    def f(m):
        return (m / 12.0) / (1.0 - math.pow(1.0 + m / 12.0, -n)) - target

    low = 1e-6
    high = 12.0
    if f(high) < 0:
        high = 50.0
        if f(high) < 0:
            return np.nan

    for _ in range(30):
        mid = 0.5 * (low + high)
        val = f(mid)
        if abs(val) < 1e-6:
            return mid
        if val > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

# Row-wise base features
def extract_row_features(df):
    feats = pd.DataFrame(index=df.index)
    
    # 1. External Sources Triad
    ext1 = df['EXT_SOURCE_1'].values
    ext2 = df['EXT_SOURCE_2'].values
    ext3 = df['EXT_SOURCE_3'].values
    
    feats['EXT_SOURCE_1'] = ext1
    feats['EXT_SOURCE_2'] = ext2
    feats['EXT_SOURCE_3'] = ext3
    feats['EXT_MEAN'] = np.nanmean(np.column_stack([ext1, ext2, ext3]), axis=1)
    feats['EXT_STD']  = np.nanstd(np.column_stack([ext1, ext2, ext3]), axis=1)
    feats['EXT_MIN']  = np.nanmin(np.column_stack([ext1, ext2, ext3]), axis=1)
    feats['EXT_MAX']  = np.nanmax(np.column_stack([ext1, ext2, ext3]), axis=1)
    feats['EXT_PROD'] = np.where(np.isnan(ext1)|np.isnan(ext2)|np.isnan(ext3), np.nan, ext1 * ext2 * ext3)
    feats['EXT_NOBS'] = (~np.isnan(ext1)).astype(int) + (~np.isnan(ext2)).astype(int) + (~np.isnan(ext3)).astype(int)
    
    feats['EXT_DIFF_1_2'] = ext1 - ext2
    feats['EXT_DIFF_2_3'] = ext2 - ext3
    feats['EXT_DIFF_1_3'] = ext1 - ext3
    feats['EXT_RATIO_1_2'] = ext1 / (ext2 + 1e-4)
    feats['EXT_RATIO_2_3'] = ext2 / (ext3 + 1e-4)
    
    # Harmonic and geometric mean
    u1 = np.clip(np.nan_to_num(ext1, nan=0.5), 1e-4, 1.0)
    u2 = np.clip(np.nan_to_num(ext2, nan=0.5), 1e-4, 1.0)
    u3 = np.clip(np.nan_to_num(ext3, nan=0.5), 1e-4, 1.0)
    feats['EXT_HARMONIC'] = 3.0 / (1.0 / u1 + 1.0 / u2 + 1.0 / u3)
    feats['EXT_GEOMETRIC'] = (u1 * u2 * u3) ** (1.0 / 3.0)
    feats['EXT_WEIGHTED'] = (u1 * 2.0 + u2 * 3.0 + u3 * 4.0) / 9.0
    
    # 2. Financial Metrics & Solvency Ratios
    inc = df['AMT_INCOME_TOTAL'].values
    cr  = df['AMT_CREDIT'].values
    an  = df['AMT_ANNUITY'].values
    gp  = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT']).values
    
    feats['AMT_INCOME_TOTAL'] = inc
    feats['AMT_CREDIT'] = cr
    feats['AMT_ANNUITY'] = an
    feats['AMT_GOODS_PRICE'] = gp
    
    pay = an / (cr + 1e-4)
    cti = cr / (inc + 1e-4)
    ati = an / (inc + 1e-4)
    ltv = cr / (gp + 1e-4)
    
    feats['PAYMENT_RATE'] = pay
    feats['CREDIT_TO_INCOME'] = cti
    feats['ANNUITY_TO_INCOME'] = ati
    feats['LTV'] = ltv
    feats['FINANCED_MARKUP'] = (cr - gp) / (gp + 1e-4)
    feats['GOODS_TO_CREDIT'] = gp / (cr + 1e-4)
    feats['CREDIT_TERM_MONTHS'] = cr / (an + 1.0)
    
    # Living expense estimate and cash surplus
    fam = df['CNT_FAM_MEMBERS'].fillna(1.0).values
    living_cost_est = (inc / 12.0) * 0.35 + (fam * 5000.0)
    surplus = np.maximum(inc / 12.0 - an - living_cost_est, 0.0)
    feats['MONTHLY_SURPLUS'] = surplus
    feats['SURPLUS_TO_INCOME'] = surplus / (inc / 12.0 + 1e-4)
    
    # DSCR
    dscr = (inc / 12.0) / (an + 1.0)
    feats['DSCR'] = dscr
    feats['DSCR_BELOW_1'] = (dscr < 1.0).astype(float)
    
    # Consumer Altman Z-score proxy
    feats['ALTMAN_Z_SCORE'] = (
        1.2 * (surplus / (an + 1.0)) +
        1.4 * (inc / (cr + 1.0)) +
        3.3 * feats['EXT_MEAN'].fillna(0.4) +
        0.6 * (gp / (cr + 1.0))
    )
    
    # 3. Clayton Copula Lower-Tail Joint Distress
    for theta in [1.5, 2.5]:
        val = np.maximum(u1**(-theta) + u2**(-theta) + u3**(-theta) - 2.0, 0.0)
        feats[f'COPULA_TAIL_{str(theta).replace(".", "")}'] = val ** (-1.0 / theta)
    
    # 4. Merton Structural Default Distance
    assets = inc * 3.0 + gp
    liabilities = cr + 12.0 * an + 1.0
    vol = np.nan_to_num(feats['EXT_STD'], nan=0.2) + 0.15
    merton_dd = (np.log(np.clip(assets / liabilities, 1e-4, None)) + 0.05) / vol
    feats['MERTON_PD'] = norm.cdf(-merton_dd)
    
    # 5. Demographics & Longevity
    db = df['DAYS_BIRTH'].values
    de = df['DAYS_EMPLOYED'].values
    feats['DAYS_BIRTH'] = db
    feats['DAYS_EMPLOYED'] = de
    feats['DAYS_EMPLOYED_ANOM'] = (de == 365243).astype(float)
    clean_emp = np.where(de == 365243, 0.0, -de)
    feats['AGE_YEARS'] = -db / 365.25
    feats['WORK_LIFE_RATIO'] = clean_emp / (-db + 1.0)
    feats['REG_TO_AGE'] = (-df['DAYS_REGISTRATION'].values) / (-db + 1.0)
    feats['ID_TO_AGE'] = (-df['DAYS_ID_PUBLISH'].values) / (-db + 1.0)
    
    ph = df['DAYS_LAST_PHONE_CHANGE'].fillna(0.0).values
    feats['PHONE_TO_AGE'] = (-ph) / (-db + 1.0)
    feats['PHONE_TO_EMP'] = (-ph) / (clean_emp + 1.0)
    
    # 6. Tariff and Discrete Decoders
    is_cash = (df['NAME_CONTRACT_TYPE'] == 'Cash loans').astype(int).values
    feats['Q_DEC'] = np.where(is_cash == 1, (an - 129.5) / np.maximum(cr, 1.0), pay)
    feats['K_DEC'] = np.where(is_cash == 1, np.round(((cr / np.maximum(gp, 1.0)) - 1.0) / 0.0132), 0.0)
    
    # 7. Bureau Inquiry Dynamics
    inq_qrt = df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0.0).values
    inq_yr  = df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(0.0).values
    feats['INQ_ACCEL'] = inq_qrt / (inq_yr / 4.0 + 0.25)
    feats['INQ_BURST'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'].fillna(0.0).values + df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0.0).values
    
    # 8. Cross Interactions
    feats['EXT_MEAN_x_PAY'] = feats['EXT_MEAN'] * pay
    feats['EXT_MEAN_x_CTI'] = feats['EXT_MEAN'] * cti
    feats['EXT_MEAN_x_SURPLUS'] = feats['EXT_MEAN'] * (surplus / 10000.0)
    
    return feats

print("Extracting features on 10,000 sample rows...")
t1 = time.time()
f_df = extract_row_features(df_tr)
print(f"Extracted {f_df.shape[1]} features in {time.time()-t1:.2f}s!")
print("Checking nulls and infinite values...")
assert not np.isinf(f_df.values).any(), "Found infinite values!"
print(f"Success! Script executed in {time.time()-t0:.2f}s.")
