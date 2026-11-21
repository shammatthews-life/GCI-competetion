import sys, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

print("=" * 80)
print("TESTING V44 ADVANCED MATHEMATICAL FEATURE MANIFOLD")
print("=" * 80)

# Load small sample of train and test
tr_path = "inpuy/train.csv"
te_path = "inpuy/test.csv"

t0 = time.time()
df_tr = pd.read_csv(tr_path, nrows=5000)
df_te = pd.read_csv(te_path, nrows=2000)
df_all = pd.concat([df_tr.drop(columns=['TARGET']), df_te], axis=0, ignore_index=True)
n_tr, n_te = len(df_tr), len(df_te)
print(f"Loaded {n_tr} train and {n_te} test rows in {time.time()-t0:.2f}s")

# 1. Russian Milestone Passport Renewal Delays
birth = df_all['DAYS_BIRTH'].values.astype(float)
id_pub = df_all['DAYS_ID_PUBLISH'].values.astype(float)
reg = df_all['DAYS_REGISTRATION'].values.astype(float)
emp = df_all['DAYS_EMPLOYED'].values.astype(float)
emp = np.where(emp == 365243, np.nan, emp)

age = -birth / 365.25
age_id = (id_pub - birth) / 365.25
age_reg = (reg - birth) / 365.25
age_emp = (emp - birth) / 365.25

id_late_20 = ((age >= 20.25) & (age_id < 20.0)).astype(np.float32)
id_late_45 = ((age >= 45.25) & (age_id < 45.0)).astype(np.float32)
id_delay = np.where(age_id >= 45.0, age_id - 45.0,
           np.where(age_id >= 20.0, age_id - 20.0, age_id - 14.0)).astype(np.float32)

reg_minus_id = ((reg - id_pub) / 365.25).astype(np.float32)
emp_minus_id = ((emp - id_pub) / 365.25).astype(np.float32)
emp_minus_reg = ((emp - reg) / 365.25).astype(np.float32)

df_milestone = pd.DataFrame({
    'AGE_AT_ID': age_id.astype(np.float32),
    'AGE_AT_REG': age_reg.astype(np.float32),
    'AGE_AT_EMP': age_emp.astype(np.float32),
    'ID_LATE_20': id_late_20,
    'ID_LATE_45': id_late_45,
    'ID_DELAY_AFTER_MILESTONE': id_delay,
    'REG_MINUS_ID': reg_minus_id,
    'EMP_MINUS_ID': emp_minus_id,
    'EMP_MINUS_REG': emp_minus_reg
})
print(f"[1] Milestone Passport Features: {df_milestone.shape[1]} columns. ID_LATE_20 mean: {id_late_20.mean():.4f}")

# 2. Alexander_L Implied Interest & Tariff Benchmarks
cr = df_all['AMT_CREDIT'].values.astype(float)
an = df_all['AMT_ANNUITY'].values.astype(float)
gp = df_all['AMT_GOODS_PRICE'].values.astype(float)
eps = 1e-5

best_rate = np.full(len(df_all), np.nan, dtype=np.float32)
best_term = np.full(len(df_all), np.nan, dtype=np.float32)
with np.errstate(all='ignore'):
    for n in [6, 12, 18, 24, 30, 36, 48, 60]:
        annual = (an / (cr + eps) - 1.0 / n) * 12.0
        mask = (annual > 0.02) & (annual < 0.90)
        closer = np.abs(annual - 0.25) < np.abs(best_rate - 0.25)
        upd = mask & (np.isnan(best_rate) | closer)
        best_rate[upd] = annual[upd]
        best_term[upd] = float(n)

df_interest = pd.DataFrame({
    'ALEX_IMPLIED_RATE': best_rate,
    'ALEX_IMPLIED_TERM': best_term,
    'ALEX_RATE_x_INCOME': (best_rate * df_all['AMT_INCOME_TOTAL'].values).astype(np.float32),
    'ALEX_RATE_x_CREDIT': (best_rate * cr).astype(np.float32)
})
print(f"[2] Alexander_L Interest Features: {df_interest.shape[1]} columns. Valid rates: {(~np.isnan(best_rate)).mean():.4f}")

# 3. Catalog Peak Clustering on C/A
t = cr / np.maximum(an, 1.0)
r1 = np.round(t, 1)
s_r1 = pd.Series(r1)
vc = s_r1.value_counts()
peaks = np.sort(vc.index[:20].values)
d_peaks = np.abs(t[:, None] - peaks[None, :])
j_peak = np.nanargmin(d_peaks, axis=1)

df_catalog = pd.DataFrame({
    'W_PROD_R1': r1.astype(np.float32),
    'W_PROD_FREQ': (s_r1.map(vc) / len(df_all)).values.astype(np.float32),
    'W_PEAK_DIST': np.abs(t - peaks[j_peak]).astype(np.float32),
    'W_PEAK_RES': (t - peaks[j_peak]).astype(np.float32),
    'W_PEAK_VAL': peaks[j_peak].astype(np.float32)
})
print(f"[3] Catalog Peak Features: {df_catalog.shape[1]} columns. Mean peak dist: {df_catalog['W_PEAK_DIST'].mean():.4f}")

# 4. Supervised OOF EXT Imputation
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
E = df_all[ext_cols].values.astype(np.float64)

# Candidate predictors for EXT: continuous applicant attributes
pred_cols = ['DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'AMT_INCOME_TOTAL',
             'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE']
X_pred = df_all[pred_cols].fillna(0.0).values.astype(np.float32)

t_imp = time.time()
EI = {}
kf = KFold(5, shuffle=True, random_state=42)
for k_idx, tcol in enumerate(ext_cols):
    yv = E[:, k_idx]
    present = ~np.isnan(yv)
    ih, im = np.where(present)[0], np.where(~present)[0]
    pred = np.zeros(len(df_all), dtype=np.float32)
    for tr_i, va_i in kf.split(ih):
        a_idx, b_idx = ih[tr_i], ih[va_i]
        m = HistGradientBoostingRegressor(max_iter=50, max_leaf_nodes=31, l2_regularization=5.0, random_state=42)
        m.fit(X_pred[a_idx], yv[a_idx])
        pred[b_idx] = m.predict(X_pred[b_idx])
        if len(im):
            pred[im] += m.predict(X_pred[im]) / 5.0
    tag = tcol.replace('EXT_SOURCE_', 'EXT')
    EI[f'{tag}_PRED'] = pred
    EI[f'{tag}_FILL'] = np.where(present, yv, pred).astype(np.float32)
    EI[f'{tag}_RESID'] = np.where(present, yv - pred, 0.0).astype(np.float32)

df_ext_imp = pd.DataFrame(EI)
df_ext_imp['EXT_FILL_MEAN'] = df_ext_imp[[c for c in df_ext_imp.columns if c.endswith('_FILL')]].mean(axis=1).astype(np.float32)
print(f"[4] OOF EXT Imputation: {df_ext_imp.shape[1]} columns computed in {time.time()-t_imp:.2f}s")

# 5. Timeline Deltas across Repeat Applicants
gender = df_all['CODE_GENDER'].astype(str)
edu = df_all['NAME_EDUCATION_TYPE'].astype(str)
k_entity = (birth - reg).astype(str) + '|' + (birth - id_pub).astype(str) + '|' + gender + '|' + edu

tmp = pd.DataFrame({
    'k': k_entity,
    'b': -birth,
    'cr': cr,
    'an': an,
    'inc': df_all['AMT_INCOME_TOTAL'].values,
    'ext': df_ext_imp['EXT_FILL_MEAN'].values,
    'row_idx': np.arange(len(df_all))
})
df_s = tmp.sort_values(['k', 'b']).copy()
df_s['DAYS_TO_NEXT'] = -df_s.groupby('k')['b'].diff(-1).fillna(9999)
df_s['D_CRED_PREV'] = df_s.groupby('k')['cr'].diff().fillna(0)
df_s['D_ANN_PREV'] = df_s.groupby('k')['an'].diff().fillna(0)
df_s['D_INC_PREV'] = df_s.groupby('k')['inc'].diff().fillna(0)
df_s['D_EXT_PREV'] = df_s.groupby('k')['ext'].diff().fillna(0)
df_rest = df_s.sort_values('row_idx')

df_timeline_deltas = pd.DataFrame({
    'DAYS_TO_NEXT_APP': df_rest['DAYS_TO_NEXT'].values.astype(np.float32),
    'TL_D_CRED_PREV': df_rest['D_CRED_PREV'].values.astype(np.float32),
    'TL_D_ANN_PREV': df_rest['D_ANN_PREV'].values.astype(np.float32),
    'TL_D_INC_PREV': df_rest['D_INC_PREV'].values.astype(np.float32),
    'TL_D_EXT_PREV': df_rest['D_EXT_PREV'].values.astype(np.float32)
})
print(f"[5] Timeline Deltas: {df_timeline_deltas.shape[1]} columns. Nonzero credit diffs: {(df_timeline_deltas['TL_D_CRED_PREV'] != 0).sum()}")

print("\nALL V44 MATHEMATICAL COMPONENTS VERIFIED SUCCESSFULLY!")
