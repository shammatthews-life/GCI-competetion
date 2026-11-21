# ==================================================================================================
# CELL 3: 143-Feature Quantitative Finance & In-Fold Relational Engineering Engine
# ==================================================================================================
print('\n' + '=' * 85)
print('  CELL 3: QUANTITATIVE FINANCE ALPHA TRINITY & IN-FOLD FEATURE ENGINE')
print('=' * 85)
t_fe_start = time.time()

df_all = pd.concat([df_train.drop(columns=['TARGET']), df_test], ignore_index=True)

CATS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'
]

emp_raw = df_all['DAYS_EMPLOYED'].values.astype(np.float64)
days_emp = np.where(emp_raw == 365243.0, np.nan, emp_raw)

inc = df_all['AMT_INCOME_TOTAL'].values.astype(np.float64)
cred = df_all['AMT_CREDIT'].values.astype(np.float64)
ann = df_all['AMT_ANNUITY'].values.astype(np.float64)
goods = df_all['AMT_GOODS_PRICE'].fillna(df_all['AMT_CREDIT']).values.astype(np.float64)

birth = df_all['DAYS_BIRTH'].values.astype(np.float64)
reg = df_all['DAYS_REGISTRATION'].values.astype(np.float64)
id_pub = df_all['DAYS_ID_PUBLISH'].values.astype(np.float64)
phone = df_all['DAYS_LAST_PHONE_CHANGE'].values.astype(np.float64)

e1 = df_all['EXT_SOURCE_1'].values.astype(np.float64)
e2 = df_all['EXT_SOURCE_2'].values.astype(np.float64)
e3 = df_all['EXT_SOURCE_3'].values.astype(np.float64)

F = {}

# 1. Cleanings & Sentinels
F['DAYS_EMPLOYED_CLEAN'] = days_emp.astype(np.float32)
F['IS_RETIRED'] = (emp_raw == 365243.0).astype(np.float32)
F['MISS_CNT'] = df_all.isna().sum(axis=1).values.astype(np.float32)
F['EXT_SOURCE_1_NA'] = np.isnan(e1).astype(np.float32)
F['EXT_SOURCE_2_NA'] = np.isnan(e2).astype(np.float32)
F['EXT_SOURCE_3_NA'] = np.isnan(e3).astype(np.float32)

# 2. Demographics & Relative Event Ages
age_years = -birth / 365.25
emp_years = -days_emp / 365.25
F['AGE_YEARS'] = age_years.astype(np.float32)
F['EMPLOYED_YEARS'] = emp_years.astype(np.float32)
F['EMP_TO_AGE'] = (days_emp / (birth + 1.0)).astype(np.float32)
F['REG_TO_AGE'] = (reg / (birth + 1.0)).astype(np.float32)
F['ID_TO_AGE'] = (id_pub / (birth + 1.0)).astype(np.float32)
F['PHONE_TO_AGE'] = (phone / (birth + 1.0)).astype(np.float32)

# 3. Financial Solvency & Banking Ratios
term = cred / (ann + 1.0)
F['CREDIT_TERM'] = term.astype(np.float32)
F['EAR'] = np.clip(24.0 * (ann * term - cred) / (cred * (term + 1.0) + 1.0), -1.0, 5.0).astype(np.float32)
surplus = inc - 12.0 * ann
F['SURPLUS'] = surplus.astype(np.float32)
F['SURPLUS_TO_INC'] = (surplus / (inc + 1.0)).astype(np.float32)
F['TRUE_COST'] = (ann * term - cred).astype(np.float32)
F['TRUE_COST_RATIO'] = ((ann * term - cred) / (cred + 1.0)).astype(np.float32)
F['LTV'] = np.clip(cred / (goods + 1.0), 0.0, 5.0).astype(np.float32)
F['DTI'] = np.clip(cred / (inc + 1.0), 0.0, 20.0).astype(np.float32)
F['DEBT_BURDEN'] = np.clip(ann / (inc / 12.0 + 1.0), 0.0, 5.0).astype(np.float32)
F['DSCR'] = np.clip(((inc / 12.0) * 0.65) / (ann + 1.0), 0.0, 20.0).astype(np.float32)
F['SOLVENCY'] = ((inc / 12.0) / (ann + 1.0)).astype(np.float32)
F['BUFFER_MONTHS'] = np.clip(surplus / (12.0 * ann + 1.0), -10.0, 50.0).astype(np.float32)
F['INC_PER_FAM'] = (inc / (df_all['CNT_FAM_MEMBERS'].fillna(1.0) + 1e-3)).values.astype(np.float32)
F['CREDIT_PER_FAM'] = (cred / (df_all['CNT_FAM_MEMBERS'].fillna(1.0) + 1e-3)).values.astype(np.float32)
F['ANN_PER_FAM'] = (ann / (df_all['CNT_FAM_MEMBERS'].fillna(1.0) + 1e-3)).values.astype(np.float32)

# Log transforms
F['LOG_INCOME'] = np.log1p(inc).astype(np.float32)
F['LOG_CREDIT'] = np.log1p(cred).astype(np.float32)
F['LOG_ANNUITY'] = np.log1p(ann).astype(np.float32)
F['LOG_GOODS'] = np.log1p(goods).astype(np.float32)

# Bureau aggregations
b_cols = [c for c in df_all.columns if c.startswith('AMT_REQ_CREDIT_BUREAU')]
F['BUR_SUM'] = df_all[b_cols].sum(axis=1, min_count=1).values.astype(np.float32)
F['BUR_RECENT'] = df_all[['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON']].sum(axis=1, min_count=1).values.astype(np.float32)
F['BUR_MON_YEAR'] = (df_all['AMT_REQ_CREDIT_BUREAU_MON'] / (df_all['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1.0)).values.astype(np.float32)

# 4. External Sources Triad & Clayton Copulas
ext_stack = np.column_stack([e1, e2, e3])
ext_mean = np.nanmean(ext_stack, axis=1)
F['EXT_MEAN'] = ext_mean.astype(np.float32)
F['EXT_MIN'] = np.nanmin(ext_stack, axis=1).astype(np.float32)
F['EXT_MAX'] = np.nanmax(ext_stack, axis=1).astype(np.float32)
F['EXT_STD'] = np.nan_to_num(np.nanstd(ext_stack, axis=1), nan=0.0).astype(np.float32)
F['EXT_RANGE'] = (F['EXT_MAX'] - F['EXT_MIN']).astype(np.float32)
F['EXT_WEIGHTED'] = (np.nan_to_num(0.2*e1) + np.nan_to_num(0.4*e2) + np.nan_to_num(0.4*e3)).astype(np.float32)
F['EXT_DIFF_12'] = (e1 - e2).astype(np.float32)
F['EXT_DIFF_23'] = (e2 - e3).astype(np.float32)
F['EXT_DIFF_13'] = (e1 - e3).astype(np.float32)
F['EXT_MULT_12'] = (e1 * e2).astype(np.float32)
F['EXT_MULT_23'] = (e2 * e3).astype(np.float32)
F['EXT_MULT_13'] = (e1 * e3).astype(np.float32)

def uniform_rank(v):
    mask = ~np.isnan(v)
    res = np.full(len(v), 0.5, dtype=np.float64)
    if mask.sum() > 0:
        res[mask] = rankdata(v[mask], method='average') / (mask.sum() + 1.0)
    return res

u1, u2, u3 = uniform_rank(e1), uniform_rank(e2), uniform_rank(e3)
for theta in [1.5, 2.5]:
    t1 = np.power(np.maximum(1e-4, u1), -theta)
    t2 = np.power(np.maximum(1e-4, u2), -theta)
    t3 = np.power(np.maximum(1e-4, u3), -theta)
    F[f'CLAYTON_COPULA_{str(theta).replace(".", "_")}'] = np.nan_to_num(np.power(np.maximum(0.0, t1 + t2 + t3 - 2.0), -1.0 / theta), nan=0.5).astype(np.float32)

# 5. Merton Structural Distance-to-Default
T_years = np.clip(term / 12.0, 0.1, 10.0)
disp_ratio = np.maximum(0.01, (inc / 12.0) / (ann + 1.0))
for mu, sigma, tag in [(0.02, 0.20, "20"), (0.03, 0.25, "25")]:
    dd = (np.log(disp_ratio) + (mu - 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    F[f'MERTON_DD_{tag}'] = np.clip(dd, -10.0, 10.0).astype(np.float32)
    F[f'MERTON_PD_{tag}'] = norm.cdf(-dd).astype(np.float32)

# 6. Discrete Tariffs & Decoded Integer Multiplier K_DEC
F['CT_R1'] = (cred / (ann + 1.0)).round(1).astype(np.float32)
F['MARKUP_R2'] = (cred / (goods + 1.0)).round(2).astype(np.float32)
cash_mask = (df_all['NAME_CONTRACT_TYPE'] == 'Cash loans').values
kf = (cred / goods - 1.0) / 0.0132
k_round = kf.round()
ok_k = cash_mask & (cred > goods) & (np.abs(kf - k_round) < 0.1)
F['K_DEC'] = np.where(ok_k, k_round, np.where(cash_mask & (cred == goods), 0.0, np.nan)).astype(np.float32)
F['Q0_FEE'] = np.where(cash_mask, (ann - 129.5) / (cred + 1.0), ann / (cred + 1.0)).astype(np.float32)

# 7. Frequency Encodings
for c in CATS + ['REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']:
    cnt_map = df_all[c].value_counts()
    F[f'CNT_{c}'] = df_all[c].map(cnt_map).values.astype(np.float32)

# 8. Peer Group Relative Deviations
for k in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']:
    grp_inc = df_all.groupby(k)['AMT_INCOME_TOTAL'].transform('mean').values
    grp_crd = df_all.groupby(k)['AMT_CREDIT'].transform('mean').values
    grp_ext = df_all.groupby(k)['EXT_SOURCE_2'].transform('mean').values
    grp_age = df_all.groupby(k)['DAYS_BIRTH'].transform('mean').values
    F[f'DEV_INC_{k[:4]}'] = (inc - grp_inc).astype(np.float32)
    F[f'RAT_INC_{k[:4]}'] = (inc / (grp_inc + 1.0)).astype(np.float32)
    F[f'DEV_CRD_{k[:4]}'] = (cred - grp_crd).astype(np.float32)
    F[f'DEV_EXT_{k[:4]}'] = (e2 - grp_ext).astype(np.float32)
    F[f'DEV_AGE_{k[:4]}'] = (birth - grp_age).astype(np.float32)

# 9. Relational Entity Graph & In-Fold UIDTE
df_all['entity_key'] = (
    (birth - reg).round(0).astype(int).astype(str) + "|" +
    (birth - id_pub).round(0).astype(int).astype(str) + "|" +
    df_all['CODE_GENDER'].astype(str) + "|" +
    df_all['NAME_EDUCATION_TYPE'].astype(str)
)
cnt_ent = df_all['entity_key'].value_counts()
F['N_APPS_ALL'] = df_all['entity_key'].map(cnt_ent).values.astype(np.float32)

df_all['row_idx'] = np.arange(n_total)
df_sorted = df_all.sort_values(['entity_key', 'DAYS_REGISTRATION'])
df_sorted['APP_ORDER'] = df_sorted.groupby('entity_key').cumcount()
df_sorted['DAYS_SINCE_PREV_APP'] = df_sorted.groupby('entity_key')['DAYS_REGISTRATION'].diff()
df_restored = df_sorted.sort_values('row_idx')
F['APP_ORDER'] = df_restored['APP_ORDER'].values.astype(np.float32)
F['DAYS_SINCE_PREV_APP'] = df_restored['DAYS_SINCE_PREV_APP'].fillna(-999.0).values.astype(np.float32)

multi_keys_set = set(cnt_ent[cnt_ent >= 2].index)
uidy_anybad = np.zeros(n_total, dtype=np.float32)
uidy_near_y = np.full(n_total, -1.0, dtype=np.float32)

skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df_all_nodes = df_all[['row_idx', 'entity_key', 'DAYS_REGISTRATION']].copy()
multi_mask = df_all_nodes['entity_key'].isin(multi_keys_set)
multi_nodes = df_all_nodes[multi_mask].copy()

for trn_idx, val_idx in skf_cv.split(np.zeros(n_train), y_train):
    trn_sub = multi_nodes[multi_nodes['row_idx'].isin(set(trn_idx))].copy()
    trn_sub['TARGET'] = y_train[trn_sub['row_idx'].values]
    val_sub = multi_nodes[multi_nodes['row_idx'].isin(set(val_idx))].copy()
    if len(val_sub) > 0 and len(trn_sub) > 0:
        val_merged = val_sub.merge(trn_sub, on='entity_key', suffixes=('', '_partner'))
        if len(val_merged) > 0:
            val_merged['delta'] = (val_merged['DAYS_REGISTRATION'] - val_merged['DAYS_REGISTRATION_partner']).abs()
            nearest = val_merged.sort_values('delta').groupby('row_idx').first()
            summary = val_merged.groupby('row_idx')['TARGET'].max()
            for r_idx in nearest.index:
                uidy_near_y[r_idx] = nearest.loc[r_idx, 'TARGET']
                uidy_anybad[r_idx] = 1.0 if summary.loc[r_idx] > 0 else 0.0

trn_full = multi_nodes[multi_nodes['row_idx'] < n_train].copy()
trn_full['TARGET'] = y_train[trn_full['row_idx'].values]
test_sub = multi_nodes[multi_nodes['row_idx'] >= n_train].copy()
if len(test_sub) > 0 and len(trn_full) > 0:
    te_merged = test_sub.merge(trn_full, on='entity_key', suffixes=('', '_partner'))
    if len(te_merged) > 0:
        te_merged['delta'] = (te_merged['DAYS_REGISTRATION'] - te_merged['DAYS_REGISTRATION_partner']).abs()
        nearest_te = te_merged.sort_values('delta').groupby('row_idx').first()
        summary_te = te_merged.groupby('row_idx')['TARGET'].max()
        for r_idx in nearest_te.index:
            uidy_near_y[r_idx] = nearest_te.loc[r_idx, 'TARGET']
            uidy_anybad[r_idx] = 1.0 if summary_te.loc[r_idx] > 0 else 0.0

F['UIDY_NEAR_Y'] = uidy_near_y
F['UIDY_ANYBAD'] = uidy_anybad

# 10. Decoded Product Schedule Target Encodings
q_ratio = ann / (cred + 1e-5)
q0_val = F['Q0_FEE']
KS = {
    'Q3': pd.Series(q_ratio).round(3).astype(str),
    'Q4': pd.Series(q_ratio).round(4).astype(str),
    'P3': pd.Series(q0_val).round(3).astype(str),
    'P4': pd.Series(q0_val).round(4).astype(str),
}
M0 = {'Q3': 100.0, 'Q4': 30.0, 'P3': 100.0, 'P4': 30.0}

def get_te_dict(keys_s, y_s, m0):
    pr = float(y_s.mean())
    df_k = pd.DataFrame({'k': keys_s, 'y': y_s})
    g = df_k.groupby('k')['y'].agg(['sum', 'count'])
    te_map = ((g['sum'] + m0 * pr) / (g['count'] + m0)).to_dict()
    cnt_map = g['count'].to_dict()
    return te_map, cnt_map, pr

for k_name in ['Q3', 'Q4', 'P3', 'P4']:
    k_series = KS[k_name].values
    te_arr = np.zeros(n_total, dtype=np.float32)
    cnt_arr = np.zeros(n_total, dtype=np.float32)
    m0_val = M0[k_name]
    for trn_idx, val_idx in skf_cv.split(np.zeros(n_train), y_train):
        te_map, cnt_map, pr = get_te_dict(k_series[trn_idx], y_train[trn_idx], m0_val)
        val_k = k_series[val_idx]
        te_arr[val_idx] = [te_map.get(x, pr) for x in val_k]
        cnt_arr[val_idx] = [cnt_map.get(x, 0) for x in val_k]
    te_map_full, cnt_map_full, pr_full = get_te_dict(k_series[:n_train], y_train, m0_val)
    te_k = k_series[n_train:]
    te_arr[n_train:] = [te_map_full.get(x, pr_full) for x in te_k]
    cnt_arr[n_train:] = [cnt_map_full.get(x, 0) for x in te_k]
    F[f'TE_{k_name}'] = te_arr
    F[f'CNT_{k_name}'] = cnt_arr

# 11. Fast Ridge Learned Expectations and Residuals
demo_mat = np.column_stack([
    birth / -365.25,
    np.nan_to_num(days_emp / -365.25, nan=0.0),
    reg / -365.25,
    id_pub / -365.25,
    np.nan_to_num(phone / -365.25, nan=0.0),
    (emp_raw == 365243.0).astype(float)
])
targets_res = {
    'EXT2': np.nan_to_num(e2, nan=0.5),
    'EXT3': np.nan_to_num(e3, nan=0.5),
    'LOGINC': np.log1p(inc),
    'LOGCR': np.log1p(cred)
}
kf_res = KFold(n_splits=5, shuffle=True, random_state=42)
for tname, y_res in targets_res.items():
    pred_res = np.zeros(n_total, dtype=np.float32)
    for tr_i, va_i in kf_res.split(demo_mat):
        rg = Ridge(alpha=100.0)
        rg.fit(demo_mat[tr_i], y_res[tr_i])
        pred_res[va_i] = rg.predict(demo_mat[va_i])
    F[f'RES_{tname}'] = (y_res - pred_res).astype(np.float32)
    F[f'PRED_{tname}'] = pred_res.astype(np.float32)

# 12. Multi-View Disagreement Differential Stacking
view_ext_cols = ['EXT_MEAN', 'EXT_MIN', 'EXT_MAX', 'EXT_STD', 'EXT_RANGE', 'EXT_WEIGHTED', 'EXT_DIFF_12', 'EXT_DIFF_23', 'EXT_DIFF_13', 'EXT_MULT_12', 'EXT_MULT_23', 'EXT_MULT_13', 'CLAYTON_COPULA_1_5', 'CLAYTON_COPULA_2_5']
view_loan_cols = ['CREDIT_TERM', 'EAR', 'SURPLUS', 'SURPLUS_TO_INC', 'TRUE_COST', 'TRUE_COST_RATIO', 'LTV', 'DTI', 'DEBT_BURDEN', 'DSCR', 'SOLVENCY', 'BUFFER_MONTHS', 'LOG_INCOME', 'LOG_CREDIT', 'LOG_ANNUITY', 'LOG_GOODS', 'K_DEC', 'Q0_FEE', 'TE_Q3', 'TE_Q4', 'TE_P3', 'TE_P4']
view_person_cols = ['AGE_YEARS', 'EMPLOYED_YEARS', 'EMP_TO_AGE', 'REG_TO_AGE', 'ID_TO_AGE', 'PHONE_TO_AGE', 'INC_PER_FAM', 'CREDIT_PER_FAM', 'ANN_PER_FAM', 'BUR_SUM', 'BUR_RECENT', 'BUR_MON_YEAR', 'N_APPS_ALL', 'APP_ORDER', 'DAYS_SINCE_PREV_APP', 'UIDY_NEAR_Y', 'UIDY_ANYBAD']

df_views = pd.DataFrame({k: F[k] for k in view_ext_cols + view_loan_cols + view_person_cols})
for c in CATS:
    df_views[c] = df_all[c].astype('category')

X_view_tr = df_views.iloc[:n_train].reset_index(drop=True)
X_view_te = df_views.iloc[n_train:].reset_index(drop=True)

p_view = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'learning_rate': 0.08, 'num_leaves': 15, 'min_child_samples': 150, 'verbose': -1, 'n_jobs': 4, 'random_state': 42}
views_oof = {}
views_test = {}

for vname, vcols in [('EXT', view_ext_cols), ('LOAN', view_loan_cols), ('PERSON', view_person_cols + CATS)]:
    voof = np.zeros(n_train, dtype=np.float32)
    vtest = np.zeros(n_test, dtype=np.float32)
    X_sub_tr = X_view_tr[vcols]
    X_sub_te = X_view_te[vcols]
    for trn_idx, val_idx in skf_cv.split(X_sub_tr, y_train):
        dtrain = lgb.Dataset(X_sub_tr.iloc[trn_idx], label=y_train[trn_idx])
        dval = lgb.Dataset(X_sub_tr.iloc[val_idx], label=y_train[val_idx], reference=dtrain)
        bst = lgb.train(p_view, dtrain, num_boost_round=150, valid_sets=[dval], callbacks=[lgb.early_stopping(20, verbose=False)])
        voof[val_idx] = bst.predict(X_sub_tr.iloc[val_idx])
        vtest += bst.predict(X_sub_te) / 5.0
    views_oof[vname] = voof
    views_test[vname] = vtest

def to_logit(p):
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))

v_ext_all = np.concatenate([to_logit(views_oof['EXT']), to_logit(views_test['EXT'])])
v_loan_all = np.concatenate([to_logit(views_oof['LOAN']), to_logit(views_test['LOAN'])])
v_pers_all = np.concatenate([to_logit(views_oof['PERSON']), to_logit(views_test['PERSON'])])

F['VIEW_EXT'] = v_ext_all.astype(np.float32)
F['VIEW_LOAN'] = v_loan_all.astype(np.float32)
F['VIEW_PERSON'] = v_pers_all.astype(np.float32)
F['VD_EXT_LOAN'] = (v_ext_all - v_loan_all).astype(np.float32)
F['VD_EXT_PERS'] = (v_ext_all - v_pers_all).astype(np.float32)
F['VD_LOAN_PERS'] = (v_loan_all - v_pers_all).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# [EXP_01 INNOVATION]: In-Memory Learned Expectation Residuals (Forensic v16/resid16 Recovery)
# --------------------------------------------------------------------------------------------------
print('\n[*] [EXP_01] Cross-Fitting In-Memory Learned Expectations for EXT_SOURCE & Income...')
from lightgbm import LGBMRegressor

resid_targets = {
    'EXT1': (df_all['EXT_SOURCE_1'], ['EXT_SOURCE_1', 'EXT_MEAN', 'EXT_MIN', 'EXT_MAX', 'EXT_12', 'EXT_13']),
    'EXT2': (df_all['EXT_SOURCE_2'], ['EXT_SOURCE_2', 'EXT_MEAN', 'EXT_MIN', 'EXT_MAX', 'EXT_12', 'EXT_23']),
    'EXT3': (df_all['EXT_SOURCE_3'], ['EXT_SOURCE_3', 'EXT_MEAN', 'EXT_MIN', 'EXT_MAX', 'EXT_13', 'EXT_23']),
    'LOGINC': (np.log1p(df_all['AMT_INCOME_TOTAL']), ['AMT_INCOME_TOTAL', 'LOG_INCOME', 'INC_PER_FAM', 'INCOME_PER_PERSON'])
}

demo_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
             'CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE']

X_demo = pd.DataFrame(index=df_all.index)
for c in demo_cols:
    if c in df_all.columns:
        s = df_all[c]
        if s.dtype.kind in ('O', 'S', 'U', 'b') or s.dtype.name in ('category', 'string'):
            codes, _ = pd.factorize(s)
            X_demo[c] = pd.Series(codes, index=df_all.index).replace(-1, np.nan).astype(float)
        else:
            X_demo[c] = pd.to_numeric(s, errors='coerce').astype(float)

kf_resid = KFold(n_splits=5, shuffle=True, random_state=42)

for tgt_name, (y_tgt, drop_c) in resid_targets.items():
    t_tgt0 = time.time()
    obs_mask = y_tgt.notna().values
    pred_tgt = np.zeros(n_total, dtype=np.float32)
    X_reg = X_demo.drop(columns=[c for c in drop_c if c in X_demo.columns], errors='ignore')
    
    for tr_idx, te_idx in kf_resid.split(X_reg):
        fit_idx = tr_idx[obs_mask[tr_idx]]
        reg = LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=20, min_child_samples=50,
                            n_jobs=4, verbose=-1, random_state=42)
        reg.fit(X_reg.iloc[fit_idx], y_tgt.iloc[fit_idx])
        pred_tgt[te_idx] = reg.predict(X_reg.iloc[te_idx])
        
    F[f'EXP_PRED_{tgt_name}'] = pred_tgt
    F[f'EXP_RESID_{tgt_name}'] = np.where(obs_mask, (y_tgt.values - pred_tgt).astype(np.float32), 0.0)
    print(f'    - Target {tgt_name:7s} expectation cross-fit complete ({time.time()-t_tgt0:.1f}s)')


# --------------------------------------------------------------------------------------------------
# [EXP_02 INNOVATION]: v63 Breakthrough Quantitative Features & Peer Standing Percentiles
# --------------------------------------------------------------------------------------------------
print('\n[*] [EXP_02] Engineering v63 Implied Interest Grid, Peer Percentiles & Decoded Product Lattice...')
t_v63_start = time.time()

# Composite EXT scores
ext_mean = df_all[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1).values
ext_min  = df_all[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1).values
ext_max  = df_all[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1).values

# 1. Alexander_L Implied Interest & Term Grid
eps = 1e-5
best_rate = np.full(n_total, np.nan)
best_term = np.full(n_total, np.nan)

with np.errstate(all="ignore"):
    for n in np.array([6, 12, 18, 24, 30, 36, 48, 60]):
        annual = (ann / (cred + eps) - 1.0 / n) * 12.0
        mask = (annual > 0.02) & (annual < 0.90)
        closer = np.abs(annual - 0.25) < np.abs(best_rate - 0.25)
        upd = mask & (np.isnan(best_rate) | closer)
        best_rate[upd] = annual[upd]
        best_term[upd] = n

v_implied_interest = pd.Series(best_rate)
v_implied_term = pd.Series(best_term)
v_payment_rate = ann / (cred + eps)
v_interest_x_ext = v_implied_interest.values * ext_mean

F['V_IMPLIED_INTEREST'] = v_implied_interest.fillna(-1.0).values.astype(np.float32)
F['V_IMPLIED_TERM']     = v_implied_term.fillna(-1.0).values.astype(np.float32)
F['V_PAYMENT_RATE']     = v_payment_rate.astype(np.float32)
F['V_INTEREST_X_EXT']   = np.nan_to_num(v_interest_x_ext, nan=0.0).astype(np.float32)

# 2. Relative Peer Standing Percentiles
occ = df_all["OCCUPATION_TYPE"].astype(str).fillna("NA")
age_bin = (-birth // (365.25 * 5) * 5).astype(str)
gender = df_all["CODE_GENDER"].astype(str).values
gen_age = gender + "_" + age_bin

s_ext = pd.Series(ext_mean)
F['EXT_PCT_OCC']     = np.nan_to_num(s_ext.groupby(occ).rank(pct=True).values, nan=0.5).astype(np.float32)
F['EXT_PCT_GEN_AGE'] = np.nan_to_num(s_ext.groupby(gen_age).rank(pct=True).values, nan=0.5).astype(np.float32)
F['P_EXTMEAN_PAY']   = np.nan_to_num(ext_mean * v_payment_rate, nan=0.0).astype(np.float32)
F['P_EXT_SPREAD']    = np.nan_to_num((ext_max - ext_min) / (ext_mean + 0.01), nan=0.0).astype(np.float32)

# 3. Decoded Product Lattice Q0 & Contract Gap
cash = (df_all["NAME_CONTRACT_TYPE"] == "Cash loans").values
q = ann / (cred + eps)
qb = (ann - 129.5) / (cred + eps)
q5 = np.round(q, 5)
qb5 = np.round(qb, 5)
vc_q5 = pd.Series(q5).value_counts()
vc_qb5 = pd.Series(qb5).value_counts()
fee_flag = ((pd.Series(qb5).map(vc_qb5).values > pd.Series(q5).map(vc_q5).values) & cash).astype(int)
q0 = np.where(fee_flag == 1, qb, q)
q0_r4 = np.round(q0, 4)

kf = np.where((cred > goods) & (goods > 0), (cred / goods - 1.0) / 0.0132, 0.0)
k_tier = np.where(cash & (cred > goods), np.round(kf), np.where(cash & (cred == goods), 0.0, -1.0))
prod_cell = pd.Series(q0_r4).astype(str) + "_K" + pd.Series(k_tier.astype(int)).astype(str) + "_F" + pd.Series(fee_flag).astype(str)

cell_ext2_bench = df_all['EXT_SOURCE_2'].groupby(prod_cell).transform('mean').values
cell_ext3_bench = df_all['EXT_SOURCE_3'].groupby(prod_cell).transform('mean').values
F['E2_CONTRACT_GAP'] = np.nan_to_num(cell_ext2_bench - e2, nan=0.0).astype(np.float32)
F['E3_CONTRACT_GAP'] = np.nan_to_num(cell_ext3_bench - e3, nan=0.0).astype(np.float32)
F['Q0_BASE_RATE']    = q0.astype(np.float32)
F['TARIFF_K_TIER']   = k_tier.astype(np.float32)

# 4. Invariant Date Keys & In-Fold Nested Target Encodings
k1 = gender + "_" + np.round(id_pub - birth, 0).astype(str)
kb = gender + "_" + np.round(id_pub - birth, 0).astype(str) + "_" + np.round(days_emp - birth, 0).astype(str)

skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_p = float(y_train.mean())

def compute_nested_te(key_series, m=30.0):
    codes, uniques = pd.factorize(key_series)
    K = len(uniques)
    oof_te = np.zeros(n_train, dtype=np.float32)
    for trn_idx, val_idx in skf_te.split(np.zeros(n_train), y_train):
        c_tr = codes[:n_train][trn_idx]
        y_tr = y_train[trn_idx]
        cnt = np.bincount(c_tr, minlength=K).astype(float)
        sums = np.bincount(c_tr, weights=y_tr, minlength=K)
        c_val = codes[:n_train][val_idx]
        oof_te[val_idx] = (sums[c_val] + m * global_p) / (cnt[c_val] + m)
    cnt_all = np.bincount(codes[:n_train], minlength=K).astype(float)
    sums_all = np.bincount(codes[:n_train], weights=y_train, minlength=K)
    c_test = codes[n_train:]
    test_te = (sums_all[c_test] + m * global_p) / (cnt_all[c_test] + m)
    return np.concatenate([oof_te, test_te])

F['TE_PROD_CELL']     = compute_nested_te(prod_cell, m=30.0).astype(np.float32)
F['TE_APPLICANT_K1']  = compute_nested_te(pd.Series(k1), m=50.0).astype(np.float32)
F['TE_APPLICANT_KB']  = compute_nested_te(pd.Series(kb), m=30.0).astype(np.float32)

print(f'    - Successfully engineered 15 high-resolution v63 features in {time.time()-t_v63_start:.1f}s')


# --------------------------------------------------------------------------------------------------
# [EXP_03 INNOVATION]: Invariant Identity Graph, Person Pooling & Merton Solvency Copulas
# --------------------------------------------------------------------------------------------------
print('\n[*] [EXP_03] Engineering Invariant Identity Graph, Person Pooling & Merton Solvency Copulas...')
t_exp3_fe = time.time()

days_reg_arr   = df_all['DAYS_REGISTRATION'].values.astype(np.float64)
days_birth_arr = df_all['DAYS_BIRTH'].values.astype(np.float64)
days_id_arr    = df_all['DAYS_ID_PUBLISH'].values.astype(np.float64)

# 1. Quantitative Merton Structural Distance-to-Default
asset_proxy = inc * 5.0 + goods
debt_proxy  = np.maximum(cred, 1.0)
sigma_asset = 0.30
merton_dd   = (np.log(asset_proxy / debt_proxy) + 0.095) / sigma_asset
F['MERTON_DD'] = merton_dd.astype(np.float32)
F['MERTON_PD'] = norm.cdf(-merton_dd).astype(np.float32)

# 2. Clayton Archimedean Copulas on normalized ranks of EXT sources
u1 = (rankdata(np.nan_to_num(e1, nan=0.5)) - 0.5) / n_total
u2 = (rankdata(np.nan_to_num(e2, nan=0.5)) - 0.5) / n_total
theta_cop = 2.0
cop_val = np.clip(u1**(-theta_cop) + u2**(-theta_cop) - 1.0, 1e-4, None)**(-1.0 / theta_cop)
F['COPULA_CLAYTON_12'] = np.asarray(cop_val, dtype=np.float32)

# 3. Invariant Repeat-Borrower Identity Fingerprints
_i1_s = np.round(days_reg_arr - days_birth_arr, 0).astype(str)
_i2_s = np.round(days_id_arr - days_birth_arr, 0).astype(str)

k_strict = (
    df_all["CODE_GENDER"].astype(str) + "|" +
    df_all["NAME_EDUCATION_TYPE"].astype(str) + "|" +
    df_all["REGION_POPULATION_RELATIVE"].astype(str) + "|" +
    pd.Series(_i1_s) + "|" +
    pd.Series(_i2_s)
)
k_strict = k_strict.where(days_reg_arr != 0)
sz_s = k_strict.map(k_strict.value_counts())
k_strict = k_strict.where(sz_s > 1)
g_s = pd.Series(pd.factorize(k_strict)[0]).replace(-1, np.nan).values

mem_s = pd.DataFrame({"g": g_s}).groupby("g").groups
_gb = days_birth_arr

PL_S = {}
for _, ix in mem_s.items():
    ix = np.asarray(ix)
    for i in ix:
        o = ix[ix != i]
        PL_S[i] = (o, _gb[o] - _gb[i])

print(f'    - Identified {len(PL_S):,} confirmed repeat borrower records across {len(mem_s):,} clusters')

# 4. Person-Level Pooled Features
E_mat = np.column_stack([e1, e2, e3])
pe = np.full((n_total, 3), np.nan)
dif = np.full((n_total, 3), np.nan)
cr_ratio = np.full(n_total, np.nan)
an_ratio = np.full(n_total, np.nan)
inc_ratio = np.full(n_total, np.nan)
same_ct = np.full(n_total, np.nan)
qd_diff = np.full(n_total, np.nan)
Qv = (ann / (cred + 1e-5))
ct_is_cash = (df_all["NAME_CONTRACT_TYPE"] == "Cash loans").values

for i, (o, g_dist) in PL_S.items():
    j = o[np.argmin(np.abs(g_dist))]
    allr = np.r_[i, o]
    for k in range(3):
        v = E_mat[allr, k]
        pe[i, k] = np.nanmean(v) if np.isfinite(v).any() else np.nan
        dif[i, k] = E_mat[i, k] - E_mat[j, k]
    cr_ratio[i] = cred[i] / (cred[j] + 1e-5)
    an_ratio[i] = ann[i] / (ann[j] + 1e-5)
    inc_ratio[i] = inc[i] / (inc[j] + 1e-5)
    same_ct[i] = float(ct_is_cash[i] == ct_is_cash[j])
    qd_diff[i] = Qv[i] - Qv[j]

F['PERS_EXT1']      = pe[:, 0].astype(np.float32)
F['PERS_EXT2']      = pe[:, 1].astype(np.float32)
F['PERS_EXT3']      = pe[:, 2].astype(np.float32)
F['PERS_EXT_MEAN']  = np.nanmean(pe, axis=1).astype(np.float32)
F['PDIF_EXT1']      = dif[:, 0].astype(np.float32)
F['PDIF_EXT2']      = dif[:, 1].astype(np.float32)
F['PDIF_EXT3']      = dif[:, 2].astype(np.float32)
F['PCR_RATIO']      = cr_ratio.astype(np.float32)
F['PAN_RATIO']      = an_ratio.astype(np.float32)
F['PINC_RATIO']     = inc_ratio.astype(np.float32)
F['PSAME_CT']       = same_ct.astype(np.float32)
F['PQ_DIFF']        = qd_diff.astype(np.float32)
F['HAS_PARTNER']    = (~np.isnan(g_s)).astype(np.float32)

# 5. In-Fold Partner Historical Outcome Encodings
partner_out = np.full((n_total, 3), np.nan, dtype=np.float32)
partner_out[:, 1] = 0.0

# Leave-self-out in-fold for train rows
for tr_f_idx, val_f_idx in skf_te.split(np.zeros(n_train), y_train):
    lab_fold = np.full(n_train, np.nan)
    lab_fold[tr_f_idx] = y_train[tr_f_idx]
    for i in val_f_idx:
        if i in PL_S:
            o, gp_ = PL_S[i]
            o_tr = o[o < n_train]
            if len(o_tr) > 0:
                v = lab_fold[o_tr]
                k_val = ~np.isnan(v)
                if k_val.any():
                    v_act = v[k_val]
                    gp_act = gp_[o < n_train][k_val]
                    partner_out[i, 0] = v_act.max()
                    partner_out[i, 1] = k_val.sum()
                    if (np.abs(gp_act) <= 180).any():
                        partner_out[i, 2] = v_act[np.abs(gp_act) <= 180].max()

# Test rows map from full train ground truth
lab_full_tr = np.full(n_total, np.nan)
lab_full_tr[:n_train] = y_train
for i in range(n_train, n_total):
    if i in PL_S:
        o, gp_ = PL_S[i]
        o_tr = o[o < n_train]
        if len(o_tr) > 0:
            v = lab_full_tr[o_tr]
            k_val = ~np.isnan(v)
            if k_val.any():
                v_act = v[k_val]
                gp_act = gp_[o < n_train][k_val]
                partner_out[i, 0] = v_act.max()
                partner_out[i, 1] = k_val.sum()
                if (np.abs(gp_act) <= 180).any():
                    partner_out[i, 2] = v_act[np.abs(gp_act) <= 180].max()

F['PY_ANY']     = partner_out[:, 0]
F['PY_N']       = partner_out[:, 1]
F['PY_NEAR180'] = partner_out[:, 2]

print(f'    - Successfully engineered 17 identity graph & solvency features in {time.time()-t_exp3_fe:.1f}s')

# Retain all raw columns (preserving document flags, region ratings, etc.)
X_raw = df_all.drop(columns=['SK_ID_CURR']).copy()
X_eng = pd.DataFrame(F)
for c in X_raw.columns:
    if c not in CATS and c not in X_eng.columns and c not in ('entity_key', 'row_idx'):
        X_eng[c] = X_raw[c].values

for c in CATS:
    X_eng[c] = df_all[c].astype('category')

X_tr = X_eng.iloc[:n_train].reset_index(drop=True)
X_te = X_eng.iloc[n_train:].reset_index(drop=True)

# Prepare numeric matrices for XGBoost and CatBoost
num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
X_tr_num = X_tr[num_cols].replace([np.inf, -np.inf], np.nan)
X_te_num = X_te[num_cols].replace([np.inf, -np.inf], np.nan)

print(f'[+] Constructed complete feature matrices in {time.time()-t_fe_start:.1f}s:')
print(f'    - Train Feature Matrix: {X_tr.shape} ({len(num_cols)} numeric columns)')
print(f'    - Test Feature Matrix : {X_te.shape} ({len(num_cols)} numeric columns)')
