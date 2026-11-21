with open('v22_champion_audit_certified_v39_sovereign_apex_zenith.py', 'r') as f:
    code = f.read()

# 0. Replace version strings
code = code.replace('V39 (SOVEREIGN APEX ZENITH)', 'V43 (GRAND APEX SOVEREIGN MASTER)')
code = code.replace('v_22_pure_champion_ensemble_certified_v39.csv', 'v_22_pure_champion_ensemble_certified_v43.csv')
code = code.replace('V39 SOVEREIGN APEX ZENITH', 'V43 GRAND APEX SOVEREIGN MASTER')

# 1. Add global CATS definition in Section 1
global_cats = '''OFFICIAL_SUB_SHA256   = "6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB"

CATS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE'
]'''
code = code.replace('OFFICIAL_SUB_SHA256   = "6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB"', global_cats)

# 2. Update function signature of train_grandmaster_suite to accept cb_df
code = code.replace(
    'def train_grandmaster_suite(X_train, y_train, X_test, df_train_raw, df_test_raw):',
    'def train_grandmaster_suite(X_train, y_train, X_test, df_train_raw, df_test_raw, cb_df):'
)

# 3. In train_grandmaster_suite, split cb_df into cb_train and cb_test with reset_index
old_setup = '''    model_names = [
        "LGB_ET_E1", "LGB_Shallow", "LGB_ET_Deep", "LGB_Conservative",
        "LGB_Wide", "LGB_Solvency", "XGB_GPU_Hist", "CB_GPU"
    ]
    oof_preds    = {m: np.zeros(n_train, dtype=np.float32) for m in model_names}
    test_cv_pred = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}
    test_refits  = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}'''

new_setup = '''    model_names = [
        "LGB_ET_E1", "LGB_Shallow", "LGB_ET_Deep", "LGB_Conservative",
        "LGB_Wide", "LGB_Solvency", "XGB_GPU_Hist", "CB_GPU"
    ]
    oof_preds    = {m: np.zeros(n_train, dtype=np.float32) for m in model_names}
    test_cv_pred = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}
    test_refits  = {m: np.zeros(n_test, dtype=np.float32)  for m in model_names}

    cb_train = cb_df.iloc[:n_train].reset_index(drop=True)
    cb_test  = cb_df.iloc[n_train:].reset_index(drop=True)
    cb_cats_cols = list(cb_df.columns)'''

code = code.replace(old_setup, new_setup)

# 4. Tune LGB_Wide slightly so it doesn't overfit
old_lgb_wide = '''        # 5. LightGBM Wide Interactor
        p_wi = dict(objective="binary", learning_rate=0.026, num_leaves=63, max_depth=8,
                    min_child_samples=250, feature_fraction=0.30, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=50.0, verbose=-1, n_jobs=6,
                    feature_pre_filter=False, seed=42+fi+40)
        m_wi = lgb.train(p_wi, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)'''

new_lgb_wide = '''        # 5. LightGBM Wide Interactor
        p_wi = dict(objective="binary", learning_rate=0.028, num_leaves=38, max_depth=6,
                    min_child_samples=160, feature_fraction=0.36, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=35.0, verbose=-1, n_jobs=6,
                    feature_pre_filter=False, seed=42+fi+40)
        m_wi = lgb.train(p_wi, lgb.Dataset(X_tr, label=y_tr, free_raw_data=False), num_boost_round=800)'''

code = code.replace(old_lgb_wide, new_lgb_wide)

# 5. Update CatBoost in fold loop to use reset_index and cb_train / cb_test
old_cb = '''        # 8. CatBoost GPU Specialist
        m_cb = CatBoostClassifier(iterations=850, learning_rate=0.035, depth=6,
                                  l2_leaf_reg=25.0, task_type="GPU", verbose=0, random_seed=42+fi)
        m_cb.fit(X_tr.replace([np.inf, -np.inf], 0.0).fillna(0.0), y_tr)
        oof_preds["CB_GPU"][vai] = m_cb.predict_proba(X_va.replace([np.inf, -np.inf], 0.0).fillna(0.0))[:, 1]
        test_cv_pred["CB_GPU"] += m_cb.predict_proba(X_test.replace([np.inf, -np.inf], 0.0).fillna(0.0))[:, 1] / FOLDS'''

new_cb = '''        # 8. CatBoost GPU Specialist with Decoded Tariff Keys
        X_tr_cb = pd.concat([X_tr.reset_index(drop=True).fillna(0.0), cb_train.iloc[tri].reset_index(drop=True)], axis=1)
        X_va_cb = pd.concat([X_va.reset_index(drop=True).fillna(0.0), cb_train.iloc[vai].reset_index(drop=True)], axis=1)
        X_te_cb = pd.concat([X_test.reset_index(drop=True).fillna(0.0), cb_test.reset_index(drop=True)], axis=1)

        m_cb = CatBoostClassifier(iterations=850, learning_rate=0.035, depth=6,
                                  l2_leaf_reg=20.0, task_type="GPU", verbose=0, random_seed=42+fi)
        m_cb.fit(X_tr_cb, y_tr, cat_features=cb_cats_cols)
        oof_preds["CB_GPU"][vai] = m_cb.predict_proba(X_va_cb)[:, 1]
        test_cv_pred["CB_GPU"] += m_cb.predict_proba(X_te_cb)[:, 1] / FOLDS'''

code = code.replace(old_cb, new_cb)

# 6. Update CatBoost in full-train re-fits
old_refit_cb = '''        # 8. CatBoost GPU Refit
        m_cbf = CatBoostClassifier(iterations=800, learning_rate=0.030, depth=6,
                                   l2_leaf_reg=25.0, task_type="GPU", verbose=0, random_seed=sd)
        m_cbf.fit(X_train_clean, y_train)
        test_refits["CB_GPU"] += m_cbf.predict_proba(X_test_clean)[:, 1] / 2.0'''

new_refit_cb = '''        # 8. CatBoost GPU Refit
        X_tr_cb_full = pd.concat([X_train.reset_index(drop=True).fillna(0.0), cb_train.reset_index(drop=True)], axis=1)
        X_te_cb_full = pd.concat([X_test.reset_index(drop=True).fillna(0.0), cb_test.reset_index(drop=True)], axis=1)
        m_cbf = CatBoostClassifier(iterations=800, learning_rate=0.030, depth=6,
                                   l2_leaf_reg=20.0, task_type="GPU", verbose=0, random_seed=sd)
        m_cbf.fit(X_tr_cb_full, y_train, cat_features=cb_cats_cols)
        test_refits["CB_GPU"] += m_cbf.predict_proba(X_te_cb_full)[:, 1] / 2.0'''

code = code.replace(old_refit_cb, new_refit_cb)

# 7. Update regime routing with L2-Regularized Softmax and Macro Structural Consensus
old_routing = '''    # Continuous Solvency & Contract Regime Routing Optimization
    print("\\n[*] Solving Continuous Solvency & Contract Regime Routing Weights (Nelder-Mead)...")
    Z_oof  = np.column_stack([to_z(rk_norm(oof_preds[m])) for m in model_names])
    Z_test = np.column_stack([to_z(test_final_members[m]) for m in model_names])
    K = len(model_names)

    def regime_loss(params):
        w_prime = np.maximum(params[:K], 0.0)
        w_dist  = np.maximum(params[K:2*K], 0.0)
        w_rev   = np.maximum(params[2*K:3*K], 0.0)

        w_prime /= (w_prime.sum() + 1e-12)
        w_dist  /= (w_dist.sum()  + 1e-12)
        w_rev   /= (w_rev.sum()   + 1e-12)

        z_p = Z_oof @ w_prime
        z_d = Z_oof @ w_dist
        z_r = Z_oof @ w_rev

        z_routed = (1.0 - rev_tr) * ((1.0 - g_dist_tr) * z_p + g_dist_tr * z_d) + rev_tr * z_r
        return -roc_auc_score(y_train, z_routed)

    init_p = np.ones(3 * K) / K
    res = minimize(regime_loss, init_p, method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4})

    opt_p = res.x
    wp = np.maximum(opt_p[:K], 0.0); wp /= (wp.sum() + 1e-12)
    wd = np.maximum(opt_p[K:2*K], 0.0); wd /= (wd.sum() + 1e-12)
    wr = np.maximum(opt_p[2*K:3*K], 0.0); wr /= (wr.sum() + 1e-12)

    best_oof_auc = -regime_loss(opt_p)

    print(f"\\n[*] Continuous Regime Routing Optimal Weights Solved (OOF AUC: {best_oof_auc:.6f}):")
    print("    Model Name           | Prime Share | Distress Share | Revolving Share")
    print("    ---------------------------------------------------------------------")
    for idx_m, m in enumerate(model_names):
        print(f"    - {m:18s} |   {wp[idx_m]*100:5.2f}%   |    {wd[idx_m]*100:5.2f}%    |    {wr[idx_m]*100:5.2f}%")

    # Apply Continuous Regime Routing to Test Probits
    z_test_p = Z_test @ wp
    z_test_d = Z_test @ wd
    z_test_r = Z_test @ wr
    z_test_routed = (1.0 - rev_te) * ((1.0 - g_dist_te) * z_test_p + g_dist_te * z_test_d) + rev_te * z_test_r

    test_ensemble = rk_norm(to_p(z_test_routed))

    print(f"\\n  >>> TOTAL POOLED ENSEMBLE OOF ROC-AUC: {best_oof_auc:.6f} <<<")
    print(f"  Total Training Runtime: {(time.time()-t_start)/60:.2f} minutes")
    print("=" * 80)

    return test_ensemble, best_oof_auc'''

new_routing = '''    # Continuous Solvency & Contract Regime Routing Optimization (L2-Regularized Softmax)
    print("\\n[*] Solving Continuous Solvency & Contract Regime Routing Weights (L2-Regularized Softmax)...")
    Z_oof  = np.column_stack([to_z(rk_norm(oof_preds[m])) for m in model_names])
    Z_test = np.column_stack([to_z(test_final_members[m]) for m in model_names])
    K = len(model_names)

    def softmax(v):
        e_v = np.exp(v - np.max(v))
        return e_v / (e_v.sum() + 1e-12)

    def regime_loss(params):
        theta_p = params[:K]
        theta_d = params[K:2*K]
        theta_r = params[2*K:3*K]

        w_p = softmax(theta_p)
        w_d = softmax(theta_d)
        w_r = softmax(theta_r)

        z_p = Z_oof @ w_p
        z_d = Z_oof @ w_d
        z_r = Z_oof @ w_r

        z_routed = (1.0 - rev_tr) * ((1.0 - g_dist_tr) * z_p + g_dist_tr * z_d) + rev_tr * z_r
        reg_penalty = 0.05 * (np.sum((theta_p - np.mean(theta_p))**2) +
                              np.sum((theta_d - np.mean(theta_d))**2) +
                              np.sum((theta_r - np.mean(theta_r))**2))
        return -roc_auc_score(y_train, z_routed) + reg_penalty

    init_p = np.zeros(3 * K)
    res = minimize(regime_loss, init_p, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4})

    opt_p = res.x
    wp = softmax(opt_p[:K])
    wd = softmax(opt_p[K:2*K])
    wr = softmax(opt_p[2*K:3*K])

    z_p_oof = Z_oof @ wp
    z_d_oof = Z_oof @ wd
    z_r_oof = Z_oof @ wr
    z_routed_oof = (1.0 - rev_tr) * ((1.0 - g_dist_tr) * z_p_oof + g_dist_tr * z_d_oof) + rev_tr * z_r_oof
    regime_oof_auc = roc_auc_score(y_train, z_routed_oof)

    print(f"\\n[*] Continuous Regime Routing Optimal Weights Solved (OOF AUC: {regime_oof_auc:.6f}):")
    print("    Model Name           | Prime Share | Distress Share | Revolving Share")
    print("    ---------------------------------------------------------------------")
    for idx_m, m in enumerate(model_names):
        print(f"    - {m:18s} |   {wp[idx_m]*100:5.2f}%   |    {wd[idx_m]*100:5.2f}%    |    {wr[idx_m]*100:5.2f}%")

    z_test_p = Z_test @ wp
    z_test_d = Z_test @ wd
    z_test_r = Z_test @ wr
    z_test_routed = (1.0 - rev_te) * ((1.0 - g_dist_te) * z_test_p + g_dist_te * z_test_d) + rev_te * z_test_r
    test_regime = rk_norm(to_p(z_test_routed))

    # Macro Structural Consensus (Balance across Tree Zoo, XGBoost, and CatBoost)
    w_tree = (oof_preds["LGB_ET_E1"] + oof_preds["LGB_Shallow"] + oof_preds["LGB_ET_Deep"] + oof_preds["LGB_Solvency"] + oof_preds["LGB_Conservative"] + oof_preds["LGB_Wide"]) / 6.0
    w_tree_te = (test_final_members["LGB_ET_E1"] + test_final_members["LGB_Shallow"] + test_final_members["LGB_ET_Deep"] + test_final_members["LGB_Solvency"] + test_final_members["LGB_Conservative"] + test_final_members["LGB_Wide"]) / 6.0

    oof_macro = rk_norm(0.65 * rk_norm(w_tree) + 0.20 * rk_norm(oof_preds["CB_GPU"]) + 0.15 * rk_norm(oof_preds["XGB_GPU_Hist"]))
    test_macro = rk_norm(0.65 * rk_norm(w_tree_te) + 0.20 * test_final_members["CB_GPU"] + 0.15 * test_final_members["XGB_GPU_Hist"])
    macro_oof_auc = roc_auc_score(y_train, oof_macro)
    print(f"\\n[*] Macro Structural Consensus OOF ROC-AUC: {macro_oof_auc:.6f}")

    # Unified Dual Consensus (65% Regime Routing + 35% Macro Structural Consensus)
    oof_unified = rk_norm(0.65 * rk_norm(to_p(z_routed_oof)) + 0.35 * oof_macro)
    unified_oof_auc = roc_auc_score(y_train, oof_unified)
    test_ensemble = rk_norm(0.65 * test_regime + 0.35 * test_macro)

    print(f"\\n  >>> TOTAL UNIFIED DUAL CONSENSUS OOF ROC-AUC: {unified_oof_auc:.6f} <<<")
    print(f"  Total Training Runtime: {(time.time()-t_start)/60:.2f} minutes")
    print("=" * 80)

    return test_ensemble, unified_oof_auc'''

code = code.replace(old_routing, new_routing)

# 8. In main(), build cb_df and pass it to train_grandmaster_suite
old_main_call = '''    X_train = X_full.iloc[:n_train].reset_index(drop=True)
    X_test  = X_full.iloc[n_train:].reset_index(drop=True)

    # 15. Train 8-Engine Grandmaster Suite with Full-Train Re-Fit & Continuous Regime Routing
    test_ensemble, best_oof_auc = train_grandmaster_suite(X_train, y_train, X_test, df_train, df_test)'''

new_main_call = '''    # Build High-Cardinality Decoded Tariff Keys for CatBoost GPU
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

    # 15. Train 8-Engine Grandmaster Suite with Full-Train Re-Fit & Continuous Regime Routing
    test_ensemble, best_oof_auc = train_grandmaster_suite(X_train, y_train, X_test, df_train, df_test, cb_df)'''

code = code.replace(old_main_call, new_main_call)

with open('v22_champion_audit_certified_v43_grand_apex_soVEREIGN_master.py', 'w') as f:
    f.write(code)

print('V43 updated successfully via scratch/update_v43.py!')
