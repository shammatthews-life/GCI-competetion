"""
Builder script to construct v22_champion_audit_certified_v20_multi_seed_gpu_solvency_master.py cleanly.
"""

with open("d:/gci/v22_champion_audit_certified_v18_probit_regime_master.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace V18 with V20 in header and target
code = code.replace("V18 (PROBIT-REGIME MASTER ENGINE)", "V20 (MULTI-SEED GPU & SOLVENCY MASTER ENGINE)")
code = code.replace("v_22_pure_champion_ensemble_certified_v18.csv", "v_22_pure_champion_ensemble_certified_v20.csv")
code = code.replace("V18 PROBIT-REGIME MASTER", "V20 MULTI-SEED GPU SOLVENCY MASTER")
code = code.replace("Re-Engineered V22 Champion V18", "Re-Engineered V22 Champion V20")
code = code.replace("v18_test", "v20_test")
code = code.replace("v18_oof", "v20_oof")
code = code.replace("v18_auc", "v20_auc")

# Add High-Order Solvency Manifold in Section 10b
old_contra = "    # 10. Contradiction Manifold"
new_contra = """    # 10. Solvency & Joint Bureau Manifold
    def compute_solvency_geo(df_):
        cr_ = df_["AMT_CREDIT"].to_numpy("float64")
        an_ = df_["AMT_ANNUITY"].to_numpy("float64")
        inc_ = df_["AMT_INCOME_TOTAL"].to_numpy("float64")
        gp_ = df_["AMT_GOODS_PRICE"].to_numpy("float64")
        e1_ = df_["EXT_SOURCE_1"].to_numpy("float64")
        e2_ = df_["EXT_SOURCE_2"].to_numpy("float64")
        e3_ = df_["EXT_SOURCE_3"].to_numpy("float64")
        em_ = np.column_stack([e1_, e2_, e3_])
        enc_ = np.isnan(em_).sum(axis=1)
        with np.errstate(all="ignore"):
            eprod_ = np.nanprod(em_, axis=1)
            egeo_ = np.power(eprod_, 1.0 / np.maximum(3 - enc_, 1.0))
            emin_ = np.nanmin(em_, axis=1)
            solv_stress_ = (an_ / np.maximum(inc_, 1.0)) / np.maximum(emin_, 0.01)
            gtc_ = gp_ / np.maximum(cr_, 1.0)
            atg_ = an_ / np.maximum(gp_, 1.0)
        return pd.DataFrame({
            "EXT_GEO_MEAN": np.where(enc_ < 3, egeo_, np.nan).astype(np.float32),
            "SOLVENCY_STRESS": np.where(enc_ < 3, solv_stress_, np.nan).astype(np.float32),
            "GOODS_TO_CREDIT_SOLV": gtc_.astype(np.float32),
            "ANNUITY_TO_GOODS_SOLV": atg_.astype(np.float32)
        }, index=df_.index)

    f_solv_geo_tr = compute_solvency_geo(df_train)
    f_solv_geo_te = compute_solvency_geo(df_test)

    # 10b. Contradiction Manifold"""

code = code.replace(old_contra, new_contra)

# Attach f_solv_geo into Xc and Xa
old_attach = "    Xc = pd.concat([f_cl_tr, f_ex_tr, f_fi_tr, f_ag_tr, f_bu_tr, f_count_tr, f_grp_tr, f_seg_tr, f_rate_tr, f_merton_tr, f_copula_tr, f_contra_tr, f_link_tr], axis=1)"
new_attach = "    Xc = pd.concat([f_cl_tr, f_ex_tr, f_fi_tr, f_ag_tr, f_bu_tr, f_count_tr, f_grp_tr, f_seg_tr, f_rate_tr, f_merton_tr, f_copula_tr, f_solv_geo_tr, f_contra_tr, f_link_tr], axis=1)"
code = code.replace(old_attach, new_attach)

old_attach_te = "    Xct = pd.concat([f_cl_te, f_ex_te, f_fi_te, f_ag_te, f_bu_te, f_count_te, f_grp_te, f_seg_te, f_rate_te, f_merton_te, f_copula_te, f_contra_te, f_link_te], axis=1)"
new_attach_te = "    Xct = pd.concat([f_cl_te, f_ex_te, f_fi_te, f_ag_te, f_bu_te, f_count_te, f_grp_te, f_seg_te, f_rate_te, f_merton_te, f_copula_te, f_solv_geo_te, f_contra_te, f_link_te], axis=1)"
code = code.replace(old_attach_te, new_attach_te)

# In CatBoost GPU, add multi-seed bagging [42, 2026]
old_cb = """        cb_model = CatBoostClassifier(**CBP)
        cb_model.fit(ca, y_train[a], cat_features=CB_CATS, verbose=False)
        pv = cb_model.predict_proba(cbv)[:, 1]
        pt = cb_model.predict_proba(cbt)[:, 1]"""

new_cb = """        pv_seeds, pt_seeds = [], []
        cb_seeds = [42] if fast else [42, 2026]
        for s_cb in cb_seeds:
            cb_model = CatBoostClassifier(**{**CBP, "random_seed": s_cb})
            cb_model.fit(ca, y_train[a], cat_features=CB_CATS, verbose=False)
            pv_seeds.append(cb_model.predict_proba(cbv)[:, 1])
            pt_seeds.append(cb_model.predict_proba(cbt)[:, 1])
        pv = np.mean(pv_seeds, axis=0)
        pt = np.mean(pt_seeds, axis=0)"""

code = code.replace(old_cb, new_cb)

# In XGBoost GPU, add 3-seed bagging [42, 2026, 777]
old_xgb = """        m_xgb = xgb.XGBClassifier(
            n_estimators=xgb_n_est,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=20.0,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        m_xgb.fit(fa, y_train[a])
        pv = m_xgb.predict_proba(fb)[:, 1]
        pt = m_xgb.predict_proba(ft)[:, 1]"""

new_xgb = """        pv_xgb_seeds, pt_xgb_seeds = [], []
        xgb_seeds = [42] if fast else [42, 2026, 777]
        for s_xgb in xgb_seeds:
            m_xgb = xgb.XGBClassifier(
                n_estimators=xgb_n_est,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.5,
                reg_lambda=20.0,
                tree_method="hist",
                device="cuda",
                random_state=s_xgb
            )
            m_xgb.fit(fa, y_train[a])
            pv_xgb_seeds.append(m_xgb.predict_proba(fb)[:, 1])
            pt_xgb_seeds.append(m_xgb.predict_proba(ft)[:, 1])
        pv = np.mean(pv_xgb_seeds, axis=0)
        pt = np.mean(pt_xgb_seeds, axis=0)"""

code = code.replace(old_xgb, new_xgb)

with open("d:/gci/v22_champion_audit_certified_v20_multi_seed_gpu_solvency_master.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully written v20! Total characters:", len(code))
