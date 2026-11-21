"""
Builder script to construct v22_champion_audit_certified_v23_10fold_gpu_master.py cleanly.
"""

with open("d:/gci/v22_champion_audit_certified_v20_multi_seed_gpu_solvency_master.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace V20 with V23 in header and target
code = code.replace("V20 (MULTI-SEED GPU & SOLVENCY MASTER ENGINE)", "V23 (10-FOLD MULTI-SEED GPU MASTER ENGINE)")
code = code.replace("v_22_pure_champion_ensemble_certified_v20.csv", "v_22_pure_champion_ensemble_certified_v23.csv")
code = code.replace("V20 MULTI-SEED GPU SOLVENCY MASTER", "V23 10-FOLD MULTI-SEED GPU MASTER")
code = code.replace("Re-Engineered V22 Champion V20", "Re-Engineered V22 Champion V23")
code = code.replace("v20_test", "v23_test")
code = code.replace("v20_oof", "v23_oof")
code = code.replace("v20_auc", "v23_auc")

# Set FOLDS = 10
code = code.replace("FOLDS = 2 if fast else 5", "FOLDS = 2 if fast else 10")

# Update compute_multiviews_inductive to accept n_splits and use it
old_mv_def = 'def compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=False):'
new_mv_def = 'def compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=False, n_splits=5):'
code = code.replace(old_mv_def, new_mv_def)

old_mv_skf = '    skf = StratifiedKFold(2 if fast else 5, shuffle=True, random_state=42)'
new_mv_skf = '    skf = StratifiedKFold(2 if fast else n_splits, shuffle=True, random_state=42)'
code = code.replace(old_mv_skf, new_mv_skf)

# Update main() calls to pass n_splits=FOLDS
old_call_o = 'views_v17_o = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="orig", fast=args.fast)'
new_call_o = 'views_v17_o = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="orig", fast=args.fast, n_splits=FOLDS)'
code = code.replace(old_call_o, new_call_o)

old_call_a = 'views_v17_a = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=args.fast)'
new_call_a = 'views_v17_a = compute_multiviews_inductive(bf, y_train, extra_views=False, variant="A", fast=args.fast, n_splits=FOLDS)'
code = code.replace(old_call_a, new_call_a)

old_call_ox = 'views_v18_ox = compute_multiviews_inductive(bf, y_train, extra_views=True, variant="orig", fast=args.fast)'
new_call_ox = 'views_v18_ox = compute_multiviews_inductive(bf, y_train, extra_views=True, variant="orig", fast=args.fast, n_splits=FOLDS)'
code = code.replace(old_call_ox, new_call_ox)

# In main(), define FOLDS before calling compute_multiviews_inductive
old_main_top = """    # 2. Extract Authentic Feature Manifolds
    bf = build_all_features_inductive(df_train, df_test)
    BASEF = {
        "Xc": bf["Xc"], "Xct": bf["Xct"],
        "Xa": bf["Xa"], "Xat": bf["Xat"],
        "Xck": bf["Xck"], "Xckt": bf["Xckt"],
        "Xak": bf["Xak"], "Xakt": bf["Xakt"]
    }
    res_tr, res_te = compute_residuals_inductive(df_train, df_test, fast=args.fast)
    F0 = {
        "Xc": pd.concat([bf["Xc"], res_tr], axis=1),
        "Xct": pd.concat([bf["Xct"], res_te], axis=1),
        "Xa": pd.concat([bf["Xa"], res_tr], axis=1),
        "Xat": pd.concat([bf["Xat"], res_te], axis=1),
        "Xck": pd.concat([bf["Xck"], res_tr], axis=1),
        "Xckt": pd.concat([bf["Xckt"], res_te], axis=1),
        "Xak": pd.concat([bf["Xak"], res_tr], axis=1),
        "Xakt": pd.concat([bf["Xakt"], res_te], axis=1)
    }
    F1 = {
        "Xc": pd.concat([bf["Xc"], res_tr], axis=1),
        "Xct": pd.concat([bf["Xct"], res_te], axis=1),
        "Xa": pd.concat([bf["Xa"], res_tr], axis=1),
        "Xat": pd.concat([bf["Xat"], res_te], axis=1),
        "Xck": pd.concat([bf["Xck"], res_tr], axis=1),
        "Xckt": pd.concat([bf["Xckt"], res_te], axis=1),
        "Xak": pd.concat([bf["Xak"], res_tr], axis=1),
        "Xakt": pd.concat([bf["Xakt"], res_te], axis=1)
    }

    # 3. Multi-View Meta-Learners"""

new_main_top = """    # 2. Extract Authentic Feature Manifolds
    FOLDS = 2 if args.fast else 10
    bf = build_all_features_inductive(df_train, df_test)
    BASEF = {
        "Xc": bf["Xc"], "Xct": bf["Xct"],
        "Xa": bf["Xa"], "Xat": bf["Xat"],
        "Xck": bf["Xck"], "Xckt": bf["Xckt"],
        "Xak": bf["Xak"], "Xakt": bf["Xakt"]
    }
    res_tr, res_te = compute_residuals_inductive(df_train, df_test, fast=args.fast)
    F0 = {
        "Xc": pd.concat([bf["Xc"], res_tr], axis=1),
        "Xct": pd.concat([bf["Xct"], res_te], axis=1),
        "Xa": pd.concat([bf["Xa"], res_tr], axis=1),
        "Xat": pd.concat([bf["Xat"], res_te], axis=1),
        "Xck": pd.concat([bf["Xck"], res_tr], axis=1),
        "Xckt": pd.concat([bf["Xckt"], res_te], axis=1),
        "Xak": pd.concat([bf["Xak"], res_tr], axis=1),
        "Xakt": pd.concat([bf["Xakt"], res_te], axis=1)
    }
    F1 = {
        "Xc": pd.concat([bf["Xc"], res_tr], axis=1),
        "Xct": pd.concat([bf["Xct"], res_te], axis=1),
        "Xa": pd.concat([bf["Xa"], res_tr], axis=1),
        "Xat": pd.concat([bf["Xat"], res_te], axis=1),
        "Xck": pd.concat([bf["Xck"], res_tr], axis=1),
        "Xckt": pd.concat([bf["Xckt"], res_te], axis=1),
        "Xak": pd.concat([bf["Xak"], res_tr], axis=1),
        "Xakt": pd.concat([bf["Xakt"], res_te], axis=1)
    }

    # 3. Multi-View Meta-Learners"""

code = code.replace(old_main_top, new_main_top)

with open("d:/gci/v22_champion_audit_certified_v23_10fold_gpu_master.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully rebuilt v23! Total length:", len(code))
