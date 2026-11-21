import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
from v22_champion_audit_certified_v44_grand_apex_soVEREIGN_master import (
    build_complete_base_manifold,
    extract_restored_raw_baseline_features,
    compute_hierarchical_tariff_exception_features,
    compute_hasan_banking_and_solvency_standards,
    compute_temporal_stability_and_inquiry_bursts,
    compute_nonlinear_external_credit_powers,
    compute_supervised_oof_ext_imputation,
    compute_russian_milestone_delays,
    compute_catalog_peaks_and_alexander_rates,
    compute_applicant_timeline_covariate_trajectories,
    compute_unsupervised_person_pooling,
    compute_group_deviations,
    compute_group_bureau_deviances,
    compute_deep_product_and_target_encodings,
    compute_covariate_expectations_and_residuals,
    compute_continuous_frequency_counts,
    compute_onehot_categoricals,
    compute_multiview_stacking,
    CATS
)

print("=" * 80)
print("TESTING V44 FEATURE PIPELINE END-TO-END (SMOKE TEST 200 ROWS)")
print("=" * 80)

tr = pd.read_csv("inpuy/train.csv", nrows=150)
te = pd.read_csv("inpuy/test.csv", nrows=50)
y_tr = tr['TARGET'].values.astype(int)
n_tr, n_te = len(tr), len(te)

cov_tr = tr.drop(columns=['TARGET'])
df_all = pd.concat([cov_tr, te], ignore_index=True)

df_base = build_complete_base_manifold(df_all)
df_raw = extract_restored_raw_baseline_features(df_all, df_base)
df_hq = compute_hierarchical_tariff_exception_features(df_all)
df_hasan = compute_hasan_banking_and_solvency_standards(df_all)
df_temp = compute_temporal_stability_and_inquiry_bursts(df_all)
df_ext_p = compute_nonlinear_external_credit_powers(df_all)
df_ext_imp = compute_supervised_oof_ext_imputation(df_all)
df_milestone = compute_russian_milestone_delays(df_all)
df_catalog = compute_catalog_peaks_and_alexander_rates(df_all)
df_timeline_deltas = compute_applicant_timeline_covariate_trajectories(df_all, df_ext_imp)
df_person = compute_unsupervised_person_pooling(df_all)
df_group = compute_group_deviations(df_all)
df_ext_group = compute_group_bureau_deviances(df_all, df_ext_imp)
df_te = compute_deep_product_and_target_encodings(df_all, y_tr, n_tr)
df_resid = compute_covariate_expectations_and_residuals(df_base, n_tr)
df_cnt = compute_continuous_frequency_counts(df_all)
df_oh = compute_onehot_categoricals(df_all)
df_views = compute_multiview_stacking(df_base, y_tr, n_tr)

X_full = pd.concat([
    df_base, df_raw, df_hq, df_hasan, df_temp, df_ext_p,
    df_ext_imp, df_milestone, df_catalog, df_timeline_deltas,
    df_person, df_group, df_ext_group, df_te, df_resid, df_cnt, df_oh, df_views
], axis=1)

print(f"X_full shape: {X_full.shape}")
dups = X_full.columns[X_full.columns.duplicated()].tolist()
print(f"Duplicate columns count: {len(dups)}")
if len(dups) > 0:
    print(f"Duplicates: {dups}")
    sys.exit(1)

print("[SUCCESS] All V44 columns are distinct and aligned!")
