import os, sys, time, gc, warnings, pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

WORK_DIR = 'd:/gci/gemini'
CACHE_DIR = os.path.join(WORK_DIR, 'cache')

print("==================================================")
print(" GCI World 2026 - EXP_15: Specialized GPU XGBoost on Enriched Manifold")
print("==================================================")
t0 = time.time()

TRAIN_PATH = 'd:/gci/inpuy/train.csv'
TEST_PATH = 'd:/gci/inpuy/test.csv'
SAMPLE_PATH = 'd:/gci/inpuy/sample_submission.csv'

tr = pd.read_csv(TRAIN_PATH)
te = pd.read_csv(TEST_PATH)
y = tr["TARGET"].values
ntr = len(tr)
nte = len(te)
n = ntr
FOLDS = 5
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y))

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

# 1. Load Features
print("Loading base features, residuals, partner dynamics, and consistent views...")
Xa = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xa.feather'))
Xat = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xat.feather'))

resid = pd.read_pickle('D:/gci/codex/v_3_cache/resid16.pkl')
R_tr = resid.iloc[:ntr].reset_index(drop=True)
R_te = resid.iloc[ntr:].reset_index(drop=True)

dyn_tr = pd.read_feather(os.path.join(CACHE_DIR, 'exp14_partner_dyn_tr.feather'))
dyn_te = pd.read_feather(os.path.join(CACHE_DIR, 'exp14_partner_dyn_te.feather'))

VIEW_A = pd.read_pickle('D:/gci/codex/v_3_cache/e1A_views.pkl')

# One-hot encoding of categoricals
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]
z = pd.concat([Xa, Xat], ignore_index=True)
oh = pd.get_dummies(z[CATS].astype("category"), dummy_na=True, dtype=np.float32)
oh.columns = [f"OH_{i}" for i in range(oh.shape[1])]
Z_base = pd.concat([z.drop(columns=CATS), oh], axis=1)

X_tr_base = pd.concat([Z_base.iloc[:ntr].reset_index(drop=True), R_tr, dyn_tr], axis=1)
X_te_base = pd.concat([Z_base.iloc[ntr:].reset_index(drop=True), R_te, dyn_te], axis=1)

# Clean column names for XGBoost
clean_cols = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in X_tr_base.columns]
X_tr_base.columns = clean_cols
X_te_base.columns = clean_cols

# Load precomputed fold encodings
enc_folds = []
for k in range(1, 6):
    with open(os.path.join(CACHE_DIR, f'v16_enc_fold_{k}.pkl'), 'rb') as f_in:
        enc_folds.append(pickle.load(f_in))

# 2. Train GPU XGBoost across 5 Folds
print("\n--- Training GPU XGBoost on Enriched Feature Matrix ---")
oof_xgb = np.zeros(n, dtype=np.float32)
test_xgb = np.zeros(nte, dtype=np.float32)

t_xgb0 = time.time()
for k, (a, b) in enumerate(folds):
    t_f0 = time.time()
    enc_data = enc_folds[k]
    
    # Encodings cleaned column names
    enc_fa = enc_data['fa_enc'].copy()
    enc_fb = enc_data['fb_enc'].copy()
    enc_ft = enc_data['ft_enc'].copy()
    enc_fa.columns = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in enc_fa.columns]
    enc_fb.columns = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in enc_fb.columns]
    enc_ft.columns = [c.replace(':', '_').replace('|', '_').replace(' ', '_').replace('-', '_') for c in enc_ft.columns]
    
    fa = pd.concat([X_tr_base.iloc[a].reset_index(drop=True), enc_fa, VIEW_A['tr'][k]], axis=1)
    fb = pd.concat([X_tr_base.iloc[b].reset_index(drop=True), enc_fb, VIEW_A['va'][k]], axis=1)
    ft = pd.concat([X_te_base.reset_index(drop=True), enc_ft, VIEW_A['te'][k]], axis=1)
    
    xgb_model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.035,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=5.0,
        tree_method='hist',
        device='cuda',
        eval_metric='auc',
        early_stopping_rounds=60,
        random_state=42 + k
    )
    
    xgb_model.fit(
        fa, y[a],
        eval_set=[(fb, y[b])],
        verbose=False
    )
    
    val_pred = xgb_model.predict_proba(fb)[:, 1]
    oof_xgb[b] = val_pred
    test_xgb += xgb_model.predict_proba(ft)[:, 1] / FOLDS
    print(f"  GPU XGBoost fold {k+1}/{FOLDS} AUC: {roc_auc_score(y[b], val_pred):.5f} (iter {xgb_model.best_iteration}) in {time.time()-t_f0:.1f}s")

auc_xgb = roc_auc_score(y, oof_xgb)
print(f"\n>>> Complete GPU XGBoost OOF AUC: {auc_xgb:.6f} ({time.time()-t_xgb0:.1f}s) <<<")

# 3. Load Standing Champion v17
v17_bundle = np.load(os.path.join(CACHE_DIR, 'v_17_full_suite_bundle.npz'))
oof_v17 = v17_bundle['final_oof']
test_v17 = v17_bundle['final_test']
auc_v17 = roc_auc_score(y, oof_v17)
print(f"--> Current Standing Champion (v17) OOF AUC: {auc_v17:.6f}")

# 4. Evaluate Blend of Standing Champion + GPU XGBoost
print("\n--- Blending GPU XGBoost with Standing Champion ---")
best_auc = auc_v17
best_w = 0.0

for w in np.linspace(0.02, 0.30, 20):
    cand = (1.0 - w) * rk(oof_v17) + w * rk(oof_xgb)
    score = roc_auc_score(y, cand)
    print(f"  Blend {100*(1-w):.1f}% v17 + {100*w:.1f}% GPU_XGB -> OOF AUC: {score:.6f} ({score - auc_v17:+.6f})")
    if score > best_auc:
        best_auc = score
        best_w = w

final_v18_oof = (1.0 - best_w) * rk(oof_v17) + best_w * rk(oof_xgb)
final_v18_test = rk((1.0 - best_w) * rk(test_v17) + best_w * rk(test_xgb))

fold_gains = []
for k, (trn, val) in enumerate(folds, 1):
    f_base = roc_auc_score(y[val], oof_v17[val])
    f_blend = roc_auc_score(y[val], final_v18_oof[val])
    fold_gains.append(f_blend - f_base)

print(f"\n==================================================")
print(f" EXP_15 GPU XGBOOST ENRICHED GRAND SUMMARY:")
print(f"   Historical Baseline (v15):       0.774041")
print(f"   Standing Champion (v17):         {auc_v17:.6f}")
print(f"   New EXP_15 Candidate OOF AUC:    {best_auc:.6f}")
print(f"   Net Gain vs Standing Champion:   {best_auc - auc_v17:+.6f}")
print(f"   Total Net Gain vs Baseline v15:  {best_auc - 0.774041:+.6f}")
print(f"   Optimal GPU XGBoost Weight:      {100*best_w:.1f}%")
print(f"   Fold Gains vs v17: {[round(g, 6) for g in fold_gains]} ({sum(g > 0 for g in fold_gains)}/5 positive)")
print(f"==================================================")

np.savez_compressed(
    os.path.join(CACHE_DIR, 'exp15_xgb_bundle.npz'),
    oof_xgb=oof_xgb, test_xgb=test_xgb, auc_xgb=auc_xgb
)

if best_auc > auc_v17 and sum(g > 0 for g in fold_gains) >= 4:
    print(">>> PASS: NEW GRAND CHAMPION CONFIRMED! Generating candidate submission...")
    sub_template = pd.read_csv(SAMPLE_PATH)
    submission = sub_template.copy()
    submission['TARGET'] = final_v18_test
    sub_file = os.path.join(WORK_DIR, 'v_18_xgboost_champion_submission.csv')
    submission.to_csv(sub_file, index=False)
    print(f"Wrote NEW grand champion submission to {sub_file}")
    
    np.savez_compressed(
        os.path.join(CACHE_DIR, 'v_18_blend_bundle.npz'),
        oof_xgb=oof_xgb, test_xgb=test_xgb,
        final_oof=final_v18_oof, final_test_rank=final_v18_test,
        final_auc=best_auc, fold_gains=fold_gains
    )
else:
    print(">>> Gate decision: v17 remains intact as reigning champion.")

print("EXP_15 completed in {:.1f}s.".format(time.time() - t0))
