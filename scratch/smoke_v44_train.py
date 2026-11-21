import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import pandas as pd
from v22_champion_audit_certified_v44_grand_apex_soVEREIGN_master import (
    train_grandmaster_suite, CATS
)

print("=" * 80)
print("RUNNING 200-ROW SMOKE TEST ON TRAIN_GRANDMASTER_SUITE")
print("=" * 80)

# Create dummy X_train, y_train, X_test, df_train_raw, df_test_raw, cb_df
np.random.seed(42)
n_tr, n_te, n_f = 100, 40, 20
X_tr = pd.DataFrame(np.random.randn(n_tr, n_f).astype(np.float32), columns=[f'f_{i}' for i in range(n_f)])
X_te = pd.DataFrame(np.random.randn(n_te, n_f).astype(np.float32), columns=[f'f_{i}' for i in range(n_f)])
y_tr = (np.random.rand(n_tr) > 0.8).astype(int)
y_tr[0] = 1; y_tr[1] = 0 # ensure both classes exist

df_tr_raw = pd.DataFrame({
    'EXT_SOURCE_1': np.random.rand(n_tr),
    'EXT_SOURCE_2': np.random.rand(n_tr),
    'EXT_SOURCE_3': np.random.rand(n_tr),
    'DAYS_EMPLOYED': np.where(np.random.rand(n_tr) > 0.8, 365243, -1000),
    'CODE_GENDER': np.where(np.random.rand(n_tr) > 0.5, 'M', 'F'),
    'NAME_CONTRACT_TYPE': np.where(np.random.rand(n_tr) > 0.8, 'Revolving loans', 'Cash loans')
})
df_te_raw = pd.DataFrame({
    'EXT_SOURCE_1': np.random.rand(n_te),
    'EXT_SOURCE_2': np.random.rand(n_te),
    'EXT_SOURCE_3': np.random.rand(n_te),
    'DAYS_EMPLOYED': np.where(np.random.rand(n_te) > 0.8, 365243, -1000),
    'CODE_GENDER': np.where(np.random.rand(n_te) > 0.5, 'M', 'F'),
    'NAME_CONTRACT_TYPE': np.where(np.random.rand(n_te) > 0.8, 'Revolving loans', 'Cash loans')
})

cb_df = pd.DataFrame(index=range(n_tr + n_te))
for c in CATS:
    cb_df[c] = 'CAT_A'
for q in ['Q3', 'Q4', 'P3', 'P4', 'PF4', 'CT1', 'CT2', 'CAT_K_DEC']:
    cb_df[q] = '0.1'

print("Starting smoke test of train_grandmaster_suite (2 folds, small rounds)...")
# Run with tiny boost rounds for smoke test
test_ens, oof_auc = train_grandmaster_suite(X_tr, y_tr, X_te, df_tr_raw, df_te_raw, cb_df)
print(f"Smoke test successful! test_ens shape: {test_ens.shape}, oof_auc: {oof_auc:.4f}")
