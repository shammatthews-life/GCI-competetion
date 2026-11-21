import re
import os
import glob

print("=== SEARCHING V10 SCORES, LOCAL AUC, AND FEATURE SPLITS ===")

files_to_check = [
    r'D:\gci\ORIGIN_MASTER_VERIFICATION_LEDGER.md',
    r'D:\gci\truth\GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md',
    r'D:\gci\truth\zip_extracted\zip\V10_ANALYSIS_TO_DATE_REPORT.md',
    r'D:\gci\past.md'
]

# Add any v10 ipynb or py
for root, dirs, files in os.walk(r'D:\gci'):
    if any(ig in root for ig in ['.git', '__pycache__', 'node_modules']): continue
    for f in files:
        if 'v10' in f.lower() and (f.endswith('.md') or f.endswith('.txt') or f.endswith('.json')):
            files_to_check.append(os.path.join(root, f))

files_to_check = sorted(list(set(files_to_check)))

for fpath in files_to_check:
    if not os.path.exists(fpath): continue
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='replace') as fp:
        lines = fp.readlines()
    
    hits = []
    for i, l in enumerate(lines):
        if any(k in l.lower() for k in ['v10', 'v_10']) and any(k in l.lower() for k in ['auc', 'score', 'oof', 'split', '0.7']):
            hits.append((i+1, l.strip()[:140]))
    
    if hits:
        print(f"\n--- {fname} ({len(hits)} hits) ---")
        for h in hits[:10]:
            print(f"  Line {h[0]}: {h[1]}")

