import json
import os
import glob
import re

print("=== READING EXISTING AUDIT METADATA ===")

with open(r'D:\gci\truth\experiments\all_63_notebooks_detailed.json', 'r', encoding='utf-8') as f:
    notebooks = json.load(f)

with open(r'D:\gci\truth\experiments\all_scripts_detailed.json', 'r', encoding='utf-8') as f:
    scripts = json.load(f)

with open(r'D:\gci\truth\experiments\all_docs_detailed.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)

base_zip = r'D:\gci\truth\zip_extracted'
csv_files = sorted(glob.glob(os.path.join(base_zip, '**/*.csv'), recursive=True))

print(f"Loaded: {len(notebooks)} notebooks, {len(scripts)} scripts, {len(docs)} docs, {len(csv_files)} CSVs.")

# Let's inspect all CSV categories
submission_csvs = []
oof_csvs = []
metric_csvs = []
sweep_csvs = []

for c in csv_files:
    fname = os.path.basename(c)
    sz = os.path.getsize(c)
    if 'submission' in fname.lower() or fname.startswith('v') and 'csv' in fname.lower() or 'champion' in fname.lower() or 'blend' in fname.lower() or fname.startswith('GCI_') and not 'fold' in fname.lower():
        submission_csvs.append((fname, sz))
    elif 'oof' in fname.lower() or 'pred' in fname.lower():
        oof_csvs.append((fname, sz))
    elif 'fold' in fname.lower() or 'summary' in fname.lower() or 'progress' in fname.lower() or 'result' in fname.lower():
        metric_csvs.append((fname, sz))
    else:
        sweep_csvs.append((fname, sz))

print(f"Categorized: {len(submission_csvs)} submissions/blends, {len(oof_csvs)} OOF arrays, {len(metric_csvs)} fold metrics, {len(sweep_csvs)} sweeps.")

