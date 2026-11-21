import os
import glob
import json
import re

base_zip = r'D:\gci\truth\zip_extracted'
truth_dir = r'D:\gci\truth'

from summarize_scripts import summarize_file

md_files = glob.glob(os.path.join(base_zip, '**/*.md'), recursive=True)
txt_files = glob.glob(os.path.join(base_zip, '**/*.txt'), recursive=True)
ipynb_files = glob.glob(os.path.join(base_zip, '**/*.ipynb'), recursive=True)

print(f"MD files: {len(md_files)}, TXT files: {len(txt_files)}, IPYNB files: {len(ipynb_files)}")

all_summaries = {'md': {}, 'txt': {}, 'ipynb': {}}

for m in md_files:
    rel = os.path.relpath(m, base_zip)
    all_summaries['md'][rel] = summarize_file(m)

for t in txt_files:
    rel = os.path.relpath(t, base_zip)
    all_summaries['txt'][rel] = summarize_file(t)

for nb in ipynb_files:
    rel = os.path.relpath(nb, base_zip)
    all_summaries['ipynb'][rel] = summarize_file(nb)

with open(r'D:\gci\truth\experiments\all_zip_summaries.json', 'w', encoding='utf-8') as f:
    json.dump(all_summaries, f, indent=2)

print("Saved all_zip_summaries.json")

