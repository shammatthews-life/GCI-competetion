import os
import glob
import re

base_zip = r'D:\gci\truth\zip_extracted'
md_files = glob.glob(os.path.join(base_zip, '**/*.md'), recursive=True)
txt_files = glob.glob(os.path.join(base_zip, '**/*.txt'), recursive=True)

print(f"Total MD: {len(md_files)}, Total TXT: {len(txt_files)}")

for fpath in sorted(md_files + txt_files):
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    lines = content.splitlines()
    aucs = re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', content)
    unique_aucs = list(dict.fromkeys(aucs))[:8]
    print(f"\nFILE: {fname} ({len(lines)} lines, {len(content)} bytes)")
    print(f"  AUCs found: {unique_aucs}")
    # print first 5 non-empty lines
    non_empty = [l.strip() for l in lines if l.strip()][:5]
    print("  Header preview:", " | ".join(non_empty))

