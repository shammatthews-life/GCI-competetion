import os
import glob

base_zip = r'D:\gci\truth\zip_extracted'
py_files = sorted(glob.glob(os.path.join(base_zip, '**/*.py'), recursive=True))

print(f"Total Python scripts: {len(py_files)}")
for p in py_files:
    fname = os.path.basename(p)
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    lines = content.splitlines()
    print(f"\n--- {fname} ({len(lines)} lines, {len(content)} bytes) ---")
    # First 10 lines
    for l in lines[:10]:
        print(" ", l)

