import os
import glob
import json
import numpy as np

base_zip = r'D:\gci\truth\zip_extracted'
truth_dir = r'D:\gci\truth'

print("=== GATHERING ALL FILES ===")
all_files = []
for root, dirs, files in os.walk(base_zip):
    for f in files:
        all_files.append(os.path.join(root, f))

truth_files = [
    os.path.join(truth_dir, 'GCI_v7_extra_trees_colab_1.ipynb'),
    os.path.join(truth_dir, 'GCI_v13_reweight_and_ET_colab_1.ipynb'),
    os.path.join(truth_dir, 'notepade.txt'),
    os.path.join(truth_dir, 'process.txt'),
    os.path.join(truth_dir, 'GCI_hgb_product_test.npy'),
    os.path.join(truth_dir, 'zip.zip'),
]

print(f"Total extracted files: {len(all_files)}")
print(f"Total direct truth files: {len(truth_files)}")

# Group files
by_ext = {}
for f in all_files:
    ext = os.path.splitext(f)[1].lower()
    by_ext.setdefault(ext, []).append(f)

for ext, flist in sorted(by_ext.items(), key=lambda x: -len(x[1])):
    print(f"Extension '{ext}': {len(flist)} files")

