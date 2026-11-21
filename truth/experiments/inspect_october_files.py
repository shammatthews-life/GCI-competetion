import json
import os
import zipfile
import numpy as np

def inspect_notebook(path):
    print("=" * 80)
    print("FILE:", path)
    print("=" * 80)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        print(f"\n--- CELL {i} ({c['cell_type']}) ---")
        src = ''.join(c.get('source', []))
        print(src)
        outputs = c.get('outputs', [])
        if outputs:
            print("--- CELL OUTPUT ---")
            for out in outputs:
                if 'text' in out:
                    print(''.join(out['text']))
                elif 'data' in out and 'text/plain' in out['data']:
                    print(''.join(out['data']['text/plain']))

def inspect_zip(path):
    print("=" * 80)
    print("ZIP FILE:", path)
    print("=" * 80)
    with zipfile.ZipFile(path, 'r') as z:
        infolist = z.infolist()
        print(f"Total entries: {len(infolist)}")
        # Group by extension
        exts = {}
        for info in infolist:
            ext = os.path.splitext(info.filename)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
        print("Extensions count:", exts)
        print("\nAll files list:")
        for info in sorted(infolist, key=lambda x: x.file_size, reverse=True)[:50]:
            print(f"  {info.filename} ({info.file_size:,} bytes)")

def inspect_txt(path, max_lines=100):
    print("=" * 80)
    print("TEXT FILE:", path)
    print("=" * 80)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print("First 30 lines:")
    for l in lines[:30]:
        print(l.rstrip())
    print("\n...\nLast 30 lines:")
    for l in lines[-30:]:
        print(l.rstrip())

if __name__ == '__main__':
    inspect_notebook(r'D:\gci\truth\GCI_v7_extra_trees_colab_1.ipynb')
    print("\n\n")
    inspect_notebook(r'D:\gci\truth\GCI_v13_reweight_and_ET_colab_1.ipynb')

