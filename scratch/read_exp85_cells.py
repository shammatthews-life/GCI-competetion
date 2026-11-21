import json

with open('champion/exp85_solo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell_idx in [6, 8, 14]:
    src = ''.join(nb['cells'][cell_idx].get('source', []))
    print(f'=' * 80)
    print(f'CELL {cell_idx} ({len(src)} chars, {len(src.splitlines())} lines):')
    print(f'=' * 80)
    for l in src.splitlines()[:60]:
        print(l)
