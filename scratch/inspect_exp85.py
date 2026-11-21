import json

with open('champion/exp85_solo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells in champion/exp85_solo.ipynb:', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    lines = src.strip().split('\n')
    first_line = lines[0] if lines else ''
    if any(k in src.lower() for k in ['def ', 'params', 'lgb', 'xgb', 'cat', 'feature', 'auc', 'partner', 'routing']):
        print(f'Cell {i:2d} ({c["cell_type"]}): {first_line[:85]}')
