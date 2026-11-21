import json

with open(r'D:\gci\truth\experiments\extracted_from_notepad\notepad_nb2_qfingerprint_0.777.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in NB2: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))
    lines = src.splitlines()
    first_line = lines[0] if lines else ''
    print(f"Cell {i} ({ctype}, {len(lines)} lines): {first_line[:90]}")

