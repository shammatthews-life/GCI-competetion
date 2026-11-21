import json

with open(r'D:\gci\truth\experiments\extracted_from_notepad\notepad_nb2_qfingerprint_0.777.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    print(f"\n==================== CELL {i} ====================")
    src = ''.join(c.get('source', []))
    print(src[:800])

