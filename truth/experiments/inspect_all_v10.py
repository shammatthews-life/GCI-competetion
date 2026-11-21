import os
import json
import re

v10_paths = [
    r'D:\gci\gemini\independentexperiments\audit_certified_v10_conservative.ipynb',
    r'D:\gci\truth\zip_extracted\zip\GCI_v10_candidate_blocks_colab.ipynb',
    r'D:\gci\truth\zip_extracted\zip\GCI_v10_standalone_colab.ipynb',
    r'D:\gci\pre info\Right_Customer_Right_Offer_Right_Time_v10.ipynb'
]

for p in v10_paths:
    if not os.path.exists(p):
        print("NOT FOUND:", p)
        continue
    fname = os.path.basename(p)
    with open(p, 'r', encoding='utf-8', errors='replace') as fp:
        data = json.load(fp)
    cells = data.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    all_text = '\n'.join(''.join(c.get('source', [])) for c in cells)
    for c in code_cells:
        for out in c.get('outputs', []):
            if 'text' in out:
                all_text += '\n' + ''.join(out['text'])
            elif 'data' in out and 'text/plain' in out['data']:
                all_text += '\n' + ''.join(out['data']['text/plain'])
                
    aucs = list(set(re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', all_text)))
    splits = [l.strip() for l in all_text.splitlines() if any(k in l.lower() for k in ['split', 'segment', 'subgroup', 'cash', 'revolving', 'goods_credit', 'contract_type']) and any(k in l for k in ['AUC', 'auc', '0.7', 'gain'])]
    
    print(f"\n==================== {fname} ====================")
    print(f"Total cells: {len(cells)}, Code cells: {len(code_cells)}")
    print(f"AUCs mentioned: {aucs[:10]}")
    print(f"Split/segment lines found: {len(splits)}")
    for s in splits[:5]:
        print(" ", s[:120])

