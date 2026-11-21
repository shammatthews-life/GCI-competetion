import os
import glob
import json
import re

base_zip = r'D:\gci\truth\zip_extracted'
nbs = sorted(glob.glob(os.path.join(base_zip, '**/*.ipynb'), recursive=True))

print(f"Parsing all {len(nbs)} notebooks...")

nb_details = []
for idx, nb_path in enumerate(nbs):
    rel_path = os.path.relpath(nb_path, base_zip)
    fname = os.path.basename(nb_path)
    file_size = os.path.getsize(nb_path)
    
    try:
        with open(nb_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except Exception as e:
        nb_details.append({
            'index': idx + 1,
            'name': fname,
            'rel_path': rel_path,
            'size': file_size,
            'error': str(e)
        })
        continue

    cells = data.get('cells', [])
    md_cells = [c for c in cells if c.get('cell_type') == 'markdown']
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    
    total_code_lines = 0
    code_sources = []
    output_texts = []
    
    for c in code_cells:
        lines = ''.join(c.get('source', [])).splitlines()
        total_code_lines += len(lines)
        code_sources.extend(lines)
        for out in c.get('outputs', []):
            if 'text' in out:
                output_texts.extend(''.join(out['text']).splitlines())
            elif 'data' in out and 'text/plain' in out['data']:
                output_texts.extend(''.join(out['data']['text/plain']).splitlines())

    all_text = '\n'.join(''.join(c.get('source', [])) for c in cells) + '\n' + '\n'.join(output_texts)
    
    # Models, algorithms, AUCs
    aucs = re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', all_text)
    aucs = list(dict.fromkeys(aucs))[:10]
    
    models = list(set(re.findall(r'\b(lgb|lightgbm|xgb|xgboost|catboost|CatBoost|ExtraTrees|HistGradientBoosting|RandomForest|LogisticRegression|RealMLP|TabM|MLP|Neural)\b', all_text, re.IGNORECASE)))
    
    # Features mentioned
    features = list(set(re.findall(r'\b(EXT_SOURCE_\d|EXT_PRODUCT|EXT_HARMONIC|CREDIT_TERM|GOODS_TO_CREDIT|ANNUITY_TO_INCOME|CREDIT_TO_INCOME|DE_SENT|Q0|Q_0|Q3|Q4|Q5|P3|P4|PF4|TE_[a-zA-Z0-9_]+|VIEW_[A-Z]+)\b', all_text)))
    
    # First markdown header / comments
    headers = []
    for c in md_cells:
        for line in ''.join(c.get('source', [])).splitlines():
            if line.strip().startswith('#'):
                headers.append(line.strip())
    
    nb_details.append({
        'index': idx + 1,
        'name': fname,
        'rel_path': rel_path,
        'size': file_size,
        'total_cells': len(cells),
        'code_cells': len(code_cells),
        'code_lines': total_code_lines,
        'headers': headers[:5],
        'aucs': aucs,
        'models': models,
        'features': features[:15],
        'first_code_snippet': code_sources[:15],
        'output_samples': [l for l in output_texts if any(k in l.lower() for k in ['auc', 'score', 'fold', 'blend', 'wrote'])][:8]
    })

with open(r'D:\gci\truth\experiments\all_63_notebooks_detailed.json', 'w', encoding='utf-8') as f:
    json.dump(nb_details, f, indent=2)

print(f"Successfully processed all {len(nb_details)} notebooks. Saved to all_63_notebooks_detailed.json")

