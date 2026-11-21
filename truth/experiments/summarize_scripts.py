import os
import glob
import json
import re

base_zip = r'D:\gci\truth\zip_extracted'
truth_dir = r'D:\gci\truth'

def summarize_file(filepath):
    size = os.path.getsize(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.py':
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.splitlines()
        # Find imports, model configs, auc mentions
        models = re.findall(r'(lgb|lightgbm|xgb|xgboost|catboost|CatBoost|ExtraTrees|HistGradientBoosting|RandomForest|LogisticRegression)', content, re.IGNORECASE)
        aucs = re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', content)
        features = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(|([A-Z0-9_]{3,}\s*=)', content)
        return {
            'type': 'python',
            'lines': len(lines),
            'size': size,
            'models': list(set(models))[:10],
            'aucs': list(set(aucs))[:10],
            'features': [f[0] or f[1] for f in features if f[0] or f[1]][:15],
            'first_20': lines[:20],
            'last_20': lines[-20:]
        }
    elif ext == '.ipynb':
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            try:
                nb = json.load(f)
            except Exception as e:
                return {'type': 'ipynb_error', 'error': str(e), 'size': size}
        cells = nb.get('cells', [])
        code_cells = [c for c in cells if c.get('cell_type') == 'code']
        md_cells = [c for c in cells if c.get('cell_type') == 'markdown']
        total_code_lines = sum(len(''.join(c.get('source', [])).splitlines()) for c in code_cells)
        # Scan for models and AUCs in code and outputs
        all_text = ''
        for c in cells:
            all_text += ''.join(c.get('source', [])) + '\n'
            for out in c.get('outputs', []):
                if 'text' in out:
                    all_text += ''.join(out['text']) + '\n'
                elif 'data' in out and 'text/plain' in out['data']:
                    all_text += ''.join(out['data']['text/plain']) + '\n'
        models = re.findall(r'(lgb|lightgbm|xgb|xgboost|catboost|CatBoost|ExtraTrees|HistGradientBoosting|RandomForest|RealMLP|TabM)', all_text, re.IGNORECASE)
        aucs = re.findall(r'(0\.[78]\d{3,5}|AUC[:=\s]+0\.[78]\d+)', all_text)
        first_md = ''.join(md_cells[0].get('source', [])) if md_cells else ''
        return {
            'type': 'ipynb',
            'cells': len(cells),
            'code_cells': len(code_cells),
            'code_lines': total_code_lines,
            'size': size,
            'models': list(set(models))[:10],
            'aucs': list(set(aucs))[:10],
            'first_md': first_md[:400]
        }
    elif ext in ['.md', '.txt']:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.splitlines()
        aucs = re.findall(r'(0\.[78]\d{3,5}|AUC[:=\s]+0\.[78]\d+)', content)
        return {
            'type': ext[1:],
            'lines': len(lines),
            'size': size,
            'aucs': list(set(aucs))[:15],
            'first_20': lines[:20],
            'last_20': lines[-20:]
        }
    return {'type': ext, 'size': size}

# Analyze all python scripts
py_files = glob.glob(os.path.join(base_zip, '**/*.py'), recursive=True)
print(f"Analyzing {len(py_files)} Python scripts...")
py_summaries = {}
for p in py_files:
    rel = os.path.relpath(p, base_zip)
    py_summaries[rel] = summarize_file(p)

with open(r'D:\gci\truth\experiments\py_summaries.json', 'w', encoding='utf-8') as f:
    json.dump(py_summaries, f, indent=2)

print("Saved py_summaries.json")

