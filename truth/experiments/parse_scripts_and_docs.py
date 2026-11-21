import os
import glob
import json
import re

base_zip = r'D:\gci\truth\zip_extracted'

def parse_scripts():
    pys = sorted(glob.glob(os.path.join(base_zip, '**/*.py'), recursive=True))
    py_details = []
    for idx, p_path in enumerate(pys):
        fname = os.path.basename(p_path)
        rel_path = os.path.relpath(p_path, base_zip)
        size = os.path.getsize(p_path)
        with open(p_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.splitlines()
        aucs = list(dict.fromkeys(re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', content)))[:10]
        models = list(set(re.findall(r'\b(lgb|lightgbm|xgb|xgboost|catboost|CatBoost|ExtraTrees|HistGradientBoosting|RandomForest|LogisticRegression|RealMLP|TabM)\b', content, re.IGNORECASE)))
        features = list(set(re.findall(r'\b(EXT_SOURCE_\d|EXT_PRODUCT|EXT_HARMONIC|CREDIT_TERM|GOODS_TO_CREDIT|ANNUITY_TO_INCOME|CREDIT_TO_INCOME|DE_SENT|Q0|Q_0|Q3|Q4|Q5|P3|P4|PF4|TE_[a-zA-Z0-9_]+|VIEW_[A-Z]+)\b', content)))
        
        py_details.append({
            'index': idx + 1,
            'name': fname,
            'rel_path': rel_path,
            'size': size,
            'lines': len(lines),
            'aucs': aucs,
            'models': models,
            'features': features[:15],
            'first_25_lines': lines[:25]
        })
    return py_details

def parse_md_and_txt():
    mds = sorted(glob.glob(os.path.join(base_zip, '**/*.md'), recursive=True))
    txts = sorted(glob.glob(os.path.join(base_zip, '**/*.txt'), recursive=True))
    doc_details = []
    for idx, d_path in enumerate(mds + txts):
        fname = os.path.basename(d_path)
        rel_path = os.path.relpath(d_path, base_zip)
        size = os.path.getsize(d_path)
        ext = os.path.splitext(d_path)[1].lower()
        with open(d_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        lines = content.splitlines()
        aucs = list(dict.fromkeys(re.findall(r'(0\.[78]\d{2,5}|AUC[:=\s]+0\.[78]\d+)', content)))[:10]
        headers = [l.strip() for l in lines if l.strip().startswith('#') or l.strip().startswith('==')][:10]
        
        doc_details.append({
            'index': idx + 1,
            'name': fname,
            'rel_path': rel_path,
            'ext': ext,
            'size': size,
            'lines': len(lines),
            'aucs': aucs,
            'headers': headers,
            'first_25_lines': [l for l in lines[:30] if l.strip()][:25]
        })
    return doc_details

scripts = parse_scripts()
docs = parse_md_and_txt()

with open(r'D:\gci\truth\experiments\all_scripts_detailed.json', 'w', encoding='utf-8') as f:
    json.dump(scripts, f, indent=2)

with open(r'D:\gci\truth\experiments\all_docs_detailed.json', 'w', encoding='utf-8') as f:
    json.dump(docs, f, indent=2)

print(f"Parsed {len(scripts)} scripts and {len(docs)} documents.")

