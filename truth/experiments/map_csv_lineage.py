import os
import glob
import re
import json

base_zip = r'D:\gci\truth\zip_extracted'
csv_files = sorted(glob.glob(os.path.join(base_zip, '**/*.csv'), recursive=True))

print(f"Total CSVs: {len(csv_files)}")

# Collect all code and text content
all_code_files = glob.glob(os.path.join(base_zip, '**/*.py'), recursive=True) + glob.glob(os.path.join(base_zip, '**/*.ipynb'), recursive=True)
all_doc_files = glob.glob(os.path.join(base_zip, '**/*.md'), recursive=True) + glob.glob(os.path.join(base_zip, '**/*.txt'), recursive=True)

code_corpus = {}
for p in all_code_files:
    fname = os.path.basename(p)
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        code_corpus[fname] = f.read()

doc_corpus = {}
for p in all_doc_files:
    fname = os.path.basename(p)
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        doc_corpus[fname] = f.read()

# Map each CSV to its creator script/notebook, formula, and score
csv_lineage = {}
for c in csv_files:
    c_name = os.path.basename(c)
    size = os.path.getsize(c)
    
    # Search in code
    creators = []
    blend_formulas = []
    for cf_name, content in code_corpus.items():
        if c_name in content or c_name.replace('.csv', '') in content:
            creators.append(cf_name)
            # Find lines around the CSV mention
            for line in content.splitlines():
                if c_name in line or 'to_csv' in line and any(w in line for w in ['blend', 'sub', 'rank', 'target', 'pred', 'out']):
                    blend_formulas.append(line.strip()[:150])
    
    # Search in docs
    doc_mentions = []
    scores_mentioned = []
    for df_name, content in doc_corpus.items():
        if c_name in content:
            doc_mentions.append(df_name)
            # Find nearby scores
            for line in content.splitlines():
                if c_name in line:
                    aucs = re.findall(r'(0\.[78]\d{2,5})', line)
                    if aucs:
                        scores_mentioned.extend(aucs)

    csv_lineage[c_name] = {
        'size': size,
        'creators': list(set(creators))[:5],
        'blend_formulas': list(set(blend_formulas))[:5],
        'doc_mentions': list(set(doc_mentions))[:5],
        'scores': list(set(scores_mentioned))[:5]
    }

print("Sample lineage for top 10 CSVs:")
for k in list(csv_lineage.keys())[:10]:
    v = csv_lineage[k]
    print(f"{k} -> Creators: {v['creators']} | Scores: {v['scores']}")

with open(r'D:\gci\truth\experiments\csv_lineage.json', 'w', encoding='utf-8') as f:
    json.dump(csv_lineage, f, indent=2)

print("Saved csv_lineage.json")

