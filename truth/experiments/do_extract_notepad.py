import json
import os

out_dir = r'D:\gci\truth\experiments\extracted_from_notepad'
os.makedirs(out_dir, exist_ok=True)

with open(r'D:\gci\truth\notepade.txt', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Notebook 1: lines 0 to 426
nb1_text = "{" + "".join(lines[1:427])
try:
    nb1_json = json.loads(nb1_text)
    nb1_path = os.path.join(out_dir, 'notepad_nb1_realmlp_tabm_0.77283.ipynb')
    with open(nb1_path, 'w', encoding='utf-8') as fp:
        json.dump(nb1_json, fp, indent=2)
    print(f"Extracted Notebook 1 (Score: 0.77283) -> {nb1_path} ({len(nb1_json.get('cells', []))} cells)")
except Exception as e:
    print("Failed to parse NB1:", e)

# Notebook 2: lines 430 to 838
nb2_text = "".join(lines[430:839])
try:
    nb2_json = json.loads(nb2_text)
    nb2_path = os.path.join(out_dir, 'notepad_nb2_qfingerprint_0.777.ipynb')
    with open(nb2_path, 'w', encoding='utf-8') as fp:
        json.dump(nb2_json, fp, indent=2)
    print(f"Extracted Notebook 2 (Score: 0.777) -> {nb2_path} ({len(nb2_json.get('cells', []))} cells)")
except Exception as e:
    print("Failed to parse NB2:", e)

# Notebook 3: line 841
nb3_text = lines[841].strip()
try:
    nb3_json = json.loads(nb3_text)
    nb3_path = os.path.join(out_dir, 'notepad_nb3_final_reproduction_0.77865.ipynb')
    with open(nb3_path, 'w', encoding='utf-8') as fp:
        json.dump(nb3_json, fp, indent=2)
    print(f"Extracted Notebook 3 (Score: 0.77865) -> {nb3_path} ({len(nb3_json.get('cells', []))} cells)")
except Exception as e:
    print("Failed to parse NB3:", e)

# Notebook 4: line 842
nb4_text = lines[842].strip()
try:
    nb4_json = json.loads(nb4_text)
    nb4_path = os.path.join(out_dir, 'notepad_nb4_v12_product_aware_AB.ipynb')
    with open(nb4_path, 'w', encoding='utf-8') as fp:
        json.dump(nb4_json, fp, indent=2)
    print(f"Extracted Notebook 4 -> {nb4_path} ({len(nb4_json.get('cells', []))} cells)")
except Exception as e:
    print("Failed to parse NB4:", e)

