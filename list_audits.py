import os

for root, dirs, files in os.walk(r"d:\gci"):
    if any(ig in root for ig in [".git", "zip_extracted", ".gemini"]): continue
    for f in files:
        if "AUDIT" in f.upper() or "FORENSIC" in f.upper():
            print(os.path.join(root, f))

