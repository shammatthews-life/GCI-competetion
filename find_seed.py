import os

targets = ["0.755334", "0.758327", "20261029", "0.77646"]
for root, dirs, files in os.walk(r"d:\gci"):
    if any(ig in root for ig in [".git", "zip_extracted"]): continue
    for f in files:
        if f.endswith(('.py', '.md', '.txt', '.json')):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    c = fp.read()
                for t in targets:
                    if t in c:
                        print(f"Match {t} in {p}")
            except:
                pass

