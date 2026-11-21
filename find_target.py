import os

target = "0.765968"
for root, dirs, files in os.walk(r"d:\gci"):
    if any(ig in root for ig in [".git", "__pycache__", "node_modules", "zip_extracted", ".gemini"]):
        continue
    for f in files:
        if f.endswith(('.py', '.md', '.txt', '.json', '.ipynb')):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                if target in content:
                    print(f"Match: {p}")
                    for line in content.splitlines():
                        if target in line:
                            print("  ", line.strip()[:140])
            except:
                pass

