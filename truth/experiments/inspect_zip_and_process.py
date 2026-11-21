import zipfile
import os

print("=== INSPECTING ZIP FILE ===")
with zipfile.ZipFile(r'D:\gci\truth\zip.zip', 'r') as z:
    names = z.namelist()
    print(f"Total files in zip.zip: {len(names)}")
    # Find notebooks, md, py, txt
    nbs = [n for n in names if n.endswith('.ipynb')]
    pys = [n for n in names if n.endswith('.py')]
    mds = [n for n in names if n.endswith('.md')]
    txts = [n for n in names if n.endswith('.txt')]
    csvs = [n for n in names if n.endswith('.csv')]
    npys = [n for n in names if n.endswith('.npy')]

    print(f"Notebooks ({len(nbs)}):")
    for n in sorted(nbs):
        print("  ", n)
    print(f"\nPython scripts ({len(pys)}):")
    for n in sorted(pys):
        print("  ", n)
    print(f"\nMarkdown files ({len(mds)}):")
    for n in sorted(mds):
        print("  ", n)
    print(f"\nText files ({len(txts)}):")
    for n in sorted(txts):
        print("  ", n)
    print(f"\nNPY files ({len(npys)}):")
    for n in sorted(npys):
        print("  ", n)

print("\n=== INSPECTING PROCESS.TXT ===")
with open(r'D:\gci\truth\process.txt', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
print(f"Total lines in process.txt: {len(lines)}")
print("First 40 lines:")
for l in lines[:40]:
    print(l.rstrip())

