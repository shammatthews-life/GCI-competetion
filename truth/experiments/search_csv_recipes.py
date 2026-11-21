import re

with open(r'D:\gci\truth\zip_extracted\zip\GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Find submission tables and blend formulas
print("=== SUBMISSIONS IN COMPLETE ARCHIVE ===")
for line in text.splitlines():
    if any(k in line.lower() for k in ['.csv', 'public auc', 'oof auc', 'champion =', 'blend =', 'rank(']):
        print(line[:120])

