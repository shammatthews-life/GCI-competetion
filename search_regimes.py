import re

keywords = ["regime", "split", "v16", "v17", "contract", "revolving", "ext_nobs", "routing", "specialist"]

def search_file(path, max_hits=25):
    print("=" * 80)
    print(f"SEARCHING: {path}")
    hits = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            low = line.lower()
            if any(k in low for k in ["regime-aware", "where v16", "where v17", "v16 vs v17", "contract_type", "ext_nobs"]):
                print(f"Line {i+1}: {line.strip()[:140]}")
                hits += 1
                if hits >= max_hits:
                    break

search_file(r"D:\gci\truth\past.md")
search_file(r"D:\gci\truth\GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md")

