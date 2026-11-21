import os
import re
import json

base_extracted = r'D:\gci\truth\experiments\extracted_from_notepad'

def audit_file_dependencies(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # 1. External dataset checks
    has_kaggle_ext = any(k in content for k in ['bureau.csv', 'previous_application.csv', 'installments_payments.csv', 'POS_CASH_balance.csv', 'credit_card_balance.csv', 'bureau_balance.csv'])
    has_external_labels = any(k in content for k in ['application_train.csv', 'application_test.csv'])
    
    # 2. Cache / pre-computed dependencies
    cache_matches = re.findall(r'(\bgci_cache[a-zA-Z0-9_]*\b|\.npy\b|\.npz\b)', content)
    has_cache_dependency = bool(cache_matches)
    
    # 3. Required packages
    pkgs = []
    for pkg in ['lightgbm', 'xgboost', 'catboost', 'torch', 'pytabkit', 'sklearn', 'scipy', 'pandas', 'numpy']:
        if pkg in content.lower():
            pkgs.append(pkg)
            
    return {
        'has_kaggle_ext': has_kaggle_ext,
        'has_external_labels': has_external_labels,
        'has_cache_dependency': has_cache_dependency,
        'cache_matches': list(set(cache_matches))[:5],
        'packages': pkgs
    }

print("=== AUDITING EXTRACTED NOTEPAD NOTEBOOKS ===")
for nb_name in os.listdir(base_extracted):
    nb_path = os.path.join(base_extracted, nb_name)
    res = audit_file_dependencies(nb_path)
    print(f"\n{nb_name}:")
    print(f"  Prohibited external data: {res['has_kaggle_ext'] or res['has_external_labels']}")
    print(f"  Pre-computed cache dependency: {res['has_cache_dependency']} -> {res['cache_matches']}")
    print(f"  Required packages: {res['packages']}")

