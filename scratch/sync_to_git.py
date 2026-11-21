import os
import shutil

src_root = r"D:\gci"
dst_root = r"D:\gci\_gci_push_tmp"

# Explicit exclusions requested by user
EXCLUDED_FOLDERS = {
    "friend code",
    "past_85",
    "pre info",
    "rules and info",
    ".venv",
    "__pycache__",
    ".git",
    "_gci_push_tmp",
    "inpuy",
    "input"
}

MAX_FILE_BYTES = 90 * 1024 * 1024  # 90 MB cutoff to protect against GitHub 100MB limit

print("=== Starting Git Sync Preparation ===")

# 1. Setup .gitignore in dst_root
gitignore_path = os.path.join(dst_root, ".gitignore")
gitignore_content = """# Data & large inputs
inpuy/
input/

# Python cache
__pycache__/
*.pyc

# User requested exclusions
friend code/
past_85/
pre info/
rules and info/

# Large binaries and archives exceeding GitHub limits
*.zip
*.rar
*.mp4
*.avi
*.mkv
*.feather
*.pkl

# Massive debug dumps (>90MB)
truth/zip/
truth/zip_extracted/
truth/experiments/phase3c_model_intelligence/OOF_ERROR_ANALYSIS.csv
truth/experiments/phase9_product_mechanism_and_catboost/runs/phase9_20261104T181500LOCAL/OOF_PREDICTIONS.csv
truth/experiments/phase21_focused_validation/runs/phase21_checkpointed_20261106/PHASE21_OOF_PREDICTIONS.csv
truth/experiments/phase13_matched_baseline/runs/phase13_20261105T075946Z/OOF_PREDICTIONS.csv
truth/experiments/phase12_label_free_product_representation/runs/phase12_20261104T173534Z/OOF_PREDICTIONS.csv
truth/experiments/phase3c_model_intelligence/PHASE3C_ADVANCED_OOF_PREDICTIONS.csv
"""

with open(gitignore_path, "w", encoding="utf-8") as f:
    f.write(gitignore_content.strip() + "\n")
print("Updated .gitignore successfully.")

# 2. Copy Root Files
print("\n--- Copying Root Files ---")
copied_root = 0
for f in os.listdir(src_root):
    src_f = os.path.join(src_root, f)
    if os.path.isfile(src_f):
        if f.startswith(".git") or f.endswith(".pyc"):
            continue
        sz = os.path.getsize(src_f)
        if sz >= MAX_FILE_BYTES:
            print(f"Skipping root file >90MB: {f} ({sz/1024/1024:.2f} MB)")
            continue
        dst_f = os.path.join(dst_root, f)
        shutil.copy2(src_f, dst_f)
        copied_root += 1
print(f"Copied/Updated {copied_root} root files.")

# Helper to copy directory tree safely
def copy_tree_safe(src_dir, dst_dir):
    copied = 0
    skipped = 0
    for root, dirs, files in os.walk(src_dir):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"} and not d.startswith(".")]
        rel_dir = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel_dir) if rel_dir != "." else dst_dir
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            if f.endswith(".pyc"):
                continue
            src_fp = os.path.join(root, f)
            sz = os.path.getsize(src_fp)
            if sz >= MAX_FILE_BYTES or f.endswith(".zip") or f.endswith(".rar") or f.endswith(".mp4"):
                skipped += 1
                continue
            dst_fp = os.path.join(target_dir, f)
            shutil.copy2(src_fp, dst_fp)
            copied += 1
    return copied, skipped

# 3. Copy `truth` folder
print("\n--- Copying truth folder ---")
truth_copied, truth_skipped = copy_tree_safe(os.path.join(src_root, "truth"), os.path.join(dst_root, "truth"))
print(f"truth: Copied {truth_copied} files, skipped {truth_skipped} large/binary files.")

# 4. Copy `5 independent`
print("\n--- Copying 5 independent ---")
ind_copied, ind_skipped = copy_tree_safe(os.path.join(src_root, "5 independent"), os.path.join(dst_root, "5 independent"))
print(f"5 independent: Copied {ind_copied} files.")

# 5. Copy `codex`
print("\n--- Copying codex ---")
codex_copied, codex_skipped = copy_tree_safe(os.path.join(src_root, "codex"), os.path.join(dst_root, "codex"))
print(f"codex: Copied {codex_copied} files, skipped {codex_skipped} large files.")

# 6. Copy `name5_dependent_code`
print("\n--- Copying name5_dependent_code ---")
n5_copied, n5_skipped = copy_tree_safe(os.path.join(src_root, "name5_dependent_code"), os.path.join(dst_root, "name5_dependent_code"))
print(f"name5_dependent_code: Copied {n5_copied} files.")

# 7. Copy `scratch`
print("\n--- Copying scratch ---")
scratch_copied, scratch_skipped = copy_tree_safe(os.path.join(src_root, "scratch"), os.path.join(dst_root, "scratch"))
print(f"scratch: Copied {scratch_copied} files.")

# 8. Create and populate `colab` from `pre info`
print("\n--- Creating and Populating colab folder from pre info ---")
pre_info_dir = os.path.join(src_root, "pre info")
dst_colab = os.path.join(dst_root, "colab")
os.makedirs(dst_colab, exist_ok=True)

colab_copied = 0
for root, dirs, files in os.walk(pre_info_dir):
    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"} and not d.startswith(".")]
    rel = os.path.relpath(root, pre_info_dir)
    target_subdir = os.path.join(dst_colab, rel) if rel != "." else dst_colab
    os.makedirs(target_subdir, exist_ok=True)
    for f in files:
        src_fp = os.path.join(root, f)
        # Check if it is a code file or colab artifact
        ext = os.path.splitext(f)[1].lower()
        if ext in [".ipynb", ".py"] or "colab" in f.lower() or ext in [".csv", ".json", ".md"]:
            sz = os.path.getsize(src_fp)
            if sz < MAX_FILE_BYTES and not f.endswith(".zip") and not f.endswith(".rar"):
                dst_fp = os.path.join(target_subdir, f)
                shutil.copy2(src_fp, dst_fp)
                colab_copied += 1
print(f"colab: Copied {colab_copied} colab codes and associated notebooks/scripts.")

# 9. Verify Excluded Folders are NOT in dst_root
print("\n--- Verifying Forbidden Exclusions ---")
for bad in ["friend code", "past_85", "pre info", "rules and info"]:
    bad_path = os.path.join(dst_root, bad)
    if os.path.exists(bad_path):
        print(f"WARNING: Forbidden folder {bad} exists in dst_root! Removing...")
        shutil.rmtree(bad_path)
    else:
        print(f"CONFIRMED: {bad} is NOT in git repo.")

print("\n=== Sync Preparation Finished Successfully ===")
