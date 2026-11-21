from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOTS = [
    Path(r"C:\Users\SHARON\Downloads"),
    Path(r"D:\exp colab"),
    Path(r"D:\gci\truth"),
    Path(r"D:\gci\rules and info"),
]
OUT = Path(r"D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery")
PRUNE = {
    ".git",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "site-packages",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SCREENSHOT_STATUS = {
    "0dd50eec-96bc-481b-a515-85ef102a2321.png": "visually inspected; unrelated wallpaper collage",
    "download.jpg": "visually inspected; unrelated image; hash duplicate of wallpaper.jpg",
    "Jesus Christ Wallpaper For Pc.jpg": "visually inspected; unrelated wallpaper",
    "wallpaper.jpg": "visually inspected; unrelated image; hash duplicate of download.jpg",
}
for _name in (
    "WhatsApp Image 2026-11-03 at 10.58.26 PM (1).jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.26 PM.jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.27 PM (1).jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.27 PM (2).jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.27 PM.jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.28 PM (1).jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.28 PM.jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.29 PM (1).jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.29 PM.jpeg",
    "WhatsApp Image 2026-11-03 at 10.58.30 PM.jpeg",
):
    SCREENSHOT_STATUS[_name] = (
        "visual transcript in SCREENSHOT_FORMULA_TRIAGE.md; no OCR engine; "
        "source corroboration pending"
    )
REPORT_TOKENS = (
    "gci",
    "exp85",
    "homecredit",
    "phase",
    "formula",
    "model",
    "feature",
    "oof",
    "auc",
    "research",
    "experiment",
    "origin",
    "screenshot",
    "whatsapp",
    "notebook",
    "competition",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_rows():
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for current, dirs, names in os.walk(root):
            dirs[:] = [
                name
                for name in dirs
                if name.lower() not in PRUNE
                and not (root == Path(r"D:\gci\truth") and name.lower() == "phase4_notebook_recovery_gpu_discovery")
            ]
            for name in names:
                path = Path(current) / name
                ext = path.suffix.lower()
                lname = name.lower()
                is_notebook = ext == ".ipynb"
                is_project_code = ext == ".py" and (
                    root in (Path(r"D:\gci\truth"), Path(r"D:\exp colab"))
                    or any(token in lname for token in REPORT_TOKENS)
                )
                is_report = ext in {".md", ".txt", ".docx", ".pdf", ".xlsx"} and any(
                    token in lname for token in REPORT_TOKENS
                )
                is_result = ext == ".csv" and any(
                    token in lname
                    for token in (
                        "manifest",
                        "oof",
                        "prediction",
                        "result",
                        "auc",
                        "fold",
                        "feature",
                        "ablation",
                        "score",
                        "benchmark",
                        "compliance",
                        "hypoth",
                        "formula",
                        "diagnostic",
                    )
                )
                is_image = ext in IMAGE_EXTENSIONS and (
                    root in (Path(r"C:\Users\SHARON\Downloads"), Path(r"D:\exp colab"), Path(r"D:\gci\truth"))
                )
                if not any((is_notebook, is_project_code, is_report, is_result, is_image)):
                    continue
                try:
                    stat = path.stat()
                    rows.append(
                        {
                            "path": str(path),
                            "kind": (
                                "notebook"
                                if is_notebook
                                else "code"
                                if is_project_code
                                else "screenshot_or_image"
                                if is_image
                                else "experiment_result"
                                if is_result
                                else "report_or_document"
                            ),
                            "bytes": stat.st_size,
                            "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
                            "sha256": sha256(path),
                        }
                    )
                except (OSError, PermissionError) as exc:
                    rows.append({"path": str(path), "kind": "unreadable", "error": repr(exc)})
    return rows


def notebook_evidence(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    code = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    errors = []
    executions = []
    strings = []
    imports = []
    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        strings.append(source)
        if source:
            imports.extend(
                line.strip()
                for line in source.splitlines()
                if re.match(r"\s*(import |from )", line)
            )
        executions.append(cell.get("execution_count"))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
            text = "".join(output.get("text", []))
            if text:
                strings.append(text)
            for value in output.get("data", {}).values():
                if isinstance(value, list) and all(isinstance(part, str) for part in value):
                    strings.append("".join(value))
    combined = "\n".join(strings)
    patterns = {
        "auc_mentions": re.findall(r"(?i)(?:auc|roc[- ]?auc).{0,80}", combined)[:40],
        "data_paths": re.findall(r"(?i)(?:[A-Z]:\\|/content/)[^\s'\"`;,)]{1,180}", combined)[:80],
        "seed_mentions": re.findall(r"(?i).{0,30}(?:random_state|random_seed|seed)\s*[=:]\s*[^,\n)]{1,50}", combined)[:60],
        "fold_mentions": re.findall(r"(?i).{0,40}(?:StratifiedKFold|KFold|folds?|n_splits).{0,70}", combined)[:80],
        "model_mentions": sorted(
            {
                token
                for token in (
                    "LightGBM",
                    "CatBoost",
                    "XGBoost",
                    "RandomForest",
                    "ExtraTrees",
                    "HistGradientBoosting",
                    "UMAP",
                    "KMeans",
                    "LogisticRegression",
                )
                if token.lower() in combined.lower()
            }
        ),
    }
    return {
        "path": str(path),
        "nbformat": data.get("nbformat"),
        "nbformat_minor": data.get("nbformat_minor"),
        "cell_count": len(cells),
        "code_cells": len(code),
        "markdown_cells": len(markdown),
        "cells_with_execution_count": sum(value is not None for value in executions),
        "saved_output_cells": sum(bool(cell.get("outputs")) for cell in code),
        "saved_output_errors": len(errors),
        "errors": json.dumps(errors, ensure_ascii=True),
        "imports": json.dumps(sorted(set(imports)), ensure_ascii=True),
        "evidence": json.dumps(patterns, ensure_ascii=True),
    }


def run_command(args):
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"command": args, "error": repr(exc)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = file_rows()
    inv = pd.DataFrame(inventory)
    inv.to_csv(OUT / "SOURCE_FILE_INVENTORY.csv", index=False)
    notebooks = [Path(row["path"]) for row in inventory if row.get("kind") == "notebook"]
    notebook_rows = []
    for path in notebooks:
        try:
            notebook_rows.append(notebook_evidence(path))
        except Exception as exc:
            notebook_rows.append(
                {"path": str(path), "inspection_error": repr(exc)}
            )
    notebook_df = pd.DataFrame(notebook_rows)
    notebook_df.to_csv(OUT / "NOTEBOOK_INVENTORY.csv", index=False)
    image_df = inv.loc[inv.kind == "screenshot_or_image"].copy() if not inv.empty else pd.DataFrame()
    if not image_df.empty:
        image_df["claim_status"] = image_df["path"].map(
            lambda value: SCREENSHOT_STATUS.get(
                Path(value).name, "inventory only; not visually reviewed"
            )
        )
    image_df.to_csv(OUT / "SCREENSHOT_INVENTORY.csv", index=False)

    nvidia = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    smi_list = run_command(["nvidia-smi", "-L"])
    python_env = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    package_info = {}
    for name in ["lightgbm", "xgboost", "catboost", "scikit-learn", "torch", "cupy"]:
        try:
            from importlib.metadata import version

            package_info[name] = version(name)
        except Exception:
            package_info[name] = "NOT_INSTALLED"
    gpu_details = {
        **python_env,
        "packages": package_info,
        "nvidia_smi_query": nvidia,
        "nvidia_smi_list": smi_list,
        "nvcc_path": __import__("shutil").which("nvcc"),
    }
    try:
        import torch

        gpu_details["torch_cuda_available"] = torch.cuda.is_available()
        gpu_details["torch_cuda_build"] = torch.version.cuda
        gpu_details["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        gpu_details["torch_error"] = repr(exc)
    try:
        import xgboost

        gpu_details["xgboost_build_info"] = xgboost.build_info()
    except Exception as exc:
        gpu_details["xgboost_error"] = repr(exc)
    try:
        import lightgbm

        gpu_details["lightgbm_version"] = lightgbm.__version__
        gpu_details["lightgbm_library_error_state"] = str(lightgbm.basic._LIB.LGBM_GetLastError())
    except Exception as exc:
        gpu_details["lightgbm_error"] = repr(exc)
    (OUT / "GPU_ENVIRONMENT_AUDIT.json").write_text(
        json.dumps(gpu_details, indent=2, default=str), encoding="utf-8"
    )

    image_count = len(image_df)
    report = f"""# Phase 4 Stage 1 inventory and initial GPU detection

## Status

Completed a metadata/source inventory pass and static review of priority notebooks and screenshots; no notebook was executed and no model was trained. The remaining notebook families, full feature-reuse matrix, and data-dictionary/provenance checks are not complete.

## Inventory coverage

Scanned these local roots: {', '.join(str(root) for root in ROOTS)}. Excluded generated caches and the current Phase 4 output folder from the recursive input inventory.

- Notebook files discovered: {len(notebooks)}.
- Notebook parse failures: {int(notebook_df.get('inspection_error', pd.Series(dtype=str)).notna().sum()) if not notebook_df.empty else 0}.
- Screenshot/image files discovered: {image_count}.
- Other source/report/result files inventoried and hashed: {max(0, len(inv) - len(notebooks) - image_count)}.
- Notebook inventory includes cell counts, execution counts, saved outputs/errors, imports, paths, seed/fold mentions, model names and AUC text matches.
- Ten formula screenshots were visually transcribed in `SCREENSHOT_FORMULA_TRIAGE.md`; `tesseract` is not installed, so this is not automated OCR. Four other images were classified as unrelated/duplicate. Formula semantics and score claims still require source/data corroboration.
- A related local source candidate, `C:\\Users\\SHARON\\Downloads\\solution_v5_gci_research.py`, was discovered (140,761 bytes). It has a 20-entry model registry and extensive CV/blending code. Its authorship/identity as Claude's exact seven-model source is unverified; its OOF vectors and supporting run artifacts were not found in the scanned inventory.

## Initial GPU finding (detection only)

- Device/driver and framework capability metadata are recorded in `GPU_ENVIRONMENT_AUDIT.json`.
- A visible NVIDIA device does not prove a Python estimator can train on it. PyTorch CUDA availability is only one probe and does not determine LightGBM/CatBoost/XGBoost capability.
- No GPU model benchmark was run in this stage. No CPU fallback or full training was started.
- Complete source review and a small representative capability fit are still gates before the requested full model reproduction.

## Next actions

1. Inspect remaining v18/v19 and research notebooks/source cell-by-cell; complete the feature/source reuse matrix.
2. Triage hashes and provenance for caches and prediction files before trusting historical metrics.
3. Perform model-specific GPU capability tests on a small, bounded representative dataset and record real device use.
4. Verify formula semantics against the official dictionary and establish permitted fold/data scope before any Phase 4 feature/model experiment.
"""
    (OUT / "STAGE1_INVENTORY_REPORT.md").write_text(report, encoding="utf-8")
    print(f"inventory files={len(inv)} notebooks={len(notebooks)} images={image_count}")
    print(f"output_directory={OUT}")
    print("gpu_query=" + json.dumps(nvidia, ensure_ascii=True))
    print("package_info=" + json.dumps(package_info, ensure_ascii=True))
    print("xgboost_cuda=" + str(gpu_details.get("xgboost_build_info", {}).get("USE_CUDA", "unknown")))


if __name__ == "__main__":
    main()
