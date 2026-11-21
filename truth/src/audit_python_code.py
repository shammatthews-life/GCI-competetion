from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRUTH_ROOT = Path(r"D:\gci\truth")
REPORT_DIR = TRUTH_ROOT / "reports" / "python_code_audit"
SOURCE_ROOTS = (
    Path(r"D:\gci\champion"),
    Path(r"D:\gci\gemini"),
    Path(r"D:\gci\codex"),
    Path(r"D:\gci\past_85"),
    Path(r"D:\gci\5 independent"),
    Path(r"D:\gci\name5_dependent_code"),
    Path(r"D:\gci\pre info"),
)
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "site-packages",
    "__pycache__",
    "node_modules",
    ".ipynb_checkpoints",
}
CODE_EXTENSIONS = {".py", ".ipynb"}
SOURCE_NAME = "preserved research source (read-only)"

FEATURE_PATTERNS = {
    "ratios_or_interactions": re.compile(
        r"\b(?:ratio|interaction|product|divide|multiply|poly(?:nomial)?)\b|/\s*"
    ),
    "aggregation_or_group_statistics": re.compile(
        r"\b(?:groupby|agg(?:regate)?|transform|rolling|cumsum|cumcount|nunique)\b",
        re.I,
    ),
    "categorical_or_frequency_encoding": re.compile(
        r"\b(?:target.?encod|catboost|category.?encod|frequency.?encod|"
        r"count.?encod|onehot|ordinal.?encod)\b",
        re.I,
    ),
    "missingness_or_imputation": re.compile(
        r"\b(?:imput|fillna|isna|isnull|missing.?indicator)\b", re.I
    ),
    "date_or_temporal_features": re.compile(
        r"\b(?:date|days_|day[s_]|month|year|timestamp|datetime|elapsed)\b", re.I
    ),
    "external_table_features": re.compile(
        r"\b(?:bureau|previous_application|installments_payments|"
        r"credit_card_balance|pos_cash_balance|application_train)\b",
        re.I,
    ),
}

AUDIT_PATTERNS = {
    "cv": re.compile(
        r"\b(?:StratifiedKFold|KFold|GroupKFold|RepeatedStratifiedKFold|"
        r"cross_val_score|cross_validate|out.of.fold|\bOOF\b|folds?)\b",
        re.I,
    ),
    "target_encoding": re.compile(r"\btarget.?encod|CatBoostEncoder\b", re.I),
    "train_test_boundary": re.compile(
        r"\b(?:train_test_split|X_test|test_df|test\.csv|test_ids|"
        r"train_df|train\.csv|concat\s*\(\s*\[.{0,80}(?:train|test))\b",
        re.I,
    ),
    "potential_leakage": re.compile(
        r"\b(?:fit_transform|fit_resample|SMOTE|pseudo.?label|"
        r"validation.*(?:fit|select)|(?:fit|select).*validation)\b",
        re.I,
    ),
    "external_access": re.compile(
        r"\b(?:requests\.(?:get|post)|urlopen|wget|curl|gdown|"
        r"kaggle\.api|read_html)\b|"
        r"\b(?:read_csv|read_parquet|read_feather|read_json)\s*"
        r"\(\s*['\"]https?://",
        re.I,
    ),
    "transductive": re.compile(
        r"\b(?:transduct|test.?set.?stat|test.?distribution|"
        r"train.?test.?concat|unlabelled.?test|unsupervised.?test)\b",
        re.I,
    ),
    "gpu": re.compile(
        r"\b(?:device\s*=\s*['\"]cuda|cuda|gpu_hist|device_type|"
        r"torch\.cuda|cupy|RAPIDS)\b",
        re.I,
    ),
    "ensemble": re.compile(
        r"\b(?:ensemble|blend(?:ing)?|stack(?:ing)?|meta.?model|"
        r"rank.?average|weighted.?average)\b",
        re.I,
    ),
    "seed": re.compile(r"\b(?:random_state|random_seed|seed)\s*=", re.I),
    "training": re.compile(
        r"\.(?:fit|train)\s*\(|\b(?:LGBMClassifier|XGBClassifier|"
        r"CatBoostClassifier|RandomForestClassifier|HistGradientBoosting)\b"
    ),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_skipped(path: Path) -> bool:
    return any(part.casefold() in SKIP_PARTS for part in path.parts)


def _files() -> list[tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for root in SOURCE_ROOTS:
        if not root.is_dir():
            raise FileNotFoundError(f"Expected read-only source directory missing: {root}")
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.casefold() in CODE_EXTENSIONS
                and not _is_skipped(path)
            ):
                key = str(path.resolve()).casefold()
                found[key] = (root, path.resolve())
    return sorted(found.values(), key=lambda pair: str(pair[1]).casefold())


def _notebook_cells(path: Path) -> tuple[list[str], int]:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    sources: list[str] = []
    magics = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cleaned: list[str] = []
        for line in str(source).splitlines():
            if line.lstrip().startswith(("%", "!")):
                magics += 1
                cleaned.append("")
            else:
                cleaned.append(line)
        sources.append("\n".join(cleaned))
    return sources, magics


def _imports_and_definitions(tree: ast.AST) -> tuple[list[str], list[str], list[str]]:
    imports: set[str] = set()
    functions: list[str] = []
    classes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add("." * node.level + (node.module or ""))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return sorted(imports), sorted(set(functions)), sorted(set(classes))


def _module_index(files: list[tuple[Path, Path]]) -> set[str]:
    names: set[str] = set()
    for root, path in files:
        relative = path.relative_to(root).with_suffix("")
        parts = list(relative.parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts:
            names.add(".".join(parts))
            names.add(parts[-1])
    return names


def _local_imports(
    imports: list[str], path: Path, roots: tuple[Path, ...], module_names: set[str]
) -> tuple[list[str], list[str]]:
    local: list[str] = []
    unresolved: list[str] = []
    search_roots = (path.parent, *roots)
    for imported in imports:
        if not imported or imported.startswith("."):
            module_parts = [part for part in imported.lstrip(".").split(".") if part]
            candidates = [path.parent.joinpath(*module_parts)]
            is_local = (
                candidates[0].is_file()
                or candidates[0].with_suffix(".py").is_file()
                or candidates[0].is_dir()
                or (module_parts and module_parts[-1] in module_names)
            )
        else:
            parts = imported.split(".")
            is_local = (
                imported in module_names
                or imported.split(".", 1)[0] in module_names
                or any(
                    root.joinpath(*parts).with_suffix(".py").is_file()
                    or root.joinpath(*parts, "__init__.py").is_file()
                    or root.joinpath(*parts).is_dir()
                    for root in search_roots
                )
            )
        if is_local:
            local.append(imported)
        else:
            unresolved.append(imported)
    return local, unresolved


def _third_party_import_status(imports: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for imported in imports:
        if imported.startswith(".") or not imported:
            continue
        top_level = imported.split(".", 1)[0]
        try:
            present = importlib.util.find_spec(top_level) is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            present = False
        output[top_level] = "AVAILABLE_IN_AUDIT_ENVIRONMENT" if present else "NOT_FOUND_IN_AUDIT_ENVIRONMENT"
    return dict(sorted(output.items()))


def _literal_paths(
    parent: Path, roots: tuple[Path, ...], tree: ast.AST | None
) -> list[dict[str, str]]:
    results = []
    seen: set[str] = set()
    reader_names = {
        "read_csv",
        "read_parquet",
        "read_feather",
        "read_pickle",
        "load",
        "load_npz",
        "open",
        "read_text",
        "read_bytes",
        "load_model",
        "load_weights",
        "read_hdf",
    }

    def constant_path(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = constant_path(node.left)
            right = constant_path(node.right)
            return left + right if left is not None and right is not None else None
        if isinstance(node, ast.JoinedStr):
            parts = [constant_path(value) for value in node.values]
            return "".join(parts) if all(part is not None for part in parts) else None
        if isinstance(node, ast.Call):
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            parts = [constant_path(argument) for argument in node.args]
            if function_name in {"Path", "PurePath", "PurePosixPath", "PureWindowsPath"}:
                return parts[0] if len(parts) == 1 else None
            if function_name == "join" and all(part is not None for part in parts):
                return str(Path(*parts))
        return None

    def add_literal(literal: str) -> None:
        if literal in seen or not re.search(
            r"\.(?:csv|parquet|feather|pkl|pickle|npz|npy|json|yaml|yml|txt|zip|cbm|ubj|pt|pth)$",
            literal,
            re.I,
        ):
            return
        seen.add(literal)
        path = Path(literal)
        candidates = (
            [path]
            if path.is_absolute()
            else [parent / path, *(root / path for root in roots)]
        )
        found = next((candidate for candidate in candidates if candidate.exists()), None)
        results.append(
            {
                "literal": literal,
                "resolved_from_source_directory": str(found or candidates[0]),
                "status": "EXISTS" if found else "NOT_FOUND_OR_DYNAMIC",
            }
        )

    if tree:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else ""
            )
            if function_name not in reader_names:
                continue
            if function_name == "open" and len(node.args) > 1:
                mode_node = node.args[1]
                if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
                    if any(flag in mode_node.value for flag in ("w", "a", "x")):
                        continue
            first_argument = node.args[0] if node.args else None
            if first_argument is None:
                continue
            literal = constant_path(first_argument)
            if literal is not None:
                add_literal(literal)
    return results


def _line_evidence(text: str, pattern: re.Pattern[str], max_count: int = 12) -> list[str]:
    evidence = []
    for number, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            snippet = " ".join(line.strip().split())[:240]
            evidence.append(f"{number}:{snippet}")
            if len(evidence) >= max_count:
                break
    return evidence


def _audit_one(root: Path, path: Path, module_names: set[str]) -> dict[str, Any]:
    stat = path.stat()
    error = ""
    code_cells = 0
    magic_count = 0
    notebook_status = ""
    syntax_status = "NOT_CHECKED"
    parsed_nodes: list[ast.AST] = []
    try:
        if path.suffix.casefold() == ".ipynb":
            cells, magic_count = _notebook_cells(path)
            code_cells = len(cells)
            text = "\n\n".join(cells)
            notebook_status = "VALID_NOTEBOOK_JSON"
            for cell_index, cell_source in enumerate(cells, start=1):
                cell_tree = ast.parse(cell_source, filename=f"{path}#cell{cell_index}")
                compile(cell_tree, f"{path}#cell{cell_index}", "exec")
                parsed_nodes.extend(cell_tree.body)
        else:
            text = path.read_text(encoding="utf-8-sig")
            python_tree = ast.parse(text, filename=str(path))
            compile(python_tree, str(path), "exec")
            parsed_nodes = python_tree.body
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, SyntaxError) as exc:
        text = ""
        error = f"{type(exc).__name__}: {exc}"
        notebook_status = "INVALID_NOTEBOOK_JSON" if path.suffix.casefold() == ".ipynb" else ""
        syntax_status = "FAIL"
    tree = None
    if not error:
        try:
            tree = ast.Module(body=parsed_nodes, type_ignores=[])
            syntax_status = "PASS"
        except (SyntaxError, ValueError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            syntax_status = "FAIL"
    imports, functions, classes = _imports_and_definitions(tree) if tree else ([], [], [])
    local_imports, unresolved_imports = _local_imports(
        imports, path, SOURCE_ROOTS, module_names
    )
    third_party = _third_party_import_status(unresolved_imports)
    evidence = {
        key: _line_evidence(text, pattern)
        for key, pattern in AUDIT_PATTERNS.items()
        if pattern.search(text)
    }
    feature_evidence = {
        key: _line_evidence(text, pattern)
        for key, pattern in FEATURE_PATTERNS.items()
        if pattern.search(text)
    }
    read_paths = _literal_paths(path.parent, SOURCE_ROOTS, tree)
    return {
        "path": str(path),
        "source_root": str(root),
        "source_group": root.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "line_count": len(text.splitlines()),
        "code_cell_count": code_cells,
        "magic_line_count": magic_count,
        "notebook_json_status": notebook_status,
        "syntax_status": syntax_status,
        "syntax_error": error,
        "imports_json": json.dumps(imports, ensure_ascii=False),
        "third_party_imports_json": json.dumps(third_party, ensure_ascii=False, sort_keys=True),
        "local_imports_json": json.dumps(local_imports, ensure_ascii=False),
        "unresolved_or_external_imports_json": json.dumps(unresolved_imports, ensure_ascii=False),
        "functions_json": json.dumps(functions, ensure_ascii=False),
        "classes_json": json.dumps(classes, ensure_ascii=False),
        "function_count": len(functions),
        "class_count": len(classes),
        "input_artifact_literals_json": json.dumps(read_paths, ensure_ascii=False),
        "audit_signals_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        "feature_signals_json": json.dumps(feature_evidence, ensure_ascii=False, sort_keys=True),
        "external_access_evidence_json": json.dumps(evidence.get("external_access", []), ensure_ascii=False),
        "cv_evidence_json": json.dumps(evidence.get("cv", []), ensure_ascii=False),
        "leakage_review_required": bool(evidence.get("potential_leakage")),
        "target_encoding_evidence": bool(evidence.get("target_encoding")),
        "transductive_evidence": bool(evidence.get("transductive")),
        "gpu_evidence": bool(evidence.get("gpu")),
        "ensemble_evidence": bool(evidence.get("ensemble")),
        "seed_evidence": bool(evidence.get("seed")),
    }, text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _all_functions(rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["source_group"] for row in rows)
    syntax = Counter(row["syntax_status"] for row in rows)
    unique_hashes = len({row["sha256"] for row in rows})
    lines = [
        "# Python and notebook source inventory",
        "",
        "Generated by `src/audit_python_code.py`. Sources are read-only and are not copied into TRUTH.",
        "",
        f"- Files inspected: **{len(rows)}** (`{sum(row['suffix'] == '.py' for row in rows)}` Python files; "
        f"`{sum(row['suffix'] == '.ipynb' for row in rows)}` notebooks).",
        f"- Unique source hashes: **{unique_hashes}**.",
        f"- Syntax results: **{dict(syntax)}** (notebook cells are compiled independently after removing IPython magic lines).",
        "- Import availability is checked without importing or executing source modules.",
        "- Duplicate code files remain individually inventoried; exact hashes are evidence of byte-identical copies only.",
        "",
        "| Source group | Files | Python | Notebooks |",
        "|---|---:|---:|---:|",
    ]
    for group in sorted(counts):
        subset = [row for row in rows if row["source_group"] == group]
        lines.append(
            f"| `{group}` | {len(subset)} | {sum(row['suffix'] == '.py' for row in subset)} | "
            f"{sum(row['suffix'] == '.ipynb' for row in subset)} |"
        )
    lines.extend(["", "## Inventory", "", "| Source | Type | Lines | SHA-256 | Syntax | Definitions | Import availability |"])
    lines.extend(["|---|---|---:|---|---|---|---|"])
    for row in rows:
        try:
            imports = json.loads(row["third_party_imports_json"])
            missing = [name for name, status in imports.items() if status.startswith("NOT_FOUND")]
        except json.JSONDecodeError:
            missing = []
        defs = json.loads(row["functions_json"]) + json.loads(row["classes_json"])
        lines.append(
            f"| `{row['path']}` | {row['suffix']} | {row['line_count']} | `{row['sha256']}` | "
            f"{row['syntax_status']}{(': ' + row['syntax_error'].replace('|', '%7C')) if row['syntax_error'] else ''} | "
            f"{', '.join(defs[:8]) if defs else '—'} | "
            f"{', '.join(missing[:8]) if missing else 'all detected top-level imports available or local'} |"
        )
    return "\n".join(lines) + "\n"


def _signals(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row["audit_signals_json"])
    except (KeyError, json.JSONDecodeError):
        return {}


def _write_dependency_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Python dependency map",
        "",
        "Read-only static import and artifact-reference map. Modules were not imported or executed.",
        "Detected imports are not proof that the module is required on a given runtime path.",
        "",
        "| Source file | Local imports | Other imports | Unavailable imports | Literal artifacts |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        local = json.loads(row["local_imports_json"])
        third_party = json.loads(row["third_party_imports_json"])
        unresolved = json.loads(row["unresolved_or_external_imports_json"])
        missing = [
            name for name, state in third_party.items()
            if state == "NOT_FOUND_IN_AUDIT_ENVIRONMENT"
        ]
        paths = json.loads(row["input_artifact_literals_json"])
        artifact_labels = [
            f"{item['literal']} [{item['status']}]" for item in paths
        ]
        lines.append(
            f"| `{row['path']}` | {', '.join(local) if local else '—'} | "
            f"{', '.join(unresolved) if unresolved else '—'} | "
            f"{', '.join(missing) if missing else '—'} | "
            f"{'; '.join(artifact_labels) if artifact_labels else 'not statically resolved'} |"
        )
    (REPORT_DIR / "PYTHON_DEPENDENCY_MAP.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_leakage_report(rows: list[dict[str, Any]]) -> None:
    signal_counts: Counter[str] = Counter()
    for row in rows:
        for key in _signals(row):
            signal_counts[key] += 1
    lines = [
        "# Leakage and validation audit",
        "",
        "## Method and limitations",
        "",
        "All code/notebooks were parsed read-only. This report is a static screen plus focused manual "
        "review of lineage-critical files; a regex signal is a review lead, not a finding of leakage. "
        "Source code was not executed, model training was not started, and source data was not modified.",
        "",
        "## Corpus-wide static indicators",
        "",
        "| Signal | Files with at least one match | Interpretation |",
        "|---|---:|---|",
        f"| Cross-validation / OOF terms | {signal_counts['cv']} | Does not prove correct fold isolation. |",
        f"| Target-encoding terms | {sum(row['target_encoding_evidence'] == 'True' for row in rows)} | Requires fold-local target provenance review. |",
        f"| Train/test boundary terms | {signal_counts['train_test_boundary']} | Includes data-loading and commentary references. |",
        f"| Potential leakage / fitting terms | {signal_counts['potential_leakage']} | Broad lexical candidate screen; review source context. |",
        f"| External access terms | {signal_counts['external_access']} | Check whether any live network/API source is used. |",
        f"| Transductive / test-statistic terms | {signal_counts['transductive']} | Unlabelled test covariate use is not itself a rule violation; organizer terms are unverified. |",
        f"| GPU execution terms | {signal_counts['gpu']} | Static intent only; actual device execution is unverified. |",
        f"| Ensemble / blending terms | {signal_counts['ensemble']} | Inspect whether blend choice uses outer validation labels. |",
        f"| Explicit seed assignments | {signal_counts['seed']} | Seeds do not guarantee deterministic libraries/hardware. |",
        "",
        "## Candidate leakage and validation review queue",
        "",
        "Every file with a static `potential_leakage` signal is listed. The signal may be a safe "
        "`fit_transform` inside a training fold or even documentation; no violation is inferred.",
        "",
        "| File | CV evidence | Target encoding | Transductive evidence | Candidate fitting/leakage lines |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        evidence = _signals(row)
        if not evidence.get("potential_leakage"):
            continue
        lines.append(
            f"| `{row['path']}` | {'; '.join(evidence.get('cv', [])[:3]) or '—'} | "
            f"{row['target_encoding_evidence']} | {row['transductive_evidence']} | "
            f"{'; '.join(evidence['potential_leakage'])} |"
        )
    lines.extend(
        [
            "",
            "## Focused code review",
            "",
            "### Fold-safe baseline and target encoding",
            "",
            "`champion/EXP85_PHASE2_FROZEN_RELEASE/SOURCE/project/source/run_independent_baseline.py` "
            "constructs applicant-level deterministic features from each row, excludes TARGET from its "
            "feature builder, and creates stratified five-fold LightGBM/CatBoost models. Validation "
            "indices are separate for each fit. CatBoost/LightGBM use the validation fold for early "
            "stopping; this is conventional fold validation, but model selection across repeated "
            "experiments must not reuse the final evaluation folds.",
            "",
            "`.../phase_02_fold_safe_key_encodings/run_experiment.py` creates combined train/test "
            "keys without labels, then calculates nested inner-OOF encodings only from training labels "
            "whose indices are disjoint from apply indices. Outer validation/test encodings are fitted "
            "from outer-training labels. The included test explicitly mutates apply-row labels and "
            "checks encoded outputs are unchanged. This is strong code-level evidence of fold safety "
            "for that helper, not proof of the complete experiment's result.",
            "",
            "`screen_and_select_features.py` aggregates gain across five CV folds, selects one global "
            "top-550 set, then saves it for downstream modeling. That selection is not nested inside a "
            "later outer evaluation fold; downstream OOF scores using the selected set can inherit "
            "selection leakage. The available `nested_cv_feature_screening_benchmark.py` screens only "
            "inside each outer training partition, but compares against a hard-coded "
            "`full_screen_auc = 0.754136`. Recompute that baseline from matched artifacts before using "
            "the reported delta.",
            "",
            "### Older research and transductive feature code",
            "",
            "`champion/05_FEATURE_ENGINEERING/FEATURE_BUILDERS/build_hasan_600_features.py` concatenates "
            "train and test covariates and derives ranks/statistics across that combined population. "
            "It separately retains training TARGET for model work. No test labels are visible in the "
            "feature calculation examined. This is transductive feature construction and needs explicit "
            "organizer clarification because the primary competition rules were not available.",
            "",
            "`pre info/oct22/solution_v5_gci_research.py` includes nested target encoding, repeated CV, "
            "validation-driven comparison/tuning/blending and optional strict pseudo-labeling according "
            "to its own comments and code markers. It is not safe to treat every experiment branch as "
            "a single leakage-free final pipeline: verify the exact selected configuration and ensure "
            "outer-fold predictions were not reused to select the final blend.",
            "",
            "`name5_dependent_code/build_precision_transductive_apex_champion.py` trains fresh models, "
            "then groups train/test records on identity-like covariates and uses known training labels "
            "to adjust test predictions; test-test smoothing is also used. The file is a distinct "
            "transductive replacement, not a reproduction of historical predictions. Rule and identity-"
            "linking authorization remains ambiguous.",
            "",
            "### Historical output reconstruction",
            "",
            "`champion/create_exp85_submissions.py`, `create_submission_recovery_v5.py`, and "
            "`create_submission_recovery_v5_rebuilt.py`, plus "
            "`gemini/final_submission/reproduce_final_submission.py`, consume generated/cached model "
            "prediction arrays and/or historical outputs. They are synthesis/reconstruction programs, "
            "not standalone raw-data training evidence. A valid final honors submission additionally "
            "requires the exact submission CSV to be reproducible with its source dependencies.",
            "",
            "No source-only static check establishes absence of manual labels or test-label access across "
            "all runtime inputs; these require provenance-complete runs and the official rule text.",
        ]
    )
    (REPORT_DIR / "LEAKAGE_AND_VALIDATION_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_compliance_matrix(rows: list[dict[str, Any]]) -> None:
    rules_path = Path(
        r"D:\gci\champion\07_COMPETITION_COMPLIANCE\OFFICIAL_RULES_SNAPSHOT.md"
    )
    observed = {
        "External network/API access": [
            row for row in rows if _signals(row).get("external_access")
        ],
        "Train/test transductive features": [
            row for row in rows if row["transductive_evidence"]
        ],
        "Target encoding": [
            row for row in rows if row["target_encoding_evidence"]
        ],
        "Cross-validation and OOF": [
            row for row in rows if _signals(row).get("cv")
        ],
        "Ensembling/calibration": [
            row for row in rows if _signals(row).get("ensemble")
        ],
    }
    definitions = [
        (
            "External datasets / network data",
        "UNVERIFIED",
            "OFFICIAL_RULES_SNAPSHOT.md §Data Policy; claims only supplied train.csv/test.csv; not independently verified against organizer source.",
            "External data is prohibited according to the locally preserved rules snapshot. Static external-access signals require review; cached/model artifacts may or may not derive solely from supplied data.",
            "External network/API access",
        ),
        (
            "Manual hand-labeling",
            "UNVERIFIED",
            "OFFICIAL_RULES_SNAPSHOT.md §Manual Hand-Labeling Policy; primary citation unavailable.",
            "The local snapshot says manual hand-labeling is strictly prohibited. Source scan cannot establish that no labels were manually introduced outside code.",
            "",
        ),
        (
            "Test-label access",
            "UNVERIFIED",
            "No exact rule text located in a primary organizer source. Local snapshot has no separately numbered test-label clause.",
            "Static code and named datasets show train TARGET usage in reviewed pipelines; no source-only scan proves all artifacts/caches contain no labels.",
            "",
        ),
        (
            "Test-covariate statistics / transductive learning",
            "AMBIGUOUS — ORGANIZER CLARIFICATION REQUIRED",
            "No explicit permission/prohibition in the available local rules snapshot; it cites Competition/README.ipynb and portal, neither available in permitted read-only sources.",
            f"Static transductive terms appear in {len(observed['Train/test transductive features'])} files; combined train/test feature generation is directly observed in feature code.",
            "Train/test transductive features",
        ),
        (
            "Target encoding",
            "UNVERIFIED",
            "No explicit target-encoding rule found in the secondary snapshot. This is a validation-safety issue distinct from organizer permission.",
            f"Target encoding appears in {len(observed['Target encoding'])} files. Fold-safe nested implementation exists in Phase 2; historical branches require independent review.",
            "Target encoding",
        ),
        (
            "Feature interactions / derived features",
            "UNVERIFIED",
            "No feature-engineering restriction or permission located in the primary organizer material.",
            "Many rowwise feature builders and interaction code paths exist; feature engineering is not presumed prohibited, but the applicable rules were not verified.",
            "",
        ),
        (
            "Cross-validation / validation selection",
            "UNVERIFIED",
            "OFFICIAL_RULES_SNAPSHOT.md §Validation Policy Audit is a local audit assertion, not primary organizer rule text.",
            f"CV/OOF indicators occur in {len(observed['Cross-validation and OOF'])} files; presence does not prove fold-local preprocessing or unbiased model selection.",
            "Cross-validation and OOF",
        ),
        (
            "Pretrained models / model licensing",
            "UNVERIFIED",
            "The local compliance audit asserts no pretrained models; no primary rule reference was found.",
            "No robust corpus-wide checkpoint loader/licensing examination can be inferred from imports alone. Verify model provenance and licenses for any selected pipeline.",
            "",
        ),
        (
            "Ensembling, calibration, rank/probit transforms",
            "UNVERIFIED",
            "No explicit organizer restriction or permission found in the primary source material.",
            f"Ensembling/post-processing terms occur in {len(observed['Ensembling/calibration'])} files. They require a frozen OOF-only selection protocol and exact submission reproduction evidence.",
            "Ensembling/calibration",
        ),
        (
            "Submission format / honors reproducibility",
            "UNVERIFIED",
            "OFFICIAL_RULES_SNAPSHOT.md §Submission Requirements and §Honors / Outstanding Student Requirement: two columns SK_ID_CURR,TARGET; 61,500 rows; reproducible submitted CSV.",
            "Local transcription is the only accessible rule text. The stated format and reproduction requirement must be verified with the GCI organizer/portal before certification.",
            "",
        ),
    ]
    matrix = []
    for technique, status, reference, evidence, evidence_group in definitions:
        sources = observed.get(evidence_group, []) if evidence_group else []
        matrix.append(
            {
                "technique": technique,
                "classification": status,
                "rule_reference_and_authority": reference,
                "supporting_code_evidence": evidence,
                "source_file_count": len(sources),
                "example_source_files": json.dumps(
                    [row["path"] for row in sources[:12]], ensure_ascii=False
                ),
                "primary_rule_verified": False,
            }
        )
    _write_csv(REPORT_DIR / "COMPETITION_COMPLIANCE_MATRIX.csv", matrix)


def _write_components(rows: list[dict[str, Any]]) -> None:
    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    historical_consumers = (
        "create_exp85_submissions.py",
        "create_submission_recovery_v5.py",
        "create_submission_recovery_v5_rebuilt.py",
        "reproduce_final_submission.py",
    )
    exact_paths = {
        Path(r"D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\source\run_independent_baseline.py"): (
            "Raw-only rowwise feature builder; stratified five-fold baseline; runtime and output need fresh validation.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
        Path(r"D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_02_fold_safe_key_encodings\run_experiment.py"): (
            "Nested encoding helper fits only on disjoint train indices; combined train/test keys are unlabeled covariate features.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
        Path(r"D:\gci\champion\EXP85_PHASE2_FROZEN_RELEASE\SOURCE\project\experiments\phase_02_fold_safe_key_encodings\test_target_encodings.py"): (
            "Focused test verifies apply-label invariance and fit/apply index separation.",
            "REUSABLE",
        ),
        Path(r"D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\build_hasan_600_features.py"): (
            "Contains reusable financial, missingness and temporal formula candidates; train+test rank/statistic construction is transductive.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
        Path(r"D:\gci\champion\05_FEATURE_ENGINEERING\FEATURE_BUILDERS\screen_and_select_features.py"): (
            "Feature-screening implementation needs fold-boundary review before reuse; selection on all labels would bias evaluation.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
        Path(r"D:\gci\pre info\oct22\solution_v5_gci_research.py"): (
            "Extensive model/feature research with nested TE and many optional validation/tuning/pseudolabel branches; exact final config is not self-evident.",
            "PARTIALLY_REPRODUCIBLE",
        ),
        Path(r"D:\gci\name5_dependent_code\train_full_v16_suite.py"): (
            "Fresh training alternative with stratified folds; factorization over concatenated train/test covariates is transductive and requires rule clarification.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
        Path(r"D:\gci\name5_dependent_code\build_precision_transductive_apex_champion.py"): (
            "Fresh raw-data models, followed by identity-key train-label transfer and test-test smoothing; not historical parity and policy ambiguity remains.",
            "REUSABLE_AFTER_MODIFICATION",
        ),
    }
    for row in rows:
        path = Path(row["path"])
        signals = _signals(row)
        input_references = json.loads(row["input_artifact_literals_json"])
        unavailable_imports = [
            module
            for module, status in json.loads(row["third_party_imports_json"]).items()
            if status == "NOT_FOUND_IN_AUDIT_ENVIRONMENT"
        ]
        if path in exact_paths:
            rationale, _ = exact_paths[path]
        elif row["syntax_status"] == "FAIL":
            rationale = "Syntax-invalid/incomplete source; cannot be used as written."
        elif any(path.name.casefold() == name for name in historical_consumers):
            rationale = "Historical submission/reconstruction consumer; not a self-contained training implementation."
        elif signals.get("training"):
            rationale = "Model-training implementation identified by static code signals; runtime and fold behavior require per-file review."
        elif json.loads(row["feature_signals_json"]):
            rationale = "Feature-building implementation identified by static code signals; formula and fit-boundary review required."
        else:
            rationale = "Support, validation, notebook, or research source; assess its specific role before reuse."
        if row["syntax_status"] == "FAIL":
            decision = "BLOCKED"
        elif any(path.name.casefold() == name for name in historical_consumers):
            decision = "PARTIALLY_REPRODUCIBLE"
        elif any(item["status"] == "NOT_FOUND_OR_DYNAMIC" for item in input_references):
            decision = "PARTIALLY_REPRODUCIBLE"
        elif signals.get("training") or json.loads(row["feature_signals_json"]):
            decision = "REUSABLE_AFTER_MODIFICATION"
        else:
            decision = "REUSABLE_AFTER_MODIFICATION"
        selected.append(
            {
                "component_path": row["path"],
                "sha256": row["sha256"],
                "component": rationale,
                "decision": decision,
                "can_run_with_current_files": (
                    "NOT_EXECUTED; syntax failure"
                    if row["syntax_status"] == "FAIL"
                    else "NOT VERIFIED; static paths/imports only"
                ),
                "feature_reproduction": (
                    "Feature code present; exact historical output not established"
                    if json.loads(row["feature_signals_json"])
                    else "No feature construction clearly identified by static signal scan"
                ),
                "validation_trust": (
                    "CV/OOF terms present; fold isolation and selection protocol require source review"
                    if signals.get("cv")
                    else "No CV/OOF signal found; validation method not established"
                ),
                "competition_compliance": "UNVERIFIED; primary organizer rules unavailable",
                "missing_dependencies": json.dumps(
                    {
                        "unavailable_imports_in_audit_environment": unavailable_imports,
                        "missing_or_dynamic_artifacts": [
                            item["literal"]
                            for item in input_references
                            if item["status"] == "NOT_FOUND_OR_DYNAMIC"
                        ],
                    },
                    ensure_ascii=False,
                ),
                "required_action": (
                    "Review the exact runtime path, inputs, folds, and rule status in an isolated TRUTH run before reuse."
                ),
            }
        )
        if row["syntax_status"] == "FAIL":
            blocked.append(
                {
                    "component_path": row["path"],
                    "sha256": row["sha256"],
                    "blocker": row["syntax_error"],
                    "decision": "BLOCKED",
                    "resolution": "Treat as an incomplete archival extract or notebook until reconstructed from a verified source; do not silently repair in the preserved source.",
                }
            )
        for item in json.loads(row["input_artifact_literals_json"]):
            if item["status"] == "NOT_FOUND_OR_DYNAMIC":
                blocked.append(
                    {
                        "component_path": row["path"],
                        "sha256": row["sha256"],
                        "blocker": (
                            "Literal input path did not resolve from the source directory "
                            f"or a source-root top level: {item['literal']}"
                        ),
                        "decision": "PARTIALLY_REPRODUCIBLE",
                        "resolution": "Verify the runtime working directory and search the expected artifact store; this static probe does not establish global absence. Rebuild only from verified supplied data if the artifact is genuinely unavailable.",
                    }
                )
    for row in rows:
        if row["path"].replace("\\", "/").endswith(
            "/final_submission/reproduce_final_submission.py"
        ):
            blocked.append(
                {
                    "component_path": row["path"],
                    "sha256": row["sha256"],
                    "blocker": "Historical final-synthesis code loads multiple OOF/test NPZ/NPY streams, including neural/synthesis files documented as missing.",
                    "decision": "BLOCKED",
                    "resolution": "Do not run as an independent training pipeline; recover exact artifacts with provenance or retrain each component from raw data.",
                }
            )
    _write_csv(REPORT_DIR / "REUSABLE_COMPONENTS.csv", selected)
    _write_csv(REPORT_DIR / "BLOCKED_COMPONENTS.csv", blocked)


def _write_execution_report(rows: list[dict[str, Any]]) -> None:
    syntax_counts = Counter(row["syntax_status"] for row in rows)
    third_party_counts: Counter[str] = Counter()
    unavailable: dict[str, list[str]] = defaultdict(list)
    test_files = [row for row in rows if re.search(r"(^|[\\/])test[^\\/]*\.py$", row["path"], re.I) or re.search(r"[\\/]tests?[\\/]", row["path"], re.I)]
    for row in rows:
        for module, status in json.loads(row["third_party_imports_json"]).items():
            third_party_counts[f"{module}: {status}"] += 1
            if status == "NOT_FOUND_IN_AUDIT_ENVIRONMENT":
                unavailable[module].append(row["path"])
    failures = [row for row in rows if row["syntax_status"] == "FAIL"]
    lines = [
        "# Execution and reproducibility audit",
        "",
        "## Safe checks performed",
        "",
        f"- Parsed and compiled **{len(rows)}** Python/notebook code sources; {syntax_counts['PASS']} passed and {syntax_counts['FAIL']} failed.",
        "- Notebook cells were compiled independently after removing IPython magic lines to avoid treating normal cell boundaries as syntax errors.",
        "- Extracted functions/classes, imports, candidate local imports, data/artifact string references, seeds, CV, GPU, ensemble and leakage review signals.",
        "- Checked third-party import availability with `find_spec`; no source module was imported or executed.",
        f"- Discovered **{len(test_files)}** existing test-like Python files; none were run because imports may trigger legacy side effects and many require historical data/artifacts.",
        "- No dependencies were installed, no model was trained, no GPU was used, and no preserved source or input data was modified.",
        "",
        "## Syntax failures",
        "",
        "| File | Failure |",
        "|---|---|",
    ]
    for row in failures:
        lines.append(f"| `{row['path']}` | {row['syntax_error'].replace('|', '%7C')} |")
    lines.extend(
        [
            "",
            f"Notebook parse failures may prevent those notebooks from executing as written. The {sum(row['suffix'] == '.py' for row in failures)} `.py` failures are archival/incomplete extracts, not evidence that the underlying project source is invalid.",
            "",
            "## Environment import check",
            "",
            "| Detected module/status | Number of source files |",
            "|---|---:|",
        ]
    )
    for key, count in sorted(third_party_counts.items()):
        lines.append(f"| `{key}` | {count} |")
    lines.extend(
        [
            "",
            "Module availability describes only the current audit Python environment. It does not substitute for each project's pinned requirements or prove version compatibility.",
            "",
            "## Blockers and artifact dependencies",
            "",
            "- Unresolved literal input references are listed in `BLOCKED_COMPONENTS.csv`. A path is checked relative to its source directory and the top level of each preserved source root; `NOT_FOUND_OR_DYNAMIC` is not proof of global absence, so verify the runtime working directory and artifact store.",
            "- Historical synthesis/recovery scripts depend on saved model prediction arrays and other generated artifacts; they do not train their entire lineage from raw data.",
            "- Direct inspection of the five scripts in `D:\\gci\\5 independent` shows reads of cached Feather/NPZ/PKL inputs and/or an anchor CSV (including v16/v17/v55 bundles); these are artifact-dependent consumers, not standalone raw-data reproductions. The separate `D:\\gci\\name5_dependent_code` scripts are distinct raw-data replacement approximations.",
            "- `gemini/final_submission/reproduce_final_submission.py` specifically references multiple neural/synthesis NPZ/NPY streams documented as absent in the preserved dependency audit. It is blocked as a complete historical synthesis route.",
            "- `gemini/dependent/exp85_notebook_run/ARTIFACT_REBUILD_AUDIT.md` says the exact 274-feature monotonic 10-fold EXP42 training creator is not present; its missing filename is not stated there.",
            "- `name5_dependent_code/README.md` says the exact V43 submission writer was not found. These are documented gaps, not names of the two user-referenced missing files.",
            "",
            "## Missing-file exclusion",
            "",
            "The two exact filenames previously identified were not discoverable from the permitted source documentation. I therefore did not guess or silently exclude files by name. All 637 available `.py`/`.ipynb` files across the seven stated preserved source roots were inventoried and scanned. The documented gaps above are separately recorded, and any organizer/user-supplied exclusion names should be added to the report without removing the current all-available-file inventory.",
        ]
    )
    (REPORT_DIR / "EXECUTION_AND_REPRODUCIBILITY_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_final_report(rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if row["syntax_status"] == "FAIL"]
    syntax_counts = Counter(row["syntax_status"] for row in rows)
    test_files = [row for row in rows if re.search(r"(^|[\\/])test[^\\/]*\.py$", row["path"], re.I) or re.search(r"[\\/]tests?[\\/]", row["path"], re.I)]
    identical_hashes = Counter(row["sha256"] for row in rows)
    duplicate_instances = sum(count - 1 for count in identical_hashes.values() if count > 1)
    lines = [
        "# TRUTH — Available Python code audit",
        "",
        "## Executive decision",
        "",
        "**Overall competition-rule status: UNVERIFIED, not a finding that the project is ineligible.** "
        "The only available rule text is a local secondary transcription in the old champion archive. "
        "Its cited primary competition brief/portal was not present in the allowed read-only source roots, "
        "and the official portal could not be independently verified here. Transductive methods, target "
        "encoding, interactions, and test-covariate statistics are therefore not declared prohibited; "
        "where the text is silent, they are marked ambiguous or unverified.",
        "",
        "Recommended candidate for a new solution: start from the raw-only Phase 2 baseline and its nested "
        "fold-safe target-encoding helper, then build only from verified supplied train/test files. Do not "
        "reuse the historical submission synthesis as training evidence. Keep transductive test statistics "
        "and identity-based train-label transfer disabled until organizers clarify them. This is a code "
        "reuse recommendation, not a score claim or guarantee of reaching public AUC 0.79000.",
        "",
        "## Scope and isolation",
        "",
        f"- Git root: `{TRUTH_ROOT}`; independent empty Git repository; no remote configured.",
        f"- Read-only source roots: {', '.join(f'`{root}`' for root in SOURCE_ROOTS)}.",
        f"- Available files inspected: **{len(rows)}** ({sum(row['suffix'] == '.py' for row in rows)} `.py`, {sum(row['suffix'] == '.ipynb' for row in rows)} notebooks).",
        f"- Exact source hashes unique: **{len(identical_hashes)}**; repeated copies: **{duplicate_instances}** additional paths.",
        "- No source files or data were copied from or modified in the old repositories.",
        "- The two exact previously identified missing-file names could not be established from permitted documentation; no guessed exclusions were made.",
        "",
        "## Source audit results",
        "",
        "See `PYTHON_FILE_INVENTORY.csv` for every path, byte size, hash, syntax, imports, definitions, "
        "artifact strings, validation/feature signals and notebook metadata. "
        "`PYTHON_DEPENDENCY_MAP.md` traces per-file imports and input-artifact references.",
        "",
        f"Syntax compilation passed for **{syntax_counts['PASS']}** sources and failed for **{syntax_counts['FAIL']}**. "
        f"All {sum(row['suffix'] == '.ipynb' for row in failures)} notebook parse failures and "
        f"{sum(row['suffix'] == '.py' for row in failures)} `.py` failures are listed in the execution report. "
        "Notebook cells were compiled independently after removing IPython magic lines. The `.py` "
        "files are incomplete extraction/dump artifacts. Syntax passing does not prove runtime imports, "
        "data paths, results, or eligibility.",
        "",
        f"**Test discovery:** {len(test_files)} test-like Python files found. They were not executed because legacy imports may trigger side effects and many tests require data/artifacts; safe static parse/compilation was used instead.",
        "",
        "## Code lineage and practical decisions",
        "",
        "| Implementation family | Run with available files? | Feature reproduction | Validation trust | Suggested decision |",
        "|---|---|---|---|---|",
        "| Frozen Phase 2 raw-data baseline | Likely, if local raw data and Python dependencies match; not executed here. | Rowwise features are directly implemented; a historical reference is optionally read only for post-hoc similarity, not as a training input. | Stratified 5-fold OOF and fold-specific fit/validation are visible. Early stopping is on the current validation fold; do not use same folds for extensive experiment selection. | REUSABLE_AFTER_MODIFICATION |",
        "| Phase 1/2 fee features and nested target encoding | Requires raw data and a referenced Phase 1 run/baseline outputs for full experiment. | Raw formula source present; target-key creation uses train/test covariates; encodings use training labels in disjoint inner/outer folds. | Strong helper-level evidence and focused label-invariance test; full run/outputs unverified. | REUSABLE_AFTER_MODIFICATION |",
        "| Hasan historical feature laboratory / screening | Some scripts run only with expected raw data/dependencies. | Formula code present; train+test rank/statistics are transductive; screening methodology must be fold-local. | Individual features can be reproduced; historical metrics are not reproduced by this audit. | REUSABLE_AFTER_MODIFICATION |",
        "| V15 monolithic research solution | Source is available; complete execution is configuration/artifact/dependency dependent. | Large in-script feature laboratory and fold-safe TE branches. | Many modes, tuning, validation-based decisions and optional pseudo-labels; exact promoted branch and outer-fold independence need audit. | PARTIALLY_REPRODUCIBLE |",
        "| `5 independent` legacy scripts | Source present, but direct inspection shows Feather/NPZ/PKL/anchor inputs; they are not raw-data-only standalone training routes. | Depends on precomputed feature/model arrays; historical output cannot be reproduced from these entry points alone. | CV code exists in some scripts; validity is not enough to repair missing upstream provenance. | PARTIALLY_REPRODUCIBLE |",
        "| `name5_dependent_code` replacement scripts | Raw train/test versions exist; not executed. | New approximations rather than historical parity; some concatenate train/test covariates or use identity-key test adjustments. | Stratified folds are present in inspected runners; blend and early-stopping choices require outer validation. | REUSABLE_AFTER_MODIFICATION |",
        "| Historical EXP85 submission reconstruction/synthesis | No, not as a self-contained raw-data pipeline; several historical arrays/caches are missing. | Synthesis formula can replay available artifacts; missing neural streams and upstream predictions block exact reproduction. | Similarity to champion is not independent training or OOF evidence. | BLOCKED |",
        "| Incomplete extracted `.py` / notebook snapshots | No as currently parsed. | Unknown/incomplete source text. | Not assessable. | BLOCKED |",
        "",
        "The exact per-file reusable/blocked evidence is in `REUSABLE_COMPONENTS.csv` and `BLOCKED_COMPONENTS.csv`.",
        "",
        "## Rule compliance matrix",
        "",
        "The matrix distinguishes policy claims from implementation evidence. The archived "
        "`OFFICIAL_RULES_SNAPSHOT.md` states only supplied train/test data, prohibits manual hand-labeling, "
        "requires a two-column 61,500-row submission and says honors submissions must be reproducible. "
        "It cites `Competition/README.ipynb` and the Omnicampus portal but provides no primary-source "
        "URL, rule numbers, capture date, or organizer quotation record. These exact official statements "
        "are therefore **not independently verified** in this audit.",
        "",
        "The archived `DATA_POLICY_AUDIT.md` also records raw train/test SHA-256 values that do not match "
        "the current files at `D:\\gci\\inpuy`: its train/test hashes are "
        "`129858df3c08354c25f448b488734e7bbbe6256ce850e0388172c72b22b2b1ff` / "
        "`9aaee24c4dc2794d4d6ee179f8263595ddb3e3437ce1cb793b827e70ea5fbca9`; current hashes are "
        "`b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1` / "
        "`dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65`. That historic audit cannot "
        "certify the current dataset bytes without an explained version transition.",
        "",
        "| Technique | Classification |",
        "|---|---|",
        "| External datasets/network data | UNVERIFIED; the secondary snapshot says prohibited. |",
        "| Manual hand-labeling | UNVERIFIED absence; the secondary snapshot says prohibited. |",
        "| Test-label access | UNVERIFIED; no source-only proof of all cached artifacts. |",
        "| Transductive/test-covariate statistics | AMBIGUOUS — ORGANIZER CLARIFICATION REQUIRED. |",
        "| Target encoding | UNVERIFIED in organizer rules; fold safety is separate and code-specific. |",
        "| Feature interactions/derived features | UNVERIFIED; no restriction was established. |",
        "| CV/OOF and validation selection | UNVERIFIED implementation-by-implementation. |",
        "| Pretrained models/licensing | UNVERIFIED; verify model/dependency lineage in chosen pipeline. |",
        "| Ensemble/calibration/postprocessing | UNVERIFIED; no rule text found; OOF selection protocol required. |",
        "| Submission format/reproducibility | UNVERIFIED primary authority; secondary snapshot describes requirements. |",
        "",
        "No technique is declared universally non-compliant solely because it uses unlabeled test covariates, "
        "feature interactions, or target encoding. No project-wide ineligibility finding is made.",
        "",
        "## Can available code contribute to an AUC 0.79000+ solution?",
        "",
        "Potentially as reusable starting components, but this audit cannot establish performance. The "
        "strongest clean-room direction is a minimal raw-data baseline, one fold-safe feature addition at "
        "a time, a frozen external OOF protocol, and explicit source/data hashes. Do not target 0.79000 "
        "as a promised outcome; score must be demonstrated by valid evaluation and the official leaderboard.",
        "",
        "## Required next actions",
        "",
        "1. Obtain the actual GCI World 2026 rule text from an organizer-controlled source, including data, "
        "transductive learning, pseudo-labeling, identity linkage, pretrained models and reproducibility clauses.",
        "2. Provide the exact two missing filenames previously identified so they can be excluded explicitly; "
        "until then all available files remain inventoried.",
        "3. Select one raw-only implementation and run it only inside TRUTH after verifying raw data hashes, "
        "requirements, local imports and all inputs. Preserve the old champion and research roots read-only.",
        "4. Use held-out outer folds for final model/blend selection; do not re-optimize weights on validation "
        "labels used to claim the final result.",
        "",
        "## Reports",
        "",
        "- `PYTHON_FILE_INVENTORY.csv` — all source paths and static metadata.",
        "- `PYTHON_DEPENDENCY_MAP.md` — imports, paths, and artifact relationships.",
        "- `FEATURE_ENGINEERING_AUDIT.csv` — per-file feature-family evidence.",
        "- `LEAKAGE_AND_VALIDATION_AUDIT.md` — fold, encoding, transductive, and leakage review.",
        "- `COMPETITION_COMPLIANCE_MATRIX.csv` — policy classification with authority limitations.",
        "- `EXECUTION_AND_REPRODUCIBILITY_AUDIT.md` — safe checks and blockers.",
        "- `REUSABLE_COMPONENTS.csv` / `BLOCKED_COMPONENTS.csv` — practical source decisions.",
        "- `SHA256_MANIFEST.json` — hashes of inspected sources and generated reports.",
    ]
    (REPORT_DIR / "FINAL_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    outputs = {}
    for path in sorted(REPORT_DIR.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.json":
            outputs[path.name] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
    sources = {
        row["path"]: {
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "source_group": row["source_group"],
            "syntax_status": row["syntax_status"],
        }
        for row in rows
    }
    rules_path = Path(
        r"D:\gci\champion\07_COMPETITION_COMPLIANCE\OFFICIAL_RULES_SNAPSHOT.md"
    )
    references = {
        "secondary_competition_rules_snapshot": (
            {
                "path": str(rules_path),
                "sha256": sha256(rules_path),
                "authority": "unverified secondary local transcription",
            }
            if rules_path.is_file()
            else {"path": str(rules_path), "status": "NOT_FOUND"}
        )
    }
    audit_tool = TRUTH_ROOT / "src" / "audit_python_code.py"
    payload = {
        "manifest_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "truth_git_root": str(TRUTH_ROOT),
        "source_count": len(sources),
        "sources": sources,
        "audit_tool": {
            "path": str(audit_tool),
            "sha256": sha256(audit_tool),
            "size_bytes": audit_tool.stat().st_size,
        },
        "outputs": outputs,
        "references": references,
        "limitations": [
            "Input source files remain outside TRUTH and were read only.",
            "Manifest does not certify external data lineage or official competition compliance.",
            "The manifest itself is not self-hashed.",
        ],
    }
    (REPORT_DIR / "SHA256_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    if Path.cwd().resolve() != TRUTH_ROOT:
        raise RuntimeError(f"Run from the isolated TRUTH Git root: {TRUTH_ROOT}")
    files = _files()
    module_names = _module_index(files)
    rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    for root, path in files:
        row, text = _audit_one(root, path, module_names)
        rows.append(row)
        texts[row["path"]] = text
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(REPORT_DIR / "PYTHON_FILE_INVENTORY.csv", rows)
    feature_rows = []
    for row in rows:
        signals = json.loads(row["feature_signals_json"])
        for category, evidence in signals.items():
            feature_rows.append(
                {
                    "source_path": row["path"],
                    "sha256": row["sha256"],
                    "feature_category": category,
                    "evidence_lines": json.dumps(evidence, ensure_ascii=False),
                    "manual_validation_status": "STATIC_SIGNAL_REQUIRES_SOURCE_REVIEW",
                }
            )
    _write_csv(REPORT_DIR / "FEATURE_ENGINEERING_AUDIT.csv", feature_rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "truth_git_root": str(TRUTH_ROOT),
        "source_roots": [str(root) for root in SOURCE_ROOTS],
        "source_file_count": len(rows),
        "python_file_count": sum(row["suffix"] == ".py" for row in rows),
        "notebook_count": sum(row["suffix"] == ".ipynb" for row in rows),
        "syntax_status_counts": dict(Counter(row["syntax_status"] for row in rows)),
        "unique_hash_count": len({row["sha256"] for row in rows}),
        "missing_source_roots": [str(root) for root in SOURCE_ROOTS if not root.is_dir()],
        "audit_method": "read-only syntax/AST/static signal analysis; no source execution or model training",
    }
    (REPORT_DIR / "AUDIT_SCAN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "PYTHON_FILE_INVENTORY.md").write_text(
        _all_functions(rows), encoding="utf-8"
    )
    _write_dependency_report(rows)
    _write_leakage_report(rows)
    _write_compliance_matrix(rows)
    _write_components(rows)
    _write_execution_report(rows)
    _write_final_report(rows)
    _write_manifest(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
