from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import hashlib
import json
import os
import platform
import pathlib as pathlib_module
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import catboost
import numpy as np
import openpyxl
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci\truth")
EXPERIMENT = ROOT / "experiments" / "phase10_catboost_product_robustness"
RUNS = EXPERIMENT / "runs"
TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
DICTIONARY_PATH = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition"
    r"\HomeCredit_columns_description.xlsx"
)
P9_RUN = ROOT / "experiments" / "phase9_product_mechanism_and_catboost" / "runs" / (
    "phase9_20261104T181500LOCAL"
)
P9_OOF = P9_RUN / "OOF_PREDICTIONS.csv"
P9_FOLDS = P9_RUN / "CATBOOST_FOLD_RESULTS.csv"
P9_RUNNER = (
    ROOT
    / "experiments"
    / "phase9_product_mechanism_and_catboost"
    / "run_phase9_product_tests.py"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD_SEED = 42
FOLD_NUMBER = 1
SEED_BASE = 20261270

sys.path.insert(0, str(EXPERIMENT))
import phase10_independent_feature_builder as feature_builder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_path(module: object) -> str:
    path = getattr(module, "__file__", None)
    return str(Path(path).resolve()) if path else "<built-in or namespace module>"


def assert_no_forbidden_module_loaded() -> None:
    forbidden = "research_archive\\exp85_phase2_source_snapshot".casefold()
    loaded = []
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin and forbidden in str(origin).replace("/", "\\").casefold():
            loaded.append((name, origin))
    if loaded:
        raise RuntimeError(f"Forbidden module loaded into Phase 10: {loaded}")


def gpu_snapshot() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_dependency_provenance(out: Path, inputs: dict[str, Path]) -> None:
    rows = []
    direct_modules = (
        ("argparse", argparse),
        ("csv", csv),
        ("hashlib", hashlib),
        ("json", json),
        ("os", os),
        ("platform", platform),
        ("pathlib", pathlib_module),
        ("subprocess", subprocess),
        ("sys", sys),
        ("time", time),
        ("datetime", datetime_module),
        ("numpy", np),
        ("pandas", pd),
        ("openpyxl", openpyxl),
        ("scikit-learn", sklearn),
        ("sklearn.metrics", __import__("sklearn.metrics", fromlist=["metrics"])),
        (
            "sklearn.model_selection",
            __import__("sklearn.model_selection", fromlist=["model_selection"]),
        ),
        ("catboost", catboost),
        ("phase10_independent_feature_builder", feature_builder),
    )
    for name, module in direct_modules:
        rows.append(
            {
                "dependency": name,
                "absolute_source_path": module_path(module),
                "provenance": "Python standard library or installed package; imported by zero-EXP85 Phase 10 runner",
                "phase10_runtime_dependency": True,
                "phase10_file_access": "module import only",
                "sha256": (
                    sha256_file(Path(module_path(module)))
                    if Path(module_path(module)).is_file()
                    else ""
                ),
            }
        )
    for name, path in inputs.items():
        rows.append(
            {
                "dependency": name,
                "absolute_source_path": str(path.resolve()),
                "provenance": {
                    "training_csv": "Permitted supplied training data; TARGET used only for model fitting/evaluation",
                    "official_dictionary": "Official local competition column-description workbook",
                    "phase9_oof_reference": "Historical Phase 9 comparison target only; never used to construct features or fit the independent model",
                    "phase9_catboost_fold_reference": "Historical Phase 9 selected iteration count only",
                    "phase9_runner_source_audit": "Read-only source-lineage inspection; never imported or executed by Phase 10",
                    "phase10_runner": "Zero-EXP85 Phase 10 audit and one-fold GPU preflight source executed for this run",
                    "phase10_spec": "Independently authored mathematical feature specification",
                    "phase10_builder": "Independent implementation under test",
                    "phase10_tests": "Formula and leakage-boundary tests",
                }.get(name, "Audited local input"),
                "phase10_runtime_dependency": name
                in {
                    "training_csv",
                    "official_dictionary",
                    "phase9_oof_reference",
                    "phase9_catboost_fold_reference",
                    "phase10_runner",
                },
                "phase10_file_access": (
                    "executed" if name == "phase10_runner" else "read-only"
                ),
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )

    p9_dir = ROOT / "experiments" / "phase9_product_mechanism_and_catboost"
    p8_run = (
        ROOT
        / "experiments"
        / "phase8_nested_product_target_encoding"
        / "runs"
        / "phase8_20261104T080000Z"
    )
    historical_edges = [
        (
            "phase9 direct source",
            P9_RUNNER,
            "Phase 9 imports the preserved row-wise helper directly; inspected, not loaded by Phase 10",
        ),
        (
            "phase9 direct model dependency",
            ROOT
            / "research_archive"
            / "exp85_phase2_source_snapshot"
            / "project"
            / "source"
            / "run_independent_baseline.py",
            "PROHIBITED historical helper named by Phase 9; no Phase 10 read/import/hash",
        ),
        (
            "phase9 indirect source",
            ROOT
            / "experiments"
            / "phase3_hidden_pattern_discovery"
            / "run_phase3_hidden_pattern_discovery.py",
            "Phase 9 loads this module; it declares the same preserved baseline helper",
        ),
        (
            "phase9 indirect source",
            ROOT
            / "experiments"
            / "phase3b_deep_validation"
            / "phase3b_ablation.py",
            "Phase 9 loads this module for its product-block compatibility check; it declares the same preserved baseline helper",
        ),
        (
            "phase9 indirect source",
            ROOT
            / "experiments"
            / "phase7_product_feature_discovery"
            / "run_phase7_product_discovery.py",
            "Phase 9 loads this module for product features; it imports Phase 3, Phase 3B and the preserved baseline helper",
        ),
        (
            "phase9 subgroup-only source",
            ROOT
            / "experiments"
            / "phase8_nested_product_target_encoding"
            / "run_phase8_nested_product_te.py",
            "Phase 9 provenance/subgroup lineage; Phase 8 itself declares the same preserved helper",
        ),
        (
            "phase9 provenance-only source",
            ROOT
            / "experiments"
            / "phase8_nested_product_target_encoding"
            / "audit_phase8_results.py",
            "Listed in Phase 9 provenance; not imported by the Phase 9 model run or Phase 10",
        ),
        (
            "phase9 subgroup-only artifact",
            p8_run / "OOF_PREDICTIONS.csv",
            "Phase 9 reads this Phase 8 OOF artifact only for historical subgroup diagnostics; not part of Phase 10",
        ),
        (
            "phase9 provenance-only artifact",
            p8_run / "SEED_METRICS.csv",
            "Listed in Phase 9 provenance; not used by the Phase 9 CatBoost model or Phase 10",
        ),
    ]
    for name, path, provenance in historical_edges:
        rows.append(
            {
                "dependency": name,
                "absolute_source_path": str(path),
                "provenance": provenance,
                "phase10_runtime_dependency": False,
                "phase10_file_access": (
                    "read-only source hash for lineage; never imported or executed"
                    if name == "phase9 direct source"
                    else "not accessed; listed solely to document the Phase 9 dependency graph"
                ),
                "sha256": "",
            }
        )
    pd.DataFrame(rows).to_csv(out / "DEPENDENCY_PROVENANCE.csv", index=False)


def run_formula_tests(out: Path) -> str:
    test_path = EXPERIMENT / "test_phase10_independent_features.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(EXPERIMENT),
            "-p",
            test_path.name,
            "-v",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    (out / "INDEPENDENT_FEATURE_BUILDER_TESTS.txt").write_text(
        output, encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(
            f"Independent feature tests failed (exit {result.returncode}):\n{output}"
        )
    return output


def run_one_fold_preflight(
    train: pd.DataFrame,
    y: np.ndarray,
    raw: pd.DataFrame,
    oof_path: Path,
    folds_path: Path,
) -> list[dict[str, object]]:
    features = feature_builder.build_rowwise_features(raw)
    keys = feature_builder.build_product_keys(raw)
    category_columns = feature_builder.categorical_columns(features)
    common = pd.concat([features, keys], axis=1)
    key_categories = category_columns + list(keys.columns)
    if common.columns.duplicated().any():
        raise AssertionError("Independent feature matrix contains duplicate columns")
    if TARGET in common.columns or ID in common.columns:
        raise AssertionError("Identifier/target leaked into independent feature matrix")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
    fit_idx, valid_idx = list(splitter.split(np.zeros(len(y)), y))[FOLD_NUMBER - 1]
    saved_folds = pd.read_csv(folds_path)
    selected = saved_folds.loc[
        (saved_folds.seed == FOLD_SEED)
        & (saved_folds.fold == FOLD_NUMBER)
        & (saved_folds.variant == "cat_product_baseline"),
        "iterations_selected_inner_fit_only",
    ]
    key_selected = saved_folds.loc[
        (saved_folds.seed == FOLD_SEED)
        & (saved_folds.fold == FOLD_NUMBER)
        & (saved_folds.variant == "cat_native_8_keys"),
        "iterations_selected_inner_fit_only",
    ]
    if len(selected) != 1 or len(key_selected) != 1:
        raise AssertionError("Phase 9 reference iteration counts are missing")
    if int(selected.iloc[0]) != int(key_selected.iloc[0]):
        raise AssertionError("Phase 9 baseline and product model did not share rounds")
    rounds = int(selected.iloc[0])

    saved = pd.read_csv(
        oof_path,
        usecols=[
            ID,
            TARGET,
            f"seed{FOLD_SEED}__cat_product_baseline",
            f"seed{FOLD_SEED}__cat_native_8_keys",
        ],
    )
    if not np.array_equal(saved[ID].to_numpy(), train[ID].to_numpy()):
        raise AssertionError("Historical Phase 9 comparison is not ID-aligned")
    if not np.array_equal(saved[TARGET].to_numpy(dtype=np.int8), y):
        raise AssertionError("Historical Phase 9 comparison target is misaligned")

    params = {
        "iterations": rounds,
        "depth": 7,
        "learning_rate": 0.045,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 5.0,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "border_count": 128,
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.80,
        "thread_count": min(8, os.cpu_count() or 1),
        "random_seed": SEED_BASE + FOLD_SEED + FOLD_NUMBER,
        "allow_writing_files": False,
        "verbose": False,
    }

    output = []
    variants = (
        (
            "cat_product_baseline",
            features,
            category_columns,
        ),
        (
            "cat_native_8_keys",
            common,
            key_categories,
        ),
    )
    for variant, frame, cats in variants:
        model = CatBoostClassifier(**params)
        model.fit(
            frame.iloc[fit_idx],
            y[fit_idx],
            cat_features=cats,
            verbose=False,
        )
        prediction = model.predict_proba(frame.iloc[valid_idx])[:, 1]
        del model
        reference = saved.loc[
            valid_idx, f"seed{FOLD_SEED}__{variant}"
        ].to_numpy(dtype=np.float64)
        independent_auc = float(roc_auc_score(y[valid_idx], prediction))
        phase9_auc = float(roc_auc_score(y[valid_idx], reference))
        output.append(
            {
                "seed": FOLD_SEED,
                "fold": FOLD_NUMBER,
                "variant": variant,
                "iterations": rounds,
                "independent_builder_feature_count": int(frame.shape[1]),
                "independent_builder_categorical_count": len(cats),
                "phase9_reference_fold_auc": phase9_auc,
                "independent_builder_fold_auc": independent_auc,
                "auc_difference_independent_minus_phase9": independent_auc
                - phase9_auc,
                "prediction_spearman": float(
                    pd.Series(prediction).corr(
                        pd.Series(reference), method="spearman"
                    )
                ),
                "prediction_max_absolute_difference": float(
                    np.max(np.abs(prediction - reference))
                ),
                "interpretation": "Diagnostic only: the independent feature matrix is intentionally not claimed to match the undocumented Phase 9 matrix.",
            }
        )
    return output


def main(run_id: str) -> int:
    out = RUNS / f"phase10_zero_exp85_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    log_path = out / "EXECUTION_LOG.txt"
    started = datetime.now(timezone.utc)
    wall = time.perf_counter()

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    try:
        inputs = {
            "training_csv": TRAIN_PATH,
            "official_dictionary": DICTIONARY_PATH,
            "phase9_oof_reference": P9_OOF,
            "phase9_catboost_fold_reference": P9_FOLDS,
            "phase9_runner_source_audit": P9_RUNNER,
            "phase10_runner": Path(__file__).resolve(),
            "phase10_spec": EXPERIMENT / "PHASE9_FEATURE_BUILDER_INDEPENDENT_SPEC.md",
            "phase10_builder": EXPERIMENT / "phase10_independent_feature_builder.py",
            "phase10_tests": EXPERIMENT / "test_phase10_independent_features.py",
        }
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(f"Required permitted input is missing: {path}")

        log("status=RUNNING")
        log(f"run_id={run_id}")
        log(f"started_utc={started.isoformat()}")
        log(f"python={sys.version}")
        log(f"platform={platform.platform()}")
        log(f"gpu={gpu_snapshot()}")
        log(f"train_sha256={sha256_file(TRAIN_PATH)}")
        log(f"official_dictionary_sha256={sha256_file(DICTIONARY_PATH)}")
        log("test_file_opened=NO")
        log("external_data=NO")
        assert_no_forbidden_module_loaded()
        log("EXP85_artifact_access_by_zero_dependency_runner=NO")
        log("EXP85_source_module_loaded=NO")

        write_dependency_provenance(out, inputs)
        (out / "PHASE9_FEATURE_BUILDER_INDEPENDENT_SPEC.md").write_text(
            inputs["phase10_spec"].read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        tests_output = run_formula_tests(out)

        train = pd.read_csv(TRAIN_PATH, low_memory=False)
        dictionary = pd.read_excel(DICTIONARY_PATH)
        dictionary_fields = set(dictionary["Row"].astype(str))
        if set(train.columns) != dictionary_fields:
            raise AssertionError("Training columns differ from the official dictionary")
        if len(train) != 171202 or train[ID].isna().any() or train[ID].duplicated().any():
            raise AssertionError("Training data row/ID validation failed")
        if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
            raise AssertionError("Training target is incomplete or nonbinary")
        y = train[TARGET].to_numpy(dtype=np.int8)
        raw = train.drop(columns=[TARGET]).reset_index(drop=True)
        preflight = run_one_fold_preflight(train, y, raw, P9_OOF, P9_FOLDS)
        pd.DataFrame(preflight).to_csv(
            out / "PHASE9_PARITY_PREFLIGHT.csv", index=False
        )

        legacy_graph = """# Phase 9 historical dependency graph and Phase 10 independence

## Result

The Phase 10 runtime has **zero EXP85 source, import, cache, model-output, or
feature-matrix dependency**. It imports only the independent Phase 10 builder
and standard/installed Python libraries, reads the permitted `train.csv` and
official dictionary, and reads the saved Phase 9 OOF / fold summaries only as
the historical parity reference. Neither the Phase 9 runner nor its legacy
feature helper is imported or executed.

## Phase 9 dependency trace

The Phase 9 runner declares and loads the frozen `run_independent_baseline.py`
helper directly. It also loads Phase 3, Phase 3B, and Phase 7. Those local
modules declare or transitively load the same helper: Phase 3 uses it for the
row-wise base features; Phase 3B loads it for its feature comparisons; Phase 7
loads it alongside Phase 3 and Phase 3B for product feature construction.
Phase 9's CatBoost matrix is the helper's row-wise matrix plus Phase 7's
product block and the separately constructed product keys.

Phase 9 provenance also includes the Phase 8 runner/audit and saved Phase 8
OOF files. The Phase 9 runner reads the Phase 8 OOF predictions for a
post-hoc subgroup diagnostic; that path is not a Phase 10 model input. Phase
8 itself also imports the frozen helper, so the Phase 8 diagnostic lineage is
not independent of that helper either.

The explicit Phase 9 dependency path
`D:\\gci\\truth\\research_archive\\exp85_phase2_source_snapshot\\project\\source\\run_independent_baseline.py`
is recorded in `DEPENDENCY_PROVENANCE.csv` as **prohibited and not accessed by
Phase 10**. Its code was not opened, hashed, copied, or imported for this
zero-dependency run.

## Parity boundary

The permitted official dictionary defines the supplied fields, but does not
give the complete mathematical specification of the historical row-wise
builder. Reconstructing that feature matrix from old source or cached outputs
would violate the zero-dependency requirement. The independent builder is
therefore a newly specified transformation, not a purported Phase 9 clone.
The one-fold GPU comparison is diagnostic only; exact Phase 9 reproduction is
**BLOCKED**. No full Phase 10 ablation/grid was launched.

## Prior pre-correction action

Before this governance correction arrived, a one-fold parity probe was run
using the then-current draft, which imported the prohibited frozen helper.
That probe was not saved as a Phase 10 artifact and is excluded from all
results in this run. No full Phase 10 run was launched. The corrected runner
and results below have no such dependency.

## Validation boundary

The corrected runtime uses the training labels only for the supervised
CatBoost fit and AUC evaluation. The Phase 9 prediction file is read solely as
a historical comparator. No Phase 7 holdout, `test.csv`, sample submission,
external data, submission generation, or champion modification was used.
"""
        (out / "ZERO_EXP85_DEPENDENCY_AUDIT.md").write_text(
            legacy_graph, encoding="utf-8"
        )

        test_report = f"""# Independent feature-builder tests

The independent row-wise and product-key builders passed all formula-level
tests from `test_phase10_independent_features.py`.

Tests cover:

- Hand-calculated ratios, age conversion, missingness counters, employment
  sentinel handling, category missing-value treatment, product-key rounding,
  KQ eligibility, and invalid sentinels.
- Rejection of TARGET-bearing inputs.
- Full supplied training-column equality with the official dictionary.
- Full-row feature construction, unique columns, finite count/indicator fields,
  numeric ratio types, and nonempty categorical strings.

Test command: `python -m unittest discover -s <Phase 10 directory> -p test_phase10_independent_features.py -v`.

Captured test output:

```text
{tests_output.strip()}
```
"""
        (out / "INDEPENDENT_FEATURE_BUILDER_TESTS.md").write_text(
            test_report, encoding="utf-8"
        )

        parity = pd.DataFrame(preflight)
        parity_blocked = """# Phase 9 one-fold parity preflight

## Decision

**BLOCKED — exact Phase 9 reproduction cannot be claimed under the
zero-EXP85-dependency rule.** The allowed official dictionary documents the
raw application columns, not the complete historical row-wise feature
builder. The independent feature matrix is intentionally specified from raw
predictors and documented row-local mathematics; it is not an attempt to
recover the prohibited helper's undocumented transformations.

## Preflight performed

One outer fold (seed 42, fold 1) was fit on the GPU with the documented Phase
9 CatBoost hyperparameters, model seed, and Phase 9 selected iteration count.
The two fits were the independent-builder baseline and the same independent
features plus the eight row-local product keys. Their validation predictions
were compared with the historical Phase 9 saved OOF columns only for
diagnostic reporting. The Phase 9 outputs were not used to construct, choose,
or fit features.

Per-variant AUCs, prediction rank correlations, maximum prediction
differences, and feature counts are in `PHASE9_PARITY_PREFLIGHT.csv`. Because
the feature matrix differs by specification, score similarity would not prove
source parity; score differences do not indict either model. The exact
reproduction status remains **BLOCKED**, regardless of numerical proximity.

No Phase 9 helper was restored, imported, or read to improve the match. No
remaining Phase 10 ablation, robustness grid, or champion decision was made.
"""
        (out / "PHASE9_PARITY_PREFLIGHT_REPORT.md").write_text(
            parity_blocked, encoding="utf-8"
        )

        recommendation = """# Next-phase recommendation

Recommend exactly one next experiment: **freeze a new, fully documented
zero-EXP85 baseline from the official application dictionary and raw
training-only formulas, then run a preregistered paired five-fold CatBoost
comparison of that baseline against the independently specified eight
product-key block.**

Treat it as a new study, not a Phase 9 reproduction. Keep the feature
definitions, fold seeds, CatBoost settings, negative controls, and eligibility
boundary fixed before running; do not use the Phase 7 holdout, test rows, or
historical EXP85-derived features. No submission or champion change follows
from the current blocked parity result.
"""
        (out / "NEXT_PHASE_RECOMMENDATION.md").write_text(
            recommendation, encoding="utf-8"
        )

        log("phase9_exact_reproduction=BLOCKED")
        log("phase9_parity_probe=one fold; independent feature matrix; GPU CatBoost")
        log("phase10_full_ablation_grid=NOT_STARTED")
        log("phase7_locked_holdout_used=NO")
        log("submission_created=NO")
        log("champion_modified=NO")
        log(f"gpu_final_snapshot={gpu_snapshot()}")
        log(f"runtime_seconds={time.perf_counter() - wall:.3f}")
        log("status=COMPLETED_WITH_PARITY_BLOCKED")

        manifest_rows = []
        for path in sorted(out.rglob("*")):
            if path.is_file() and path.name != "SHA256_MANIFEST.csv":
                manifest_rows.append(
                    {
                        "file": str(path.relative_to(out)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        pd.DataFrame(manifest_rows).to_csv(
            out / "SHA256_MANIFEST.csv", index=False
        )
        return 0
    except Exception as error:
        log(f"status=FAILED; exception={type(error).__name__}; message={error}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("T", "").replace("Z", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only alphanumeric characters, T, Z, and underscore")
    raise SystemExit(main(args.run_id))
