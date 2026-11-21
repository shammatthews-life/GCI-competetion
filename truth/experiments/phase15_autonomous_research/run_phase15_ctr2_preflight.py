from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PHASE13_DIR = Path(__file__).resolve().parents[1] / "phase13_matched_baseline"
PHASE14_DIR = Path(__file__).resolve().parents[1] / "phase14_label_free_model_development"
PHASE15_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE13_DIR))
sys.path.insert(0, str(PHASE15_DIR))
import run_phase13 as phase13
import run_phase14 as phase14
from phase15_features import build_time_lifecycle_features


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_OOF = (
    PHASE13_DIR / "runs" / "phase13_20261105T075946Z" / "OOF_PREDICTIONS.csv"
)
DEFAULT_PRIMARY_OOF = (
    PHASE15_DIR
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT_ROOT = PHASE15_DIR / "runs"
SEED = 20261113
FOLD = 1
ITERATIONS = 700


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(output_dir: Path, sources: list[Path], inputs: list[Path]) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            rows.append(
                {
                    "kind": "output",
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for kind, paths in (("source", sources), ("input", inputs)):
        for path in paths:
            rows.append(
                {
                    "kind": kind,
                    "file": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with (output_dir / "SHA256_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("kind", "file", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    train_path: Path, reference_path: Path, primary_path: Path, output_root: Path
) -> Path:
    for path in (train_path, reference_path, primary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run input missing: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase15_ctr2_preflight_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = logging.getLogger("phase15_ctr2_preflight")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    for handler in (
        logging.FileHandler(output_dir / "EXECUTION_LOG.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    started = time.perf_counter()
    logger.info("Starting matched-fold CatBoost CTR-complexity-2 preflight")
    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(reference_path)
    primary = pd.read_csv(primary_path)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not (
        len(reference) == len(primary) == len(train)
        and np.array_equal(
            train[phase13.ID_COLUMN].to_numpy(),
            reference[phase13.ID_COLUMN].to_numpy(),
        )
        and np.array_equal(
            train[phase13.ID_COLUMN].to_numpy(),
            primary[phase13.ID_COLUMN].to_numpy(),
        )
        and np.array_equal(y, reference[phase13.TARGET_COLUMN].to_numpy())
        and np.array_equal(y, primary[phase13.TARGET_COLUMN].to_numpy())
    ):
        raise ValueError("Training and saved prediction alignment failed")
    folds = np.full(len(y), -1, dtype=np.int8)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    split_rows = list(split.split(np.zeros(len(y), dtype=np.int8), y))
    for fold, (_, apply) in enumerate(split_rows, start=1):
        folds[apply] = fold
    if not np.array_equal(reference["primary_fold"].to_numpy(), folds):
        raise ValueError("Phase 13 fold assignment mismatch")

    raw = train.drop(columns=[phase13.TARGET_COLUMN]).copy()
    numeric, categorical, rowwise = phase13.build_raw_feature_blocks(raw)
    keys = phase13.build_product_keys(raw)
    decompositions = phase13.build_label_free_decompositions(raw)
    enhancements = phase14.build_numeric_enhancements(
        raw.drop(columns=[phase13.ID_COLUMN])
    )
    time_features = build_time_lifecycle_features(
        raw.drop(columns=[phase13.ID_COLUMN])
    )
    enhancements = pd.concat(
        [
            enhancements.reset_index(drop=True),
            time_features.reset_index(drop=True),
        ],
        axis=1,
    )
    fit, apply = split_rows[FOLD - 1]
    blocks = phase14._feature_blocks(
        raw,
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        enhancements,
        {"augmented": True},
    )
    x_fit, x_apply, info = phase14._matrix_pair(
        fit,
        apply,
        blocks,
        phase13.PRODUCT_KEYS,
        "catboost",
    )
    model_seed = SEED + FOLD * 100
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.78,
        border_count=128,
        one_hot_max_size=10,
        max_ctr_complexity=2,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=model_seed,
        allow_writing_files=False,
        verbose=100,
    )
    logger.info(
        "fit_rows=%d valid_rows=%d feature_count=%d categorical_count=%d seed=%d",
        len(fit),
        len(apply),
        info["feature_count"],
        len(info["cat_features"]),
        model_seed,
    )
    fit_started = time.perf_counter()
    model.fit(
        x_fit,
        y[fit],
        cat_features=list(info["cat_features"]),
        verbose=100,
    )
    prediction = model.predict_proba(x_apply)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("CTR-complexity-2 model returned invalid predictions")
    auc = float(roc_auc_score(y[apply], prediction))
    primary_prediction = primary.loc[apply, "native_catboost_phase14_plus_time"].to_numpy(
        dtype=float
    )
    baseline_auc = float(roc_auc_score(y[apply], primary_prediction))
    row = {
        "variant": "native_catboost_phase14_plus_time_ctr_complexity_2",
        "fold": FOLD,
        "seed": model_seed,
        "iterations": ITERATIONS,
        "fit_rows": len(fit),
        "validation_rows": len(apply),
        "auc": auc,
        "ctr1_baseline_auc": baseline_auc,
        "paired_delta_vs_ctr1": auc - baseline_auc,
        "fit_seconds": time.perf_counter() - fit_started,
        "feature_count": info["feature_count"],
        "max_fit_category_count": info["max_fit_category_count"],
        "catboost_device": "GPU",
        "max_ctr_complexity": 2,
        "eligibility": "native CTR eligibility unresolved; research-only",
    }
    pd.DataFrame([row]).to_csv(output_dir / "PREFLIGHT_RESULTS.csv", index=False)
    pd.DataFrame(
        {
            "row_index": apply,
            phase13.ID_COLUMN: train.iloc[apply][phase13.ID_COLUMN].to_numpy(),
            phase13.TARGET_COLUMN: y[apply],
            "fold": FOLD,
            "ctr_complexity_1": primary_prediction,
            "ctr_complexity_2": prediction,
        }
    ).to_csv(output_dir / "PREFLIGHT_OOF.csv", index=False)
    report = {
        "status": "one-fold model hyperparameter preflight only",
        "training_sha256": sha256_file(train_path),
        "phase13_oof_sha256": sha256_file(reference_path),
        "primary_oof_sha256": sha256_file(primary_path),
        "result": row,
        "runtime_seconds": time.perf_counter() - started,
        "eligibility": (
            "CatBoost CTR uses training labels internally and remains "
            "unresolved; product-key constants remain unresolved."
        ),
        "no_test_external_or_submission_data": True,
    }
    (output_dir / "PREFLIGHT_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "fold_auc=%.6f ctr1_auc=%.6f delta=%.6f runtime_seconds=%.1f",
        auc,
        baseline_auc,
        auc - baseline_auc,
        time.perf_counter() - started,
    )
    sources = [
        Path(__file__).resolve(),
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
        PHASE14_DIR / "run_phase14.py",
        PHASE14_DIR / "phase14_features.py",
        PHASE15_DIR / "phase15_features.py",
    ]
    _manifest(
        output_dir,
        sources,
        [train_path, reference_path, primary_path],
    )
    for handler in logger.handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, default=DEFAULT_PHASE13_OOF)
    parser.add_argument("--primary-oof", type=Path, default=DEFAULT_PRIMARY_OOF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(
        run(
            args.train.resolve(),
            args.phase13_oof.resolve(),
            args.primary_oof.resolve(),
            args.output_root.resolve(),
        )
    )


if __name__ == "__main__":
    main()
