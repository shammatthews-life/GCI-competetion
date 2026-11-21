from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PHASE13_DIR = Path(__file__).resolve().parents[1] / "phase13_matched_baseline"
PHASE14_DIR = Path(__file__).resolve().parents[1] / "phase14_label_free_model_development"
PHASE14_RUN = PHASE14_DIR / "run_phase14.py"
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE13_DIR))
import run_phase14 as phase14
import run_phase13 as phase13
from phase15_features import build_time_lifecycle_features


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_OOF = (
    PHASE13_DIR / "runs" / "phase13_20261105T075946Z" / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
SEED = 20261113
FOLD = 1
ITERATIONS = 700


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase15_catboost_preflight")
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
    return logger


def _model(variant_seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
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
        max_ctr_complexity=1,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=variant_seed,
        allow_writing_files=False,
        verbose=100,
    )


def _manifest(output_dir: Path, source_files: list[Path], input_files: list[Path]) -> None:
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
    for kind, paths in (("source", source_files), ("input", input_files)):
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


def run(train_path: Path, reference_path: Path, output_root: Path) -> Path:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Phase 13 reference OOF does not exist: {reference_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase15_native_catboost_preflight_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(output_dir)
    started = time.perf_counter()
    logger.info("Starting one-fold native-category GPU CatBoost preflight")
    logger.info("train=%s", train_path)
    logger.info("reference=%s", reference_path)
    logger.info("Only the supplied train file and saved Phase 13 OOF are read")

    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(reference_path)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if len(reference) != len(train):
        raise ValueError("Phase 13 OOF length does not match training rows")
    if not np.array_equal(reference["row_index"].to_numpy(), np.arange(len(train))):
        raise ValueError("Phase 13 OOF row index does not match train order")
    if not np.array_equal(
        reference[phase13.ID_COLUMN].to_numpy(),
        train[phase13.ID_COLUMN].to_numpy(),
    ):
        raise ValueError("Phase 13 OOF IDs do not match train order")
    if not np.array_equal(reference[phase13.TARGET_COLUMN].to_numpy(), y):
        raise ValueError("Phase 13 OOF target does not match train target")
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, apply) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        folds[apply] = fold
    if not np.array_equal(reference["primary_fold"].to_numpy(), folds):
        raise ValueError("Phase 13 folds do not match the preregistered seed")

    raw = train.drop(columns=[phase13.TARGET_COLUMN]).copy()
    numeric, categorical, rowwise = phase13.build_raw_feature_blocks(raw)
    keys = phase13.build_product_keys(raw)
    decompositions = phase13.build_label_free_decompositions(raw)
    enhancements = phase14.build_numeric_enhancements(
        raw.drop(columns=[phase13.ID_COLUMN])
    )
    time_features = build_time_lifecycle_features(raw.drop(columns=[phase13.ID_COLUMN]))
    if enhancements.columns.intersection(time_features.columns).size:
        raise ValueError("Phase 15 feature names collide with Phase 14")
    enhancements_plus_time = pd.concat(
        [enhancements.reset_index(drop=True), time_features.reset_index(drop=True)],
        axis=1,
    )
    fit = np.flatnonzero(folds != FOLD)
    apply = np.flatnonzero(folds == FOLD)
    variants: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {
        "native_catboost_base_all8": (
            {"augmented": False},
            enhancements,
        ),
        "native_catboost_phase14_numeric": (
            {"augmented": True},
            enhancements,
        ),
        "native_catboost_phase14_plus_time": (
            {"augmented": True},
            enhancements_plus_time,
        ),
    }

    output_rows = []
    prediction_columns: dict[str, np.ndarray] = {}
    logger.info(
        "fold=%d fit_rows=%d valid_rows=%d seed=%d GPU CatBoost iterations=%d",
        FOLD,
        len(fit),
        len(apply),
        SEED,
        ITERATIONS,
    )
    for index, (name, (variant, numeric_enhancements)) in enumerate(
        variants.items(), start=1
    ):
        blocks = phase14._feature_blocks(
            raw,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            numeric_enhancements,
            variant,
        )
        x_fit, x_apply, matrix_info = phase14._matrix_pair(
            fit,
            apply,
            blocks,
            phase13.PRODUCT_KEYS,
            "catboost",
        )
        cat_features = list(matrix_info["cat_features"])
        logger.info(
            "fit %s: numeric+frequency=%d categorical=%d max_fit_category=%d",
            name,
            x_fit.shape[1] - len(cat_features),
            len(cat_features),
            matrix_info["max_fit_category_count"],
        )
        model = _model(SEED + FOLD * 100 + index)
        model_started = time.perf_counter()
        model.fit(
            x_fit,
            y[fit],
            cat_features=cat_features,
            verbose=100,
        )
        prediction = model.predict_proba(x_apply)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} produced non-finite validation predictions")
        auc = float(roc_auc_score(y[apply], prediction))
        elapsed = time.perf_counter() - model_started
        prediction_columns[name] = prediction
        output_rows.append(
            {
                "variant": name,
                "outer_fold": FOLD,
                "seed": SEED + FOLD * 100 + index,
                "iterations": ITERATIONS,
                "fit_rows": len(fit),
                "validation_rows": len(apply),
                "auc": auc,
                "fit_seconds": elapsed,
                "feature_count": matrix_info["feature_count"],
                "categorical_feature_count": len(cat_features),
                "max_fit_category_count": matrix_info["max_fit_category_count"],
                "catboost_device": "GPU",
                "catboost_one_hot_max_size": 10,
                "catboost_max_ctr_complexity": 1,
                "eligibility": "CTR use unresolved; research-only Track B",
            }
        )
        logger.info("%s fold_auc=%.6f runtime_seconds=%.1f", name, auc, elapsed)
        del model, x_fit, x_apply

    phase13_e = reference.loc[apply, "E_C_plus_keys_frequency_ohe"].to_numpy(float)
    phase14_cat = pd.read_csv(
        PHASE14_DIR
        / "runs"
        / "phase14_20261105T114015Z"
        / "OOF_PREDICTIONS.csv",
        usecols=["catboost_all8_augmented"],
    ).iloc[apply]["catboost_all8_augmented"].to_numpy(float)
    for row in output_rows:
        row["paired_delta_vs_phase13_E_fold"] = row["auc"] - float(
            roc_auc_score(y[apply], phase13_e)
        )
        row["phase13_E_prediction_spearman"] = float(
            pd.Series(prediction_columns[row["variant"]]).corr(
                pd.Series(phase13_e), method="spearman"
            )
        )
        row["phase14_onehot_catboost_fold_auc"] = float(
            roc_auc_score(y[apply], phase14_cat)
        )
    pd.DataFrame(output_rows).to_csv(
        output_dir / "PREFLIGHT_RESULTS.csv", index=False
    )
    predictions = pd.DataFrame(
        {
            "row_index": apply,
            phase13.ID_COLUMN: train.iloc[apply][phase13.ID_COLUMN].to_numpy(),
            phase13.TARGET_COLUMN: y[apply],
            "fold": FOLD,
            **prediction_columns,
        }
    )
    predictions.to_csv(output_dir / "PREFLIGHT_OOF.csv", index=False)
    report = {
        "status": "one-fold controlled preflight; exploratory only",
        "train_rows": len(train),
        "training_sha256": sha256_file(train_path),
        "phase13_oof_sha256": sha256_file(reference_path),
        "phase13_fold_seed": SEED,
        "fold": FOLD,
        "fit_rows": len(fit),
        "validation_rows": len(apply),
        "target_rate": float(y.mean()),
        "models": output_rows,
        "phase13_E_fold_auc": float(roc_auc_score(y[apply], phase13_e)),
        "phase14_onehot_catboost_fold_auc": float(
            roc_auc_score(y[apply], phase14_cat)
        ),
        "runtime_seconds": time.perf_counter() - started,
        "eligibility": (
            "Product-key constants are unresolved; GPU CatBoost native CTR "
            "target-statistics eligibility is unresolved. Not a Track A or "
            "competition-ready candidate."
        ),
        "no_test_external_or_submission_data": True,
    }
    (output_dir / "PREFLIGHT_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("phase15_features.py"),
        PHASE14_RUN,
        PHASE14_DIR / "phase14_features.py",
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
    ]
    _manifest(
        output_dir,
        source_files,
        [train_path, reference_path],
    )
    logger.info("Completed one-fold preflight in %.1f seconds", time.perf_counter() - started)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, default=DEFAULT_PHASE13_OOF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(args.train.resolve(), args.phase13_oof.resolve(), args.output_root.resolve())
    print(result)


if __name__ == "__main__":
    main()
