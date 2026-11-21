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
from typing import Any

import numpy as np
import pandas as pd
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
import run_phase15_catboost_preflight as preflight
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
SECONDARY_SEEDS = (20261114, 20261115)
FOLDS = 5
VARIANT = "native_catboost_phase14_plus_time"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase15_catboost_multiseed")
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
    train_path: Path,
    reference_path: Path,
    primary_path: Path,
    output_root: Path,
) -> Path:
    for path in (train_path, reference_path, primary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run input missing: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase15_native_catboost_multiseed_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = _logger(output_dir)
    started = time.perf_counter()
    logger.info("Starting two-secondary-seed bagging experiment")
    logger.info("This reuses no saved validation predictions for fitting")

    train = pd.read_csv(train_path, low_memory=False)
    phase13_oof = pd.read_csv(reference_path)
    primary_oof = pd.read_csv(primary_path)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not (
        len(phase13_oof) == len(primary_oof) == len(train)
        and np.array_equal(
            train[phase13.ID_COLUMN].to_numpy(),
            phase13_oof[phase13.ID_COLUMN].to_numpy(),
        )
        and np.array_equal(
            train[phase13.ID_COLUMN].to_numpy(),
            primary_oof[phase13.ID_COLUMN].to_numpy(),
        )
        and np.array_equal(y, phase13_oof[phase13.TARGET_COLUMN].to_numpy())
        and np.array_equal(y, primary_oof[phase13.TARGET_COLUMN].to_numpy())
    ):
        raise ValueError("Training, Phase 13 and primary OOF alignment failed")
    primary_name = "native_catboost_phase14_plus_time"
    if primary_name not in primary_oof:
        raise ValueError(f"Primary OOF lacks selected variant: {primary_name}")
    if not np.isfinite(primary_oof[primary_name].to_numpy()).all():
        raise ValueError("Primary OOF has invalid predictions")

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
    enhancements_plus_time = pd.concat(
        [
            enhancements.reset_index(drop=True),
            time_features.reset_index(drop=True),
        ],
        axis=1,
    )
    if enhancements.columns.intersection(time_features.columns).size:
        raise ValueError("Phase 15 feature names collide with Phase 14")

    secondary_predictions: dict[int, np.ndarray] = {
        seed: np.full(len(y), np.nan, dtype=np.float32) for seed in SECONDARY_SEEDS
    }
    fold_rows: list[dict[str, Any]] = []
    for seed in SECONDARY_SEEDS:
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y), dtype=np.int8), y))
        logger.info("seed=%d", seed)
        for fold, (fit, apply) in enumerate(splits, start=1):
            blocks = phase14._feature_blocks(
                raw,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                enhancements_plus_time,
                {"augmented": True},
            )
            x_fit, x_apply, matrix_info = phase14._matrix_pair(
                fit,
                apply,
                blocks,
                phase13.PRODUCT_KEYS,
                "catboost",
            )
            model_seed = seed + fold * 100
            model = preflight._model(model_seed)
            fit_started = time.perf_counter()
            model.fit(
                x_fit,
                y[fit],
                cat_features=list(matrix_info["cat_features"]),
                verbose=100,
            )
            prediction = model.predict_proba(x_apply)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite OOF predictions for seed {seed}, fold {fold}"
                )
            secondary_predictions[seed][apply] = prediction.astype(np.float32)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "model_seed": model_seed,
                    "fit_rows": len(fit),
                    "validation_rows": len(apply),
                    "fold_auc": float(roc_auc_score(y[apply], prediction)),
                    "fit_seconds": time.perf_counter() - fit_started,
                    "feature_count": matrix_info["feature_count"],
                    "max_fit_category_count": matrix_info[
                        "max_fit_category_count"
                    ],
                    "iterations": preflight.ITERATIONS,
                    "paired_seed_across_variants": "single feature variant",
                }
            )
            logger.info(
                "seed=%d fold=%d auc=%.6f runtime_seconds=%.1f",
                seed,
                fold,
                fold_rows[-1]["fold_auc"],
                fold_rows[-1]["fit_seconds"],
            )
            del model, x_fit, x_apply, blocks

    if any(not np.isfinite(pred).all() for pred in secondary_predictions.values()):
        raise ValueError("A secondary seed has incomplete OOF predictions")
    primary = primary_oof[primary_name].to_numpy(dtype=np.float64)
    all_seeds = [primary] + [
        secondary_predictions[seed].astype(np.float64) for seed in SECONDARY_SEEDS
    ]
    bagged = np.mean(np.column_stack(all_seeds), axis=1)
    if not np.isfinite(bagged).all():
        raise ValueError("Seed-bagged OOF prediction is not finite")
    summary_rows = []
    for seed, prediction in zip((20261113, *SECONDARY_SEEDS), all_seeds):
        split_seed = seed
        fold_ids = np.full(len(y), -1, dtype=np.int8)
        for fold, (_, apply) in enumerate(
            StratifiedKFold(
                n_splits=FOLDS, shuffle=True, random_state=split_seed
            ).split(np.zeros(len(y), dtype=np.int8), y),
            start=1,
        ):
            fold_ids[apply] = fold
        fold_aucs = [
            float(
                roc_auc_score(
                    y[fold_ids == fold], prediction[fold_ids == fold]
                )
            )
            for fold in range(1, FOLDS + 1)
        ]
        summary_rows.append(
            {
                "seed": seed,
                "pooled_oof_auc": float(roc_auc_score(y, prediction)),
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "fold_auc_std": float(np.std(fold_aucs, ddof=1)),
                "fold_aucs": json.dumps(fold_aucs),
            }
        )
    summary_rows.append(
        {
            "seed": "three_seed_oof_average",
            "pooled_oof_auc": float(roc_auc_score(y, bagged)),
            "mean_fold_auc": float("nan"),
            "fold_auc_std": float("nan"),
            "fold_aucs": "Not a single fold assignment; each component is OOF",
        }
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "SECONDARY_FOLD_METRICS.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "MULTISEED_RESULTS.csv", index=False
    )
    output = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=np.int32),
            phase13.ID_COLUMN: train[phase13.ID_COLUMN].to_numpy(),
            phase13.TARGET_COLUMN: y,
            "primary_fold_seed_20261113": primary_oof["fold"].to_numpy(),
            "phase15_primary_seed_20261113": primary,
            "phase15_secondary_seed_20261114": secondary_predictions[20261114],
            "phase15_secondary_seed_20261115": secondary_predictions[20261115],
            "phase15_three_seed_average": bagged,
        }
    )
    output.to_csv(output_dir / "MULTISEED_OOF_PREDICTIONS.csv", index=False)
    integrity = {
        "rows": len(output),
        "unique_ids": int(output[phase13.ID_COLUMN].nunique()),
        "id_order_exact": bool(
            np.array_equal(
                output[phase13.ID_COLUMN].to_numpy(),
                train[phase13.ID_COLUMN].to_numpy(),
            )
        ),
        "target_order_exact": bool(np.array_equal(output[phase13.TARGET_COLUMN], y)),
        "all_seed_predictions_finite": bool(
            np.isfinite(
                output[
                    [
                        "phase15_primary_seed_20261113",
                        "phase15_secondary_seed_20261114",
                        "phase15_secondary_seed_20261115",
                        "phase15_three_seed_average",
                    ]
                ].to_numpy()
            ).all()
        ),
        "each_row_out_of_fold_in_all_three_seeds": True,
    }
    (output_dir / "OOF_INTEGRITY_REPORT.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = {
        "status": "seed robustness/bagging experiment on exposed labels",
        "training_sha256": sha256_file(train_path),
        "phase13_oof_sha256": sha256_file(reference_path),
        "primary_oof_sha256": sha256_file(primary_path),
        "primary_seed": 20261113,
        "secondary_seeds": list(SECONDARY_SEEDS),
        "model": {
            "type": "CatBoost GPU native categories",
            "iterations": preflight.ITERATIONS,
            "depth": 6,
            "learning_rate": 0.05,
            "one_hot_max_size": 10,
            "max_ctr_complexity": 1,
        },
        "results": summary_rows,
        "fold_results": fold_rows,
        "integrity": integrity,
        "runtime_seconds": time.perf_counter() - started,
        "eligibility": (
            "Internal CTR target statistics and the constants in product keys "
            "remain unresolved. This remains research-only Track B."
        ),
        "no_test_external_or_submission_data": True,
        "independent_confirmation": False,
    }
    (output_dir / "PHASE15_MULTISEED_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "Completed multi-seed experiment in %.1f seconds; bagged_auc=%.6f",
        time.perf_counter() - started,
        summary_rows[-1]["pooled_oof_auc"],
    )
    source_files = [
        Path(__file__).resolve(),
        PHASE15_DIR / "run_phase15_catboost_preflight.py",
        PHASE15_DIR / "phase15_features.py",
        PHASE14_DIR / "run_phase14.py",
        PHASE14_DIR / "phase14_features.py",
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
    ]
    _manifest(output_dir, source_files, [train_path, reference_path, primary_path])
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
