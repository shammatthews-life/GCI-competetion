from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
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
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE13_DIR))
import run_phase14 as phase14
import run_phase13 as phase13
import run_phase15_catboost_preflight as preflight
from phase15_features import build_time_lifecycle_features


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_OOF = (
    PHASE13_DIR / "runs" / "phase13_20261105T075946Z" / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
SEED = 20261113
FOLDS = 5
BOOTSTRAP_REPETITIONS = 1000
VARIANTS = (
    "native_catboost_base_all8",
    "native_catboost_phase14_numeric",
    "native_catboost_phase14_plus_time",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase15_catboost_oof")
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


def _nvidia_smi() -> str:
    result = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def _bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sample = np.concatenate(
            (
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            )
        )
        deltas[index] = roc_auc_score(
            y[sample], candidate[sample]
        ) - roc_auc_score(y[sample], baseline[sample])
    return {
        "paired_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "paired_delta_ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    }


def _manifest(
    output_dir: Path, source_files: list[Path], input_files: list[Path]
) -> None:
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
        "phase15_native_catboost_oof_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = configure_logger(output_dir)
    started = time.perf_counter()
    gpu_before = _nvidia_smi()
    (output_dir / "GPU_DEVICE_INFO_BEFORE.txt").write_text(
        gpu_before, encoding="utf-8"
    )
    logger.info("Starting matched 5-fold native-category GPU CatBoost OOF")
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
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(
        splitter.split(np.zeros(len(y), dtype=np.int8), y)
    )
    for fold, (_, apply) in enumerate(fold_splits, start=1):
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
    variant_blocks = {
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
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float32) for name in VARIANTS
    }
    fit_rows: list[dict[str, Any]] = []
    logger.info(
        "rows=%d positives=%d folds=%d seed=%d iterations=%d",
        len(y),
        int(y.sum()),
        FOLDS,
        SEED,
        preflight.ITERATIONS,
    )

    for fold, (fit, apply) in enumerate(fold_splits, start=1):
        fold_started = time.perf_counter()
        logger.info(
            "fold=%d fit_rows=%d valid_rows=%d", fold, len(fit), len(apply)
        )
        for name in VARIANTS:
            variant, enhancement_block = variant_blocks[name]
            blocks = phase14._feature_blocks(
                raw,
                numeric,
                categorical,
                rowwise,
                keys,
                decompositions,
                enhancement_block,
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
            model_seed = SEED + fold * 100
            model = preflight._model(model_seed)
            model_started = time.perf_counter()
            model.fit(
                x_fit,
                y[fit],
                cat_features=cat_features,
                verbose=100,
            )
            prediction = model.predict_proba(x_apply)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"{name} produced non-finite predictions in fold {fold}"
                )
            predictions[name][apply] = prediction.astype(np.float32)
            fold_auc = float(roc_auc_score(y[apply], prediction))
            fit_rows.append(
                {
                    "variant": name,
                    "fold": fold,
                    "seed": model_seed,
                    "iterations": preflight.ITERATIONS,
                    "fit_rows": len(fit),
                    "validation_rows": len(apply),
                    "fold_auc": fold_auc,
                    "fit_seconds": time.perf_counter() - model_started,
                    "feature_count": matrix_info["feature_count"],
                    "categorical_feature_count": len(cat_features),
                    "max_fit_category_count": matrix_info[
                        "max_fit_category_count"
                    ],
                    "catboost_device": "GPU",
                    "one_hot_max_size": 10,
                    "max_ctr_complexity": 1,
                    "paired_seed_across_variants": True,
                    "peak_working_set_mb": phase13.peak_working_set_mb(),
                    "eligibility": "unresolved native CTRs; research-only Track B",
                }
            )
            logger.info(
                "%s fold=%d auc=%.6f runtime_seconds=%.1f",
                name,
                fold,
                fold_auc,
                time.perf_counter() - model_started,
            )
            del model, x_fit, x_apply, blocks
        logger.info(
            "fold=%d completed runtime_seconds=%.1f",
            fold,
            time.perf_counter() - fold_started,
        )

    if any(not np.isfinite(prediction).all() for prediction in predictions.values()):
        raise ValueError("At least one candidate has incomplete OOF predictions")
    phase13_e = reference["E_C_plus_keys_frequency_ohe"].to_numpy(dtype=float)
    result_rows = []
    for name in VARIANTS:
        pred = predictions[name].astype(np.float64)
        fold_auc = [
            float(roc_auc_score(y[folds == fold], pred[folds == fold]))
            for fold in range(1, FOLDS + 1)
        ]
        result_rows.append(
            {
                "variant": name,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "mean_fold_auc": float(np.mean(fold_auc)),
                "fold_auc_std": float(np.std(fold_auc, ddof=1)),
                "fold1_auc": fold_auc[0],
                "fold2_auc": fold_auc[1],
                "fold3_auc": fold_auc[2],
                "fold4_auc": fold_auc[3],
                "fold5_auc": fold_auc[4],
                "spearman_vs_phase13_E": float(
                    pd.Series(pred).corr(pd.Series(phase13_e), method="spearman")
                ),
                "auc_delta_vs_phase13_E": float(
                    roc_auc_score(y, pred) - roc_auc_score(y, phase13_e)
                ),
            }
        )
    base = predictions["native_catboost_base_all8"].astype(np.float64)
    for row in result_rows:
        candidate = predictions[row["variant"]].astype(np.float64)
        row["auc_delta_vs_native_base"] = float(
            roc_auc_score(y, candidate) - roc_auc_score(y, base)
        )
        row.update(
            _bootstrap_delta(
                y,
                base,
                candidate,
                seed=SEED + 2025,
            )
        )
    pd.DataFrame(fit_rows).to_csv(output_dir / "FOLD_LEVEL_METRICS.csv", index=False)
    pd.DataFrame(result_rows).to_csv(output_dir / "MODEL_RESULTS.csv", index=False)
    correlations = []
    names = list(predictions)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            correlations.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": float(
                        np.corrcoef(predictions[left], predictions[right])[0, 1]
                    ),
                    "spearman": float(
                        pd.Series(predictions[left]).corr(
                            pd.Series(predictions[right]), method="spearman"
                        )
                    ),
                }
            )
    pd.DataFrame(correlations).to_csv(
        output_dir / "PREDICTION_CORRELATIONS.csv", index=False
    )
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=np.int32),
            phase13.ID_COLUMN: train[phase13.ID_COLUMN].to_numpy(),
            phase13.TARGET_COLUMN: y,
            "fold": folds,
            "phase13_E_C_plus_keys_frequency_ohe": phase13_e,
            **predictions,
        }
    )
    oof.to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    integrity = {
        "rows": len(y),
        "unique_ids": int(train[phase13.ID_COLUMN].nunique()),
        "id_order_exact": bool(
            np.array_equal(
                train[phase13.ID_COLUMN].to_numpy(),
                oof[phase13.ID_COLUMN].to_numpy(),
            )
        ),
        "target_order_exact": bool(
            np.array_equal(y, oof[phase13.TARGET_COLUMN].to_numpy())
        ),
        "fold_rows": {
            str(fold): int(np.sum(folds == fold)) for fold in range(1, FOLDS + 1)
        },
        "all_prediction_values_finite": bool(
            np.isfinite(oof[list(VARIANTS)].to_numpy()).all()
        ),
        "no_missing_oof_predictions": bool(oof[list(VARIANTS)].notna().all().all()),
        "fold_seed": SEED,
    }
    (output_dir / "OOF_INTEGRITY_REPORT.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    gpu_after = _nvidia_smi()
    (output_dir / "GPU_DEVICE_INFO.txt").write_text(
        gpu_after, encoding="utf-8"
    )
    report = {
        "status": "controlled development OOF; prior label exposure means not independent confirmation",
        "training_sha256": sha256_file(train_path),
        "phase13_oof_sha256": sha256_file(reference_path),
        "fold_seed": SEED,
        "folds": FOLDS,
        "config": {
            "model": "CatBoost GPU native categories",
            "iterations": preflight.ITERATIONS,
            "depth": 6,
            "learning_rate": 0.05,
            "one_hot_max_size": 10,
            "max_ctr_complexity": 1,
            "matched_seed_across_feature_variants_within_fold": True,
        },
        "results": result_rows,
        "fold_fit_results": fit_rows,
        "integrity": integrity,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "eligibility": (
            "Raw and Phase 14 numeric features are deterministic rowwise transforms. "
            "Product-key constants 130.5 and 0.0066 are unresolved; CatBoost CTR "
            "target-statistics eligibility is also unresolved. Track B only."
        ),
        "no_test_external_or_submission_data": True,
    }
    (output_dir / "PHASE15_CATBOOST_OOF_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    source_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("run_phase15_catboost_preflight.py"),
        Path(__file__).with_name("phase15_features.py"),
        PHASE14_DIR / "run_phase14.py",
        PHASE14_DIR / "phase14_features.py",
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
    ]
    _manifest(output_dir, source_files, [train_path, reference_path])
    logger.info(
        "Completed five-fold OOF in %.1f seconds; results=%s",
        time.perf_counter() - started,
        output_dir / "MODEL_RESULTS.csv",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, default=DEFAULT_PHASE13_OOF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(
        args.train.resolve(),
        args.phase13_oof.resolve(),
        args.output_root.resolve(),
    )
    print(result)


if __name__ == "__main__":
    main()
