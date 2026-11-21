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
from phase15_interactions import build_organization_external_bin_cross


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_OOF = (
    PHASE13_DIR / "runs" / "phase13_20261105T075946Z" / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT_ROOT = PHASE15_DIR / "runs"
SEED = 20261113
FOLDS = 5
ITERATIONS = 700
BOOTSTRAP_REPETITIONS = 400
BASE_VARIANT = "catboost_ctr1_time_base"
CROSS_VARIANT = "catboost_ctr1_time_plus_org_external_cross"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase15_org_ext_cross_oof")
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


def _bootstrap_delta(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 101)
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


def run(train_path: Path, reference_path: Path, output_root: Path) -> Path:
    for path in (train_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run input missing: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase15_org_ext_cross_oof_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = _logger(output_dir)
    started = time.perf_counter()
    logger.info("Starting matched five-fold organization x external-risk-bin OOF")
    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(reference_path)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not (
        len(reference) == len(train)
        and np.array_equal(
            train[phase13.ID_COLUMN].to_numpy(),
            reference[phase13.ID_COLUMN].to_numpy(),
        )
        and np.array_equal(y, reference[phase13.TARGET_COLUMN].to_numpy())
    ):
        raise ValueError("Training and Phase 13 OOF alignment failed")
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y), dtype=np.int8), y))
    folds = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, apply) in enumerate(splits, start=1):
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
    predictions = {
        BASE_VARIANT: np.full(len(y), np.nan, dtype=np.float32),
        CROSS_VARIANT: np.full(len(y), np.nan, dtype=np.float32),
    }
    fold_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []

    for fold, (fit, apply) in enumerate(splits, start=1):
        cross, cuts = build_organization_external_bin_cross(raw, fit, apply)
        categorical_cross = pd.concat(
            [categorical, cross.rename(cross.name)], axis=1
        )
        base_blocks = phase14._feature_blocks(
            raw,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            enhancements,
            {"augmented": True},
        )
        cross_blocks = phase14._feature_blocks(
            raw,
            numeric,
            categorical_cross,
            rowwise,
            keys,
            decompositions,
            enhancements,
            {"augmented": True},
        )
        for variant, blocks in (
            (BASE_VARIANT, base_blocks),
            (CROSS_VARIANT, cross_blocks),
        ):
            x_fit, x_apply, info = phase14._matrix_pair(
                fit,
                apply,
                blocks,
                phase13.PRODUCT_KEYS,
                "catboost",
            )
            seed = SEED + fold * 100
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
                max_ctr_complexity=1,
                l2_leaf_reg=5.0,
                random_strength=1.0,
                random_seed=seed,
                allow_writing_files=False,
                verbose=100,
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
                raise ValueError(f"{variant} produced invalid OOF rows in fold {fold}")
            predictions[variant][apply] = prediction.astype(np.float32)
            fold_rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "seed": seed,
                    "fit_rows": len(fit),
                    "validation_rows": len(apply),
                    "fold_auc": float(roc_auc_score(y[apply], prediction)),
                    "fit_seconds": time.perf_counter() - fit_started,
                    "feature_count": info["feature_count"],
                    "categorical_feature_count": len(info["cat_features"]),
                    "max_fit_category_count": info["max_fit_category_count"],
                    "peak_working_set_mb": phase13.peak_working_set_mb(),
                }
            )
            logger.info(
                "%s fold=%d auc=%.6f runtime_seconds=%.1f",
                variant,
                fold,
                fold_rows[-1]["fold_auc"],
                fold_rows[-1]["fit_seconds"],
            )
            del model, x_fit, x_apply, blocks
        support_rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit),
                "validation_rows": len(apply),
                "fit_fold_quantile_cuts": json.dumps(cuts.tolist()),
                "fit_cross_category_count": int(
                    categorical_cross.iloc[fit][cross.name].nunique()
                ),
                "validation_cross_category_count": int(
                    categorical_cross.iloc[apply][cross.name].nunique()
                ),
            }
        )

    if any(not np.isfinite(pred).all() for pred in predictions.values()):
        raise ValueError("Incomplete or non-finite cross-model OOF predictions")
    baseline = predictions[BASE_VARIANT].astype(np.float64)
    candidate = predictions[CROSS_VARIANT].astype(np.float64)
    paired_deltas = []
    for fold in range(1, FOLDS + 1):
        mask = folds == fold
        base_auc = float(roc_auc_score(y[mask], baseline[mask]))
        cross_auc = float(roc_auc_score(y[mask], candidate[mask]))
        paired_deltas.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "cross_auc": cross_auc,
                "delta": cross_auc - base_auc,
                "cross_is_better": cross_auc > base_auc,
            }
        )
    results = {
        "baseline_pooled_auc": float(roc_auc_score(y, baseline)),
        "cross_pooled_auc": float(roc_auc_score(y, candidate)),
        "pooled_auc_delta": float(
            roc_auc_score(y, candidate) - roc_auc_score(y, baseline)
        ),
        "positive_fold_count": int(
            sum(row["cross_is_better"] for row in paired_deltas)
        ),
        "fold_deltas": paired_deltas,
        "spearman_correlation": float(
            pd.Series(baseline).corr(pd.Series(candidate), method="spearman")
        ),
        "paired_bootstrap": _bootstrap_delta(y, baseline, candidate),
        "runtime_seconds": time.perf_counter() - started,
        "status": "controlled development OOF on previously exposed labels",
        "eligibility": (
            "The category cross is label-free; quantile cut points are fit only "
            "on the outer training fold. Product-key constants and CatBoost "
            "native CTR eligibility remain unresolved."
        ),
    }
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=np.int32),
            phase13.ID_COLUMN: train[phase13.ID_COLUMN].to_numpy(),
            phase13.TARGET_COLUMN: y,
            "fold": folds,
            **predictions,
        }
    )
    oof.to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "FOLD_LEVEL_METRICS.csv", index=False
    )
    pd.DataFrame(support_rows).to_csv(
        output_dir / "FOLD_FITTED_CATEGORY_SUPPORT.csv", index=False
    )
    (output_dir / "PHASE15_INTERACTION_RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    integrity = {
        "rows": len(oof),
        "id_order_exact": bool(
            np.array_equal(
                oof[phase13.ID_COLUMN].to_numpy(),
                train[phase13.ID_COLUMN].to_numpy(),
            )
        ),
        "target_order_exact": bool(np.array_equal(oof[phase13.TARGET_COLUMN], y)),
        "unique_ids": int(oof[phase13.ID_COLUMN].nunique()),
        "fold_counts": {
            str(fold): int(np.sum(folds == fold)) for fold in range(1, FOLDS + 1)
        },
        "predictions_finite_and_complete": bool(
            np.isfinite(oof[[BASE_VARIANT, CROSS_VARIANT]].to_numpy()).all()
        ),
    }
    (output_dir / "OOF_INTEGRITY_REPORT.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "completed pooled_delta=%.6f positive_folds=%d/5 runtime_seconds=%.1f",
        results["pooled_auc_delta"],
        results["positive_fold_count"],
        time.perf_counter() - started,
    )
    source_files = [
        Path(__file__).resolve(),
        PHASE15_DIR / "phase15_interactions.py",
        PHASE15_DIR / "phase15_features.py",
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
        PHASE14_DIR / "run_phase14.py",
        PHASE14_DIR / "phase14_features.py",
    ]
    _manifest(output_dir, source_files, [train_path, reference_path])
    for handler in logger.handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, default=DEFAULT_PHASE13_OOF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(
        run(
            args.train.resolve(),
            args.phase13_oof.resolve(),
            args.output_root.resolve(),
        )
    )


if __name__ == "__main__":
    main()
