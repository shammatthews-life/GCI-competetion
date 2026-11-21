from __future__ import annotations

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


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PHASE13_DIR = REPO / "experiments" / "phase13_matched_baseline"
PHASE14_DIR = REPO / "experiments" / "phase14_label_free_model_development"
PHASE15_DIR = REPO / "experiments" / "phase15_autonomous_research"
sys.path.insert(0, str(PHASE13_DIR))
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE15_DIR))
import run_phase13 as phase13
import run_phase14 as phase14
from phase15_features import build_time_lifecycle_features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
REFERENCE_PATH = (
    PHASE13_DIR / "runs" / "phase13_20261105T075946Z" / "OOF_PREDICTIONS.csv"
)
PHASE15_OOF_PATH = (
    PHASE15_DIR
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
OUTER_SEED = 20261113
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 20261601

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "historical_d6_lr005_i700",
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 700,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "name": "shallower_d5_lr005_i850",
        "depth": 5,
        "learning_rate": 0.05,
        "iterations": 850,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
    },
    {
        "name": "deeper_d7_lr004_i800",
        "depth": 7,
        "learning_rate": 0.04,
        "iterations": 800,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.0,
    },
    {
        "name": "regularized_d6_lr0035_i1000",
        "depth": 6,
        "learning_rate": 0.035,
        "iterations": 1000,
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(config: dict[str, Any], seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.78,
        border_count=128,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase16_nested_catboost")
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


def feature_matrix(
    fit: np.ndarray,
    apply: np.ndarray,
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rowwise: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    enhancements: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
    return x_fit, x_apply, info


def _manifest(output_dir: Path, inputs: tuple[Path, ...]) -> None:
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
    for path in inputs:
        rows.append(
            {
                "kind": "input",
                "file": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for path in (
        Path(__file__).resolve(),
        ROOT / "PHASE16_PROTOCOL.md",
        PHASE13_DIR / "run_phase13.py",
        PHASE14_DIR / "run_phase14.py",
        PHASE15_DIR / "phase15_features.py",
    ):
        rows.append(
            {
                "kind": "source",
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


def paired_bootstrap(
    y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sample = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        deltas[index] = roc_auc_score(
            y[sample], candidate[sample]
        ) - roc_auc_score(y[sample], baseline[sample])
    return {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "paired_delta_ci95_high": float(np.quantile(deltas, 0.975)),
    }


def run() -> Path:
    for path in (TRAIN_PATH, REFERENCE_PATH, PHASE15_OOF_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    run_name = datetime.now(timezone.utc).strftime(
        "phase16_nested_catboost_%Y%m%dT%H%M%SZ"
    )
    output_dir = ROOT / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logger(output_dir)
    started = time.perf_counter()
    logger.info("Starting nested GPU CatBoost hyperparameter comparison")
    logger.info("Outer folds are frozen to Phase 13 seed %d", OUTER_SEED)
    logger.info("Four fixed parameter sets; 3 inner folds per outer fold")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    reference = pd.read_csv(REFERENCE_PATH)
    prior_oof = pd.read_csv(PHASE15_OOF_PATH)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    ids = train[phase13.ID_COLUMN].to_numpy()
    expected_index = np.arange(len(train))
    if not (
        len(reference) == len(prior_oof) == len(train)
        and np.array_equal(reference["row_index"].to_numpy(), expected_index)
        and np.array_equal(prior_oof["row_index"].to_numpy(), expected_index)
        and np.array_equal(reference[phase13.ID_COLUMN].to_numpy(), ids)
        and np.array_equal(prior_oof[phase13.ID_COLUMN].to_numpy(), ids)
        and np.array_equal(reference[phase13.TARGET_COLUMN].to_numpy(), y)
        and np.array_equal(prior_oof[phase13.TARGET_COLUMN].to_numpy(), y)
    ):
        raise ValueError("Train and saved OOF references do not align by row/ID/target")

    folds = np.full(len(y), -1, dtype=np.int8)
    outer_splitter = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    outer_splits = list(
        outer_splitter.split(np.zeros(len(y), dtype=np.int8), y)
    )
    for fold, (_, apply) in enumerate(outer_splits, start=1):
        folds[apply] = fold
    if not np.array_equal(reference["primary_fold"].to_numpy(), folds):
        raise ValueError("The frozen Phase 13 primary fold assignment changed")
    if not np.array_equal(prior_oof["fold"].to_numpy(), folds):
        raise ValueError("The Phase 15 reference uses different outer folds")

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
        [enhancements.reset_index(drop=True), time_features.reset_index(drop=True)],
        axis=1,
    )
    if not enhancements.columns.is_unique:
        raise ValueError("Feature builder produced duplicate enhanced feature names")

    baseline = prior_oof[
        "native_catboost_phase14_plus_time"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("Saved Phase 15 reference has non-finite predictions")
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    selection_rows: list[dict[str, Any]] = []
    outer_rows: list[dict[str, Any]] = []
    inner_seed_base = 20261600

    for outer_fold, (outer_fit, outer_apply) in enumerate(outer_splits, start=1):
        logger.info(
            "Outer fold %d/%d: fit=%d validation=%d",
            outer_fold,
            OUTER_FOLDS,
            len(outer_fit),
            len(outer_apply),
        )
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=inner_seed_base + outer_fold,
        )
        inner_splits = [
            (outer_fit[fit], outer_fit[valid])
            for fit, valid in inner_splitter.split(
                np.zeros(len(outer_fit), dtype=np.int8), y[outer_fit]
            )
        ]
        candidate_scores: list[tuple[float, dict[str, Any], list[float]]] = []
        for config_index, config in enumerate(CONFIGS):
            inner_oof = np.full(len(outer_fit), np.nan, dtype=np.float64)
            inner_fold_scores = []
            inner_started = time.perf_counter()
            for inner_fold, (inner_fit, inner_apply) in enumerate(
                inner_splits, start=1
            ):
                x_fit, x_apply, info = feature_matrix(
                    inner_fit,
                    inner_apply,
                    raw,
                    numeric,
                    categorical,
                    rowwise,
                    keys,
                    decompositions,
                    enhancements,
                )
                model = make_model(
                    config,
                    seed=OUTER_SEED + outer_fold * 1000 + config_index * 10 + inner_fold,
                )
                model.fit(
                    x_fit,
                    y[inner_fit],
                    cat_features=info["cat_features"],
                    verbose=False,
                )
                prediction = model.predict_proba(x_apply)[:, 1]
                if not np.isfinite(prediction).all():
                    raise ValueError(
                        f"Non-finite inner predictions: outer={outer_fold}, "
                        f"config={config['name']}, inner={inner_fold}"
                    )
                positions = np.searchsorted(outer_fit, inner_apply)
                inner_oof[positions] = prediction
                fold_auc = float(roc_auc_score(y[inner_apply], prediction))
                inner_fold_scores.append(fold_auc)
                logger.info(
                    "outer=%d config=%s inner=%d auc=%.6f features=%d cat=%d",
                    outer_fold,
                    config["name"],
                    inner_fold,
                    fold_auc,
                    info["feature_count"],
                    len(info["cat_features"]),
                )
                del model, x_fit, x_apply
            if not np.isfinite(inner_oof).all():
                raise ValueError(
                    f"Incomplete inner OOF for outer fold {outer_fold}, "
                    f"config {config['name']}"
                )
            pooled_inner_auc = float(roc_auc_score(y[outer_fit], inner_oof))
            candidate_scores.append((pooled_inner_auc, config, inner_fold_scores))
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "config": config["name"],
                    "pooled_inner_oof_auc": pooled_inner_auc,
                    "mean_inner_fold_auc": float(np.mean(inner_fold_scores)),
                    "inner_fold_aucs": json.dumps(inner_fold_scores),
                    "fit_seconds": time.perf_counter() - inner_started,
                    **config,
                }
            )
        chosen_score, chosen_config, chosen_inner_folds = max(
            candidate_scores, key=lambda row: row[0]
        )
        logger.info(
            "outer=%d selected=%s pooled_inner_auc=%.6f",
            outer_fold,
            chosen_config["name"],
            chosen_score,
        )

        x_fit, x_apply, info = feature_matrix(
            outer_fit,
            outer_apply,
            raw,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            enhancements,
        )
        final_seed = OUTER_SEED + outer_fold * 1000 + 999
        final_model = make_model(chosen_config, seed=final_seed)
        fit_started = time.perf_counter()
        final_model.fit(
            x_fit,
            y[outer_fit],
            cat_features=info["cat_features"],
            verbose=False,
        )
        fold_prediction = final_model.predict_proba(x_apply)[:, 1]
        if not np.isfinite(fold_prediction).all():
            raise ValueError(f"Non-finite outer OOF predictions in fold {outer_fold}")
        predictions[outer_apply] = fold_prediction
        fold_auc = float(roc_auc_score(y[outer_apply], fold_prediction))
        baseline_fold_auc = float(roc_auc_score(y[outer_apply], baseline[outer_apply]))
        outer_rows.append(
            {
                "fold": outer_fold,
                "selected_config": chosen_config["name"],
                "pooled_inner_oof_auc": chosen_score,
                "inner_fold_aucs": json.dumps(chosen_inner_folds),
                "outer_auc": fold_auc,
                "phase15_baseline_auc": baseline_fold_auc,
                "paired_fold_delta": fold_auc - baseline_fold_auc,
                "fit_rows": len(outer_fit),
                "validation_rows": len(outer_apply),
                "features": info["feature_count"],
                "categorical_features": len(info["cat_features"]),
                "fit_seconds": time.perf_counter() - fit_started,
                "seed": final_seed,
            }
        )
        logger.info(
            "outer=%d auc=%.6f baseline=%.6f delta=%+.6f",
            outer_fold,
            fold_auc,
            baseline_fold_auc,
            fold_auc - baseline_fold_auc,
        )
        del final_model, x_fit, x_apply

    if not np.isfinite(predictions).all():
        raise ValueError("Nested CatBoost run did not produce exactly one OOF per row")
    pooled_auc = float(roc_auc_score(y, predictions))
    baseline_auc = float(roc_auc_score(y, baseline))
    interval = paired_bootstrap(y, baseline, predictions)
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "INNER_CONFIG_SELECTION.csv", index=False
    )
    pd.DataFrame(outer_rows).to_csv(output_dir / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(
        {
            "row_index": expected_index,
            phase13.ID_COLUMN: ids,
            phase13.TARGET_COLUMN: y,
            "fold": folds,
            "phase16_nested_tuned_catboost": predictions,
            "phase15_matched_seed_baseline": baseline,
        }
    ).to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)

    summary = {
        "status": "nested hyperparameter selection, exploratory development OOF",
        "run_id": run_name,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "outer_seed": OUTER_SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "configurations": list(CONFIGS),
        "selected_config_by_outer_fold": {
            str(row["fold"]): row["selected_config"] for row in outer_rows
        },
        "pooled_outer_auc": pooled_auc,
        "phase15_baseline_pooled_auc": baseline_auc,
        "pooled_auc_delta_vs_phase15_baseline": pooled_auc - baseline_auc,
        "positive_outer_fold_count": int(
            sum(row["paired_fold_delta"] > 0 for row in outer_rows)
        ),
        **interval,
        "training_sha256": sha256_file(TRAIN_PATH),
        "phase13_oof_sha256": sha256_file(REFERENCE_PATH),
        "phase15_oof_sha256": sha256_file(PHASE15_OOF_PATH),
        "feature_scope": (
            "Supplied train features, eight historical product keys, "
            "Phase 14 deterministic numeric block, and Phase 15 date-offset "
            "features. Product-key constants and CatBoost native CTR eligibility "
            "remain unresolved."
        ),
        "governance": (
            "No test rows, external data, or submissions used. All results reuse "
            "previously exposed labels; this is not independent confirmation or "
            "an eligibility determination."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "Completed: pooled_auc=%.6f baseline=%.6f delta=%+.6f CI=[%.6f, %.6f]",
        pooled_auc,
        baseline_auc,
        pooled_auc - baseline_auc,
        interval["paired_delta_ci95_low"],
        interval["paired_delta_ci95_high"],
    )
    _manifest(output_dir, (TRAIN_PATH, REFERENCE_PATH, PHASE15_OOF_PATH))
    return output_dir


if __name__ == "__main__":
    print(run())
