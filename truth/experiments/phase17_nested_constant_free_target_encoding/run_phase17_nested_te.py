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
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PHASE13_DIR = REPO / "experiments" / "phase13_matched_baseline"
sys.path.insert(0, str(PHASE13_DIR))
import run_phase13 as phase13


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
OUTER_SEED = 20261113
FOLDS = 5
INNER_TE_FOLDS = 5
TE_ALPHA = 30.0
TE_KEYS = (
    "PRODUCT_KEY_Q3",
    "PRODUCT_KEY_Q4",
    "PRODUCT_KEY_Q5",
    "PRODUCT_KEY_AG3",
)
VARIANT = "E_C_plus_keys_frequency_ohe"
CONFIG: dict[str, Any] = {"name": "leaves31_child80", "num_leaves": 31, "min_child_samples": 80}
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 20261701


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logger_for(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase17_nested_te")
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


def encode_one(
    values: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    apply: np.ndarray,
    alpha: float = TE_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    if np.intersect1d(fit, apply).size:
        raise ValueError("Target-map fit and apply indices overlap")
    prior = float(y[fit].mean())
    frame = pd.DataFrame(
        {"key": values[fit], "target": y[fit].astype(np.float64)}
    )
    grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
    rates = (grouped["sum"] + alpha * prior) / (grouped["count"] + alpha)
    apply_keys = pd.Series(values[apply])
    encoded = apply_keys.map(rates).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(encoded).all() or not np.isfinite(counts).all():
        raise ValueError("Non-finite target encoding or count")
    return encoded, counts


def nested_training_te(
    keys: pd.DataFrame,
    y: np.ndarray,
    outer_fit: np.ndarray,
    fold: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.full((len(outer_fit), 2 * len(TE_KEYS)), np.nan, dtype=np.float32)
    local_y = y[outer_fit]
    splitter = StratifiedKFold(
        n_splits=INNER_TE_FOLDS,
        shuffle=True,
        random_state=OUTER_SEED + 100 + fold,
    )
    diagnostics = []
    for inner_fold, (fit_local, apply_local) in enumerate(
        splitter.split(np.zeros(len(outer_fit), dtype=np.int8), local_y), start=1
    ):
        fit = outer_fit[fit_local]
        apply = outer_fit[apply_local]
        for key_idx, name in enumerate(TE_KEYS):
            encoded, count = encode_one(keys[name].to_numpy(dtype=str), y, fit, apply)
            values[apply_local, 2 * key_idx] = encoded
            values[apply_local, 2 * key_idx + 1] = np.log1p(count)
        diagnostics.append(
            {
                "inner_fold": inner_fold,
                "map_fit_rows": int(len(fit)),
                "encoded_rows": int(len(apply)),
                "fit_apply_disjoint": True,
            }
        )
    if not np.isfinite(values).all():
        raise ValueError("Nested training target encodings are incomplete")
    return values, diagnostics


def apply_te(
    keys: pd.DataFrame, y: np.ndarray, fit: np.ndarray, apply: np.ndarray
) -> np.ndarray:
    values = np.empty((len(apply), 2 * len(TE_KEYS)), dtype=np.float32)
    for key_idx, name in enumerate(TE_KEYS):
        encoded, count = encode_one(keys[name].to_numpy(dtype=str), y, fit, apply)
        values[:, 2 * key_idx] = encoded
        values[:, 2 * key_idx + 1] = np.log1p(count)
    return values


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
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Training file does not exist: {TRAIN_PATH}")
    output_root = ROOT / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase17_nested_te_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = logger_for(output_dir)
    started = time.perf_counter()
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    ids = train[phase13.ID_COLUMN].to_numpy()
    if set(y.tolist()) != {0, 1} or not np.isfinite(y).all():
        raise ValueError("Training labels must be complete binary values")
    raw = train.drop(columns=[phase13.TARGET_COLUMN]).reset_index(drop=True)
    numeric, categorical, rowwise = phase13.build_raw_feature_blocks(raw)
    keys = phase13.build_product_keys(raw)
    decompositions = phase13.build_label_free_decompositions(raw)
    if set(TE_KEYS).difference(keys.columns):
        raise ValueError("A preregistered constant-free product key is missing")

    baseline_oof = np.full(len(y), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(y), np.nan, dtype=np.float64)
    folds = np.full(len(y), -1, dtype=np.int8)
    fold_rows = []
    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=OUTER_SEED)
    outer_splits = list(split.split(np.zeros(len(y), dtype=np.int8), y))
    for fold, (outer_fit, outer_apply) in enumerate(outer_splits, start=1):
        folds[outer_apply] = fold
        logger.info(
            "fold=%d/%d fit=%d validation=%d", fold, FOLDS, len(outer_fit), len(outer_apply)
        )

        # Select one shared iteration count using only a baseline inner split.
        inner_fit, inner_stop = train_test_split(
            outer_fit,
            test_size=0.1,
            random_state=OUTER_SEED + 700 + fold,
            stratify=y[outer_fit],
        )
        x_inner_fit, x_inner_stop, _ = phase13.make_matrices(
            inner_fit,
            inner_stop,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            VARIANT,
            key_columns=TE_KEYS,
        )
        _, selected_rounds, selection_seconds = phase13._early_stopped_predict(
            x_inner_fit,
            x_inner_stop,
            y[inner_fit],
            y[inner_stop],
            CONFIG,
            seed=OUTER_SEED + fold * 10000 + 1701,
        )

        x_fit, x_apply, matrix_info = phase13.make_matrices(
            outer_fit,
            outer_apply,
            numeric,
            categorical,
            rowwise,
            keys,
            decompositions,
            VARIANT,
            key_columns=TE_KEYS,
        )
        train_te, inner_map_diagnostics = nested_training_te(keys, y, outer_fit, fold)
        valid_te = apply_te(keys, y, outer_fit, outer_apply)
        x_fit_te = hstack([x_fit, csr_matrix(train_te)], format="csr")
        x_apply_te = hstack([x_apply, csr_matrix(valid_te)], format="csr")
        model_seed = OUTER_SEED + fold * 10000 + 2701

        fit_started = time.perf_counter()
        baseline_prediction, _ = phase13._fit_predict(
            x_fit, x_apply, y[outer_fit], CONFIG, model_seed, selected_rounds
        )
        candidate_prediction, _ = phase13._fit_predict(
            x_fit_te, x_apply_te, y[outer_fit], CONFIG, model_seed, selected_rounds
        )
        baseline_oof[outer_apply] = baseline_prediction
        candidate_oof[outer_apply] = candidate_prediction
        base_auc = float(roc_auc_score(y[outer_apply], baseline_prediction))
        candidate_auc = float(roc_auc_score(y[outer_apply], candidate_prediction))
        apply_support = {
            name: {
                "fit_categories": int(keys.iloc[outer_fit][name].nunique()),
                "validation_unseen_categories": int(
                    (~keys.iloc[outer_apply][name].isin(set(keys.iloc[outer_fit][name]))).sum()
                ),
            }
            for name in TE_KEYS
        }
        fold_rows.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "target_encoded_auc": candidate_auc,
                "paired_delta": candidate_auc - base_auc,
                "selected_rounds": selected_rounds,
                "inner_stopping_seconds": selection_seconds,
                "paired_final_fit_seconds": time.perf_counter() - fit_started,
                "baseline_feature_count": int(matrix_info["feature_count"]),
                "target_encoding_feature_count": int(2 * len(TE_KEYS)),
                "te_alpha": TE_ALPHA,
                "inner_map_folds": json.dumps(inner_map_diagnostics),
                "key_support": json.dumps(apply_support, sort_keys=True),
                "seed": model_seed,
            }
        )
        logger.info(
            "fold=%d rounds=%d baseline_auc=%.6f te_auc=%.6f delta=%+.6f",
            fold,
            selected_rounds,
            base_auc,
            candidate_auc,
            candidate_auc - base_auc,
        )

    if not (
        np.isfinite(baseline_oof).all()
        and np.isfinite(candidate_oof).all()
        and (folds > 0).all()
    ):
        raise ValueError("OOF coverage is incomplete or contains non-finite values")

    baseline_auc = float(roc_auc_score(y, baseline_oof))
    candidate_auc = float(roc_auc_score(y, candidate_oof))
    interval = paired_bootstrap(y, baseline_oof, candidate_oof)
    summary = {
        "status": "nested constant-free-key target encoding, exploratory OOF",
        "run_id": run_name,
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "outer_seed": OUTER_SEED,
        "outer_folds": FOLDS,
        "inner_target_encoding_folds": INNER_TE_FOLDS,
        "target_encoding_alpha": TE_ALPHA,
        "target_encoded_keys": list(TE_KEYS),
        "pooled_baseline_auc": baseline_auc,
        "pooled_target_encoded_auc": candidate_auc,
        "pooled_auc_delta": candidate_auc - baseline_auc,
        "positive_fold_count": int(sum(row["paired_delta"] > 0 for row in fold_rows)),
        **interval,
        "training_sha256": sha256_file(TRAIN_PATH),
        "feature_scope": (
            "Raw and rowwise application features; Q3/Q4/Q5/AG3 constant-free "
            "product keys; nested fold-local smoothed target rates and log counts. "
            "No 130.5 or 0.0066 hypotheses."
        ),
        "governance": (
            "Target encoding uses training labels only and is cross-fitted within "
            "each outer training fold. Rule status remains unresolved. No test rows, "
            "external data, submissions, or champion changes."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    pd.DataFrame(fold_rows).to_csv(output_dir / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            phase13.ID_COLUMN: ids,
            phase13.TARGET_COLUMN: y,
            "fold": folds,
            "baseline_constant_free_ohe": baseline_oof,
            "nested_te_constant_free": candidate_oof,
        }
    ).to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "Completed baseline=%.6f nested_te=%.6f delta=%+.6f CI=[%.6f, %.6f]",
        baseline_auc,
        candidate_auc,
        candidate_auc - baseline_auc,
        interval["paired_delta_ci95_low"],
        interval["paired_delta_ci95_high"],
    )

    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "kind": "output",
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for kind, path in (
        ("input", TRAIN_PATH),
        ("source", Path(__file__).resolve()),
        ("source", ROOT / "PHASE17_PROTOCOL.md"),
        ("source", PHASE13_DIR / "run_phase13.py"),
        ("source", PHASE13_DIR / "phase13_features.py"),
    ):
        manifest_rows.append(
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
        writer.writerows(manifest_rows)
    return output_dir


if __name__ == "__main__":
    print(run())
