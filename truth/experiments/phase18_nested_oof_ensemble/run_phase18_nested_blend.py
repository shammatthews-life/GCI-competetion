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
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
PHASE13_DIR = REPO / "experiments" / "phase13_matched_baseline"
PHASE14_DIR = REPO / "experiments" / "phase14_label_free_model_development"
PHASE15_DIR = REPO / "experiments" / "phase15_autonomous_research"
PHASE17_DIR = REPO / "experiments" / "phase17_nested_constant_free_target_encoding"
sys.path.insert(0, str(PHASE13_DIR))
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE15_DIR))
sys.path.insert(0, str(PHASE17_DIR))
import run_phase13 as phase13
import run_phase14 as phase14
import run_phase17_nested_te as phase17
from phase15_features import build_time_lifecycle_features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
PHASE15_OOF = (
    PHASE15_DIR
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE17_OOF = (
    PHASE17_DIR / "runs" / "phase17_nested_te_20261105T140620Z"
    / "OOF_PREDICTIONS.csv"
)
OUTER_SEED = 20261113
OUTER_FOLDS = 5
INNER_FOLDS = 3
TE_INNER_ITERATIONS = 1200
TE_INNER_EARLY_STOPPING = 80
TE_CONFIG = {"name": "leaves31_child80", "num_leaves": 31, "min_child_samples": 80}
CATBOOST_CONFIG: dict[str, Any] = {
    "iterations": 700,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
}
CATBOOST_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 20261801


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logger_for(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase18_nested_blend")
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


def make_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        **CATBOOST_CONFIG,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.78,
        border_count=128,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )


def catboost_pair(
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
    return phase14._matrix_pair(
        fit, apply, blocks, phase13.PRODUCT_KEYS, "catboost"
    )


def fit_blend_components(
    fit: np.ndarray,
    apply: np.ndarray,
    fold: int,
    inner_fold: int,
    y: np.ndarray,
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rowwise: pd.DataFrame,
    keys: pd.DataFrame,
    decompositions: pd.DataFrame,
    enhancements: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    stop_fit, stop_apply = train_test_split(
        fit,
        test_size=0.1,
        random_state=OUTER_SEED + fold * 1000 + inner_fold,
        stratify=y[fit],
    )
    x_stop_fit, x_stop_apply, _ = phase13.make_matrices(
        stop_fit,
        stop_apply,
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        "E_C_plus_keys_frequency_ohe",
        key_columns=phase17.TE_KEYS,
    )
    _, rounds, stop_seconds = phase13._early_stopped_predict(
        x_stop_fit,
        x_stop_apply,
        y[stop_fit],
        y[stop_apply],
        TE_CONFIG,
        seed=OUTER_SEED + fold * 10000 + inner_fold * 100 + 1801,
    )

    lgb_fit, lgb_apply, _ = phase13.make_matrices(
        fit,
        apply,
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        "E_C_plus_keys_frequency_ohe",
        key_columns=phase17.TE_KEYS,
    )
    fit_te, _ = phase17.nested_training_te(
        keys, y, fit, fold=fold * 10 + inner_fold
    )
    apply_te = phase17.apply_te(keys, y, fit, apply)
    lgb_fit = hstack([lgb_fit, csr_matrix(fit_te)], format="csr")
    lgb_apply = hstack([lgb_apply, csr_matrix(apply_te)], format="csr")
    lgb_prediction, _ = phase13._fit_predict(
        lgb_fit,
        lgb_apply,
        y[fit],
        TE_CONFIG,
        seed=OUTER_SEED + fold * 10000 + inner_fold * 100 + 1802,
        rounds=rounds,
    )

    cat_fit, cat_apply, cat_info = catboost_pair(
        fit,
        apply,
        raw,
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        enhancements,
    )
    cat_model = make_catboost(
        OUTER_SEED + fold * 10000 + inner_fold * 100 + 1803
    )
    cat_started = time.perf_counter()
    cat_model.fit(
        cat_fit,
        y[fit],
        cat_features=cat_info["cat_features"],
        verbose=False,
    )
    cat_prediction = cat_model.predict_proba(cat_apply)[:, 1]
    cat_seconds = time.perf_counter() - cat_started
    if not np.isfinite(cat_prediction).all() or not np.isfinite(lgb_prediction).all():
        raise ValueError(f"Non-finite inner component predictions in fold {fold}")
    return cat_prediction, lgb_prediction, rounds, stop_seconds, cat_seconds


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
    for path in (TRAIN_PATH, PHASE15_OOF, PHASE17_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    output_root = ROOT / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase18_nested_blend_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    logger = logger_for(output_dir)
    started = time.perf_counter()

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    phase15_oof = pd.read_csv(PHASE15_OOF)
    phase17_oof = pd.read_csv(PHASE17_OOF)
    y = train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    ids = train[phase13.ID_COLUMN].to_numpy()
    folds = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    outer_splits = list(splitter.split(np.zeros(len(y), dtype=np.int8), y))
    for fold, (_, apply) in enumerate(outer_splits, start=1):
        folds[apply] = fold
    for name, frame in (("Phase 15", phase15_oof), ("Phase 17", phase17_oof)):
        if not (
            len(frame) == len(train)
            and np.array_equal(frame["row_index"].to_numpy(), np.arange(len(y)))
            and np.array_equal(frame[phase13.ID_COLUMN].to_numpy(), ids)
            and np.array_equal(frame[phase13.TARGET_COLUMN].to_numpy(), y)
            and np.array_equal(frame["fold"].to_numpy(), folds)
        ):
            raise ValueError(f"{name} OOF rows, IDs, targets, or folds do not align")
    outer_cat = phase15_oof[
        "native_catboost_phase14_plus_time"
    ].to_numpy(dtype=np.float64)
    outer_lgb = phase17_oof["nested_te_constant_free"].to_numpy(dtype=np.float64)
    if not np.isfinite(outer_cat).all() or not np.isfinite(outer_lgb).all():
        raise ValueError("A saved component OOF vector contains non-finite values")

    raw = train.drop(columns=[phase13.TARGET_COLUMN]).reset_index(drop=True)
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

    blend_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    selection_rows = []
    for outer_fold, (outer_fit, outer_apply) in enumerate(outer_splits, start=1):
        logger.info(
            "outer=%d/%d fit=%d validation=%d",
            outer_fold,
            OUTER_FOLDS,
            len(outer_fit),
            len(outer_apply),
        )
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=OUTER_SEED + 500 + outer_fold,
        )
        inner_cat = np.full(len(outer_fit), np.nan, dtype=np.float64)
        inner_lgb = np.full(len(outer_fit), np.nan, dtype=np.float64)
        inner_fit_details = []
        inner_splits = [
            (outer_fit[fit], outer_fit[apply])
            for fit, apply in inner_splitter.split(
                np.zeros(len(outer_fit), dtype=np.int8), y[outer_fit]
            )
        ]
        for inner_fold, (inner_fit, inner_apply) in enumerate(inner_splits, start=1):
            cat_prediction, lgb_prediction, rounds, stop_seconds, cat_seconds = (
                fit_blend_components(
                    inner_fit,
                    inner_apply,
                    outer_fold,
                    inner_fold,
                    y,
                    raw,
                    numeric,
                    categorical,
                    rowwise,
                    keys,
                    decompositions,
                    enhancements,
                )
            )
            positions = np.searchsorted(outer_fit, inner_apply)
            inner_cat[positions] = cat_prediction
            inner_lgb[positions] = lgb_prediction
            inner_fit_details.append(
                {
                    "inner_fold": inner_fold,
                    "fit_rows": int(len(inner_fit)),
                    "validation_rows": int(len(inner_apply)),
                    "lgb_selected_rounds": int(rounds),
                    "lgb_inner_stopping_seconds": float(stop_seconds),
                    "catboost_fit_seconds": float(cat_seconds),
                }
            )
            logger.info(
                "outer=%d inner=%d CatBoost_auc=%.6f TE_LGBM_auc=%.6f rounds=%d",
                outer_fold,
                inner_fold,
                roc_auc_score(y[inner_apply], cat_prediction),
                roc_auc_score(y[inner_apply], lgb_prediction),
                rounds,
            )
        if not np.isfinite(inner_cat).all() or not np.isfinite(inner_lgb).all():
            raise ValueError(f"Inner OOF predictions incomplete in fold {outer_fold}")
        candidate_weights = []
        for cat_weight in CATBOOST_WEIGHTS:
            blended_inner = (
                cat_weight * inner_cat + (1.0 - cat_weight) * inner_lgb
            )
            inner_auc = float(roc_auc_score(y[outer_fit], blended_inner))
            candidate_weights.append((inner_auc, cat_weight))
            selection_rows.append(
                {
                    "outer_fold": outer_fold,
                    "catboost_weight": cat_weight,
                    "target_encoded_lightgbm_weight": 1.0 - cat_weight,
                    "pooled_inner_oof_auc": inner_auc,
                }
            )
        selected_auc, selected_weight = max(candidate_weights, key=lambda row: row[0])
        selected_lgb_weight = 1.0 - selected_weight
        blended_outer = selected_weight * outer_cat[outer_apply] + selected_lgb_weight * outer_lgb[outer_apply]
        blend_oof[outer_apply] = blended_outer
        cat_fold_auc = float(roc_auc_score(y[outer_apply], outer_cat[outer_apply]))
        lgb_fold_auc = float(roc_auc_score(y[outer_apply], outer_lgb[outer_apply]))
        blend_fold_auc = float(roc_auc_score(y[outer_apply], blended_outer))
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_catboost_weight": selected_weight,
                "selected_te_lightgbm_weight": selected_lgb_weight,
                "pooled_inner_oof_auc_at_selected_weight": selected_auc,
                "outer_catboost_auc": cat_fold_auc,
                "outer_te_lightgbm_auc": lgb_fold_auc,
                "outer_nested_blend_auc": blend_fold_auc,
                "delta_vs_catboost": blend_fold_auc - cat_fold_auc,
                "delta_vs_te_lightgbm": blend_fold_auc - lgb_fold_auc,
                "inner_fit_details": json.dumps(inner_fit_details),
            }
        )
        logger.info(
            "outer=%d selected_weight_cat=%.2f blend_auc=%.6f CatBoost=%.6f TE_LGBM=%.6f",
            outer_fold,
            selected_weight,
            blend_fold_auc,
            cat_fold_auc,
            lgb_fold_auc,
        )

    if not np.isfinite(blend_oof).all():
        raise ValueError("Nested blend did not predict every outer row exactly once")
    cat_auc = float(roc_auc_score(y, outer_cat))
    lgb_auc = float(roc_auc_score(y, outer_lgb))
    blend_auc = float(roc_auc_score(y, blend_oof))
    cat_interval = paired_bootstrap(y, outer_cat, blend_oof)
    lgb_interval = paired_bootstrap(y, outer_lgb, blend_oof)
    summary = {
        "status": "nested fold-local blend selection, exploratory outer OOF",
        "run_id": run_name,
        "rows": int(len(y)),
        "outer_seed": OUTER_SEED,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "candidate_catboost_weights": list(CATBOOST_WEIGHTS),
        "phase15_catboost_auc": cat_auc,
        "phase17_nested_te_lightgbm_auc": lgb_auc,
        "nested_blend_auc": blend_auc,
        "delta_vs_catboost": blend_auc - cat_auc,
        "delta_vs_phase17_te_lightgbm": blend_auc - lgb_auc,
        "selected_weights_by_fold": {
            str(row["outer_fold"]): {
                "catboost": row["selected_catboost_weight"],
                "te_lightgbm": row["selected_te_lightgbm_weight"],
            }
            for row in fold_rows
        },
        "paired_interval_vs_catboost": cat_interval,
        "paired_interval_vs_phase17": lgb_interval,
        "training_sha256": sha256_file(TRAIN_PATH),
        "phase15_oof_sha256": sha256_file(PHASE15_OOF),
        "phase17_oof_sha256": sha256_file(PHASE17_OOF),
        "eligibility": (
            "Research only. Phase 15 contains unresolved product constants and "
            "native CTRs; Phase 17 uses label-derived target statistics."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "INNER_BLEND_SELECTION.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(output_dir / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            phase13.ID_COLUMN: ids,
            phase13.TARGET_COLUMN: y,
            "fold": folds,
            "phase15_catboost": outer_cat,
            "phase17_nested_te_lightgbm": outer_lgb,
            "phase18_nested_blend": blend_oof,
        }
    ).to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info(
        "Completed: nested_blend=%.6f Phase15=%.6f Phase17=%.6f",
        blend_auc,
        cat_auc,
        lgb_auc,
    )
    sources = (
        Path(__file__).resolve(),
        ROOT / "PHASE18_PROTOCOL.md",
        PHASE13_DIR / "run_phase13.py",
        PHASE13_DIR / "phase13_features.py",
        PHASE14_DIR / "run_phase14.py",
        PHASE15_DIR / "run_phase15_catboost_preflight.py",
        PHASE15_DIR / "phase15_features.py",
        PHASE17_DIR / "run_phase17_nested_te.py",
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
    for kind, paths in (("input", (TRAIN_PATH, PHASE15_OOF, PHASE17_OOF)), ("source", sources)):
        for path in paths:
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
