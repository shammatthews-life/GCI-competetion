from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE34 = Path(__file__).resolve().parents[1] / "phase34_xgboost_rules_first"
sys.path[:0] = [str(PHASE33), str(PHASE34)]
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE33_OOF_DEFAULT = (
    PHASE33
    / "runs"
    / "phase33_rules_first_20261107T170931Z"
    / "PHASE33_OOF_PREDICTIONS.csv"
)
PHASE34_OOF_DEFAULT = (
    PHASE34
    / "runs"
    / "phase34_xgb_20261107T171823Z"
    / "PHASE34_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
MAX_ITERATIONS = 2400
STOPPING_ROUNDS = 120


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(iterations: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.72,
        l2_leaf_reg=6.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )


def run(
    train_path: Path,
    phase33_oof_path: Path,
    phase34_oof_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase33_oof_path, phase34_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    lgbm_oof = pd.read_csv(phase33_oof_path)
    xgb_oof = pd.read_csv(phase34_oof_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must include unique IDs and TARGET")
    for name, reference in (("Phase 33", lgbm_oof), ("Phase 34", xgb_oof)):
        if (
            len(reference) != len(train)
            or not reference[ID].equals(train[ID])
            or not reference[TARGET].equals(train[TARGET])
            or not reference["fold"].between(1, FOLDS).all()
        ):
            raise ValueError(f"{name} OOF artifact is not aligned")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    for name, reference in (("Phase 33", lgbm_oof), ("Phase 34", xgb_oof)):
        if not np.array_equal(fold_ids, reference["fold"].to_numpy()):
            raise ValueError(f"{name} OOF fold assignment differs")
    if not {"oof_keys_frequency"}.issubset(lgbm_oof.columns):
        raise ValueError("Phase 33 OOF lacks the LightGBM vector")
    if "oof_xgboost" not in xgb_oof:
        raise ValueError("Phase 34 OOF lacks the XGBoost vector")

    oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_fit_idx, inner_valid_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_idx]
        inner_valid = fit_rows[inner_valid_idx]
        x_inner_fit, x_inner_valid = phase33.make_matrices(
            raw, blocks, inner_fit, inner_valid, "keys_frequency"
        )
        selector = _model(MAX_ITERATIONS, fold_seed)
        selector.fit(
            x_inner_fit.toarray(),
            y[inner_fit],
            eval_set=(x_inner_valid.toarray(), y[inner_valid]),
            early_stopping_rounds=STOPPING_ROUNDS,
            use_best_model=True,
        )
        iterations = max(100, int(selector.get_best_iteration()) + 1)
        del selector, x_inner_fit, x_inner_valid

        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, fit_rows, valid_rows, "keys_frequency"
        )
        model = _model(iterations, fold_seed)
        model.fit(x_fit.toarray(), y[fit_rows])
        prediction = model.predict_proba(x_valid.toarray())[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite Phase 35 prediction in fold {fold_number}")
        oof[valid_rows] = prediction.astype(np.float32)
        auc = float(roc_auc_score(y[valid_rows], prediction))
        records.append(
            {
                "fold": fold_number,
                "auc": auc,
                "iterations": iterations,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} AUC={auc:.6f} iterations={iterations}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions are incomplete")
    lgbm = lgbm_oof["oof_keys_frequency"].to_numpy(dtype=np.float32)
    xgb_prediction = xgb_oof["oof_xgboost"].to_numpy(dtype=np.float32)
    equal_three = ((lgbm + xgb_prediction + oof) / 3.0).astype(np.float32)
    equal_pair_cat_xgb = ((xgb_prediction + oof) / 2.0).astype(np.float32)

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE35_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE35_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE35_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    result = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_catboost_onehot": oof,
            "oof_equal_catboost_xgboost": equal_pair_cat_xgb,
            "oof_equal_three_model_blend": equal_three,
        }
    )
    result.to_csv(oof_path, index=False)
    metric_vectors = {
        "catboost_onehot": oof,
        "equal_catboost_xgboost": equal_pair_cat_xgb,
        "equal_three_model_blend": equal_three,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, prediction in metric_vectors.items():
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(
                np.mean(
                    [
                        roc_auc_score(y[valid], prediction[valid])
                        for _, valid in folds
                    ]
                )
            ),
            "log_loss": float(log_loss(y, prediction)),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 35 one-hot CatBoost and fixed-blend screen",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "feature_set": "Phase 33 constant-free ratio keys and fit-fold frequencies",
        "parameters": {
            "depth": 6,
            "learning_rate": 0.04,
            "max_iterations": MAX_ITERATIONS,
            "early_stopping_rounds": STOPPING_ROUNDS,
            "categorical_target_statistics": "none; dense numerical one-hot matrix only",
            "task_type": "GPU",
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; CatBoost receives only numerical one-hot columns",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "blends": "fixed equal weights; OOF diagnostics only, no weights fitted",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                phase33_oof_path,
                phase34_oof_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                fold_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--phase33-oof", type=Path, default=PHASE33_OOF_DEFAULT)
    parser.add_argument("--phase34-oof", type=Path, default=PHASE34_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase35_cat_ohe_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase33_oof, args.phase34_oof, args.output_dir)
