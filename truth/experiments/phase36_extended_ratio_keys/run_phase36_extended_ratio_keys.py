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
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE34 = Path(__file__).resolve().parents[1] / "phase34_xgboost_rules_first"
sys.path[:0] = [str(PHASE33), str(PHASE34)]
import run_phase33_rules_first_features as phase33


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
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
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
EXTRA_KEYS = (
    "EXT_KEY_CREDIT_INCOME_3",
    "EXT_KEY_ANNUITY_INCOME_3",
    "EXT_KEY_GOODS_CREDIT_3",
    "EXT_KEY_GOODS_INCOME_3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_extra_keys(raw: pd.DataFrame) -> pd.DataFrame:
    if TARGET in raw or ID not in raw:
        raise ValueError("Feature input must include ID and exclude TARGET")
    credit = phase33._numeric(raw, "AMT_CREDIT")
    annuity = phase33._numeric(raw, "AMT_ANNUITY")
    goods = phase33._numeric(raw, "AMT_GOODS_PRICE")
    income = phase33._numeric(raw, "AMT_INCOME_TOTAL")
    return pd.DataFrame(
        {
            EXTRA_KEYS[0]: phase33._integer_key(
                phase33._ratio(credit, income), 3
            ),
            EXTRA_KEYS[1]: phase33._integer_key(
                phase33._ratio(annuity, income), 3
            ),
            EXTRA_KEYS[2]: phase33._integer_key(
                phase33._ratio(goods, credit), 3
            ),
            EXTRA_KEYS[3]: phase33._integer_key(
                phase33._ratio(goods, income), 3
            ),
        }
    )


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return csr_matrix(values)


def make_matrices(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    extra_keys: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[csr_matrix, csr_matrix]:
    numeric, categorical, rowwise, _, safe_keys, _ = blocks
    fit_numeric = pd.concat(
        [
            numeric.iloc[fit_rows].reset_index(drop=True),
            rowwise.iloc[fit_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    apply_numeric = pd.concat(
        [
            numeric.iloc[apply_rows].reset_index(drop=True),
            rowwise.iloc[apply_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    fit_blocks = [_numeric_matrix(fit_numeric)]
    apply_blocks = [_numeric_matrix(apply_numeric)]
    raw_encoder = OneHotEncoder(
        handle_unknown="ignore", dtype=np.float32, sparse_output=True
    )
    fit_cat = raw_encoder.fit_transform(
        categorical.iloc[fit_rows].astype(str)
    ).tocsr()
    apply_cat = raw_encoder.transform(
        categorical.iloc[apply_rows].astype(str)
    ).tocsr()
    fit_blocks.append(fit_cat)
    apply_blocks.append(apply_cat)

    all_keys = pd.concat(
        [
            safe_keys.reset_index(drop=True),
            extra_keys.reset_index(drop=True),
        ],
        axis=1,
    )
    key_columns = tuple(all_keys.columns)
    fit_key_values = all_keys.iloc[fit_rows].astype(str)
    apply_key_values = all_keys.iloc[apply_rows].astype(str)
    key_encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=2,
        dtype=np.float32,
        sparse_output=True,
    )
    fit_key = key_encoder.fit_transform(fit_key_values).tocsr()
    apply_key = key_encoder.transform(apply_key_values).tocsr()
    fit_blocks.append(fit_key)
    apply_blocks.append(apply_key)
    fit_counts: list[np.ndarray] = []
    apply_counts: list[np.ndarray] = []
    for column in key_columns:
        counts = fit_key_values[column].value_counts()
        fit_counts.append(fit_key_values[column].map(counts).to_numpy(np.float32))
        apply_counts.append(
            apply_key_values[column].map(counts).fillna(0).to_numpy(np.float32)
        )
    fit_blocks.append(csr_matrix(np.column_stack(fit_counts), dtype=np.float32))
    apply_blocks.append(
        csr_matrix(np.column_stack(apply_counts), dtype=np.float32)
    )
    return (
        hstack(fit_blocks, format="csr", dtype=np.float32),
        hstack(apply_blocks, format="csr", dtype=np.float32),
    )


def _model(seed: int, iterations: int, stopping: bool) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=iterations,
        max_depth=4,
        learning_rate=0.035,
        min_child_weight=40,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=8,
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
        verbosity=0,
    )


def run(
    train_path: Path,
    phase34_oof_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase34_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    reference = pd.read_csv(phase34_oof_path)
    if (
        ID not in train
        or TARGET not in train
        or not train[ID].is_unique
        or len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or "oof_xgboost" not in reference
    ):
        raise ValueError("Training or Phase 34 reference artifacts are invalid")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    extra_keys = build_extra_keys(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = fold_number
    if not np.array_equal(fold_ids, reference["fold"].to_numpy()):
        raise ValueError("Phase 34 fold assignments differ from current protocol")

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
        x_inner_fit, x_inner_valid = make_matrices(
            raw, blocks, extra_keys, inner_fit, inner_valid
        )
        selector = _model(fold_seed, MAX_ITERATIONS, stopping=True)
        selector.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_valid, y[inner_valid])],
            verbose=False,
        )
        rounds = max(100, int(selector.best_iteration) + 1)
        del selector, x_inner_fit, x_inner_valid

        x_fit, x_valid = make_matrices(
            raw, blocks, extra_keys, fit_rows, valid_rows
        )
        model = _model(fold_seed, rounds, stopping=False)
        model.fit(x_fit, y[fit_rows], verbose=False)
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite Phase 36 prediction in fold {fold_number}")
        oof[valid_rows] = prediction.astype(np.float32)
        auc = float(roc_auc_score(y[valid_rows], prediction))
        records.append(
            {
                "fold": fold_number,
                "auc": auc,
                "iterations": rounds,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} AUC={auc:.6f} iterations={rounds}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions are incomplete")
    base = reference["oof_xgboost"].to_numpy(dtype=np.float32)
    blend_with_base = ((base + oof) / 2.0).astype(np.float32)

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE36_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE36_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE36_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    results = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_xgboost_extended_ratio_keys": oof,
            "oof_equal_blend_base_and_extended": blend_with_base,
        }
    )
    results.to_csv(oof_path, index=False)
    metrics: dict[str, dict[str, float]] = {}
    for name, prediction in (
        ("extended_ratio_keys", oof),
        ("fixed_equal_blend_base_extended", blend_with_base),
    ):
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
        "phase": "Phase 36 extended constant-free ratio categories",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "new_features": {
            key: "round(rowwise ratio, 3 decimals), sourced only from supplied amount columns"
            for key in EXTRA_KEYS
        },
        "feature_processing": {
            "one_hot": "vocabulary and minimum-frequency grouping fit on each training partition only",
            "frequency": "category counts fit on each training partition only; unseen values map to zero",
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "comparison": "same seed and five outer folds as Phase 34",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
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
    parser.add_argument("--phase34-oof", type=Path, default=PHASE34_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase36_ratio_keys_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase34_oof, args.output_dir)
