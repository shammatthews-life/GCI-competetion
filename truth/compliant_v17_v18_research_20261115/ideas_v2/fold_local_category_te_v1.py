from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
PHASE33_DIR = Path(r"D:\gci\truth\experiments\phase33_rules_first_features")
sys.path.insert(0, str(PHASE33_DIR))
import run_phase33_rules_first_features as phase33


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE_OOF = (
    ROOT
    / "ideas_v1"
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "fold_local_category_te_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
TE_ALPHA = 30.0
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005
MODEL_TREES = 600
INTERACTION_PAIRS = (
    ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    ("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"),
    ("CODE_GENDER", "NAME_EDUCATION_TYPE"),
    ("REGION_RATING_CLIENT", "NAME_HOUSING_TYPE"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alignment(
    train: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with train.csv")
    if BASELINE_COLUMN not in baseline or "fold" not in baseline:
        raise ValueError("Baseline OOF file is missing required columns")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_rows) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Baseline OOF fold assignment differs from the declared split")
    return y, fold_ids


def build_encoding_keys(raw: pd.DataFrame) -> dict[str, pd.Series]:
    categorical = list(
        raw.select_dtypes(include=["object", "category", "string"]).columns
    )
    keys: dict[str, pd.Series] = {}
    for column in categorical:
        keys[f"CAT_{column}"] = (
            raw[column].astype("string").fillna("__MISSING__").astype(str)
        )
    for left, right in INTERACTION_PAIRS:
        if left not in raw or right not in raw:
            raise ValueError(f"Missing declared category interaction: {left}, {right}")
        left_values = raw[left].astype("string").fillna("__MISSING__").astype(str)
        right_values = raw[right].astype("string").fillna("__MISSING__").astype(str)
        keys[f"PAIR_{left}_{right}"] = left_values + "|" + right_values
    return keys


def encode_key(
    values: pd.Series,
    y: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.intersect1d(fit_rows, apply_rows).size:
        raise ValueError("Target-map fit and apply rows overlap")
    prior = float(y[fit_rows].mean())
    frame = pd.DataFrame(
        {
            "key": values.iloc[fit_rows].astype(str).to_numpy(),
            "target": y[fit_rows].astype(np.float64),
        }
    )
    grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
    rates = (grouped["sum"] + TE_ALPHA * prior) / (
        grouped["count"] + TE_ALPHA
    )
    apply_keys = values.iloc[apply_rows].astype(str)
    encoded = apply_keys.map(rates).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(encoded).all() or not np.isfinite(counts).all():
        raise ValueError("Non-finite target encoding or count")
    return encoded, np.log1p(counts)


def nested_training_encodings(
    keys: dict[str, pd.Series],
    y: np.ndarray,
    outer_fit: np.ndarray,
    fold: int,
) -> np.ndarray:
    values = np.full(
        (len(outer_fit), 2 * len(keys)), np.nan, dtype=np.float32
    )
    local_y = y[outer_fit]
    split = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED + 100 + fold,
    )
    for fit_local, apply_local in split.split(
        np.zeros(len(outer_fit), dtype=np.int8), local_y
    ):
        fit_rows = outer_fit[fit_local]
        apply_rows = outer_fit[apply_local]
        for key_index, key_values in enumerate(keys.values()):
            encoded, count = encode_key(key_values, y, fit_rows, apply_rows)
            values[apply_local, 2 * key_index] = encoded
            values[apply_local, 2 * key_index + 1] = count
    if not np.isfinite(values).all():
        raise ValueError("Nested category encodings are incomplete")
    return values


def apply_training_encodings(
    keys: dict[str, pd.Series],
    y: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(apply_rows), 2 * len(keys)), dtype=np.float32)
    for key_index, key_values in enumerate(keys.values()):
        encoded, count = encode_key(key_values, y, fit_rows, apply_rows)
        values[:, 2 * key_index] = encoded
        values[:, 2 * key_index + 1] = count
    return values


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=MODEL_TREES,
        num_leaves=63,
        min_child_samples=100,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        device_type="cpu",
        n_jobs=8,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    for path in (TRAIN, BASELINE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    y, fold_ids = validate_alignment(train, baseline)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_scores).all():
        raise ValueError("Baseline OOF predictions contain non-finite scores")

    blocks = phase33.build_safe_blocks(raw)
    keys = build_encoding_keys(raw)
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold in range(1, FOLDS + 1):
        fold_started = time.perf_counter()
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        train_te = nested_training_encodings(keys, y, fit_rows, fold)
        valid_te = apply_training_encodings(keys, y, fit_rows, valid_rows)
        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, fit_rows, valid_rows, "all_safe"
        )
        meta_fit = hstack(
            (x_fit, csr_matrix(train_te, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        meta_valid = hstack(
            (x_valid, csr_matrix(valid_te, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        model = make_model(SEED + fold * 100)
        model.fit(meta_fit, y[fit_rows])
        prediction = model.predict_proba(meta_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: category-TE expert scores are non-finite")
        expert_oof[valid_rows] = prediction.astype(np.float32)

        baseline_rank = fold_rank(baseline_scores[valid_rows])
        expert_rank = fold_rank(prediction)
        blend_oof[valid_rows] = (
            (1.0 - BLEND_WEIGHT) * baseline_rank
            + BLEND_WEIGHT * expert_rank
        ).astype(np.float32)
        baseline_auc = float(roc_auc_score(y[valid_rows], baseline_scores[valid_rows]))
        expert_auc = float(roc_auc_score(y[valid_rows], prediction))
        blend_auc = float(roc_auc_score(y[valid_rows], blend_oof[valid_rows]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": baseline_auc,
                "category_te_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - baseline_auc,
                "blend_delta": blend_auc - baseline_auc,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "category_te_keys": int(len(keys)),
                "target_encoding_feature_count": int(train_te.shape[1]),
                "feature_count": int(meta_fit.shape[1]),
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={baseline_auc:.6f} "
            f"category_te_expert={expert_auc:.6f} blend={blend_auc:.6f} "
            f"seconds={records[-1]['runtime_seconds']:.1f}",
            flush=True,
        )
        del model, meta_fit, meta_valid, x_fit, x_valid

    if not np.isfinite(expert_oof).all() or not np.isfinite(blend_oof).all():
        raise ValueError("Outer OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            BASELINE_COLUMN: baseline_scores,
            "fold_local_category_te_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["category_te_expert_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated categorical target-encoding experiment; not an independent estimate",
        "hypothesis": (
            "Smoothed target encodings of supplied categorical fields and "
            "predeclared category interactions may complement ratio-key "
            "encodings already used by the TE20 benchmark."
        ),
        "categorical_key_names": list(keys),
        "smoothing_alpha": TE_ALPHA,
        "inner_crossfit_folds": INNER_FOLDS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "model": {
            "family": "LightGBM",
            "trees": MODEL_TREES,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "max_bin": 127,
        },
        "fixed_blend": (
            f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{BLEND_WEIGHT:.2f} category-TE expert within-fold rank"
        ),
        "minimum_promotion_gain": PROMOTION_DELTA,
        "baseline_mean_fold_auc": baseline_mean,
        "expert_mean_fold_auc": expert_mean,
        "blend_mean_fold_auc": blend_mean,
        "blend_mean_fold_gain": gain,
        "positive_blend_folds": int(sum(row["blend_delta"] > 0 for row in records)),
        "fold_metrics": records,
        "promotion_gate_passed": bool(
            gain >= PROMOTION_DELTA
            and all(row["blend_delta"] > 0 for row in records)
        ),
        "submission_generated": False,
        "versions": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "target_encoding_scope": (
                "training-row encodings are three-way inner out-of-fold; "
                "outer-validation encodings use only outer-fit labels"
            ),
            "encoding_keys": (
                "supplied categorical fields plus four fixed pairwise "
                "interactions; no ID, applicant matching, or external data"
            ),
            "smoothing": f"fixed alpha={TE_ALPHA}; no per-row or per-fold override",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only for alignment, fold checks, and benchmark comparison"
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                TRAIN,
                BASELINE_OOF,
                Path(__file__).resolve(),
                oof_path,
                fold_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
