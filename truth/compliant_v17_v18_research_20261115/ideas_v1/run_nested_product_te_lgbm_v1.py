from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASE_OOF = ROOT / "runs" / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "nested_product_te_lgbm_v1"
SEED = 20261113
FOLDS = 5
INNER_TE_FOLDS = 5
TE_ALPHA = 30.0
TE_KEYS = (
    "CF_KEY_ANNUITY_CREDIT_3",
    "CF_KEY_ANNUITY_CREDIT_4",
    "CF_KEY_ANNUITY_CREDIT_5",
    "CF_KEY_ANNUITY_GOODS_3",
)
ID = "SK_ID_CURR"
TARGET = "TARGET"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_key(
    values: pd.Series,
    y: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.intersect1d(fit_rows, apply_rows).size:
        raise ValueError("Target-map fit and apply rows overlap")
    prior = float(y[fit_rows].mean())
    fit_frame = pd.DataFrame(
        {
            "key": values.iloc[fit_rows].astype(str).to_numpy(),
            "target": y[fit_rows].astype(np.float64),
        }
    )
    grouped = fit_frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
    rates = (grouped["sum"] + TE_ALPHA * prior) / (
        grouped["count"] + TE_ALPHA
    )
    apply_keys = values.iloc[apply_rows].astype(str)
    encoded = apply_keys.map(rates).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(encoded).all() or not np.isfinite(counts).all():
        raise ValueError("Non-finite target encoding or count")
    return encoded, np.log1p(counts)


def nested_training_te(
    features: pd.DataFrame,
    y: np.ndarray,
    outer_fit: np.ndarray,
    fold: int,
) -> np.ndarray:
    values = np.full((len(outer_fit), 2 * len(TE_KEYS)), np.nan, dtype=np.float32)
    local_y = y[outer_fit]
    split = StratifiedKFold(
        n_splits=INNER_TE_FOLDS,
        shuffle=True,
        random_state=SEED + 100 + fold,
    )
    for fit_local, apply_local in split.split(
        np.zeros(len(outer_fit), dtype=np.int8), local_y
    ):
        fit_rows = outer_fit[fit_local]
        apply_rows = outer_fit[apply_local]
        for key_index, key_name in enumerate(TE_KEYS):
            encoded, count = encode_key(
                features[key_name], y, fit_rows, apply_rows
            )
            values[apply_local, 2 * key_index] = encoded
            values[apply_local, 2 * key_index + 1] = count
    if not np.isfinite(values).all():
        raise ValueError("Nested training encodings are incomplete")
    return values


def apply_training_te(
    features: pd.DataFrame,
    y: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> np.ndarray:
    values = np.empty((len(apply_rows), 2 * len(TE_KEYS)), dtype=np.float32)
    for key_index, key_name in enumerate(TE_KEYS):
        encoded, count = encode_key(
            features[key_name], y, fit_rows, apply_rows
        )
        values[:, 2 * key_index] = encoded
        values[:, 2 * key_index + 1] = count
    return values


def make_sparse_pair(
    features: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    fit_te: np.ndarray,
    apply_te: np.ndarray,
) -> tuple[csr_matrix, csr_matrix, int]:
    key_set = set(TE_KEYS)
    categorical = [
        column
        for column in features.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        if column not in key_set
    ]
    numeric = [
        column
        for column in features.select_dtypes(include=[np.number]).columns
        if column not in key_set
    ]
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=2,
        dtype=np.float32,
        sparse_output=True,
    )
    fit_categories = features.iloc[fit_rows][categorical].astype(str)
    apply_categories = features.iloc[apply_rows][categorical].astype(str)
    fit_ohe = encoder.fit_transform(fit_categories).tocsr()
    apply_ohe = encoder.transform(apply_categories).tocsr()

    fit_numeric = features.iloc[fit_rows][numeric].to_numpy(dtype=np.float32)
    apply_numeric = features.iloc[apply_rows][numeric].to_numpy(dtype=np.float32)
    fit_numeric[~np.isfinite(fit_numeric)] = np.nan
    apply_numeric[~np.isfinite(apply_numeric)] = np.nan
    fit_matrix = hstack(
        (
            csr_matrix(fit_numeric),
            fit_ohe,
            csr_matrix(fit_te, dtype=np.float32),
        ),
        format="csr",
    )
    apply_matrix = hstack(
        (
            csr_matrix(apply_numeric),
            apply_ohe,
            csr_matrix(apply_te, dtype=np.float32),
        ),
        format="csr",
    )
    if fit_matrix.shape[1] != apply_matrix.shape[1]:
        raise ValueError("Fit/apply matrix feature dimensions differ")
    return fit_matrix, apply_matrix, int(fit_matrix.shape[1])


def new_model(iterations: int, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        random_state=seed,
        n_jobs=8,
        verbosity=-1,
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite run: {OUTPUT}")
    if not TRAIN.is_file() or not BASE_OOF.is_file():
        raise FileNotFoundError("Required training or baseline OOF file is missing")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    base = pd.read_csv(BASE_OOF, usecols=[ID, TARGET, "fold"])
    if (
        len(train) != len(base)
        or not train[ID].equals(base[ID])
        or not train[TARGET].equals(base[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows are not aligned")
    y = train[TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    features, _ = make_features(raw)
    if set(TE_KEYS).difference(features.columns):
        raise ValueError("A required constant-free product key is missing")

    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    records: list[dict[str, int | float]] = []
    for fold, (outer_fit, outer_valid) in enumerate(
        split.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        fold_ids[outer_valid] = fold
        inner_fit, inner_stop = train_test_split(
            outer_fit,
            test_size=0.1,
            random_state=SEED + 700 + fold,
            stratify=y[outer_fit],
        )

        inner_fit_te = nested_training_te(features, y, inner_fit, fold + 100)
        inner_stop_te = apply_training_te(
            features, y, inner_fit, inner_stop
        )
        x_inner_fit, x_inner_stop, _ = make_sparse_pair(
            features, inner_fit, inner_stop, inner_fit_te, inner_stop_te
        )
        selector = new_model(2200, SEED + fold * 100)
        selector.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_stop, y[inner_stop])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=120, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        selected_rounds = max(100, int(selector.best_iteration_ or 2200))
        del selector, x_inner_fit, x_inner_stop

        outer_fit_te = nested_training_te(features, y, outer_fit, fold)
        outer_valid_te = apply_training_te(
            features, y, outer_fit, outer_valid
        )
        x_outer_fit, x_outer_valid, feature_count = make_sparse_pair(
            features,
            outer_fit,
            outer_valid,
            outer_fit_te,
            outer_valid_te,
        )
        model = new_model(selected_rounds, SEED + fold * 100)
        model.fit(x_outer_fit, y[outer_fit])
        prediction = model.predict_proba(x_outer_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite prediction in fold {fold}")
        oof[outer_valid] = prediction.astype(np.float32)
        auc = float(roc_auc_score(y[outer_valid], prediction))
        records.append(
            {
                "fold": fold,
                "fit_rows": int(len(outer_fit)),
                "valid_rows": int(len(outer_valid)),
                "selected_rounds": selected_rounds,
                "features": feature_count,
                "auc": auc,
            }
        )
        print(
            f"fold={fold}/{FOLDS} AUC={auc:.6f} rounds={selected_rounds}",
            flush=True,
        )
        del model, x_outer_fit, x_outer_valid

    if not np.array_equal(fold_ids, base["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Experiment folds differ from baseline folds")
    if not np.isfinite(oof).all():
        raise ValueError("OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_nested_product_te_lgbm": oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    summary = {
        "status": "exploratory nested fold-local target-encoding OOF; repeated-label development",
        "inputs": "official train.csv and row-alignment metadata from prior OOF; test.csv not read",
        "outer_folds": FOLDS,
        "inner_target_encoding_folds": INNER_TE_FOLDS,
        "seed": SEED,
        "target_encoding_keys": list(TE_KEYS),
        "target_encoding_smoothing": TE_ALPHA,
        "target_encoding": "Training rows receive inner-OOF statistics; each map uses only disjoint rows from the same outer training partition. Outer validation statistics use outer training rows only.",
        "model": {
            "library": "LightGBM",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 80,
            "max_bin": 127,
            "early_stopping_rounds": 120,
            "early_stopping_split": "10% stratified split of each outer training partition",
        },
        "versions": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fold_metrics": records,
        "pooled_auc": float(roc_auc_score(y, oof)),
        "mean_fold_auc": float(np.mean([row["auc"] for row in records])),
        "rule_controls": {
            "external_data": "none",
            "label_transfer": "none; only supervised category statistics from the official TARGET column, fold-local and cross-fitted",
            "train_test_pooling": "none",
            "test_rows": "not read",
            "manual_prediction_changes": "none",
            "outer_validation_labels": "not used for target maps, preprocessing, or stopping-round selection",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (TRAIN, BASE_OOF, Path(__file__).resolve(), oof_path, fold_path)
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
