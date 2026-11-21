from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASE_OOF = ROOT / "runs" / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "clean_external_geometry_catboost_v1"
SEED = 20261113
FOLDS = 5
ID = "SK_ID_CURR"
TARGET = "TARGET"
EXT_COLUMNS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_external_geometry(
    features: pd.DataFrame,
    categorical: list[str],
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    scores = raw.loc[:, EXT_COLUMNS].to_numpy(dtype=np.float64)
    for left, right, name in (
        (0, 1, "EXT_RATIO_12"),
        (0, 2, "EXT_RATIO_13"),
        (1, 2, "EXT_RATIO_23"),
    ):
        ratio = np.full(len(scores), np.nan, dtype=np.float64)
        valid = (
            np.isfinite(scores[:, left])
            & np.isfinite(scores[:, right])
            & (scores[:, right] != 0)
        )
        np.divide(
            scores[:, left],
            scores[:, right],
            out=ratio,
            where=valid,
        )
        ratio[~np.isfinite(ratio)] = np.nan
        features[name] = ratio

    order: list[str] = []
    for row in scores:
        present = np.flatnonzero(np.isfinite(row))
        if not len(present):
            order.append("__NO_SCORES__")
            continue
        ranked = sorted(present, key=lambda index: (-row[index], index))
        order.append(">".join(str(index + 1) for index in ranked))
    features["EXT_SCORE_ORDER"] = np.asarray(order, dtype=str)
    categorical = [*categorical, "EXT_SCORE_ORDER"]
    return features, categorical


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite run: {OUTPUT}")
    if not TRAIN.is_file() or not BASE_OOF.is_file():
        raise FileNotFoundError("Required training or baseline OOF file is missing")

    gpu_count = int(get_gpu_device_count())
    if gpu_count < 1:
        raise RuntimeError("CatBoost cannot see a CUDA GPU; refusing CPU fallback")

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
    features, categorical = make_features(raw)
    features, categorical = add_external_geometry(features, categorical, raw)
    category_indices = [features.columns.get_loc(column) for column in categorical]
    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predictions = np.full(len(y), np.nan, dtype=np.float32)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    records: list[dict[str, int | float]] = []

    for fold, (fit_rows, valid_rows) in enumerate(
        split.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        fold_ids[valid_rows] = fold
        model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.72,
            border_count=128,
            one_hot_max_size=10,
            max_ctr_complexity=1,
            random_seed=SEED + fold,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            features.iloc[fit_rows],
            y[fit_rows],
            cat_features=category_indices,
        )
        prediction = model.predict_proba(features.iloc[valid_rows])[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite prediction in fold {fold}")
        predictions[valid_rows] = prediction.astype(np.float32)
        auc = float(roc_auc_score(y[valid_rows], prediction))
        records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "auc": auc,
                "iterations": 700,
            }
        )
        print(f"fold={fold}/{FOLDS} AUC={auc:.6f}", flush=True)
        del model

    if not np.array_equal(fold_ids, base["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Experiment folds differ from baseline folds")
    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_external_geometry_catboost": predictions,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    summary = {
        "status": "exploratory rules-conscious OOF; repeated-label development",
        "inputs": "official train.csv and row-alignment metadata from prior OOF; test.csv not read",
        "feature_hypothesis": "row-wise pairwise ratios and score-order category for the three supplied external scores",
        "outer_folds": FOLDS,
        "seed": SEED,
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": int(len(categorical)),
        "gpu": "CatBoost CUDA; no CPU fallback",
        "versions": {
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fold_metrics": records,
        "pooled_auc": float(roc_auc_score(y, predictions)),
        "mean_fold_auc": float(np.mean([row["auc"] for row in records])),
        "rule_controls": {
            "label_transfer": "none",
            "train_test_pooling": "none",
            "external_data": "none",
            "test_rows": "not read",
            "manual_prediction_changes": "none",
            "cross_row_feature_fitting": "none; added features are row-wise",
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
