from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASE_OOF = ROOT / "runs" / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "compliant_multiview_catboost_v1"
SEED = 20261113
FOLDS = 5
ID = "SK_ID_CURR"
TARGET = "TARGET"
VIEWS = ("external", "loan", "person")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_views(features: pd.DataFrame) -> dict[str, list[str]]:
    external = [
        column
        for column in features.columns
        if column.startswith("EXT_SOURCE")
        or column.startswith("EXT_")
    ]
    loan_prefixes = (
        "AMT_",
        "CREDIT_",
        "ANNUITY_",
        "GOODS_",
        "INCOME_PER_FAMILY",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "NAME_CONTRACT_TYPE",
        "CF_KEY_",
    )
    loan = [
        column
        for column in features.columns
        if column.startswith(loan_prefixes)
    ]
    person = [
        column
        for column in features.columns
        if column not in set(external).union(loan)
    ]
    views = {"external": external, "loan": loan, "person": person}
    if any(not columns for columns in views.values()):
        raise ValueError(f"A feature view is empty: { {k: len(v) for k, v in views.items()} }")
    if len(set(external).intersection(loan)):
        raise ValueError("External and loan views overlap unexpectedly")
    if set(external).union(loan, person) != set(features.columns):
        raise ValueError("The view partition does not cover every feature")
    return views


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
    baseline = pd.read_csv(BASE_OOF, usecols=[ID, TARGET, "fold"])
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows are not aligned")

    y = train[TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    features, categorical = make_features(raw)
    views = split_views(features)
    predictions = {
        view: np.full(len(y), np.nan, dtype=np.float32) for view in VIEWS
    }
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    records: list[dict[str, int | float | str]] = []
    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    for fold, (fit_rows, valid_rows) in enumerate(
        split.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        fold_ids[valid_rows] = fold
        for view in VIEWS:
            columns = views[view]
            view_categories = [
                columns.index(column)
                for column in categorical
                if column in columns
            ]
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
                features.iloc[fit_rows][columns],
                y[fit_rows],
                cat_features=view_categories,
            )
            prediction = model.predict_proba(
                features.iloc[valid_rows][columns]
            )[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(f"Nonfinite {view} prediction in fold {fold}")
            predictions[view][valid_rows] = prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], prediction))
            records.append(
                {
                    "view": view,
                    "fold": fold,
                    "rows": int(len(valid_rows)),
                    "positives": int(y[valid_rows].sum()),
                    "auc": auc,
                    "features": int(len(columns)),
                    "categorical_features": int(len(view_categories)),
                }
            )
            print(
                f"view={view} fold={fold}/{FOLDS} AUC={auc:.6f} "
                f"features={len(columns)}",
                flush=True,
            )
            del model

    if not np.array_equal(fold_ids, baseline["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Multi-view folds differ from baseline folds")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Multi-view OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {ID: train[ID], TARGET: y, "fold": fold_ids, **{
            f"oof_{view}": values for view, values in predictions.items()
        }}
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    summary = {
        "status": "exploratory fold-isolated multi-view OOF; repeated-label development",
        "inputs": "official train.csv and row-alignment metadata from prior OOF; test.csv not read",
        "view_features": {name: columns for name, columns in views.items()},
        "outer_folds": FOLDS,
        "seed": SEED,
        "model": {
            "family": "CatBoost GPU",
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.05,
        },
        "versions": {
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fold_metrics": records,
        "mean_fold_auc": {
            view: float(
                np.mean(
                    [record["auc"] for record in records if record["view"] == view]
                )
            )
            for view in VIEWS
        },
        "rule_controls": {
            "external_data": "none",
            "test_rows": "not read",
            "label_transfer": "none",
            "train_test_pooling": "none",
            "row_routing": "none; views are global models with fixed feature sets",
            "manual_prediction_changes": "none",
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
