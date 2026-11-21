from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE_OOF = (
    ROOT
    / "ideas_v1"
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "catboost_engineered_expert_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005


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


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    if not TRAIN.is_file() or not BASELINE_OOF.is_file():
        raise FileNotFoundError("Required official training or benchmark OOF file is missing")
    gpu_count = int(get_gpu_device_count())
    if gpu_count < 1:
        raise RuntimeError("CatBoost cannot see a CUDA GPU; refusing CPU fallback")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    y, fold_ids = validate_alignment(train, baseline)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_scores).all():
        raise ValueError("Baseline OOF scores are non-finite")

    features, categorical = make_features(raw)
    categorical_indices = [features.columns.get_loc(column) for column in categorical]
    if not np.isfinite(
        features.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).all():
        # CatBoost accepts NaNs, but not infinities; NaNs remain as native missing values.
        numeric = features.select_dtypes(include=[np.number]).to_numpy(
            dtype=np.float64
        )
        if np.isinf(numeric).any():
            raise ValueError("Engineered feature matrix contains infinities")

    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in range(1, FOLDS + 1):
        fold_started = time.perf_counter()
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
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
            cat_features=categorical_indices,
        )
        prediction = model.predict_proba(features.iloc[valid_rows])[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: CatBoost predictions are non-finite")
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
                "catboost_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - baseline_auc,
                "blend_delta": blend_auc - baseline_auc,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "feature_count": int(features.shape[1]),
                "categorical_feature_count": int(len(categorical_indices)),
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={baseline_auc:.6f} "
            f"catboost_expert={expert_auc:.6f} blend={blend_auc:.6f} "
            f"seconds={records[-1]['runtime_seconds']:.1f}",
            flush=True,
        )
        del model

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
            "catboost_engineered_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["catboost_expert_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated CatBoost expert; not an independent estimate",
        "hypothesis": (
            "CatBoost's native ordered categorical processing with established "
            "safe financial and external-score features may provide a "
            "complementary ranking stream."
        ),
        "outer_folds": FOLDS,
        "seed": SEED,
        "model": {
            "family": "CatBoost GPU",
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 5.0,
            "border_count": 128,
            "max_ctr_complexity": 1,
        },
        "fixed_blend": (
            f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{BLEND_WEIGHT:.2f} CatBoost expert within-fold rank"
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
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "features": (
                "existing constant-free feature builder using supplied fields, "
                "guarded formulas, categorical native features, and ratio keys"
            ),
            "labels": "used only for outer-fit CatBoost training and OOF scoring",
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only for alignment, fold checks, and benchmark comparison"
            ),
            "gpu": "CatBoost CUDA required; no CPU fallback",
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
