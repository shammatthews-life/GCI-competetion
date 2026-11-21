from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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
OUTPUT = IDEAS_DIR / "runs" / "xgboost_maxbin512_rank_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
MAX_BIN = 512
BASELINE_COLUMN = "candidate_te20_blend"
RANK_BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def make_model(seed: int, iterations: int, early_stopping: bool) -> xgb.XGBClassifier:
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
        max_bin=MAX_BIN,
        random_state=seed,
        n_jobs=8,
        early_stopping_rounds=STOPPING_ROUNDS if early_stopping else None,
        verbosity=0,
    )


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


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    if xgb.__version__ != "3.4.1":
        raise RuntimeError(f"Expected XGBoost 3.4.1, found {xgb.__version__}")
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
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold in range(1, FOLDS + 1):
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        fold_seed = SEED + fold * 100
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.12,
            random_state=fold_seed,
        )
        inner_fit_local, inner_valid_local = next(
            inner.split(np.zeros(len(fit_rows), dtype=np.int8), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_local]
        inner_valid = fit_rows[inner_valid_local]
        x_inner_fit, x_inner_valid = phase33.make_matrices(
            raw, blocks, inner_fit, inner_valid, "keys_frequency"
        )

        selector = make_model(fold_seed, MAX_ITERATIONS, early_stopping=True)
        selector.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_valid, y[inner_valid])],
            verbose=False,
        )
        rounds = max(100, int(selector.best_iteration) + 1)
        del selector, x_inner_fit, x_inner_valid

        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, fit_rows, valid_rows, "keys_frequency"
        )
        model = make_model(fold_seed, rounds, early_stopping=False)
        model.fit(x_fit, y[fit_rows], verbose=False)
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: XGBoost predictions are non-finite")
        expert_oof[valid_rows] = prediction.astype(np.float32)

        base_rank = fold_rank(baseline_scores[valid_rows])
        expert_rank = fold_rank(prediction)
        blend_oof[valid_rows] = (
            (1.0 - RANK_BLEND_WEIGHT) * base_rank
            + RANK_BLEND_WEIGHT * expert_rank
        ).astype(np.float32)
        base_auc = float(roc_auc_score(y[valid_rows], baseline_scores[valid_rows]))
        expert_auc = float(roc_auc_score(y[valid_rows], prediction))
        blend_auc = float(roc_auc_score(y[valid_rows], blend_oof[valid_rows]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "xgboost_maxbin512_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - base_auc,
                "blend_delta": blend_auc - base_auc,
                "selected_rounds": rounds,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "feature_count": int(x_fit.shape[1]),
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={base_auc:.6f} "
            f"xgb_maxbin512={expert_auc:.6f} "
            f"fixed_blend={blend_auc:.6f} rounds={rounds}",
            flush=True,
        )
        del model, x_fit, x_valid

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
            "candidate_te20_blend": baseline_scores,
            "xgboost_maxbin512": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(
        np.mean([row["xgboost_maxbin512_auc"] for row in records])
    )
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated exploratory experiment; not an independent estimate",
        "hypothesis": (
            "A finer histogram grid may preserve useful split thresholds in "
            "skewed financial ratios and bureau-score features."
        ),
        "max_bin": MAX_BIN,
        "n_estimators_cap": MAX_ITERATIONS,
        "early_stopping_rounds": STOPPING_ROUNDS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "fixed_blend": (
            f"{1.0 - RANK_BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{RANK_BLEND_WEIGHT:.2f} XGBoost within-fold rank"
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
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "labels": "used only for fold-local training and AUC reporting",
            "preprocessing": "one-hot/key-frequency matrices fit on each training partition",
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only as a fold-aligned benchmark; never used as a model feature"
            ),
            "gpu": "CUDA required; XGBoost configured with device=cuda",
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
    summary_path = OUTPUT / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
