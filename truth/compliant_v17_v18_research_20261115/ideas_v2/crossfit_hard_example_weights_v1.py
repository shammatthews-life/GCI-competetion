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
OUTPUT = IDEAS_DIR / "runs" / "crossfit_hard_example_weights_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
BASELINE_COLUMN = "candidate_te20_blend"
FEATURE_VARIANT = "all_safe"
BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005
WEIGHT_STRENGTH = 3.0
MODEL_TREES = 700


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


def make_model(seed: int, iterations: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
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
        raise ValueError("Baseline OOF scores are non-finite")
    blocks = phase33.build_safe_blocks(raw)

    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in range(1, FOLDS + 1):
        fold_started = time.perf_counter()
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)

        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold * 1000,
        )
        inner_predictions = np.full(len(fit_rows), np.nan, dtype=np.float32)
        for inner_number, (train_local, valid_local) in enumerate(
            inner_splitter.split(np.zeros(len(fit_rows), dtype=np.int8), y[fit_rows]),
            start=1,
        ):
            inner_fit = fit_rows[train_local]
            inner_valid = fit_rows[valid_local]
            x_inner_fit, x_inner_valid = phase33.make_matrices(
                raw, blocks, inner_fit, inner_valid, FEATURE_VARIANT
            )
            model = make_model(SEED + fold * 10000 + inner_number, MODEL_TREES)
            model.fit(x_inner_fit, y[inner_fit])
            prediction = model.predict_proba(x_inner_valid)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Fold {fold}, inner {inner_number}: non-finite OOF probabilities"
                )
            inner_predictions[valid_local] = prediction.astype(np.float32)
            del model, x_inner_fit, x_inner_valid
            print(
                f"fold={fold} inner={inner_number}/{INNER_FOLDS} OOF complete",
                flush=True,
            )

        if not np.isfinite(inner_predictions).all():
            raise ValueError(f"Fold {fold}: cross-fitted training scores are incomplete")
        fit_labels = y[fit_rows].astype(np.float32)
        weights = 1.0 + WEIGHT_STRENGTH * np.abs(
            fit_labels - inner_predictions.astype(np.float32)
        )
        if (
            not np.isfinite(weights).all()
            or weights.min() < 1.0
            or weights.max() > 1.0 + WEIGHT_STRENGTH + 1e-6
        ):
            raise ValueError(f"Fold {fold}: hard-example weights are invalid")

        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, fit_rows, valid_rows, FEATURE_VARIANT
        )
        weighted_model = make_model(SEED + fold * 100, MODEL_TREES)
        weighted_model.fit(x_fit, y[fit_rows], sample_weight=weights)
        prediction = weighted_model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: weighted-expert scores are non-finite")
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
                "hard_example_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - baseline_auc,
                "blend_delta": blend_auc - baseline_auc,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "mean_weight": float(weights.mean()),
                "mean_positive_weight": float(weights[fit_labels == 1].mean()),
                "mean_negative_weight": float(weights[fit_labels == 0].mean()),
                "max_weight": float(weights.max()),
                "feature_count": int(x_fit.shape[1]),
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={baseline_auc:.6f} "
            f"hard_example_expert={expert_auc:.6f} blend={blend_auc:.6f} "
            f"mean_weight={weights.mean():.3f} "
            f"seconds={records[-1]['runtime_seconds']:.1f}",
            flush=True,
        )
        del weighted_model, x_fit, x_valid

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
            "crossfit_hard_example_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["hard_example_expert_auc"] for row in records]))
    blend_mean = float(np.mean([row["fixed_90_10_rank_blend_auc"] for row in records]))
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated cross-fitted hard-example weighted expert; exploratory",
        "hypothesis": (
            "Supervised training weights based on each outer-training row's "
            "inner-OOF absolute prediction error may improve pairwise ordering "
            "of hard cases while avoiding in-sample weighting."
        ),
        "weight_formula": (
            "w_i = 1 + 3 * abs(y_i - p_i), with p_i generated for row i by "
            "an inner model that did not train on row i"
        ),
        "weight_strength": WEIGHT_STRENGTH,
        "weight_bounds": [1.0, 1.0 + WEIGHT_STRENGTH],
        "feature_variant": FEATURE_VARIANT,
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
            f"{BLEND_WEIGHT:.2f} hard-example expert within-fold rank"
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
            "weight_generation": (
                "within each outer-fit partition, predictions for training "
                "weights are inner out-of-fold; outer validation is excluded "
                "from all weight generation and model fitting"
            ),
            "label_use": (
                "training labels used for model fitting and per-row training "
                "error weights only; outer labels used only for AUC reporting"
            ),
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": "read only for alignment and benchmark comparison",
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
