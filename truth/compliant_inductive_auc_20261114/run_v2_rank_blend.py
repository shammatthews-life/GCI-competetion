from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLDS = 5
PRIMARY_OOF_DEFAULT = Path(
    r"D:\gci\truth\experiments\phase37_multiseed_xgboost_rules_first"
    r"\runs\phase37_multiseed_20261107T173127Z\PHASE37_OOF_PREDICTIONS.csv"
)
AUXILIARY_OOF_DEFAULT = (
    ROOT
    / "runs"
    / "v1_class_weight_verified_20261114T214500Z"
    / "V1_OOF_PREDICTIONS.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_aligned_oof(
    primary_path: Path, auxiliary_path: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    primary = pd.read_csv(primary_path)
    auxiliary = pd.read_csv(auxiliary_path)
    required_primary = {
        ID,
        TARGET,
        "fold_seed_20261113",
        "oof_equal_blend_lgbm_xgb_three_seeds",
    }
    required_auxiliary = {
        ID,
        TARGET,
        "fold",
        "oof_scale_pos_weight_2",
    }
    if not required_primary.issubset(primary.columns):
        raise ValueError("Phase 37 OOF artifact is missing required columns")
    if not required_auxiliary.issubset(auxiliary.columns):
        raise ValueError("V1 OOF artifact is missing required columns")
    if (
        len(primary) != len(auxiliary)
        or not primary[ID].is_unique
        or not auxiliary[ID].is_unique
        or not primary[ID].equals(auxiliary[ID])
        or not primary[TARGET].equals(auxiliary[TARGET])
        or not primary["fold_seed_20261113"].equals(auxiliary["fold"])
    ):
        raise ValueError("OOF artifacts do not have identical IDs, targets, and folds")

    y = pd.to_numeric(primary[TARGET], errors="raise").to_numpy(dtype=np.int8)
    folds = primary["fold_seed_20261113"].to_numpy(dtype=np.int8)
    primary_score = primary[
        "oof_equal_blend_lgbm_xgb_three_seeds"
    ].to_numpy(dtype=np.float64)
    auxiliary_score = auxiliary["oof_scale_pos_weight_2"].to_numpy(
        dtype=np.float64
    )
    for label, scores in (
        ("Phase 37", primary_score),
        ("V1 class-weight", auxiliary_score),
    ):
        if not np.isfinite(scores).all():
            raise ValueError(f"{label} OOF predictions contain nonfinite values")
    if not np.isin(y, (0, 1)).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("OOF target must contain both binary classes")
    if set(np.unique(folds)) != set(range(1, FOLDS + 1)):
        raise ValueError("Unexpected fold assignments")
    return primary, y, folds, primary_score, auxiliary_score


def rank_within_folds(scores: np.ndarray, folds: np.ndarray) -> np.ndarray:
    ranked = np.full(len(scores), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = folds == fold
        if int(mask.sum()) < 2:
            raise ValueError(f"Fold {fold} is too small to rank")
        ranked[mask] = (rankdata(scores[mask], method="average") - 1.0) / (
            int(mask.sum()) - 1.0
        )
    if not np.isfinite(ranked).all():
        raise ValueError("Foldwise rank transform produced invalid values")
    return ranked


def fold_metrics(
    y: np.ndarray, scores: np.ndarray, folds: np.ndarray
) -> dict[str, object]:
    per_fold = [
        float(roc_auc_score(y[folds == fold], scores[folds == fold]))
        for fold in range(1, FOLDS + 1)
    ]
    return {
        "mean_fold_auc": float(np.mean(per_fold)),
        "fold_auc": per_fold,
    }


def run(primary_path: Path, auxiliary_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (primary_path, auxiliary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required OOF artifact does not exist: {path}")

    primary, y, folds, primary_score, auxiliary_score = load_aligned_oof(
        primary_path, auxiliary_path
    )
    primary_rank = rank_within_folds(primary_score, folds)
    auxiliary_rank = rank_within_folds(auxiliary_score, folds)
    rank_blend = 0.5 * primary_rank + 0.5 * auxiliary_rank
    probability_blend = 0.5 * primary_score + 0.5 * auxiliary_score

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    oof_path = output_dir / "V2_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "V2_SUMMARY.json"
    pd.DataFrame(
        {
            ID: primary[ID],
            TARGET: y,
            "fold": folds,
            "oof_phase37_reference": primary_score,
            "oof_v1_weight2": auxiliary_score,
            "oof_rank_blend_50_50": rank_blend,
            "oof_probability_blend_50_50": probability_blend,
        }
    ).to_csv(oof_path, index=False)
    summary = {
        "experiment": "V2 fixed rank and probability blending",
        "status": "exploratory OOF ensemble comparison; not a leaderboard score",
        "rows": int(len(y)),
        "fold_assignment": "shared seed-20261113 five-fold stratification",
        "weights": "predeclared equal 0.5/0.5; not fitted on OOF labels",
        "metrics": {
            "phase37_reference": fold_metrics(y, primary_score, folds),
            "v1_fixed_weight_2": fold_metrics(y, auxiliary_score, folds),
            "rank_blend_50_50": fold_metrics(y, rank_blend, folds),
            "probability_blend_50_50": fold_metrics(y, probability_blend, folds),
        },
        "rank_blend_definition": (
            "Within each outer validation fold separately, rank each model's "
            "OOF predictions to [0,1] and average the two ranks."
        ),
        "rule_controls": {
            "inputs": "two row-aligned OOF artifacts derived from supplied training data",
            "test_or_external_data_read": False,
            "OOF_labels_used_for_blend_weights": False,
            "manual_prediction_adjustments": False,
            "outer_validation_rows_used_to_fit_base_models": False,
        },
        "limitations": [
            "The historical Phase 37 model is retained as a fixed OOF component; this is a development comparison, not untouched confirmation.",
            "Only mean-fold AUC is reported because the OOF components were produced by fold-specific models.",
            "No test predictions or leaderboard result are generated.",
        ],
        "sha256": {
            str(primary_path): sha256_file(primary_path),
            str(auxiliary_path): sha256_file(auxiliary_path),
            str(oof_path): sha256_file(oof_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-oof", type=Path, default=PRIMARY_OOF_DEFAULT)
    parser.add_argument(
        "--auxiliary-oof", type=Path, default=AUXILIARY_OOF_DEFAULT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / datetime.now(timezone.utc).strftime("v2_rank_blend_%Y%m%dT%H%M%SZ"),
    )
    arguments = parser.parse_args()
    run(arguments.primary_oof, arguments.auxiliary_oof, arguments.output_dir)
