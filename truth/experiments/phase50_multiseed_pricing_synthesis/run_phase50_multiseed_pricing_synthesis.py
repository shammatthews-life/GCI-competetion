from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


PHASE49 = Path(__file__).resolve().parents[1] / "phase49_unified_contract_pricing_specialists"
PHASE37_OOF = (
    Path(__file__).resolve().parents[1]
    / "phase37_multiseed_xgboost_rules_first"
    / "runs"
    / "phase37_multiseed_20261107T173127Z"
    / "PHASE37_OOF_PREDICTIONS.csv"
)
PHASE49_SEED13_OOF = (
    PHASE49
    / "runs"
    / "phase49_pricing_experts_20261108T112704Z"
    / "PHASE49_OOF_PREDICTIONS.csv"
)
sys.path.insert(0, str(PHASE49))
import run_phase49_unified_contract_pricing_specialists as phase49


ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"
SEEDS = (20261113, 20261114, 20261115)
EXPERT_PREDICTION = "oof_cross_model_pricing_expert_blend"
GLOBAL_PREDICTION = "oof_cross_model_pricing_global_blend"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_oof(frame: pd.DataFrame, reference: pd.DataFrame, label: str) -> None:
    if (
        len(frame) != len(reference)
        or not frame[ID].is_unique
        or not frame[ID].equals(reference[ID])
        or not frame[TARGET].equals(reference[TARGET])
    ):
        raise ValueError(f"{label} OOF rows do not align with reference IDs/targets")


def metric_record(
    y: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains nonfinite values")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("Prediction values must be in [0, 1]")
    fold_auc = [
        float(roc_auc_score(y[folds == fold], prediction[folds == fold]))
        for fold in sorted(np.unique(folds))
    ]
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "pooled_log_loss": float(log_loss(y, prediction)),
        "mean_fold_auc": float(np.mean(fold_auc)),
        "std_fold_auc": float(np.std(fold_auc)),
    }


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, PHASE37_OOF, PHASE49_SEED13_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must include unique IDs and TARGET")
    reference = train[[ID, TARGET]].reset_index(drop=True)
    phase37 = pd.read_csv(PHASE37_OOF)
    phase49_seed13 = pd.read_csv(PHASE49_SEED13_OOF)
    validate_oof(phase37, reference, "Phase 37")
    validate_oof(phase49_seed13, reference, "Phase 49 seed 13")

    seed_frames: dict[int, pd.DataFrame] = {20261113: phase49_seed13}
    for seed in SEEDS[1:]:
        seed_output = output_dir / f"seed_{seed}"
        phase49.SEED = seed
        phase49.run(train_path, seed_output)
        seed_frames[seed] = pd.read_csv(seed_output / "PHASE49_OOF_PREDICTIONS.csv")
        validate_oof(seed_frames[seed], reference, f"Phase 49 seed {seed}")

    seed_predictions = {
        seed: frame[EXPERT_PREDICTION].to_numpy(dtype=np.float32)
        for seed, frame in seed_frames.items()
    }
    seed_global_predictions = {
        seed: frame[GLOBAL_PREDICTION].to_numpy(dtype=np.float32)
        for seed, frame in seed_frames.items()
    }
    phase49_expert_average = np.mean(
        np.vstack([seed_predictions[seed] for seed in SEEDS]), axis=0
    ).astype(np.float32)
    phase49_global_average = np.mean(
        np.vstack([seed_global_predictions[seed] for seed in SEEDS]), axis=0
    ).astype(np.float32)
    phase37_prediction = phase37["oof_equal_blend_lgbm_xgb_three_seeds"].to_numpy(
        dtype=np.float32
    )

    # Equal weights are fixed in advance; no OOF labels are used for weight selection.
    candidate_expert = ((phase37_prediction + phase49_expert_average) / 2.0).astype(
        np.float32
    )
    candidate_global = ((phase37_prediction + phase49_global_average) / 2.0).astype(
        np.float32
    )

    y = reference[TARGET].to_numpy(dtype=np.int8)
    folds = phase37["fold_seed_20261113"].to_numpy(dtype=np.int8)
    output = pd.DataFrame(
        {
            ID: reference[ID],
            TARGET: y,
            FOLD: folds,
            "oof_phase37_reference": phase37_prediction,
            "oof_phase49_expert_seed_20261113": seed_predictions[20261113],
            "oof_phase49_expert_seed_20261114": seed_predictions[20261114],
            "oof_phase49_expert_seed_20261115": seed_predictions[20261115],
            "oof_phase49_expert_three_seed_average": phase49_expert_average,
            "oof_phase49_global_three_seed_average": phase49_global_average,
            "oof_equal_phase37_phase49_expert": candidate_expert,
            "oof_equal_phase37_phase49_global": candidate_global,
        }
    )
    if output.isna().any().any():
        raise ValueError("Phase 50 OOF output contains missing values")
    oof_path = output_dir / "PHASE50_OOF_PREDICTIONS.csv"
    output.to_csv(oof_path, index=False)

    metrics = {
        "phase37_reference": metric_record(y, phase37_prediction, folds),
        "phase49_expert_seed_20261113": metric_record(
            y, seed_predictions[20261113], folds
        ),
        "phase49_expert_three_seed_average": metric_record(
            y, phase49_expert_average, folds
        ),
        "phase49_global_three_seed_average": metric_record(
            y, phase49_global_average, folds
        ),
        "equal_phase37_phase49_expert": metric_record(y, candidate_expert, folds),
        "equal_phase37_phase49_global": metric_record(y, candidate_global, folds),
    }
    run_summaries = {
        seed: json.loads(
            (output_dir / f"seed_{seed}" / "PHASE49_SUMMARY.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS[1:]
    }
    summary: dict[str, Any] = {
        "phase": "Phase 50 multi-seed Phase 49 pricing synthesis",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": len(reference),
        "seeds": list(SEEDS),
        "outer_folds": 5,
        "weight_selection": "fixed equal blends; no OOF-label-based weight tuning",
        "phase49_model": "unchanged Phase 49 runner/configuration; seed varied only",
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV plus prior OOF artifacts; no test or external data",
            "base_predictions": "each component is OOF for every row in its own seed split",
            "blend_weights": "fixed 0.5/0.5",
            "target_encoding": "none",
            "test_statistics": "none",
            "manual_prediction_edits": "none",
            "empirical_pricing_constants": "Phase 49 constants retained; rule interpretation requires organizer clarification",
        },
        "input_sha256": {
            str(path): sha256_file(path)
            for path in (train_path, PHASE37_OOF, PHASE49_SEED13_OOF)
        },
        "seed_runs": {
            str(seed): {
                "summary_path": str(output_dir / f"seed_{seed}" / "PHASE49_SUMMARY.json"),
                "metrics": run_summaries[seed]["metrics"],
            }
            for seed in SEEDS[1:]
        },
        "output_sha256": {str(oof_path): sha256_file(oof_path)},
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "PHASE50_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, result in metrics.items():
        print(
            f"{name}: AUC={result['pooled_auc']:.9f} "
            f"logloss={result['pooled_log_loss']:.9f}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, empty output directory for this Phase 50 run",
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
