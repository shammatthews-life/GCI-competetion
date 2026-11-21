from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent / "experiments"
PHASE33 = EXPERIMENTS / "phase33_rules_first_features"
PHASE38 = EXPERIMENTS / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
EXPECTED_TRAIN_SHA256 = (
    "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
CLASS_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
MINIMUM_ITERATIONS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_data(train: pd.DataFrame) -> np.ndarray:
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must contain unique IDs and TARGET")
    values = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or not np.isin(values, (0, 1)).all()
        or set(np.unique(values)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    return values.astype(np.int8)


def require_cuda(model: Any) -> None:
    config = json.loads(model.get_booster().save_config())
    device = config["learner"]["generic_param"].get("device", "")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"XGBoost did not train on CUDA; detected device={device!r}")


def fit_selector(
    x_fit: Any,
    y_fit: np.ndarray,
    x_stop: Any,
    y_stop: np.ndarray,
    seed: int,
    weight: float,
) -> tuple[int, float]:
    model = phase38._model(seed, MAX_ITERATIONS, stopping=True)
    model.set_params(
        scale_pos_weight=weight,
        early_stopping_rounds=STOPPING_ROUNDS,
    )
    model.fit(x_fit, y_fit, eval_set=[(x_stop, y_stop)], verbose=False)
    require_cuda(model)
    rounds = max(MINIMUM_ITERATIONS, int(model.best_iteration) + 1)
    prediction = model.predict_proba(x_stop)[:, 1]
    score = float(roc_auc_score(y_stop, prediction))
    del model
    return rounds, score


def fit_outer_model(
    x_fit: Any,
    y_fit: np.ndarray,
    x_valid: Any,
    seed: int,
    weight: float,
    rounds: int,
) -> np.ndarray:
    model = phase38._model(seed, rounds, stopping=False)
    model.set_params(scale_pos_weight=weight, early_stopping_rounds=None)
    model.fit(x_fit, y_fit, verbose=False)
    require_cuda(model)
    prediction = model.predict_proba(x_valid)[:, 1]
    del model
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite prediction for class weight {weight}")
    return prediction.astype(np.float32)


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    train_hash = sha256_file(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            "Training file hash does not match the official supplied train.csv: "
            f"{train_hash}"
        )

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    y = validate_training_data(train)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    fold_ids = np.full(len(y), -1, dtype=np.int8)
    fixed_oof = {
        weight: np.full(len(y), np.nan, dtype=np.float32)
        for weight in CLASS_WEIGHTS
    }
    selected_oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_fit_idx, inner_stop_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_idx]
        inner_stop = fit_rows[inner_stop_idx]

        x_inner_fit, x_inner_stop = phase38.make_matrices(
            blocks, inner_fit, inner_stop, KEY_MIN_FREQUENCY
        )
        x_outer_fit, x_outer_valid = phase38.make_matrices(
            blocks, fit_rows, valid_rows, KEY_MIN_FREQUENCY
        )

        inner_scores: dict[float, float] = {}
        inner_rounds: dict[float, int] = {}
        outer_predictions: dict[float, np.ndarray] = {}
        for weight in CLASS_WEIGHTS:
            rounds, inner_auc = fit_selector(
                x_inner_fit,
                y[inner_fit],
                x_inner_stop,
                y[inner_stop],
                fold_seed,
                weight,
            )
            inner_scores[weight] = inner_auc
            inner_rounds[weight] = rounds
            outer_predictions[weight] = fit_outer_model(
                x_outer_fit,
                y[fit_rows],
                x_outer_valid,
                fold_seed,
                weight,
                rounds,
            )
            fixed_oof[weight][valid_rows] = outer_predictions[weight]
            fold_auc = float(
                roc_auc_score(y[valid_rows], outer_predictions[weight])
            )
            fold_records.append(
                {
                    "fold": fold_number,
                    "scale_pos_weight": weight,
                    "inner_auc": inner_auc,
                    "iterations": rounds,
                    "outer_auc": fold_auc,
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"fold={fold_number}/{FOLDS} weight={weight:g} "
                f"inner_auc={inner_auc:.6f} outer_auc={fold_auc:.6f} "
                f"iterations={rounds}",
                flush=True,
            )

        selected_weight = max(
            CLASS_WEIGHTS, key=lambda value: (inner_scores[value], -value)
        )
        selected_oof[valid_rows] = outer_predictions[selected_weight]
        selection_records.append(
            {
                "fold": fold_number,
                "selected_scale_pos_weight": selected_weight,
                "selected_inner_auc": inner_scores[selected_weight],
                "selected_iterations": inner_rounds[selected_weight],
                "outer_auc": float(
                    roc_auc_score(y[valid_rows], selected_oof[valid_rows])
                ),
                "inner_auc_by_weight": {
                    str(weight): inner_scores[weight]
                    for weight in CLASS_WEIGHTS
                },
            }
        )

    if np.any(fold_ids < 1) or not np.isfinite(selected_oof).all():
        raise ValueError("Nested-selection OOF output is incomplete")
    if any(not np.isfinite(values).all() for values in fixed_oof.values()):
        raise ValueError("A fixed-weight OOF output is incomplete")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    oof_path = output_dir / "V1_OOF_PREDICTIONS.csv"
    fold_path = output_dir / "V1_FOLD_RESULTS.csv"
    selection_path = output_dir / "V1_WEIGHT_SELECTION.json"
    summary_path = output_dir / "V1_SUMMARY.json"

    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{
                f"oof_scale_pos_weight_{weight:g}": prediction
                for weight, prediction in fixed_oof.items()
            },
            "oof_nested_weight_selection": selected_oof,
        }
    )
    oof_frame.to_csv(oof_path, index=False)
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    pd.DataFrame(selection_records).to_json(
        selection_path, orient="records", indent=2
    )

    metrics: dict[str, dict[str, Any]] = {}
    candidates = {
        **{
            f"fixed_weight_{weight:g}": prediction
            for weight, prediction in fixed_oof.items()
        },
        "nested_weight_selection": selected_oof,
    }
    for name, prediction in candidates.items():
        fold_auc = [
            float(roc_auc_score(y[valid_rows], prediction[valid_rows]))
            for _, valid_rows in folds
        ]
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(np.mean(fold_auc)),
            "fold_auc": fold_auc,
        }
    metrics["nested_weight_selection"]["pooled_auc_note"] = (
        "Different weights may be selected in different outer folds, so this "
        "pooled score is sensitive to between-fold score-scale shifts; "
        "mean_fold_auc is the primary nested-selection metric."
    )

    summary = {
        "experiment": "V1 nested positive-class weighting",
        "status": "exploratory development evidence; not a leaderboard score",
        "rows": int(len(train)),
        "positive_rate": float(np.mean(y)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_selection": {
            "method": "one stratified 12% split per outer training fold",
            "selection_metric": "ROC-AUC",
            "tie_break": "choose the smaller scale_pos_weight",
            "candidate_weights": list(CLASS_WEIGHTS),
        },
        "matrix": {
            "helper": "Phase 38 fold-fitted one-hot/key-frequency matrices",
            "key_min_frequency": KEY_MIN_FREQUENCY,
            "feature_builder": "Phase 33 deterministic supplied-column features",
        },
        "metrics": metrics,
        "fold_selection": selection_records,
        "rule_controls": {
            "input": "supplied training CSV only",
            "test_csv_read": False,
            "external_data_or_prediction_cache": False,
            "target_encoding_or_peer_label_transfer": False,
            "train_test_covariate_pooling": False,
            "outer_validation_used_for_selection_or_stopping": False,
            "manual_prediction_adjustments": False,
            "category_vocabularies_and_key_frequencies": (
                "fit on the applicable model training partition"
            ),
            "gpu_required": True,
        },
        "limitations": [
            "All historical training labels have prior project exposure; this is not an untouched holdout.",
            "The fixed-weight OOF candidates are comparisons, not independent confirmations.",
            "Only the nested-selection estimate evaluates the weight-selection procedure without using outer-fold labels for selection.",
            "No public or private leaderboard score is inferred.",
        ],
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost")
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(train_path): train_hash,
            **{
                str(path): sha256_file(path)
                for path in (
                    Path(__file__).resolve(),
                    PHASE33 / "run_phase33_rules_first_features.py",
                    PHASE38 / "run_phase38_key_frequency_thresholds.py",
                    oof_path,
                    fold_path,
                    selection_path,
                )
            },
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / datetime.now(timezone.utc).strftime("v1_class_weight_%Y%m%dT%H%M%SZ"),
    )
    arguments = parser.parse_args()
    run(arguments.train, arguments.output_dir)
