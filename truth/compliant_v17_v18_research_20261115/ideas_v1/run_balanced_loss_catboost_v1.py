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
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE = ROOT / "runs" / "fold_aligned_catboost_blend_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "balanced_loss_catboost_v1"
EXPECTED_TRAIN_SHA256 = "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
SEED = 20261113
FOLDS = 5
EXPERT_WEIGHT = 0.10
ID = "SK_ID_CURR"
TARGET = "TARGET"
BASELINE_PREDICTION = "fixed_50_50_within_fold_rank_blend"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within_fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite run: {OUTPUT}")
    if not TRAIN.is_file() or not BASELINE.is_file():
        raise FileNotFoundError("Required official train or aligned OOF comparator is missing")
    train_hash = sha256_file(TRAIN)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Training file hash differs from the verified input: {train_hash}")
    if int(get_gpu_device_count()) < 1:
        raise RuntimeError("CatBoost cannot see a CUDA GPU; refusing CPU fallback")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE, usecols=[ID, TARGET, "fold", BASELINE_PREDICTION])
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with official train")

    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Training labels must be binary")
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    features, categorical = make_features(raw)
    categorical_indices = [features.columns.get_loc(name) for name in categorical]
    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(split.split(np.zeros(len(y), dtype=np.int8), y))
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        expected_fold_ids[valid_rows] = fold_number
    if not np.array_equal(expected_fold_ids, baseline["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Comparator fold assignments do not match this experiment")

    predictions = np.full(len(y), np.nan, dtype=np.float32)
    fold_records: list[dict[str, int | float]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fit_y = y[fit_rows]
        negative_count = int(np.sum(fit_y == 0))
        positive_count = int(np.sum(fit_y == 1))
        if positive_count == 0:
            raise ValueError(f"Fold {fold_number} has no positive training rows")
        positive_weight = negative_count / positive_count
        model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, positive_weight],
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.72,
            border_count=128,
            one_hot_max_size=10,
            max_ctr_complexity=1,
            random_seed=SEED + fold_number,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            features.iloc[fit_rows],
            fit_y,
            cat_features=categorical_indices,
        )
        prediction = model.predict_proba(features.iloc[valid_rows])[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite prediction in fold {fold_number}")
        predictions[valid_rows] = prediction.astype(np.float32)
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "fit_positive_count": positive_count,
                "fit_negative_count": negative_count,
                "positive_loss_weight": positive_weight,
                "auc_balanced_loss_catboost": float(
                    roc_auc_score(y[valid_rows], prediction)
                ),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} "
            f"positive_weight={positive_weight:.6f} "
            f"AUC={fold_records[-1]['auc_balanced_loss_catboost']:.6f}",
            flush=True,
        )
        del model

    if not np.isfinite(predictions).all():
        raise ValueError("OOF predictions are incomplete or non-finite")
    base_prediction = baseline[BASELINE_PREDICTION].to_numpy(dtype=np.float64)
    candidate = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, int | float]] = []
    base_aucs: list[float] = []
    candidate_aucs: list[float] = []
    for fold_number in range(1, FOLDS + 1):
        valid_rows = np.flatnonzero(expected_fold_ids == fold_number)
        base_rank = within_fold_rank(base_prediction[valid_rows])
        expert_rank = within_fold_rank(predictions[valid_rows])
        candidate[valid_rows] = (1.0 - EXPERT_WEIGHT) * base_rank + EXPERT_WEIGHT * expert_rank
        base_auc = float(roc_auc_score(y[valid_rows], base_prediction[valid_rows]))
        candidate_auc = float(roc_auc_score(y[valid_rows], candidate[valid_rows]))
        base_aucs.append(base_auc)
        candidate_aucs.append(candidate_auc)
        fold_metrics.append(
            {
                "fold": fold_number,
                "baseline_auc": base_auc,
                "candidate_auc": candidate_auc,
                "delta": candidate_auc - base_auc,
            }
        )

    mean_base = float(np.mean(base_aucs))
    mean_candidate = float(np.mean(candidate_aucs))
    fold_deltas = np.asarray(candidate_aucs) - np.asarray(base_aucs)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    model_path = OUTPUT / "MODEL_FOLD_METRICS.csv"
    blend_path = OUTPUT / "BLEND_FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": expected_fold_ids,
            "balanced_loss_catboost": predictions,
            "aligned_baseline": base_prediction,
            "fixed_90_10_rank_blend": candidate,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(fold_records).to_csv(model_path, index=False)
    pd.DataFrame(fold_metrics).to_csv(blend_path, index=False)

    manifest = {
        "status": "training-only exploratory OOF experiment",
        "data_scope": "verified official train.csv only; no test or sample-submission rows read",
        "input_sha256": {
            str(TRAIN): train_hash,
            str(BASELINE): sha256_file(BASELINE),
        },
        "hypothesis": (
            "Balancing total positive and negative Logloss weight may change "
            "tree splits and improve ranking under class imbalance."
        ),
        "loss_weight_formula": "negative_count / positive_count within each outer training fold",
        "blend": {
            "rule": "90% within-fold rank of aligned reference + 10% within-fold rank of weighted CatBoost",
            "expert_weight": EXPERT_WEIGHT,
            "weight_selected_on_current_folds": False,
        },
        "validation": {
            "type": "five-fold stratified OOF, seed 20261113",
            "baseline_mean_fold_auc": mean_base,
            "candidate_mean_fold_auc": mean_candidate,
            "mean_fold_auc_gain": mean_candidate - mean_base,
            "fold_delta_sample_sd": float(np.std(fold_deltas, ddof=1)),
            "positive_fold_deltas": int(np.sum(fold_deltas > 0)),
            "model_fold_metrics": fold_records,
            "blend_fold_metrics": fold_metrics,
            "selection_note": (
                "The fixed 10% blend is an exploratory test on repeatedly examined labels; "
                "no leaderboard result is inferred."
            ),
        },
        "model": {
            "family": "CatBoost GPU",
            "iterations": 700,
            "depth": 6,
            "learning_rate": 0.05,
            "seed": SEED,
            "fold_seeds": [SEED + i for i in range(1, FOLDS + 1)],
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "external_data": "none",
            "matched_label_transfer": "none",
            "manual_predictions": "none",
            "cross_dataset_statistics": "none; row-wise features are built from each training row independently",
            "validation_labels": "not used in model fitting, feature building, or fold-specific loss weights",
            "cached_predictions": "aligned OOF used only as a comparator, never as model input",
        },
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": {
            str(path): sha256_file(path) for path in (oof_path, model_path, blend_path)
        },
    }
    manifest_path = OUTPUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
