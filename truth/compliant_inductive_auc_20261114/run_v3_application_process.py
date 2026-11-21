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
from scipy.stats import rankdata
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
PHASE37_OOF_DEFAULT = Path(
    r"D:\gci\truth\experiments\phase37_multiseed_xgboost_rules_first"
    r"\runs\phase37_multiseed_20261107T173127Z\PHASE37_OOF_PREDICTIONS.csv"
)
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
MINIMUM_ITERATIONS = 100
PROCESS_MISSING_COLUMNS = (
    "OWN_CAR_AGE",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_process_features(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = set(PROCESS_MISSING_COLUMNS).difference(raw.columns)
    required = missing_columns.union(
        {"DAYS_BIRTH", "DAYS_EMPLOYED", "NAME_CONTRACT_TYPE"}
    )
    absent = required.difference(raw.columns)
    if absent:
        raise ValueError(f"Application-process fields missing: {sorted(absent)}")
    missing = raw.loc[:, PROCESS_MISSING_COLUMNS].isna()
    flags = {
        f"PROCESS_MISSING_{column}": missing[column].to_numpy(dtype=np.int8)
        for column in PROCESS_MISSING_COLUMNS
    }
    employment = pd.to_numeric(
        raw["DAYS_EMPLOYED"], errors="raise"
    ).to_numpy(dtype=np.float64)
    flags["PROCESS_EMPLOYMENT_SENTINEL"] = (employment == 365243.0).astype(
        np.int8
    )
    age = -pd.to_numeric(raw["DAYS_BIRTH"], errors="raise").to_numpy(
        dtype=np.float64
    ) / 365.25
    cash = raw["NAME_CONTRACT_TYPE"].astype("string").eq("Cash loans").to_numpy(
        dtype=np.int8
    )
    flags["PROCESS_MISSING_COUNT"] = missing.sum(axis=1).to_numpy(
        dtype=np.int8
    )
    for column in PROCESS_MISSING_COLUMNS:
        flag = missing[column].to_numpy(dtype=np.float64)
        flags[f"PROCESS_MISSING_{column}_X_AGE"] = flag * age
        flags[f"PROCESS_MISSING_{column}_X_CASH"] = flag * cash
    result = pd.DataFrame(flags).replace([np.inf, -np.inf], np.nan)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise ValueError("Application-process features contain nonfinite values")
    return result


def require_cuda(model: Any) -> None:
    config = json.loads(model.get_booster().save_config())
    device = config["learner"]["generic_param"].get("device", "")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"XGBoost did not train on CUDA; detected device={device!r}")


def fit_predict_fold(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    y: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, int]:
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
    selector = phase38._model(fold_seed, MAX_ITERATIONS, stopping=True)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_stop, y[inner_stop])],
        verbose=False,
    )
    require_cuda(selector)
    rounds = max(MINIMUM_ITERATIONS, int(selector.best_iteration) + 1)
    del selector, x_inner_fit, x_inner_stop

    x_fit, x_valid = phase38.make_matrices(
        blocks, fit_rows, valid_rows, KEY_MIN_FREQUENCY
    )
    model = phase38._model(fold_seed, rounds, stopping=False)
    model.fit(x_fit, y[fit_rows], verbose=False)
    require_cuda(model)
    prediction = model.predict_proba(x_valid)[:, 1]
    del model
    if not np.isfinite(prediction).all():
        raise ValueError("Application-process model produced nonfinite predictions")
    return prediction.astype(np.float32), rounds


def fold_metrics(
    y: np.ndarray, scores: np.ndarray, folds: np.ndarray
) -> dict[str, object]:
    per_fold = [
        float(roc_auc_score(y[folds == fold], scores[folds == fold]))
        for fold in range(1, FOLDS + 1)
    ]
    return {"mean_fold_auc": float(np.mean(per_fold)), "fold_auc": per_fold}


def rank_within_folds(scores: np.ndarray, folds: np.ndarray) -> np.ndarray:
    result = np.full(len(scores), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = folds == fold
        result[mask] = (rankdata(scores[mask], method="average") - 1.0) / (
            int(mask.sum()) - 1.0
        )
    if not np.isfinite(result).all():
        raise ValueError("Foldwise rank transform produced invalid values")
    return result


def run(train_path: Path, phase37_oof_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase37_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    train_hash = sha256_file(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Official training data hash mismatch: {train_hash}")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must contain unique IDs and TARGET")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, (0, 1)).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks_list = list(phase33.build_safe_blocks(raw))
    blocks_list[2] = pd.concat(
        [
            blocks_list[2].reset_index(drop=True),
            make_process_features(raw).reset_index(drop=True),
        ],
        axis=1,
    )
    blocks = tuple(blocks_list)

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        prediction, rounds = fit_predict_fold(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        oof[valid_rows] = prediction
        fold_auc = float(roc_auc_score(y[valid_rows], prediction))
        fold_records.append(
            {
                "fold": fold_number,
                "outer_auc": fold_auc,
                "iterations": rounds,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} application_process_auc="
            f"{fold_auc:.6f} iterations={rounds}",
            flush=True,
        )
    if np.any(fold_ids < 1) or not np.isfinite(oof).all():
        raise ValueError("Application-process OOF predictions are incomplete")

    reference = pd.read_csv(phase37_oof_path)
    required_reference = {
        ID,
        TARGET,
        "fold_seed_20261113",
        "oof_equal_blend_lgbm_xgb_three_seeds",
    }
    if not required_reference.issubset(reference.columns):
        raise ValueError("Phase 37 OOF is missing required columns")
    if (
        len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or not np.array_equal(
            reference["fold_seed_20261113"].to_numpy(dtype=np.int8), fold_ids
        )
    ):
        raise ValueError("Phase 37 OOF rows or folds do not align")
    reference_oof = reference[
        "oof_equal_blend_lgbm_xgb_three_seeds"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(reference_oof).all():
        raise ValueError("Phase 37 OOF contains nonfinite predictions")

    rank_blend = 0.5 * rank_within_folds(reference_oof, fold_ids) + 0.5 * (
        rank_within_folds(oof, fold_ids)
    )
    probability_blend = 0.5 * reference_oof + 0.5 * oof

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    oof_path = output_dir / "V3_OOF_PREDICTIONS.csv"
    fold_path = output_dir / "V3_FOLD_RESULTS.csv"
    summary_path = output_dir / "V3_SUMMARY.json"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_application_process_xgb": oof,
            "oof_phase37_reference": reference_oof,
            "oof_rank_blend_50_50": rank_blend,
            "oof_probability_blend_50_50": probability_blend,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    summary = {
        "experiment": "V3 fold-local application-process feature family",
        "status": "exploratory development evidence; not a leaderboard score",
        "rows": int(len(train)),
        "positive_rate": float(np.mean(y)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "features": {
            "source": "deterministic row-wise values from supplied train columns",
            "missingness_flags": list(PROCESS_MISSING_COLUMNS),
            "interactions": "each missingness flag times age and cash-contract indicator",
            "target_or_other_row_labels_used": False,
        },
        "metrics": {
            "application_process_xgb": fold_metrics(y, oof, fold_ids),
            "phase37_reference": fold_metrics(y, reference_oof, fold_ids),
            "rank_blend_50_50": fold_metrics(y, rank_blend, fold_ids),
            "probability_blend_50_50": fold_metrics(
                y, probability_blend, fold_ids
            ),
        },
        "rule_controls": {
            "training_data": "official train.csv only; SHA-256 enforced",
            "test_or_external_data_read": False,
            "target_encoding_or_peer_label_transfer": False,
            "train_test_covariate_pooling": False,
            "outer_validation_used_for_fitting_or_stopping": False,
            "blend_weights": "fixed equal weights; not fitted on labels",
            "manual_prediction_adjustments": False,
            "gpu_required": True,
        },
        "limitations": [
            "All historical training labels have prior project exposure; this is not an untouched holdout.",
            "The Phase 37 reference has prior development exposure and is used only as a fixed OOF comparison component.",
            "Mean-fold AUC is primary because the predictions come from fold-specific models.",
            "No public or private leaderboard score is inferred.",
        ],
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost")
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(train_path): train_hash,
            str(phase37_oof_path): sha256_file(phase37_oof_path),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(PHASE33 / "run_phase33_rules_first_features.py"): sha256_file(
                PHASE33 / "run_phase33_rules_first_features.py"
            ),
            str(PHASE38 / "run_phase38_key_frequency_thresholds.py"): sha256_file(
                PHASE38 / "run_phase38_key_frequency_thresholds.py"
            ),
            str(oof_path): sha256_file(oof_path),
            str(fold_path): sha256_file(fold_path),
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
        "--phase37-oof", type=Path, default=PHASE37_OOF_DEFAULT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / datetime.now(timezone.utc).strftime("v3_application_process_%Y%m%dT%H%M%SZ"),
    )
    arguments = parser.parse_args()
    run(arguments.train, arguments.phase37_oof, arguments.output_dir)
