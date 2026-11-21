from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE49 = (
    Path(__file__).resolve().parents[1]
    / "phase49_unified_contract_pricing_specialists"
)
PHASE49_OOF = (
    PHASE49
    / "runs"
    / "phase49_pricing_experts_20261108T112704Z"
    / "PHASE49_OOF_PREDICTIONS.csv"
)
sys.path.insert(0, str(PHASE49))
import run_phase49_unified_contract_pricing_specialists as phase49


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_TE_FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
STOPPING_ROUNDS = 120
TWIN_SMOOTHING = 10.0
TWIN_KEY_COLUMNS = (
    "birth_registration_gap",
    "birth_id_publication_gap",
    "gender",
    "education",
    "region_rating",
    "unique_missing_key",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_twin_keys(raw: pd.DataFrame) -> pd.DataFrame:
    required = (
        "DAYS_BIRTH",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "REGION_RATING_CLIENT",
    )
    missing = set(required).difference(raw.columns)
    if missing:
        raise ValueError(f"Twin-key columns missing: {sorted(missing)}")
    birth = pd.to_numeric(raw["DAYS_BIRTH"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    registration = pd.to_numeric(
        raw["DAYS_REGISTRATION"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    id_published = pd.to_numeric(
        raw["DAYS_ID_PUBLISH"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid_dates = (
        np.isfinite(birth)
        & np.isfinite(registration)
        & np.isfinite(id_published)
    )
    birth_registration_gap = np.full(len(raw), np.nan, dtype=np.float64)
    birth_id_gap = np.full(len(raw), np.nan, dtype=np.float64)
    birth_registration_gap[valid_dates] = np.abs(
        birth[valid_dates] - registration[valid_dates]
    )
    birth_id_gap[valid_dates] = np.abs(birth[valid_dates] - id_published[valid_dates])
    keys = pd.DataFrame(
        {
            "birth_registration_gap": birth_registration_gap,
            "birth_id_publication_gap": birth_id_gap,
            "gender": raw["CODE_GENDER"].astype("string").fillna("MISSING"),
            "education": raw["NAME_EDUCATION_TYPE"]
            .astype("string")
            .fillna("MISSING"),
            "region_rating": raw["REGION_RATING_CLIENT"]
            .astype("string")
            .fillna("MISSING"),
            "unique_missing_key": "",
        }
    )
    # Missing date components cannot define a reliable shared fingerprint.
    keys.loc[~valid_dates, "unique_missing_key"] = np.arange(
        len(raw), dtype=np.int64
    )[~valid_dates].astype(str)
    return keys


def _apply_group_statistics(
    source_keys: pd.DataFrame,
    source_y: np.ndarray,
    apply_keys: pd.DataFrame,
) -> np.ndarray:
    if len(source_keys) != len(source_y) or len(source_keys) == 0:
        raise ValueError("Group-statistic source rows and targets must align")
    if not np.isfinite(source_y).all() or not np.isin(source_y, (0, 1)).all():
        raise ValueError("Group-statistic targets must be finite binary values")
    prior = float(np.mean(source_y))
    source = source_keys.loc[:, TWIN_KEY_COLUMNS].copy()
    source["target"] = source_y.astype(np.float64)
    grouped = source.groupby(
        list(TWIN_KEY_COLUMNS), sort=False, dropna=False
    )["target"].agg(["sum", "count"])

    apply_index = pd.MultiIndex.from_frame(
        apply_keys.loc[:, TWIN_KEY_COLUMNS], names=TWIN_KEY_COLUMNS
    )
    counts = grouped["count"].reindex(apply_index).fillna(0).to_numpy(
        dtype=np.float64
    )
    sums = grouped["sum"].reindex(apply_index).fillna(0).to_numpy(
        dtype=np.float64
    )
    rates = (sums + TWIN_SMOOTHING * prior) / (
        counts + TWIN_SMOOTHING
    )
    return np.column_stack(
        [
            rates.astype(np.float32),
            np.log1p(counts).astype(np.float32),
            (counts > 0).astype(np.float32),
        ]
    )


def crossfit_group_features(
    keys: pd.DataFrame, y: np.ndarray, seed: int
) -> np.ndarray:
    if len(keys) != len(y):
        raise ValueError("Group keys and targets must align")
    splitter = StratifiedKFold(
        n_splits=INNER_TE_FOLDS, shuffle=True, random_state=seed
    )
    result = np.full((len(y), 3), np.nan, dtype=np.float32)
    for source_rows, apply_rows in splitter.split(np.zeros(len(y)), y):
        result[apply_rows] = _apply_group_statistics(
            keys.iloc[source_rows].reset_index(drop=True),
            y[source_rows],
            keys.iloc[apply_rows].reset_index(drop=True),
        )
    if not np.isfinite(result).all():
        raise ValueError("Cross-fitted group features are incomplete")
    return result


def make_model(seed: int, iterations: int, stopping: bool) -> xgb.XGBClassifier:
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
        random_state=seed,
        n_jobs=8,
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
        verbosity=0,
    )


def make_matrix_pair(
    blocks: tuple[pd.DataFrame, ...],
    raw: pd.DataFrame,
    source_rows: np.ndarray,
    apply_rows: np.ndarray,
    source_y: np.ndarray,
    seed: int,
) -> tuple[Any, Any]:
    base_source, base_apply = phase49.make_matrices_with_pricing(
        blocks,
        raw,
        source_rows,
        apply_rows,
        KEY_MIN_FREQUENCY,
    )
    source_keys = build_twin_keys(raw.iloc[source_rows].reset_index(drop=True))
    apply_keys = build_twin_keys(raw.iloc[apply_rows].reset_index(drop=True))
    source_features = crossfit_group_features(source_keys, source_y, seed)
    apply_features = _apply_group_statistics(
        source_keys, source_y, apply_keys
    )
    return (
        hstack([base_source, source_features], format="csr", dtype=np.float32),
        hstack([base_apply, apply_features], format="csr", dtype=np.float32),
    )


def metric_record(
    y: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> dict[str, float]:
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains nonfinite values")
    if np.any((prediction < 0.0) | (prediction > 1.0)):
        raise ValueError("Predictions must be in [0, 1]")
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
    if not train_path.is_file() or not PHASE49_OOF.is_file():
        raise FileNotFoundError("Training CSV or Phase 49 reference OOF is missing")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must include unique IDs and TARGET")
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
    baseline = pd.read_csv(PHASE49_OOF)
    if (
        len(baseline) != len(train)
        or not baseline[ID].equals(train[ID])
        or not baseline[TARGET].equals(train[TARGET])
        or "oof_xgb_pricing_global" not in baseline
    ):
        raise ValueError("Phase 49 baseline OOF does not align with training data")

    blocks = phase49.phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if "fold" not in baseline or not np.array_equal(
        fold_ids, baseline["fold"].to_numpy(dtype=np.int8)
    ):
        raise ValueError("Phase 49 baseline fold IDs do not match Phase 51")
    predictions = np.full(len(y), np.nan, dtype=np.float32)
    iterations_by_fold: list[int] = []
    twin_feature_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_fit_idx, inner_valid_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_idx]
        inner_valid = fit_rows[inner_valid_idx]

        x_inner_fit, x_inner_valid = make_matrix_pair(
            blocks, raw, inner_fit, inner_valid, y[inner_fit], fold_seed
        )
        selector = make_model(fold_seed, MAX_ITERATIONS, stopping=True)
        selector.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_valid, y[inner_valid])],
            verbose=False,
        )
        iterations = max(100, int(selector.best_iteration) + 1)
        iterations_by_fold.append(iterations)
        del selector, x_inner_fit, x_inner_valid

        x_fit, x_valid = make_matrix_pair(
            blocks, raw, fit_rows, valid_rows, y[fit_rows], fold_seed + 1
        )
        model = make_model(fold_seed, iterations, stopping=False)
        model.fit(x_fit, y[fit_rows], verbose=False)
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite prediction in outer fold {fold_number}")
        predictions[valid_rows] = prediction.astype(np.float32)

        valid_keys = build_twin_keys(raw.iloc[valid_rows].reset_index(drop=True))
        applied_twin = _apply_group_statistics(
            build_twin_keys(raw.iloc[fit_rows].reset_index(drop=True)),
            y[fit_rows],
            valid_keys,
        )
        twin_feature_records.append(
            {
                "fold": fold_number,
                "validation_rows": int(len(valid_rows)),
                "rows_with_training_key_match": int((applied_twin[:, 2] > 0).sum()),
                "matched_row_fraction": float((applied_twin[:, 2] > 0).mean()),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} AUC="
            f"{roc_auc_score(y[valid_rows], prediction):.6f} "
            f"iterations={iterations}",
            flush=True,
        )

    if not np.isfinite(predictions).all():
        raise ValueError("Phase 51 OOF predictions are incomplete")
    baseline_prediction = baseline["oof_xgb_pricing_global"].to_numpy(
        dtype=np.float32
    )
    fixed_blend = ((predictions + baseline_prediction) / 2.0).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = output_dir / "PHASE51_OOF_PREDICTIONS.csv"
    output = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_phase49_xgb_pricing_global": baseline_prediction,
            "oof_xgb_with_fold_local_twin_features": predictions,
            "oof_equal_blend_twin_and_phase49_xgb": fixed_blend,
        }
    )
    output.to_csv(oof_path, index=False)
    metrics = {
        "phase49_xgb_pricing_global_reference": metric_record(
            y, baseline_prediction, fold_ids
        ),
        "xgb_with_fold_local_twin_features": metric_record(
            y, predictions, fold_ids
        ),
        "fixed_50_50_twin_reference_blend": metric_record(
            y, fixed_blend, fold_ids
        ),
    }
    summary: dict[str, Any] = {
        "phase": "Phase 51 fold-local fingerprint label-statistic experiment",
        "status": "exploratory; label-derived feature; rule interpretation unresolved",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_target_statistic_folds": INNER_TE_FOLDS,
        "smoothing": TWIN_SMOOTHING,
        "fingerprint": (
            "absolute DAYS_BIRTH-DAYS_REGISTRATION and "
            "DAYS_BIRTH-DAYS_ID_PUBLISH gaps, CODE_GENDER, "
            "NAME_EDUCATION_TYPE, REGION_RATING_CLIENT; incomplete date keys "
            "are row-unique and cannot match"
        ),
        "features": [
            "smoothed group default rate, cross-fitted for model-fit rows",
            "log1p(group count), cross-fitted for model-fit rows",
            "group-match indicator",
        ],
        "metrics": metrics,
        "fold_iteration_counts": iterations_by_fold,
        "validation_match_support": twin_feature_records,
        "rule_controls": {
            "input": "supplied training CSV and Phase 49 training OOF only",
            "outer_validation_labels": "not used to fit features, model, or stopping",
            "training_target_statistics": (
                "each training row's group label rate/count is cross-fitted "
                "inside its applicable model-fit partition"
            ),
            "test_rows": "not read; no test prediction or test-to-train matching performed",
            "blend_weights": "fixed 0.5/0.5; no label-fitted weight search",
            "manual_prediction_edits": "none",
            "competition_rule_status": (
                "training-label group statistics are not explicitly resolved "
                "in the supplied local rule text; organizer clarification is required"
            ),
        },
        "input_sha256": {
            str(path): sha256_file(path) for path in (train_path, PHASE49_OOF)
        },
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "oof_sha256": sha256_file(oof_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "PHASE51_SUMMARY.json").write_text(
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
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase51_twin_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
