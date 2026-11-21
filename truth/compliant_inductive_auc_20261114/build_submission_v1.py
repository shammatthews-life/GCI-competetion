from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent / "experiments"
PHASE33 = EXPERIMENTS / "phase33_rules_first_features"
PHASE38 = EXPERIMENTS / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38
from run_v3_application_process import (
    EXPECTED_TRAIN_SHA256,
    ID,
    TARGET,
    make_process_features,
    require_cuda,
)


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
TEST_DEFAULT = Path(r"D:\gci\inpuy\test.csv")
SAMPLE_DEFAULT = Path(r"D:\gci\inpuy\sample_submission.csv")
EXPECTED_INPUT_SHA256 = {
    "train.csv": EXPECTED_TRAIN_SHA256,
    "test.csv": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    "sample_submission.csv": "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}
XGB_SEEDS = (20261113, 20261114, 20261115)
PROCESS_SEED = 20261113
LGB_SEED = 20261113
KEY_MIN_FREQUENCY = 50
XGB_MAX_ITERATIONS = 3000
XGB_STOPPING_ROUNDS = 120
LGB_MAX_ITERATIONS = 1600
LGB_STOPPING_ROUNDS = 100
MINIMUM_ITERATIONS = 100
MatrixBuilder = Callable[
    [pd.DataFrame, tuple[pd.DataFrame, ...], np.ndarray, np.ndarray],
    tuple[Any, Any],
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(
    train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame
) -> np.ndarray:
    if TARGET not in train or ID not in train or TARGET in test or ID not in test:
        raise ValueError("Train/test input columns are invalid")
    if test.columns.tolist() != [
        column for column in train.columns if column != TARGET
    ]:
        raise ValueError("Train/test schemas differ")
    if sample.columns.tolist() != [ID, TARGET]:
        raise ValueError("Sample submission columns must be SK_ID_CURR,TARGET")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("Training and test IDs must be unique")
    if not test[ID].reset_index(drop=True).equals(
        sample[ID].reset_index(drop=True)
    ):
        raise ValueError("Test IDs do not match sample submission order")
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


def combined_raw(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    train_raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    test_raw = test.reset_index(drop=True)
    raw = pd.concat([train_raw, test_raw], ignore_index=True)
    train_rows = np.arange(len(train_raw), dtype=np.int64)
    test_rows = np.arange(len(train_raw), len(raw), dtype=np.int64)
    return raw, train_rows, test_rows


def make_phase37_xgb_matrices(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[Any, Any]:
    return phase33.make_matrices(
        raw, blocks, fit_rows, apply_rows, "keys_frequency"
    )


def make_process_xgb_matrices(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[Any, Any]:
    return phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )


def select_xgb_rounds(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    y: np.ndarray,
    train_rows: np.ndarray,
    seed: int,
    weight: float,
    matrix_builder: MatrixBuilder,
) -> int:
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.12, random_state=seed
    )
    fit_idx, stop_idx = next(split.split(np.zeros(len(y)), y))
    inner_fit = train_rows[fit_idx]
    inner_stop = train_rows[stop_idx]
    x_fit, x_stop = matrix_builder(raw, blocks, inner_fit, inner_stop)
    selector = phase38._model(seed, XGB_MAX_ITERATIONS, stopping=True)
    selector.set_params(
        scale_pos_weight=weight,
        early_stopping_rounds=XGB_STOPPING_ROUNDS,
    )
    selector.fit(x_fit, y[fit_idx], eval_set=[(x_stop, y[stop_idx])], verbose=False)
    require_cuda(selector)
    rounds = max(MINIMUM_ITERATIONS, int(selector.best_iteration) + 1)
    del selector, x_fit, x_stop
    return rounds


def fit_xgb_test(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    y: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    seed: int,
    weight: float,
    matrix_builder: MatrixBuilder,
) -> tuple[np.ndarray, int]:
    rounds = select_xgb_rounds(
        raw, blocks, y, train_rows, seed, weight, matrix_builder
    )
    x_fit, x_test = matrix_builder(raw, blocks, train_rows, test_rows)
    model = phase38._model(seed, rounds, stopping=False)
    model.set_params(scale_pos_weight=weight, early_stopping_rounds=None)
    model.fit(x_fit, y, verbose=False)
    require_cuda(model)
    prediction = model.predict_proba(x_test)[:, 1]
    del model, x_fit, x_test
    if not np.isfinite(prediction).all():
        raise ValueError(f"XGBoost seed {seed} produced nonfinite scores")
    return prediction.astype(np.float64), rounds


def select_lgb_rounds(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    y: np.ndarray,
    train_rows: np.ndarray,
    seed: int,
) -> int:
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.12, random_state=seed
    )
    fit_idx, stop_idx = next(split.split(np.zeros(len(y)), y))
    inner_fit = train_rows[fit_idx]
    inner_stop = train_rows[stop_idx]
    x_fit, x_stop = phase33.make_matrices(
        raw, blocks, inner_fit, inner_stop, "keys_frequency"
    )
    selector = phase33._model(seed, LGB_MAX_ITERATIONS)
    selector.fit(
        x_fit,
        y[fit_idx],
        eval_X=x_stop,
        eval_y=y[stop_idx],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(LGB_STOPPING_ROUNDS, verbose=False)],
    )
    rounds = max(MINIMUM_ITERATIONS, int(selector.best_iteration_ or 0))
    del selector, x_fit, x_stop
    return rounds


def fit_lgb_test(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    y: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    rounds = select_lgb_rounds(raw, blocks, y, train_rows, seed)
    x_fit, x_test = phase33.make_matrices(
        raw, blocks, train_rows, test_rows, "keys_frequency"
    )
    model = phase33._model(seed, rounds)
    model.fit(x_fit, y)
    prediction = model.predict_proba(x_test)[:, 1]
    del model, x_fit, x_test
    if not np.isfinite(prediction).all():
        raise ValueError("LightGBM produced nonfinite scores")
    return prediction.astype(np.float64), rounds


def run(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    input_paths = (train_path, test_path, sample_path)
    input_hashes: dict[str, str] = {}
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
        digest = sha256_file(path)
        expected = EXPECTED_INPUT_SHA256.get(path.name)
        if expected is None or digest != expected:
            raise ValueError(f"Official input hash mismatch for {path.name}: {digest}")
        input_hashes[str(path)] = digest

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    y = validate_inputs(train, test, sample)
    raw, train_rows, test_rows = combined_raw(train, test)

    base_blocks = phase33.build_safe_blocks(raw)
    process_blocks_list = list(base_blocks)
    process_blocks_list[2] = pd.concat(
        [
            process_blocks_list[2].reset_index(drop=True),
            make_process_features(raw).reset_index(drop=True),
        ],
        axis=1,
    )
    process_blocks = tuple(process_blocks_list)

    xgb_predictions: list[np.ndarray] = []
    xgb_rounds: dict[str, int] = {}
    for seed in XGB_SEEDS:
        prediction, rounds = fit_xgb_test(
            raw,
            base_blocks,
            y,
            train_rows,
            test_rows,
            seed,
            weight=1.0,
            matrix_builder=make_phase37_xgb_matrices,
        )
        xgb_predictions.append(prediction)
        xgb_rounds[str(seed)] = rounds
        print(f"xgboost seed={seed} rounds={rounds}", flush=True)

    lgb_prediction, lgb_rounds = fit_lgb_test(
        raw, base_blocks, y, train_rows, test_rows, LGB_SEED
    )
    print(f"lightgbm seed={LGB_SEED} rounds={lgb_rounds}", flush=True)
    phase37_prediction = (
        np.sum(np.vstack(xgb_predictions), axis=0) + lgb_prediction
    ) / 4.0

    process_prediction, process_rounds = fit_xgb_test(
        raw,
        process_blocks,
        y,
        train_rows,
        test_rows,
        PROCESS_SEED,
        weight=1.0,
        matrix_builder=make_process_xgb_matrices,
    )
    print(f"application-process xgboost rounds={process_rounds}", flush=True)
    final_prediction = 0.5 * phase37_prediction + 0.5 * process_prediction

    if (
        len(final_prediction) != len(sample)
        or not np.isfinite(final_prediction).all()
        or np.any(final_prediction < 0)
        or np.any(final_prediction > 1)
    ):
        raise ValueError("Final predictions violate submission length or range")
    submission = pd.DataFrame(
        {ID: sample[ID].to_numpy(), TARGET: final_prediction}
    )
    if submission.columns.tolist() != [ID, TARGET]:
        raise ValueError("Submission columns must be exactly SK_ID_CURR,TARGET")
    if not submission[ID].equals(sample[ID]):
        raise ValueError("Submission IDs or ordering differ from sample")
    if not submission[ID].is_unique:
        raise ValueError("Submission IDs are not unique")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    submission_path = output_dir / "submission_candidate_v1.csv"
    summary_path = output_dir / "SUBMISSION_SUMMARY.json"
    submission.to_csv(submission_path, index=False)
    summary: dict[str, Any] = {
        "candidate": "strictly inductive Phase 37 plus application-process blend",
        "status": "generated reproducible candidate; leaderboard score unknown",
        "submission": {
            "path": str(submission_path),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "ids_unique": bool(submission[ID].is_unique),
            "ids_match_sample_in_order": bool(submission[ID].equals(sample[ID])),
            "prediction_min": float(submission[TARGET].min()),
            "prediction_max": float(submission[TARGET].max()),
        },
        "fixed_ensemble": {
            "phase37_probability_component": {
                "definition": "equal average of three XGBoost seeds and one LightGBM model",
                "xgboost_seeds": list(XGB_SEEDS),
                "xgboost_rounds": xgb_rounds,
                "lightgbm_seed": LGB_SEED,
                "lightgbm_rounds": lgb_rounds,
            },
            "application_process_component": {
                "model": "XGBoost CUDA; row-wise missingness and interactions",
                "seed": PROCESS_SEED,
                "rounds": process_rounds,
            },
            "final_weights": "0.5 Phase 37 probability average + 0.5 application-process probability",
            "reason_for_probability_blend": "avoids fitting score transforms or statistics on the test prediction batch",
        },
        "training_boundary": {
            "official_train_test_sample_hashes_enforced": True,
            "external_data_or_prediction_cache_read": False,
            "target_encoding_or_peer_label_transfer": False,
            "learned_category_vocabulary_and_frequency_fit_rows": "training rows only",
            "test_covariates_used_for_fitted_preprocessing": False,
            "test_covariates_used_only_for_rowwise_features_and_inference": True,
            "model_round_selection": "stratified inner holdout from training rows",
            "manual_prediction_adjustments": False,
            "xgboost_device": "CUDA required and checked after each fit",
            "lightgbm_device": "CPU",
        },
        "validation_context": {
            "development_oof_reference_mean_fold_auc": 0.7660807122170806,
            "development_oof_reference": "V3 fixed 50/50 probability blend on seed-20261113 outer folds",
            "note": "This OOF score is not a test AUC or leaderboard estimate; the public score is unknown until submitted.",
        },
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm")
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            **input_hashes,
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(PHASE33 / "run_phase33_rules_first_features.py"): sha256_file(
                PHASE33 / "run_phase33_rules_first_features.py"
            ),
            str(PHASE38 / "run_phase38_key_frequency_thresholds.py"): sha256_file(
                PHASE38 / "run_phase38_key_frequency_thresholds.py"
            ),
            str(Path(__file__).with_name("run_v3_application_process.py")): sha256_file(
                Path(__file__).with_name("run_v3_application_process.py")
            ),
            str(submission_path): sha256_file(submission_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / datetime.now(timezone.utc).strftime("submission_v1_%Y%m%dT%H%M%SZ"),
    )
    arguments = parser.parse_args()
    run(
        arguments.train,
        arguments.test,
        arguments.sample,
        arguments.output_dir,
    )
