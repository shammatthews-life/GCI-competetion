from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pipeline_core_v2 import (
    DEFAULT_DATA_DIR,
    EXPECTED_INPUT_SHA256,
    ID_COLUMN,
    TARGET_COLUMN,
    _assert_inputs,
    _xgb_matrix,
    compute_fold_target_encoding,
    make_features,
    seed_everything,
    sha256_file,
)


def build_submission(
    output_dir: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 314159,
    folds: int = 5,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output_dir}")
    output_dir.mkdir(parents=True)
    seed_everything(seed)

    input_hashes = {}
    for filename, expected_hash in EXPECTED_INPUT_SHA256.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required organizer input not found: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Input hash mismatch for {filename}: {actual_hash}")
        input_hashes[filename] = actual_hash

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    _assert_inputs(train, test, sample)
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    x_all = make_features(train, "rowwise_v1")
    x_test = make_features(test, "rowwise_v1")
    if list(x_all.columns) != list(x_test.columns):
        raise AssertionError("Training and test base-feature columns differ.")

    splits = list(
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y
        )
    )
    oof = np.full(len(y), np.nan, dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int8)
    test_fold_predictions = []
    fold_results = []
    started = time.time()
    print(
        f"[v06_submission_predict_once] rows={len(train)} folds={folds} "
        f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}",
        flush=True,
    )

    for fold_number, (fit_idx, valid_idx) in enumerate(splits, start=1):
        if np.intersect1d(fit_idx, valid_idx).size:
            raise AssertionError("Outer training and validation rows overlap.")
        frame_fit = train.iloc[fit_idx].reset_index(drop=True)
        frame_valid = train.iloc[valid_idx].reset_index(drop=True)
        y_fit = y[fit_idx]
        te_fit, te_valid = compute_fold_target_encoding(
            frame_fit, y_fit, frame_valid, seed=seed + fold_number
        )
        _, te_test = compute_fold_target_encoding(
            frame_fit, y_fit, test.reset_index(drop=True), seed=seed + fold_number
        )

        x_fit = pd.concat(
            [x_all.iloc[fit_idx].reset_index(drop=True), te_fit], axis=1
        )
        x_valid = pd.concat(
            [x_all.iloc[valid_idx].reset_index(drop=True), te_valid], axis=1
        )
        x_test_fold = pd.concat([x_test.reset_index(drop=True), te_test], axis=1)
        x_query = pd.concat([x_valid, x_test_fold], ignore_index=True)
        fit_matrix, query_matrix = _xgb_matrix(x_fit, x_query)

        model_seed = seed + fold_number * 100
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=60,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=15.0,
            max_bin=256,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            device="cuda",
            n_jobs=8,
            random_state=model_seed,
        )
        model.fit(fit_matrix, y_fit, verbose=False)
        booster = model.get_booster()
        booster.set_param({"device": "cpu"})
        query_predictions = booster.inplace_predict(query_matrix)
        valid_predictions = np.asarray(query_predictions[: len(valid_idx)], dtype=np.float64)
        test_predictions = np.asarray(query_predictions[len(valid_idx) :], dtype=np.float64)
        if valid_predictions.shape != (len(valid_idx),) or not np.isfinite(valid_predictions).all():
            raise AssertionError(f"Invalid OOF predictions in fold {fold_number}.")
        if test_predictions.shape != (len(test),) or not np.isfinite(test_predictions).all():
            raise AssertionError(f"Invalid test predictions in fold {fold_number}.")

        oof[valid_idx] = valid_predictions
        oof_count[valid_idx] += 1
        test_fold_predictions.append(test_predictions)
        fold_auc = float(roc_auc_score(y[valid_idx], valid_predictions))
        fold_results.append(
            {
                "fold": fold_number,
                "n_fit": len(fit_idx),
                "n_valid": len(valid_idx),
                "xgboost_gpu_auc": fold_auc,
            }
        )
        print(
            f"  fold {fold_number}/{folds}: auc={fold_auc:.6f} "
            "(one GPU fit predicts validation and test)",
            flush=True,
        )

    if not np.all(oof_count == 1) or not np.isfinite(oof).all():
        raise AssertionError("OOF predictions do not cover every training row exactly once.")
    if len(test_fold_predictions) != folds:
        raise AssertionError("Test prediction folds are incomplete.")

    averaged_test = np.mean(np.column_stack(test_fold_predictions), axis=1)
    if not np.isfinite(averaged_test).all() or np.any((averaged_test < 0) | (averaged_test > 1)):
        raise AssertionError("Averaged test predictions are invalid.")
    submission = pd.DataFrame(
        {ID_COLUMN: sample[ID_COLUMN].to_numpy(), TARGET_COLUMN: averaged_test}
    )
    if list(submission.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise AssertionError("Submission columns do not match the official schema.")
    if not np.array_equal(submission[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy()):
        raise AssertionError("Submission IDs/order do not match test.csv.")

    oof_frame = pd.DataFrame({ID_COLUMN: train[ID_COLUMN].to_numpy(), "OOF_xgboost_gpu": oof})
    oof_path = output_dir / "oof_predictions.csv"
    submission_path = output_dir / "submission_xgboost_gpu.csv"
    oof_frame.to_csv(oof_path, index=False)
    submission.to_csv(submission_path, index=False)
    source_hashes = {
        "pipeline_core_v2.py": sha256_file(Path(__file__).with_name("pipeline_core_v2.py")),
        "build_v06_submission.py": sha256_file(Path(__file__).resolve()),
        "configs\\v06_nested_te_xgboost_submission.json": sha256_file(
            Path(__file__).parent
            / "configs"
            / "v06_nested_te_xgboost_submission.json"
        ),
    }
    metrics = {
        "candidate_id": "v06_nested_te_xgboost_submission",
        "seed": seed,
        "folds": folds,
        "feature_set": "rowwise_v1",
        "target_encoding": True,
        "fit_count_per_fold": 1,
        "fold_results": fold_results,
        "model_summary": {
            "xgboost_gpu": {
                "pooled_oof_auc": float(roc_auc_score(y, oof)),
                "mean_fold_auc": float(
                    np.mean([result["xgboost_gpu_auc"] for result in fold_results])
                ),
                "std_fold_auc": float(
                    np.std([result["xgboost_gpu_auc"] for result in fold_results], ddof=1)
                ),
                "fold_auc": [result["xgboost_gpu_auc"] for result in fold_results],
            }
        },
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "oof_sha256": sha256_file(oof_path),
        "submission_sha256": sha256_file(submission_path),
        "submission_files": [submission_path.name],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cuda_runtime": torch.version.cuda,
        },
        "seconds": round(time.time() - started, 3),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["model_summary"], indent=2), flush=True)
    print(f"submission={submission_path.resolve()}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    build_submission(args.output_dir, args.data_dir, args.seed, args.folds)
