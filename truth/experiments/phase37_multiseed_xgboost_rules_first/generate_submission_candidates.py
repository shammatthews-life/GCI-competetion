from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
PHASE33 = EXPERIMENTS / "phase33_rules_first_features"
PHASE34_RUN = (
    EXPERIMENTS
    / "phase34_xgboost_rules_first"
    / "runs"
    / "phase34_xgb_20261107T171823Z"
)
PHASE37_RUN = (
    Path(__file__).resolve().parent
    / "runs"
    / "phase37_multiseed_20261107T173127Z"
)
PHASE33_RUN = (
    PHASE33
    / "runs"
    / "phase33_rules_first_20261107T170931Z"
)
TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
TEST_DEFAULT = Path(r"D:\gci\inpuy\test.csv")
SAMPLE_DEFAULT = Path(r"D:\gci\inpuy\sample_submission.csv")
V17_DEFAULT = Path(r"D:\gci\pre info\oct22\GCI_submission_v17.csv")
V18_DEFAULT = Path(r"D:\gci\pre info\oct22\GCI_submission_v18.csv")
HISTORICAL_V22_DEFAULT = Path(r"D:\gci\gemini\v_22_pure_champion_ensemble.csv")

ID = "SK_ID_CURR"
TARGET = "TARGET"
XGB_SEEDS = (20261113, 20261114, 20261115)
XGB_MAX_ITERATIONS = 3000
XGB_STOPPING_ROUNDS = 120
LGBM_MAX_ITERATIONS = 1600
LGBM_STOPPING_ROUNDS = 100
LGBM_CONFIG = {
    "num_leaves": 63,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "max_bin": 127,
}

sys.path.insert(0, str(PHASE33))
import run_phase33_rules_first_features as phase33


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return pd.read_csv(path)


def _median_iterations(
    path: Path, variant: str | None = None, seed: int | None = None
) -> int:
    folds = _required_csv(path, "Fold-results file")
    if variant is not None:
        folds = folds.loc[folds["variant"] == variant]
    if seed is not None:
        if "seed" not in folds:
            raise ValueError(f"No seed column found in {path}")
        folds = folds.loc[folds["seed"] == seed]
    if folds.empty or "iterations" not in folds:
        raise ValueError(f"No iteration counts found in {path}")
    counts = pd.to_numeric(folds["iterations"], errors="raise").to_numpy()
    if not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError(f"Invalid iteration counts in {path}")
    return max(1, int(np.median(counts)))


def _xgb_model(seed: int, iterations: int) -> xgb.XGBClassifier:
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
        verbosity=0,
    )


def _lgbm_model(seed: int, iterations: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        **LGBM_CONFIG,
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def _validate_submission(
    frame: pd.DataFrame, expected_ids: pd.Series, label: str
) -> None:
    if list(frame.columns) != [ID, TARGET]:
        raise ValueError(f"{label} must have exactly {ID},{TARGET} columns")
    if len(frame) != len(expected_ids) or not frame[ID].equals(expected_ids):
        raise ValueError(f"{label} IDs/order do not match test/sample rows")
    target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or ((target < 0) | (target > 1)).any():
        raise ValueError(f"{label} predictions must be finite probabilities")


def _make_phase37_submission(
    train_path: Path,
    test_path: Path,
    sample_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    train = _required_csv(train_path, "Training CSV")
    test = _required_csv(test_path, "Test CSV")
    sample = _required_csv(sample_path, "Sample submission")
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training data must have unique IDs and TARGET")
    if ID not in test or not test[ID].is_unique:
        raise ValueError("Test data must have unique IDs")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("Training predictors and test columns/order differ")
    if list(sample.columns) != [ID, TARGET] or not sample[ID].equals(test[ID]):
        raise ValueError("Sample submission IDs/order do not match test CSV")

    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, (0, 1)).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training TARGET must contain both binary classes")

    oof_path = PHASE37_RUN / "PHASE37_OOF_PREDICTIONS.csv"
    oof = _required_csv(oof_path, "Phase 37 OOF predictions")
    if (
        len(oof) != len(train)
        or not oof[ID].equals(train[ID])
        or not oof[TARGET].equals(train[TARGET])
        or "oof_equal_blend_lgbm_xgb_three_seeds" not in oof
    ):
        raise ValueError("Phase 37 OOF artifact is not aligned with training rows")
    oof_auc = float(
        roc_auc_score(y, oof["oof_equal_blend_lgbm_xgb_three_seeds"].to_numpy())
    )
    if abs(oof_auc - 0.7659681876381996) > 1e-10:
        raise ValueError(f"Unexpected Phase 37 reference OOF AUC: {oof_auc:.12f}")

    lgbm_iterations = _median_iterations(
        PHASE33_RUN / "PHASE33_FOLD_RESULTS.csv", "keys_frequency"
    )
    xgb_iterations = {
        20261113: _median_iterations(PHASE34_RUN / "PHASE34_FOLD_RESULTS.csv"),
        20261114: _median_iterations(
            PHASE37_RUN / "PHASE37_FOLD_RESULTS.csv", seed=20261114
        ),
        20261115: _median_iterations(
            PHASE37_RUN / "PHASE37_FOLD_RESULTS.csv", seed=20261115
        ),
    }

    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    raw = pd.concat([raw_train, raw_test], ignore_index=True)
    blocks = phase33.build_safe_blocks(raw)
    fit_rows = np.arange(len(train), dtype=np.int64)
    apply_rows = np.arange(len(train), len(raw), dtype=np.int64)
    x_fit, x_test = phase33.make_matrices(
        raw, blocks, fit_rows, apply_rows, "keys_frequency"
    )

    started = time.perf_counter()
    lgbm = _lgbm_model(20261113, lgbm_iterations)
    lgbm.fit(x_fit, y)
    lgbm_prediction = lgbm.predict_proba(x_test)[:, 1].astype(np.float64)
    del lgbm

    xgb_predictions: list[np.ndarray] = []
    for seed in XGB_SEEDS:
        model = _xgb_model(seed, xgb_iterations[seed])
        model.fit(x_fit, y, verbose=False)
        prediction = model.predict_proba(x_test)[:, 1].astype(np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite XGBoost predictions for seed {seed}")
        xgb_predictions.append(prediction)
        del model
        print(
            f"Fitted XGBoost seed={seed}, rounds={xgb_iterations[seed]}",
            flush=True,
        )
    xgb_average = np.mean(np.vstack(xgb_predictions), axis=0)
    final_prediction = 0.5 * lgbm_prediction + 0.5 * xgb_average
    if (
        not np.isfinite(final_prediction).all()
        or ((final_prediction < 0) | (final_prediction > 1)).any()
    ):
        raise ValueError("Phase 37 ensemble produced invalid probabilities")

    submission = pd.DataFrame({ID: test[ID], TARGET: final_prediction})
    _validate_submission(submission, sample[ID], "Phase 37 submission")
    submission.to_csv(output_path, index=False)
    print(
        "Fitted one LightGBM and three XGBoost scratch models; "
        f"Phase 37 OOF reference AUC={oof_auc:.9f}",
        flush=True,
    )
    return {
        "file": str(output_path),
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "oof_auc_reference": oof_auc,
        "model_components": [
            {"model": "LightGBM", "seed": 20261113, "iterations": lgbm_iterations},
            *[
                {
                    "model": "XGBoost",
                    "seed": seed,
                    "iterations": xgb_iterations[seed],
                }
                for seed in XGB_SEEDS
            ],
        ],
        "fit_seconds": time.perf_counter() - started,
        "sha256": _sha256(output_path),
    }


def _make_v22_reproduction(
    test_path: Path,
    sample_path: Path,
    v17_path: Path,
    v18_path: Path,
    historical_v22_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    test = _required_csv(test_path, "Test CSV")
    sample = _required_csv(sample_path, "Sample submission")
    v17 = _required_csv(v17_path, "Historical V17 predictions")
    v18 = _required_csv(v18_path, "Historical V18 predictions")
    for label, frame in (("V17", v17), ("V18", v18)):
        if list(frame.columns) != [ID, TARGET]:
            raise ValueError(f"{label} prediction file must contain {ID},{TARGET}")
        if (
            len(frame) != len(test)
            or not frame[ID].equals(test[ID])
            or not frame[ID].is_unique
        ):
            raise ValueError(f"{label} IDs/order do not match test rows")
        prediction = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(
            dtype=float
        )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{label} predictions contain nonfinite values")

    def normalized_rank(values: np.ndarray) -> np.ndarray:
        if len(values) < 2:
            raise ValueError("Rank fusion requires at least two test rows")
        return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)

    r17 = normalized_rank(v17[TARGET].to_numpy(dtype=float))
    r18 = normalized_rank(v18[TARGET].to_numpy(dtype=float))
    fused = normalized_rank(0.55 * r17 + 0.45 * r18)
    submission = pd.DataFrame({ID: test[ID], TARGET: fused})
    _validate_submission(submission, sample[ID], "V22 reproduction")
    submission.to_csv(output_path, index=False)

    match: dict[str, Any] = {"historical_file_available": False}
    if historical_v22_path.is_file():
        historical = _required_csv(historical_v22_path, "Historical V22 output")
        _validate_submission(historical, sample[ID], "Historical V22 output")
        historical_values = historical[TARGET].to_numpy(dtype=float)
        exact = bool(np.array_equal(fused, historical_values))
        rank_corr = float(np.corrcoef(rankdata(fused), rankdata(historical_values))[0, 1])
        match = {
            "historical_file_available": True,
            "exact_value_match": exact,
            "spearman_rank_correlation": rank_corr,
        }
    return {
        "file": str(output_path),
        "rows": int(len(submission)),
        "columns": list(submission.columns),
        "method": "rank((0.55 * rank(V17) + 0.45 * rank(V18)))",
        "public_score_status": "historical claim not independently verified",
        "source_predictions_regenerated": False,
        "historical_output_comparison": match,
        "sha256": _sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate Phase 37 conservative and historical V22 "
            "submission candidates without overwriting existing files."
        )
    )
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--sample", type=Path, default=SAMPLE_DEFAULT)
    parser.add_argument("--v17", type=Path, default=V17_DEFAULT)
    parser.add_argument("--v18", type=Path, default=V18_DEFAULT)
    parser.add_argument("--historical-v22", type=Path, default=HISTORICAL_V22_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("submission_candidates_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    phase37_path = args.output_dir / "phase37_rules_first_submission.csv"
    v22_path = args.output_dir / "v22_historical_rank_blend_reproduction.csv"
    phase37_result = _make_phase37_submission(
        args.train, args.test, args.sample, phase37_path
    )
    v22_result = _make_v22_reproduction(
        args.test,
        args.sample,
        args.v17,
        args.v18,
        args.historical_v22,
        v22_path,
    )
    summary = {
        "status": "generated; public/private leaderboard performance unknown",
        "rule_boundary": {
            "phase37": (
                "lower-ambiguity recipe from supplied train labels and columns; "
                "no target encoding, pooled test statistics, external data, or "
                "manual prediction edits. Not organizer-certified."
            ),
            "v22": (
                "historical rank blend uses pre-existing V17/V18 test predictions; "
                "not represented as a clean-room or rule-certified submission."
            ),
        },
        "phase37_submission": phase37_result,
        "v22_reproduction": v22_result,
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {
            str(path): _sha256(path)
            for path in (
                args.train,
                args.test,
                args.sample,
                args.v17,
                args.v18,
            )
        },
    }
    summary_path = args.output_dir / "submission_candidates_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
