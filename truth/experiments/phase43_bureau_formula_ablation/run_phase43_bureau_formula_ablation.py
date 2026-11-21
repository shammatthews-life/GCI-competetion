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
from scipy.sparse import hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
PHASE38_OOF_DEFAULT = (
    PHASE38
    / "runs"
    / "phase38_thresholds_20261107T173546Z"
    / "PHASE38_OOF_PREDICTIONS.csv"
)
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
CONFIGS = {
    "ext_product": ("SAFE_EXT_PRODUCT",),
    "ext_harmonic": ("SAFE_EXT_HARMONIC_MEAN",),
    "solvency_x_ext23": ("SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ",),
    "combined": (
        "SAFE_EXT_PRODUCT",
        "SAFE_EXT_HARMONIC_MEAN",
        "SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_bureau_formulas(
    raw: pd.DataFrame, rowwise: pd.DataFrame
) -> pd.DataFrame:
    columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    missing = [column for column in columns if column not in raw]
    if missing or "AMT_INCOME_TOTAL" not in raw or "AMT_ANNUITY" not in raw:
        raise ValueError(f"Required formula inputs are missing: {missing}")
    scores = raw[columns].apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64
    )
    complete = np.isfinite(scores).all(axis=1)
    product = np.full(len(raw), np.nan, dtype=np.float64)
    product[complete] = np.prod(scores[complete], axis=1)
    positive = complete & (scores > 0).all(axis=1)
    harmonic = np.full(len(raw), np.nan, dtype=np.float64)
    harmonic[positive] = 3.0 / np.sum(1.0 / scores[positive], axis=1)

    income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
        dtype=np.float64
    )
    annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64
    )
    ext23 = scores[:, 1] * scores[:, 2]
    solvency_valid = (
        np.isfinite(income)
        & np.isfinite(annuity)
        & (annuity > 0)
        & np.isfinite(ext23)
    )
    solvency = np.full(len(raw), np.nan, dtype=np.float64)
    solvency[solvency_valid] = (
        income[solvency_valid]
        / annuity[solvency_valid]
        * np.square(ext23[solvency_valid])
    )

    result = rowwise.copy()
    result["SAFE_EXT_PRODUCT"] = product
    result["SAFE_EXT_HARMONIC_MEAN"] = harmonic
    result["SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ"] = solvency
    return result


def make_matrices(
    blocks: tuple[pd.DataFrame, ...],
    formula_features: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    formula_columns: tuple[str, ...],
) -> tuple[Any, Any]:
    base_fit, base_apply = phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    if not formula_columns:
        return base_fit, base_apply
    fit_extra = phase38._numeric_matrix(
        formula_features.iloc[fit_rows][list(formula_columns)].reset_index(
            drop=True
        )
    )
    apply_extra = phase38._numeric_matrix(
        formula_features.iloc[apply_rows][list(formula_columns)].reset_index(
            drop=True
        )
    )
    return (
        hstack([base_fit, fit_extra], format="csr", dtype=np.float32),
        hstack([base_apply, apply_extra], format="csr", dtype=np.float32),
    )


def run(train_path: Path, phase38_oof_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    for path in (train_path, phase38_oof_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    reference = pd.read_csv(phase38_oof_path)
    if (
        ID not in train
        or TARGET not in train
        or not train[ID].is_unique
        or len(reference) != len(train)
        or not reference[ID].equals(train[ID])
        or not reference[TARGET].equals(train[TARGET])
        or "fold" not in reference
        or "oof_xgb_min_frequency_50" not in reference
    ):
        raise ValueError("Training data or Phase 38 OOF is invalid")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    numeric, categorical, rowwise, financial, keys, peer = (
        phase33.build_safe_blocks(raw)
    )
    formula_features = add_bureau_formulas(raw, pd.DataFrame(index=raw.index))
    blocks = (numeric, categorical, rowwise, financial, keys, peer)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if not np.array_equal(fold_ids, reference["fold"].to_numpy()):
        raise ValueError("Phase 38 fold assignment does not match")
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float32) for name in CONFIGS
    }
    records: list[dict[str, Any]] = []
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
        for name, formula_columns in CONFIGS.items():
            x_inner_fit, x_inner_valid = make_matrices(
                blocks,
                formula_features,
                inner_fit,
                inner_valid,
                formula_columns,
            )
            selector = phase38._model(
                fold_seed, MAX_ITERATIONS, stopping=True
            )
            selector.fit(
                x_inner_fit,
                y[inner_fit],
                eval_set=[(x_inner_valid, y[inner_valid])],
                verbose=False,
            )
            rounds = max(100, int(selector.best_iteration) + 1)
            del selector, x_inner_fit, x_inner_valid

            x_fit, x_valid = make_matrices(
                blocks,
                formula_features,
                fit_rows,
                valid_rows,
                formula_columns,
            )
            model = phase38._model(fold_seed, rounds, stopping=False)
            model.fit(x_fit, y[fit_rows], verbose=False)
            prediction = model.predict_proba(x_valid)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite prediction for formula={name}, fold={fold_number}"
                )
            predictions[name][valid_rows] = prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], prediction))
            records.append(
                {
                    "formula_set": name,
                    "fold": fold_number,
                    "auc": auc,
                    "iterations": rounds,
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"formula_set={name} fold={fold_number}/{FOLDS} "
                f"AUC={auc:.6f} iterations={rounds}",
                flush=True,
            )
            del model, x_fit, x_valid
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("One or more OOF vectors are incomplete")

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE43_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE43_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE43_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    output = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "oof_xgb_keys_frequency_50_reference": reference[
                "oof_xgb_min_frequency_50"
            ].to_numpy(dtype=np.float32),
            **{
                f"oof_xgb_{name}": values
                for name, values in predictions.items()
            },
        }
    )
    output.to_csv(oof_path, index=False)
    vectors = {
        "keys_frequency_50_reference": reference[
            "oof_xgb_min_frequency_50"
        ].to_numpy(dtype=np.float32),
        **predictions,
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, vector)),
            "pooled_log_loss": float(log_loss(y, vector)),
            "mean_fold_auc": float(
                np.mean(
                    [roc_auc_score(y[valid], vector[valid]) for _, valid in folds]
                )
            ),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 43 safe bureau formula ablation",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "formula_definitions": {
            "SAFE_EXT_PRODUCT": "EXT_SOURCE_1 * EXT_SOURCE_2 * EXT_SOURCE_3; missing if any source is missing",
            "SAFE_EXT_HARMONIC_MEAN": "3 / sum(1 / EXT_SOURCE_i); missing unless all three sources are finite and positive",
            "SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ": "(AMT_INCOME_TOTAL / AMT_ANNUITY) * (EXT_SOURCE_2 * EXT_SOURCE_3)^2; missing if required scores are missing or annuity is nonpositive",
        },
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "median_imputation_for_formula_inputs": "none; missing values remain missing",
            "manual_prediction_edits": "none",
            "category_vocabularies_and_frequencies": "fit on each model training partition only",
        },
        "caveats": [
            "The historical within-split formula AUCs in ORIGIN_MASTER_VERIFICATION_LEDGER.md are not directly comparable to this pooled outer-OOF evaluation.",
            "The report's explanation that bureau scores are independent probabilities is not established by the data; products and harmonic means are tested only as candidate nonlinear transforms.",
            "All candidate selection uses repeatedly examined labels and is exploratory, not an unbiased final estimate.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                phase38_oof_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
                fold_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--phase38-oof", type=Path, default=PHASE38_OOF_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase43_bureau_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.phase38_oof, args.output_dir)
