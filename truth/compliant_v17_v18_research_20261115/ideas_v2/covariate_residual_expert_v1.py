from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
PHASE33_DIR = Path(r"D:\gci\truth\experiments\phase33_rules_first_features")
sys.path.insert(0, str(PHASE33_DIR))
import run_phase33_rules_first_features as phase33


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE_OOF = (
    ROOT
    / "ideas_v1"
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "covariate_residual_expert_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
RESIDUAL_TARGETS = (
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
)
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005
RESIDUAL_TREES = 160


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_values(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for column in RESIDUAL_TARGETS:
        values = pd.to_numeric(raw[column], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if column in {"AMT_CREDIT", "AMT_INCOME_TOTAL"}:
            values = np.where(values > 0, np.log(values), np.nan)
        if column == "DAYS_EMPLOYED":
            values[values == 365243] = np.nan
        result[column] = values
    return result


def numeric_covariates(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    categorical = list(
        raw.select_dtypes(include=["object", "category", "string"]).columns
    )
    numeric = raw.drop(columns=[ID, TARGET, *categorical], errors="ignore").copy()
    numeric = numeric.apply(pd.to_numeric, errors="raise").astype(np.float32)
    if "DAYS_EMPLOYED" in numeric:
        numeric.loc[numeric["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.reset_index(drop=True), categorical


def make_covariate_pair(
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical_columns: list[str],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    response_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_columns = list(numeric.columns)
    if response_column not in numeric_columns:
        raise ValueError(f"Residual target is not a numeric input: {response_column}")
    response_index = numeric_columns.index(response_column)
    fit_numeric = numeric.iloc[fit_rows].to_numpy(dtype=np.float32, copy=True)
    apply_numeric = numeric.iloc[apply_rows].to_numpy(dtype=np.float32, copy=True)
    fit_numeric[:, response_index] = np.nan
    apply_numeric[:, response_index] = np.nan

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        dtype=np.float32,
        sparse_output=False,
    )
    fit_categories = (
        raw.iloc[fit_rows][categorical_columns]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )
    apply_categories = (
        raw.iloc[apply_rows][categorical_columns]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )
    fit_encoded = encoder.fit_transform(fit_categories).astype(np.float32)
    apply_encoded = encoder.transform(apply_categories).astype(np.float32)
    fit_matrix = np.column_stack((fit_numeric, fit_encoded)).astype(np.float32)
    apply_matrix = np.column_stack((apply_numeric, apply_encoded)).astype(np.float32)
    if fit_matrix.shape[1] != apply_matrix.shape[1]:
        raise ValueError("Residual fit/apply feature widths differ")
    return fit_matrix, apply_matrix


def residual_regressor(seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=RESIDUAL_TREES,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=40,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        max_bin=256,
        random_state=seed,
        n_jobs=8,
        verbosity=0,
    )


def crossfit_residual_views(
    raw: pd.DataFrame,
    numeric: pd.DataFrame,
    categorical_columns: list[str],
    responses: dict[str, np.ndarray],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    fit_local = np.full(
        (len(fit_rows), 2 * len(RESIDUAL_TARGETS)), np.nan, dtype=np.float32
    )
    apply_predictions = np.zeros(
        (len(apply_rows), len(RESIDUAL_TARGETS)), dtype=np.float64
    )

    for inner_number, (train_local, valid_local) in enumerate(
        splitter.split(np.zeros(len(fit_rows), dtype=np.int8)), start=1
    ):
        inner_fit = fit_rows[train_local]
        inner_valid = fit_rows[valid_local]
        for target_index, response in enumerate(RESIDUAL_TARGETS):
            combined_apply = np.concatenate((inner_valid, apply_rows))
            x_fit, x_combined = make_covariate_pair(
                raw,
                numeric,
                categorical_columns,
                inner_fit,
                combined_apply,
                response,
            )
            response_fit = responses[response][inner_fit]
            observed = np.isfinite(response_fit)
            if int(observed.sum()) < 1000:
                raise ValueError(f"Too few observed values for {response}")
            model = residual_regressor(seed + inner_number * 100 + target_index)
            model.fit(x_fit[observed], response_fit[observed])
            prediction_combined = model.predict(x_combined)
            prediction_valid = prediction_combined[: len(inner_valid)]
            prediction_apply = prediction_combined[len(inner_valid) :]
            actual_valid = responses[response][inner_valid]
            local_pred = np.full(len(inner_valid), np.nan, dtype=np.float32)
            local_residual = np.full(len(inner_valid), np.nan, dtype=np.float32)
            local_pred[:] = prediction_valid.astype(np.float32)
            observed_valid = np.isfinite(actual_valid)
            local_residual[observed_valid] = (
                actual_valid[observed_valid] - prediction_valid[observed_valid]
            ).astype(np.float32)
            fit_local[valid_local, 2 * target_index] = local_pred
            fit_local[valid_local, 2 * target_index + 1] = local_residual
            apply_predictions[:, target_index] += prediction_apply / INNER_FOLDS
            del model, x_fit, x_combined
        print(
            f"  residual inner fold {inner_number}/{INNER_FOLDS} complete",
            flush=True,
        )

    apply_features = np.full(
        (len(apply_rows), 2 * len(RESIDUAL_TARGETS)), np.nan, dtype=np.float32
    )
    for target_index, response in enumerate(RESIDUAL_TARGETS):
        prediction = apply_predictions[:, target_index]
        actual = responses[response][apply_rows]
        observed_apply = np.isfinite(actual)
        apply_features[:, 2 * target_index] = prediction.astype(np.float32)
        apply_features[observed_apply, 2 * target_index + 1] = (
            actual[observed_apply] - prediction[observed_apply]
        ).astype(np.float32)

    if np.isinf(fit_local).any() or np.isinf(apply_features).any():
        raise ValueError("Residual feature construction produced infinities")
    if np.isnan(fit_local[:, ::2]).any() or np.isnan(apply_features[:, ::2]).any():
        raise ValueError("Residual prediction features are incomplete")
    return fit_local, apply_features


def fold_rank(values: np.ndarray) -> np.ndarray:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def make_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        num_leaves=63,
        min_child_samples=100,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        device_type="cpu",
        n_jobs=8,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("XGBoost CUDA is required; refusing CPU fallback")
    for path in (TRAIN, BASELINE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
        or BASELINE_COLUMN not in baseline
        or "fold" not in baseline
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with train.csv")

    y = train[TARGET].to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(splitter.split(np.zeros(len(y)), y))
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid_rows) in enumerate(outer_splits, start=1):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Baseline OOF fold assignment differs from the declared split")

    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    numeric, categorical_columns = numeric_covariates(raw)
    responses = target_values(raw)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold, (outer_fit, outer_valid) in enumerate(outer_splits, start=1):
        started_fold = time.perf_counter()
        fit_residuals, valid_residuals = crossfit_residual_views(
            raw,
            numeric,
            categorical_columns,
            responses,
            outer_fit,
            outer_valid,
            SEED + fold * 1000,
        )
        base_fit, base_valid = phase33.make_matrices(
            raw,
            phase33.build_safe_blocks(raw),
            outer_fit,
            outer_valid,
            "keys_frequency",
        )
        x_fit = hstack(
            (base_fit, csr_matrix(fit_residuals, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        x_valid = hstack(
            (base_valid, csr_matrix(valid_residuals, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        model = make_classifier(SEED + fold * 100)
        model.fit(x_fit, y[outer_fit])
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: residual-expert scores are non-finite")
        expert_oof[outer_valid] = prediction.astype(np.float32)

        base_rank = fold_rank(baseline_scores[outer_valid])
        expert_rank = fold_rank(prediction)
        blend_oof[outer_valid] = (
            (1.0 - BLEND_WEIGHT) * base_rank + BLEND_WEIGHT * expert_rank
        ).astype(np.float32)
        base_auc = float(roc_auc_score(y[outer_valid], baseline_scores[outer_valid]))
        expert_auc = float(roc_auc_score(y[outer_valid], prediction))
        blend_auc = float(roc_auc_score(y[outer_valid], blend_oof[outer_valid]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "residual_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - base_auc,
                "blend_delta": blend_auc - base_auc,
                "fit_rows": int(len(outer_fit)),
                "valid_rows": int(len(outer_valid)),
                "feature_count": int(x_fit.shape[1]),
                "runtime_seconds": time.perf_counter() - started_fold,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={base_auc:.6f} "
            f"residual_expert={expert_auc:.6f} blend={blend_auc:.6f} "
            f"seconds={records[-1]['runtime_seconds']:.1f}",
            flush=True,
        )
        del model, x_fit, x_valid, base_fit, base_valid

    if not np.isfinite(expert_oof).all() or not np.isfinite(blend_oof).all():
        raise ValueError("Outer OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "candidate_te20_blend": baseline_scores,
            "covariate_residual_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["residual_expert_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated exploratory residual-feature experiment; not an independent estimate",
        "hypothesis": (
            "Fold-local residuals from covariate-only regressors may expose "
            "applicant deviations from demographic/financial expectation "
            "without using TARGET."
        ),
        "residual_targets": list(RESIDUAL_TARGETS),
        "residual_regressor": {
            "family": "XGBoost CUDA regressor",
            "trees": RESIDUAL_TREES,
            "max_depth": 3,
            "max_bin": 256,
        },
        "inner_crossfit_folds": INNER_FOLDS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "fixed_blend": (
            f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{BLEND_WEIGHT:.2f} residual expert within-fold rank"
        ),
        "minimum_promotion_gain": PROMOTION_DELTA,
        "baseline_mean_fold_auc": baseline_mean,
        "expert_mean_fold_auc": expert_mean,
        "blend_mean_fold_auc": blend_mean,
        "blend_mean_fold_gain": gain,
        "positive_blend_folds": int(sum(row["blend_delta"] > 0 for row in records)),
        "fold_metrics": records,
        "promotion_gate_passed": bool(
            gain >= PROMOTION_DELTA
            and all(row["blend_delta"] > 0 for row in records)
        ),
        "submission_generated": False,
        "versions": {
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "target_for_residuals": (
                "supplied covariate columns only; competition TARGET is not used "
                "to fit residual regressors"
            ),
            "residual_fit": (
                "each training row's residual is cross-fitted; outer-validation "
                "residual predictions use regressors trained only on outer-fit rows"
            ),
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only as a fold-aligned benchmark; never used as a model feature"
            ),
            "gpu": "CUDA required for residual regressors; no CPU fallback",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                TRAIN,
                BASELINE_OOF,
                Path(__file__).resolve(),
                oof_path,
                fold_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
