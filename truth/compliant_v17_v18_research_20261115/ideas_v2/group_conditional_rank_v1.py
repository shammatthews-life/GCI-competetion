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
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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
OUTPUT = IDEAS_DIR / "runs" / "group_conditional_rank_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 5
MIN_GROUP_ROWS = 30
N_ESTIMATORS = 600
BASELINE_COLUMN = "candidate_te20_blend"
EXPERT_WEIGHT = 0.10
PROMOTION_DELTA = 0.005

GROUP_COLUMNS = (
    "OCCUPATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "REGION_RATING_CLIENT",
    "NAME_CONTRACT_TYPE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def numeric_views(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    credit = pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    employed = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    employed[employed == 365243] = np.nan
    credit_income = np.divide(
        credit,
        income,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(credit) & np.isfinite(income) & (income > 0),
    )
    annuity_credit = np.divide(
        annuity,
        credit,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(annuity) & np.isfinite(credit) & (credit > 0),
    )
    return {
        "EXT_SOURCE_1": pd.to_numeric(raw["EXT_SOURCE_1"], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        ),
        "EXT_SOURCE_2": pd.to_numeric(raw["EXT_SOURCE_2"], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        ),
        "EXT_SOURCE_3": pd.to_numeric(raw["EXT_SOURCE_3"], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        ),
        "CREDIT_TO_INCOME": credit_income,
        "ANNUITY_TO_CREDIT": annuity_credit,
        "DAYS_BIRTH": pd.to_numeric(raw["DAYS_BIRTH"], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        ),
        "DAYS_EMPLOYED_CLEAN": employed,
    }


def conditional_rank_features(
    raw: pd.DataFrame,
    values: dict[str, np.ndarray],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> np.ndarray:
    if np.intersect1d(fit_rows, apply_rows).size:
        raise ValueError("Conditional-rank fit and apply rows overlap")

    rank_columns: list[np.ndarray] = []
    support_columns: list[np.ndarray] = []
    apply_groups_by_column: dict[str, np.ndarray] = {}
    for column in GROUP_COLUMNS:
        apply_groups_by_column[column] = (
            raw.iloc[apply_rows][column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )

    for group_column in GROUP_COLUMNS:
        fit_groups = (
            raw.iloc[fit_rows][group_column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .to_numpy()
        )
        apply_groups = apply_groups_by_column[group_column]
        fit_frame = pd.DataFrame({"group": fit_groups})
        group_counts = fit_frame["group"].value_counts()
        apply_counts = (
            pd.Series(apply_groups)
            .map(group_counts)
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        support_columns.append(np.log1p(apply_counts).astype(np.float32))

        for feature_name, feature_values in values.items():
            fit_values = feature_values[fit_rows]
            apply_values = feature_values[apply_rows]
            finite_fit = np.isfinite(fit_values)
            global_reference = np.sort(fit_values[finite_fit])
            if not len(global_reference):
                raise ValueError(f"No finite values available for {feature_name}")

            grouped_values: dict[str, np.ndarray] = {}
            valid_frame = pd.DataFrame(
                {"group": fit_groups[finite_fit], "value": fit_values[finite_fit]}
            )
            for group, indices in valid_frame.groupby("group", sort=False).indices.items():
                if int(group_counts.get(group, 0)) >= MIN_GROUP_ROWS:
                    grouped_values[str(group)] = np.sort(
                        valid_frame.iloc[np.asarray(indices, dtype=np.int64)][
                            "value"
                        ].to_numpy(dtype=np.float64)
                    )

            ranks = np.full(len(apply_rows), np.nan, dtype=np.float32)
            finite_apply = np.isfinite(apply_values)
            if finite_apply.any():
                apply_frame = pd.DataFrame(
                    {
                        "position": np.flatnonzero(finite_apply),
                        "group": apply_groups[finite_apply],
                        "value": apply_values[finite_apply],
                    }
                )
                for group, positions in apply_frame.groupby(
                    "group", sort=False
                ).indices.items():
                    positions = np.asarray(positions, dtype=np.int64)
                    current = apply_frame.iloc[positions]
                    reference = grouped_values.get(str(group), global_reference)
                    current_values = current["value"].to_numpy(dtype=np.float64)
                    left = np.searchsorted(reference, current_values, side="left")
                    right = np.searchsorted(reference, current_values, side="right")
                    row_positions = current["position"].to_numpy(dtype=np.int64)
                    ranks[row_positions] = (
                        (left + right) / (2.0 * len(reference))
                    ).astype(np.float32)
            rank_columns.append(ranks)

    result = np.column_stack(rank_columns + support_columns).astype(np.float32)
    if result.shape[1] != len(GROUP_COLUMNS) * (len(values) + 1):
        raise ValueError("Unexpected conditional-rank feature width")
    if np.isinf(result).any():
        raise ValueError("Conditional-rank features contain infinities")
    return result


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=N_ESTIMATORS,
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


def verify_inputs(
    train: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with train.csv")
    if BASELINE_COLUMN not in baseline or "fold" not in baseline:
        raise ValueError("Baseline OOF file is missing required columns")

    y = train[TARGET].to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_rows) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Baseline OOF folds differ from the declared outer split")
    return y, fold_ids


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    for path in (TRAIN, BASELINE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    y, fold_ids = verify_inputs(train, baseline)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_scores).all():
        raise ValueError("Baseline OOF predictions contain non-finite scores")

    missing_columns = set(GROUP_COLUMNS).difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing group columns: {sorted(missing_columns)}")
    values = numeric_views(raw)
    blocks = phase33.build_safe_blocks(raw)
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold in range(1, FOLDS + 1):
        outer_fit = np.flatnonzero(fold_ids != fold)
        outer_valid = np.flatnonzero(fold_ids == fold)
        inner_oof = np.full(
            (len(outer_fit), len(GROUP_COLUMNS) * (len(values) + 1)),
            np.nan,
            dtype=np.float32,
        )
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + 1000 + fold,
        )
        local_y = y[outer_fit]
        support_start = len(GROUP_COLUMNS) * len(values)
        for inner_fit_local, inner_valid_local in inner_splitter.split(
            np.zeros(len(outer_fit), dtype=np.int8), local_y
        ):
            inner_fit = outer_fit[inner_fit_local]
            inner_valid = outer_fit[inner_valid_local]
            inner_oof[inner_valid_local] = conditional_rank_features(
                raw, values, inner_fit, inner_valid
            )
        if np.isinf(inner_oof).any() or np.isnan(inner_oof[:, support_start:]).any():
            raise ValueError(f"Fold {fold}: inner conditional features are invalid")

        outer_features = conditional_rank_features(
            raw, values, outer_fit, outer_valid
        )
        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, outer_fit, outer_valid, "keys_frequency"
        )
        x_fit = hstack(
            (x_fit, csr_matrix(inner_oof, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        x_valid = hstack(
            (x_valid, csr_matrix(outer_features, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        if x_fit.shape[1] != x_valid.shape[1]:
            raise ValueError(f"Fold {fold}: model feature widths differ")

        model = make_model(SEED + fold * 100)
        model.fit(x_fit, y[outer_fit])
        prediction = model.predict_proba(x_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: expert predictions are non-finite")
        expert_oof[outer_valid] = prediction.astype(np.float32)

        base_rank = fold_rank(baseline_scores[outer_valid])
        expert_rank = fold_rank(prediction)
        blend_oof[outer_valid] = (
            (1.0 - EXPERT_WEIGHT) * base_rank + EXPERT_WEIGHT * expert_rank
        ).astype(np.float32)
        base_auc = float(roc_auc_score(y[outer_valid], baseline_scores[outer_valid]))
        expert_auc = float(roc_auc_score(y[outer_valid], prediction))
        blend_auc = float(roc_auc_score(y[outer_valid], blend_oof[outer_valid]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "group_conditional_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - base_auc,
                "blend_delta": blend_auc - base_auc,
                "fit_rows": int(len(outer_fit)),
                "valid_rows": int(len(outer_valid)),
                "feature_count": int(x_fit.shape[1]),
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={base_auc:.6f} "
            f"group_expert={expert_auc:.6f} "
            f"fixed_blend={blend_auc:.6f}",
            flush=True,
        )
        del model, x_fit, x_valid, inner_oof, outer_features

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
            "group_conditional_rank_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(
        np.mean([row["group_conditional_expert_auc"] for row in records])
    )
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated exploratory experiment; not an independent estimate",
        "hypothesis": (
            "Training-fold conditional empirical ranks expose within-group "
            "relative position while preserving raw/global features for "
            "cross-group positive-negative ordering."
        ),
        "conditional_rank_definition": (
            "midrank empirical CDF within each supplied categorical group; "
            f"groups with fewer than {MIN_GROUP_ROWS} fit rows use the fit-fold "
            "global empirical CDF; missing numeric values remain missing."
        ),
        "group_columns": list(GROUP_COLUMNS),
        "ranked_numeric_views": list(values),
        "inner_crossfit_folds": INNER_FOLDS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "fixed_blend": (
            f"{1.0 - EXPERT_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{EXPERT_WEIGHT:.2f} group expert within-fold rank"
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
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "labels": "used only for model fitting on outer training rows and AUC reporting",
            "group_rank_fit": (
                "inner rank features use only the corresponding inner-fit rows; "
                "outer-validation features use only outer-fit rows"
            ),
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only as a fold-aligned benchmark; never used as a model feature"
            ),
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
    summary_path = OUTPUT / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
