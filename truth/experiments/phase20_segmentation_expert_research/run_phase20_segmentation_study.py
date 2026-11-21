from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
sys.path.insert(0, str(PHASE13))
import phase13_features as features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
OUTPUT_DIR = ROOT / "runs" / "phase20_stage1_20261106"
OUTER_SEED = 20261113
DISCOVERY_SEED = 20261106
DISCOVERY_FRACTION = 0.60
FOLDS = 5
INNER_GATE_FOLDS = 3
GLOBAL_BLEND_WEIGHT = 0.50
MIN_EXPERT_VALID_ROWS = 200
MIN_EXPERT_CLASS_ROWS = 300
GPU_SAMPLE_SECONDS = 2.0
TARGET = features.TARGET_COLUMN
ID = features.ID_COLUMN


FAMILIES: tuple[dict[str, str], ...] = (
    {
        "id": "F01",
        "name": "contract_type",
        "kind": "static",
        "route": "NAME_CONTRACT_TYPE == 'Cash loans' versus all other values",
        "hypothesis": "Loan contract products may have different risk mechanisms.",
    },
    {
        "id": "F02",
        "name": "employment_sentinel",
        "kind": "static",
        "route": "DAYS_EMPLOYED == 365243 versus non-sentinel",
        "hypothesis": "The supplied employment sentinel may define a distinct applicant state.",
    },
    {
        "id": "F03",
        "name": "external_score_completeness",
        "kind": "static",
        "route": "All three EXT_SOURCE values observed versus at least one missing",
        "hypothesis": "Availability of external risk scores may alter the useful decision boundary.",
    },
    {
        "id": "F04",
        "name": "age_band",
        "kind": "static",
        "route": "Age in years <= 35 versus > 35",
        "hypothesis": "Repayment risk may differ by life stage.",
    },
    {
        "id": "F05",
        "name": "employment_duration",
        "kind": "static",
        "route": "Employment sentinel or cleaned employment duration < 2 years versus other",
        "hypothesis": "Short or unrecorded employment history may require a different expert.",
    },
    {
        "id": "F06",
        "name": "income_band",
        "kind": "median",
        "route": "AMT_INCOME_TOTAL above the outer-fit median versus at or below it",
        "hypothesis": "Income scale may change the relationship between burden and default.",
    },
    {
        "id": "F07",
        "name": "credit_amount",
        "kind": "static",
        "route": "AMT_CREDIT > 500000 versus <= 500000",
        "hypothesis": "Large credit requests may have distinct risk structure.",
    },
    {
        "id": "F08",
        "name": "annuity_income_burden",
        "kind": "static",
        "route": "AMT_ANNUITY / AMT_INCOME_TOTAL > 0.20 versus <= 0.20",
        "hypothesis": "High payment burden may change risk-feature interactions.",
    },
    {
        "id": "F09",
        "name": "credit_income",
        "kind": "static",
        "route": "AMT_CREDIT / AMT_INCOME_TOTAL > 3.5 versus <= 3.5",
        "hypothesis": "High leverage applicants may benefit from specialized ranking.",
    },
    {
        "id": "F10",
        "name": "goods_credit_ratio",
        "kind": "static",
        "route": "AMT_GOODS_PRICE / AMT_CREDIT < 0.90 versus >= 0.90",
        "hypothesis": "The collateral-to-credit relationship may identify product regimes.",
    },
    {
        "id": "F11",
        "name": "missingness_profile",
        "kind": "static",
        "route": "At least five missing raw predictors versus fewer than five",
        "hypothesis": "Broad application-data completeness may modify score reliability.",
    },
    {
        "id": "F12",
        "name": "external_score_risk_profile",
        "kind": "static",
        "route": "Mean of observed EXT_SOURCE values < 0.45 versus >= 0.45; all-missing rows assigned to low-score group",
        "hypothesis": "A low external-score regime may need different interactions.",
    },
    {
        "id": "F13",
        "name": "financial_profile_gpu_clustering",
        "kind": "gpu_cluster",
        "route": "Two clusters from fold-fitted standardized financial variables using a GPU clustering implementation",
        "hypothesis": "Unsupervised financial profiles may separate borrower repayment regimes.",
    },
    {
        "id": "F14",
        "name": "demographic_employment_gpu_clustering",
        "kind": "gpu_cluster",
        "route": "Two clusters from fold-fitted standardized age, employment, family and external-score variables using a GPU clustering implementation",
        "hypothesis": "Joint demographic and employment profiles may be more informative than single thresholds.",
    },
    {
        "id": "F15",
        "name": "combined_domain_and_crossfit_gate",
        "kind": "combined",
        "route": "Cash loan AND age <= 40 AND credit/income > 2.0; selected configurations replace the exact route with a cross-fitted GPU gate predicting this feature-only membership",
        "hypothesis": "A joint domain segment or a cross-fitted gate may capture smooth subgroup boundaries.",
    },
)


CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "C01",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 4,
        "iterations": 160,
        "learning_rate": 0.08,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C02",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 5,
        "iterations": 180,
        "learning_rate": 0.07,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C03",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 6,
        "iterations": 200,
        "learning_rate": 0.06,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C04",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 7,
        "iterations": 220,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C05",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 8,
        "iterations": 240,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C06",
        "model": "CatBoost_GPU",
        "feature_set": "raw_plus_financial_decompositions",
        "depth": 6,
        "iterations": 200,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C07",
        "model": "CatBoost_GPU",
        "feature_set": "raw_plus_external_score_polynomials",
        "depth": 6,
        "iterations": 200,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C08",
        "model": "CatBoost_GPU",
        "feature_set": "raw_plus_fold_local_constant_free_key_frequency",
        "depth": 6,
        "iterations": 200,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    },
    {
        "id": "C09",
        "model": "CatBoost_GPU",
        "feature_set": "raw_rowwise",
        "depth": 6,
        "iterations": 300,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
    },
    {
        "id": "C10",
        "model": "CatBoost_GPU",
        "feature_set": "raw_plus_financial_and_external_score_interactions",
        "depth": 7,
        "iterations": 250,
        "learning_rate": 0.04,
        "l2_leaf_reg": 7.0,
    },
)


def build_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for config in CONFIGS:
            learned_gate = (
                family["id"] == "F15" and config["id"] in {"C09", "C10"}
            )
            rows.append(
                {
                    "experiment_id": f"{family['id']}_{config['id']}",
                    "kind": "real",
                    "family_id": family["id"],
                    "family": family["name"],
                    "hypothesis": family["hypothesis"],
                    "route_definition": family["route"],
                    "config_id": config["id"],
                    "architecture": (
                        "R5_crossfit_gate_mixture"
                        if learned_gate
                        else "R2_fixed_50_50_global_plus_routed_experts"
                    ),
                    "model": config["model"],
                    "feature_set": config["feature_set"],
                    "depth": config["depth"],
                    "iterations": config["iterations"],
                    "learning_rate": config["learning_rate"],
                    "l2_leaf_reg": config["l2_leaf_reg"],
                    "device": "GPU",
                    "one_hot_max_size": 256,
                    "max_ctr_complexity": 1,
                    "gpu_ram_part": 0.82,
                    "random_seed": "20261113 + 100*fold + config_index",
                    "expert_seed_policy": (
                        "20261113 + 100*fold + config_number*10 + group"
                    ),
                    "minimum_expert_fit_rows": 5000,
                    "minimum_target_class_rows_per_expert_fit_fold": MIN_EXPERT_CLASS_ROWS,
                    "gate_strategy": (
                        "cross-fitted GPU CatBoost membership gate"
                        if learned_gate
                        else "exact feature-only rule"
                    ),
                    "eligibility": (
                        "Track B: deterministic constant-free keys have not been independently cleared"
                        if config["id"] == "C08"
                        else "Track A candidate: supplied columns and fold-local preprocessing; organizer approval not asserted"
                    ),
                    "status": (
                        "GPU_BLOCKED_NO_CUDA_CLUSTERING_LIBRARY"
                        if family["kind"] == "gpu_cluster"
                        else "PREREGISTERED"
                    ),
                }
            )
    placebo_proportions = (0.50, 0.70, 0.90, 0.50, 0.70, 0.90, 0.50, 0.70, 0.90, "ID_PARITY")
    for i in range(30):
        config = CONFIGS[i % len(CONFIGS)]
        proportion = placebo_proportions[i % len(placebo_proportions)]
        rows.append(
            {
                "experiment_id": f"P{i + 1:02d}_{config['id']}",
                "kind": "placebo",
                "family_id": "P",
                "family": "random_or_id_placebo",
                "hypothesis": "Null routing split should not consistently beat the matched global model.",
                "route_definition": (
                    f"Random row assignment seed={8000 + i}, group_A_proportion={proportion}"
                    if proportion != "ID_PARITY"
                    else "SK_ID_CURR parity; placebo diagnostic only"
                ),
                "config_id": config["id"],
                "architecture": "R2_fixed_50_50_global_plus_routed_experts",
                "model": config["model"],
                "feature_set": config["feature_set"],
                "depth": config["depth"],
                "iterations": config["iterations"],
                "learning_rate": config["learning_rate"],
                "l2_leaf_reg": config["l2_leaf_reg"],
                "device": "GPU",
                "one_hot_max_size": 256,
                "max_ctr_complexity": 1,
                "gpu_ram_part": 0.82,
                "random_seed": "20261113 + 100*fold + config_index",
                "expert_seed_policy": (
                    "20261113 + 100*fold + config_number*10 + group"
                ),
                "minimum_expert_fit_rows": 5000,
                "minimum_target_class_rows_per_expert_fit_fold": MIN_EXPERT_CLASS_ROWS,
                "gate_strategy": "label-independent placebo assignment",
                "eligibility": "Null diagnostic only; not a deployable routing rule",
                "status": "PREREGISTERED",
            }
        )
    registry = pd.DataFrame(rows)
    if len(registry[registry["kind"] == "real"]) != 150:
        raise ValueError("The real-experiment registry must contain exactly 150 rows")
    if registry.loc[registry.kind == "real", "experiment_id"].nunique() != 150:
        raise ValueError("Real experiment identifiers are not unique")
    if len(registry[registry["kind"] == "placebo"]) != 30:
        raise ValueError("The placebo registry must contain exactly 30 rows")
    return registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_registry(registry: pd.DataFrame, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "PHASE20_EXPERIMENT_REGISTRY.csv"
    text = registry.to_csv(index=False, lineterminator="\n")
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("Existing preregistered registry differs; refusing overwrite")
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    print(f"PHASE20_REGISTRY_SHA256={digest}", flush=True)
    print(f"PHASE20_REAL_EXPERIMENTS={(registry.kind == 'real').sum()}", flush=True)
    print(f"PHASE20_PLACEBOS={(registry.kind == 'placebo').sum()}", flush=True)
    return digest


def raw_feature_frames(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric, categorical, rowwise = features.build_raw_feature_blocks(raw)
    decompositions = features.build_label_free_decompositions(raw)
    return numeric, categorical, rowwise, decompositions


def build_constant_free_keys(raw: pd.DataFrame) -> pd.DataFrame:
    credit = pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
        dtype=np.float64
    )
    annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64
    )
    goods = pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="raise").to_numpy(
        dtype=np.float64
    )

    def integerized_ratio(
        numerator: np.ndarray, denominator: np.ndarray, decimals: int
    ) -> np.ndarray:
        ratio = _division(numerator, denominator)
        scaled = np.rint(ratio * (10**decimals))
        values = np.full(len(ratio), "MISSING", dtype=object)
        valid = np.isfinite(scaled)
        limits = np.iinfo(np.int64)
        if np.any(valid & ((scaled < limits.min) | (scaled > limits.max))):
            raise ValueError("Constant-free product key exceeds int64 range")
        values[valid] = scaled[valid].astype(np.int64).astype(str)
        return values.astype(str)

    return pd.DataFrame(
        {
            "PRODUCT_KEY_Q3": integerized_ratio(annuity, credit, 3),
            "PRODUCT_KEY_Q4": integerized_ratio(annuity, credit, 4),
            "PRODUCT_KEY_Q5": integerized_ratio(annuity, credit, 5),
            "PRODUCT_KEY_AG3": integerized_ratio(annuity, goods, 3),
        }
    )


def extra_numeric_features(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(
        pd.to_numeric, errors="raise"
    )
    polynomial: dict[str, np.ndarray] = {}
    for column in ext.columns:
        values = ext[column].to_numpy(dtype=np.float64)
        polynomial[f"{column}_SQUARED_P20"] = values**2
        polynomial[f"{column}_CUBED_P20"] = values**3
    for left, right in (
        ("EXT_SOURCE_1", "EXT_SOURCE_2"),
        ("EXT_SOURCE_1", "EXT_SOURCE_3"),
        ("EXT_SOURCE_2", "EXT_SOURCE_3"),
    ):
        polynomial[f"{left}_X_{right}_P20"] = (
            ext[left].to_numpy(dtype=np.float64)
            * ext[right].to_numpy(dtype=np.float64)
        )
    poly = pd.DataFrame(polynomial).replace([np.inf, -np.inf], np.nan)
    return {"external_polynomials": poly}


def _division(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def make_route(
    family: dict[str, str],
    raw: pd.DataFrame,
    fit: np.ndarray,
) -> np.ndarray:
    kind = family["kind"]
    if kind == "gpu_cluster":
        raise RuntimeError("No compatible CUDA clustering library was detected")
    if kind == "combined":
        credit_income = _division(
            pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
                dtype=np.float64
            ),
            pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
                dtype=np.float64
            ),
        )
        age = -pd.to_numeric(raw["DAYS_BIRTH"], errors="raise").to_numpy(
            dtype=np.float64
        ) / 365.25
        return (
            (raw["NAME_CONTRACT_TYPE"].astype(str).to_numpy() == "Cash loans")
            & (age <= 40)
            & (credit_income > 2.0)
        )
    if family["id"] == "F01":
        return raw["NAME_CONTRACT_TYPE"].astype(str).to_numpy() == "Cash loans"
    if family["id"] == "F02":
        return (
            pd.to_numeric(raw["DAYS_EMPLOYED"], errors="raise").to_numpy()
            == 365243
        )
    if family["id"] == "F03":
        return raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().all(
            axis=1
        ).to_numpy()
    if family["id"] == "F04":
        age = -pd.to_numeric(raw["DAYS_BIRTH"], errors="raise").to_numpy(
            dtype=np.float64
        ) / 365.25
        return age <= 35
    if family["id"] == "F05":
        employed = pd.to_numeric(
            raw["DAYS_EMPLOYED"], errors="raise"
        ).to_numpy(dtype=np.float64)
        duration = -employed / 365.25
        return (employed == 365243) | (duration < 2.0)
    if family["id"] == "F06":
        income = pd.to_numeric(
            raw["AMT_INCOME_TOTAL"], errors="raise"
        ).to_numpy(dtype=np.float64)
        threshold = float(np.nanmedian(income[fit]))
        return income > threshold
    if family["id"] == "F07":
        return (
            pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
                dtype=np.float64
            )
            > 500000.0
        )
    if family["id"] == "F08":
        burden = _division(
            pd.to_numeric(raw["AMT_ANNUITY"], errors="raise").to_numpy(
                dtype=np.float64
            ),
            pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
                dtype=np.float64
            ),
        )
        return burden > 0.20
    if family["id"] == "F09":
        ratio = _division(
            pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
                dtype=np.float64
            ),
            pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
                dtype=np.float64
            ),
        )
        return ratio > 3.5
    if family["id"] == "F10":
        ratio = _division(
            pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="raise").to_numpy(
                dtype=np.float64
            ),
            pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
                dtype=np.float64
            ),
        )
        return ratio < 0.90
    if family["id"] == "F11":
        return raw.isna().sum(axis=1).to_numpy() >= 5
    if family["id"] == "F12":
        ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
            dtype=np.float64
        )
        observed = np.isfinite(ext)
        counts = observed.sum(axis=1)
        mean = np.divide(
            np.nansum(ext, axis=1),
            counts,
            out=np.full(len(ext), np.nan, dtype=np.float64),
            where=counts > 0,
        )
        return (~np.isfinite(mean)) | (mean < 0.45)
    raise ValueError(f"Unknown segmentation family: {family['id']}")


def feature_pair(
    config: dict[str, Any],
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    rowwise: pd.DataFrame,
    extra: dict[str, pd.DataFrame],
    keys: pd.DataFrame,
    raw: pd.DataFrame,
    fit: np.ndarray,
    apply: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    num_parts = [numeric, rowwise]
    feature_set = str(config["feature_set"])
    if feature_set in {
        "raw_plus_financial_decompositions",
        "raw_plus_financial_and_external_score_interactions",
    }:
        num_parts.append(extra["decompositions"])
    if feature_set in {
        "raw_plus_external_score_polynomials",
        "raw_plus_financial_and_external_score_interactions",
    }:
        num_parts.append(extra["external_polynomials"])
    if feature_set == "raw_plus_fold_local_constant_free_key_frequency":
        key_counts_fit = pd.DataFrame(index=np.arange(len(fit)))
        key_counts_apply = pd.DataFrame(index=np.arange(len(apply)))
        for column in (
            "PRODUCT_KEY_Q3",
            "PRODUCT_KEY_Q4",
            "PRODUCT_KEY_Q5",
            "PRODUCT_KEY_AG3",
        ):
            counts = keys.iloc[fit][column].astype(str).value_counts()
            key_counts_fit[f"FIT_COUNT_{column}"] = (
                keys.iloc[fit][column].astype(str).map(counts).to_numpy(
                    dtype=np.float32
                )
            )
            key_counts_apply[f"FIT_COUNT_{column}"] = (
                keys.iloc[apply][column]
                .astype(str)
                .map(counts)
                .fillna(0)
                .to_numpy(dtype=np.float32)
            )
        fit_numeric = pd.concat(
            [
                *(part.iloc[fit].reset_index(drop=True) for part in num_parts),
                key_counts_fit.reset_index(drop=True),
            ],
            axis=1,
        )
        apply_numeric = pd.concat(
            [
                *(part.iloc[apply].reset_index(drop=True) for part in num_parts),
                key_counts_apply.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        fit_numeric = pd.concat(
            [part.iloc[fit].reset_index(drop=True) for part in num_parts],
            axis=1,
        )
        apply_numeric = pd.concat(
            [part.iloc[apply].reset_index(drop=True) for part in num_parts],
            axis=1,
        )
    if fit_numeric.columns.duplicated().any():
        raise ValueError("Duplicate numeric feature names in the experiment matrix")
    if categorical.columns.duplicated().any():
        raise ValueError("Duplicate categorical feature names in the experiment matrix")
    fit_categorical = categorical.iloc[fit].reset_index(drop=True).copy()
    apply_categorical = categorical.iloc[apply].reset_index(drop=True).copy()
    for column in fit_categorical.columns:
        fit_categorical[column] = (
            fit_categorical[column].astype("string").fillna("MISSING").astype(str)
        )
        apply_categorical[column] = (
            apply_categorical[column].astype("string").fillna("MISSING").astype(str)
        )
    if not fit_categorical.empty and int(fit_categorical.nunique().max()) > 256:
        raise ValueError(
            "A categorical feature exceeds the configured GPU one-hot ceiling; "
            "refusing to enable CatBoost CTRs implicitly"
        )
    x_fit = pd.concat([fit_numeric, fit_categorical], axis=1)
    x_apply = pd.concat([apply_numeric, apply_categorical], axis=1)
    cat_columns = list(fit_categorical.columns)
    if x_fit.columns.duplicated().any():
        raise ValueError("Feature names collide between numeric and categorical blocks")
    if x_fit.shape[1] != x_apply.shape[1] or list(x_fit.columns) != list(x_apply.columns):
        raise ValueError("Fit/apply feature schemas do not match")
    numeric_values = x_fit.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64
    )
    if np.isinf(numeric_values).any():
        raise ValueError("Infinite numeric features are not allowed")
    return x_fit, x_apply, cat_columns


def fit_predict(
    config: dict[str, Any],
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_apply: pd.DataFrame,
    cat_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, float]:
    if len(np.unique(y_fit)) != 2:
        raise ValueError("A model fit partition contains only one target class")
    model = CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.82,
        one_hot_max_size=256,
        max_ctr_complexity=1,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    cat_indices = [x_fit.columns.get_loc(column) for column in cat_columns]
    started = time.perf_counter()
    model.fit(x_fit, y_fit, cat_features=cat_indices, verbose=False)
    prediction = model.predict_proba(x_apply)[:, 1]
    elapsed = time.perf_counter() - started
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("CatBoost did not run with the required GPU task type")
    if not np.isfinite(prediction).all():
        raise ValueError("GPU model generated non-finite predictions")
    return prediction.astype(np.float64), elapsed


def make_placebo(
    row: pd.Series, ids: np.ndarray, n_rows: int
) -> np.ndarray:
    definition = str(row.route_definition)
    if "parity" in definition.casefold():
        return (ids % 2) == 0
    proportion_text = definition.split("group_A_proportion=")[1]
    proportion = float(proportion_text)
    seed = int(definition.split("seed=")[1].split(",")[0])
    rng = np.random.default_rng(seed)
    return rng.random(n_rows) < proportion


def safe_group_auc(
    y: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float | None:
    selected_y = y[mask]
    if len(selected_y) == 0 or len(np.unique(selected_y)) < 2:
        return None
    return float(roc_auc_score(selected_y, prediction[mask]))


def monitor_gpu(samples: list[dict[str, float]], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            fields = [float(value.strip()) for value in result.stdout.split(",")]
            if len(fields) == 4:
                samples.append(
                    {
                        "gpu_utilization_percent": fields[0],
                        "memory_used_mib": fields[1],
                        "memory_total_mib": fields[2],
                        "power_draw_watts": fields[3],
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(GPU_SAMPLE_SECONDS)


def result_row(
    experiment: pd.Series,
    status: str,
    reason: str,
    global_oof: np.ndarray | None = None,
    routed_oof: np.ndarray | None = None,
    mixture_oof: np.ndarray | None = None,
    route_oof: np.ndarray | None = None,
    y: np.ndarray | None = None,
    fold_ids: np.ndarray | None = None,
    fit_seconds: list[float] | None = None,
) -> dict[str, Any]:
    row = experiment.to_dict()
    row.update(
        {
            "status": status,
            "reason": reason,
            "global_pooled_auc": "",
            "r1_hard_route_pooled_auc": "",
            "r2_global_expert_mixture_pooled_auc": "",
            "delta_r2_vs_global": "",
            "r5_gate_mixture_pooled_auc": "",
            "delta_r5_gate_vs_global": "",
            "architecture_pooled_auc": "",
            "architecture_delta_vs_global": "",
            "architecture_positive_folds": "",
            "architecture_fold_auc": "",
            "architecture_fold_delta": "",
            "expert_mixture_pooled_auc": "",
            "delta_expert_mixture_vs_global": "",
            "positive_folds_r2": "",
            "fold_global_auc": "",
            "fold_r1_auc": "",
            "fold_r2_auc": "",
            "fold_delta_r2": "",
            "group_a_rows": "",
            "group_a_positives": "",
            "group_a_auc": "",
            "group_b_rows": "",
            "group_b_positives": "",
            "group_b_auc": "",
            "global_logloss": "",
            "r1_logloss": "",
            "r2_logloss": "",
            "mean_gpu_fit_seconds": (
                float(np.mean(fit_seconds)) if fit_seconds else ""
            ),
            "total_gpu_fit_seconds": float(np.sum(fit_seconds)) if fit_seconds else "",
        }
    )
    if all(
        value is not None
        for value in (global_oof, routed_oof, mixture_oof, route_oof, y, fold_ids)
    ):
        assert global_oof is not None
        assert routed_oof is not None
        assert mixture_oof is not None
        assert route_oof is not None
        assert y is not None
        assert fold_ids is not None
        global_auc = float(roc_auc_score(y, global_oof))
        r1_auc = float(roc_auc_score(y, routed_oof))
        r2_auc = float(roc_auc_score(y, mixture_oof))
        fold_global = []
        fold_r1 = []
        fold_r2 = []
        fold_delta = []
        for fold in range(1, FOLDS + 1):
            mask = fold_ids == fold
            global_score = float(roc_auc_score(y[mask], global_oof[mask]))
            r1_score = float(roc_auc_score(y[mask], routed_oof[mask]))
            r2_score = float(roc_auc_score(y[mask], mixture_oof[mask]))
            fold_global.append(global_score)
            fold_r1.append(r1_score)
            fold_r2.append(r2_score)
            fold_delta.append(r2_score - global_score)
        group_a = route_oof.astype(bool)
        architecture_delta = r2_auc - global_auc
        positive_folds = int(sum(delta > 0 for delta in fold_delta))
        architecture_name = str(row.get("architecture", ""))
        row.update(
            {
                "global_pooled_auc": global_auc,
                "r1_hard_route_pooled_auc": r1_auc,
                "architecture_pooled_auc": r2_auc,
                "architecture_delta_vs_global": architecture_delta,
                "architecture_positive_folds": positive_folds,
                "architecture_fold_auc": json.dumps(fold_r2),
                "architecture_fold_delta": json.dumps(fold_delta),
                "expert_mixture_pooled_auc": r2_auc,
                "delta_expert_mixture_vs_global": architecture_delta,
                "fold_global_auc": json.dumps(fold_global),
                "fold_r1_auc": json.dumps(fold_r1),
                "fold_r2_auc": json.dumps(fold_r2),
                "fold_delta_r2": json.dumps(fold_delta),
                "group_a_rows": int(group_a.sum()),
                "group_a_positives": int(y[group_a].sum()),
                "group_a_auc": safe_group_auc(y, mixture_oof, group_a),
                "group_b_rows": int((~group_a).sum()),
                "group_b_positives": int(y[~group_a].sum()),
                "group_b_auc": safe_group_auc(y, mixture_oof, ~group_a),
                "global_logloss": float(log_loss(y, global_oof)),
                "r1_logloss": float(log_loss(y, routed_oof)),
                "r2_logloss": float(log_loss(y, mixture_oof)),
            }
        )
        if architecture_name.startswith("R5_"):
            row["r5_gate_mixture_pooled_auc"] = r2_auc
            row["delta_r5_gate_vs_global"] = architecture_delta
        else:
            row["r2_global_expert_mixture_pooled_auc"] = r2_auc
            row["delta_r2_vs_global"] = architecture_delta
        row["positive_folds_r2"] = positive_folds
    return row


def make_matrix_context(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    numeric, categorical, rowwise, decompositions = raw_feature_frames(raw)
    keys = build_constant_free_keys(raw)
    extra = extra_numeric_features(raw)
    extra["decompositions"] = decompositions
    return numeric, categorical, rowwise, keys, extra


def fit_global_predictions(
    config: dict[str, Any],
    context: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, pd.DataFrame],
    ],
    raw: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config_index: int,
) -> tuple[np.ndarray, list[float]]:
    numeric, categorical, rowwise, keys, extra = context
    oof = np.full(len(y), np.nan, dtype=np.float64)
    timings: list[float] = []
    for fold, (fit, valid) in enumerate(folds, start=1):
        x_fit, x_valid, cat_columns = feature_pair(
            config,
            numeric,
            categorical,
            rowwise,
            extra,
            keys,
            raw,
            fit,
            valid,
        )
        prediction, elapsed = fit_predict(
            config,
            x_fit,
            y[fit],
            x_valid,
            cat_columns,
            seed=OUTER_SEED + fold * 100 + config_index,
        )
        oof[valid] = prediction
        timings.append(elapsed)
        del x_fit, x_valid
        print(
            f"GLOBAL config={config['id']} fold={fold} auc="
            f"{roc_auc_score(y[valid], prediction):.6f} fit_seconds={elapsed:.2f}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise ValueError(f"Global OOF coverage incomplete for {config['id']}")
    return oof, timings


def crossfit_gate(
    x_fit: pd.DataFrame,
    membership: np.ndarray,
    x_apply: pd.DataFrame,
    cat_columns: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    if len(np.unique(membership)) != 2:
        raise ValueError("Gate membership has only one class")
    inner = KFold(n_splits=INNER_GATE_FOLDS, shuffle=True, random_state=seed)
    train_gate_probability = np.full(len(membership), np.nan, dtype=np.float64)
    timings: list[float] = []
    gate_config = {
        "iterations": 100,
        "depth": 4,
        "learning_rate": 0.08,
        "l2_leaf_reg": 5.0,
    }
    for inner_fit, inner_apply in inner.split(np.zeros(len(membership))):
        prediction, elapsed = fit_predict(
            gate_config,
            x_fit.iloc[inner_fit].reset_index(drop=True),
            membership[inner_fit].astype(np.int8),
            x_fit.iloc[inner_apply].reset_index(drop=True),
            cat_columns,
            seed=seed + len(timings) + 1,
        )
        train_gate_probability[inner_apply] = prediction
        timings.append(elapsed)
    if not np.isfinite(train_gate_probability).all():
        raise ValueError("Cross-fitted gate predictions are incomplete")
    full_gate, elapsed = fit_predict(
        gate_config,
        x_fit,
        membership.astype(np.int8),
        x_apply,
        cat_columns,
        seed=seed + 99,
    )
    timings.append(elapsed)
    return train_gate_probability, full_gate, timings


def exact_route_predictions(
    group_a_prediction: np.ndarray,
    group_b_prediction: np.ndarray,
    group_a: np.ndarray,
) -> np.ndarray:
    if (
        len(group_a_prediction) != len(group_b_prediction)
        or len(group_a_prediction) != len(group_a)
    ):
        raise ValueError("Expert predictions and route mask must have equal lengths")
    selected = np.where(group_a, group_a_prediction, group_b_prediction)
    if not np.isfinite(selected).all():
        raise ValueError("The selected expert prediction is incomplete")
    return selected


def train_experiment(
    experiment: pd.Series,
    config: dict[str, Any],
    family: dict[str, str],
    global_oof: np.ndarray,
    context: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, pd.DataFrame],
    ],
    raw: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    if family["kind"] == "gpu_cluster":
        return result_row(
            experiment,
            "GPU_BLOCKED",
            "No CUDA-enabled k-means implementation is installed (PyTorch is CPU-only; cuML/CuPy unavailable).",
        )
    numeric, categorical, rowwise, keys, extra = context
    exact_route_oof = np.zeros(len(y), dtype=bool)
    routed_oof = np.full(len(y), np.nan, dtype=np.float64)
    mixture_oof = np.full(len(y), np.nan, dtype=np.float64)
    timings: list[float] = []
    learned_gate = str(experiment.architecture).startswith("R5_")

    for fold, (fit, valid) in enumerate(folds, start=1):
        exact = make_route(family, raw, fit)
        exact_route_oof[valid] = exact[valid]
        x_fit, x_valid, cat_columns = feature_pair(
            config,
            numeric,
            categorical,
            rowwise,
            extra,
            keys,
            raw,
            fit,
            valid,
        )
        if learned_gate:
            gate_train_prob, gate_valid_prob, gate_times = crossfit_gate(
                x_fit,
                exact[fit].astype(np.int8),
                x_valid,
                cat_columns,
                seed=OUTER_SEED + fold * 1000,
            )
            timings.extend(gate_times)
            fit_membership = gate_train_prob >= 0.5
            valid_membership = gate_valid_prob >= 0.5
        else:
            fit_membership = exact[fit]
            valid_membership = exact[valid]

        if len(np.unique(fit_membership)) != 2:
            return result_row(
                experiment,
                "UNSUPPORTED_GROUP_SIZE",
                f"Fold {fold} routed training groups collapsed to one group.",
            )

        expert_predictions: dict[int, np.ndarray] = {}
        for group in (False, True):
            fit_rows = np.flatnonzero(fit_membership == group)
            fit_y = y[fit[fit_rows]]
            if len(fit_rows) < 5000:
                return result_row(
                    experiment,
                    "UNSUPPORTED_GROUP_SIZE",
                    f"Fold {fold} group {group} has only {len(fit_rows)} expert fit rows.",
                )
            class_counts = np.bincount(fit_y, minlength=2)
            if int(class_counts.min()) < MIN_EXPERT_CLASS_ROWS:
                return result_row(
                    experiment,
                    "UNSUPPORTED_GROUP_SIZE",
                    f"Fold {fold} group {group} has fewer than {MIN_EXPERT_CLASS_ROWS} examples of one target class.",
                )
            expert_predictions[int(group)] = np.full(
                len(valid), np.nan, dtype=np.float64
            )
            if learned_gate:
                apply_rows = np.arange(len(valid))
            else:
                apply_rows = np.flatnonzero(valid_membership == group)
                if len(apply_rows) < MIN_EXPERT_VALID_ROWS:
                    return result_row(
                        experiment,
                        "UNSUPPORTED_GROUP_SIZE",
                        f"Fold {fold} group {group} has only {len(apply_rows)} validation rows.",
                    )
            x_expert_fit = x_fit.iloc[fit_rows].reset_index(drop=True)
            x_expert_apply = x_valid.iloc[apply_rows].reset_index(drop=True)
            pred, elapsed = fit_predict(
                config,
                x_expert_fit,
                y[fit[fit_rows]],
                x_expert_apply,
                cat_columns,
                seed=OUTER_SEED + fold * 100 + int(str(config["id"])[1:]) * 10 + int(group),
            )
            timings.append(elapsed)
            expert_predictions[int(group)][apply_rows] = pred

        if learned_gate:
            p_a = expert_predictions[1]
            p_b = expert_predictions[0]
            routed_oof[valid] = np.where(valid_membership, p_a, p_b)
            mixture_oof[valid] = (
                gate_valid_prob * p_a + (1.0 - gate_valid_prob) * p_b
            )
        else:
            routed_oof[valid] = exact_route_predictions(
                expert_predictions[1], expert_predictions[0], valid_membership
            )
            mixture_oof[valid] = (
                GLOBAL_BLEND_WEIGHT * routed_oof[valid]
                + (1.0 - GLOBAL_BLEND_WEIGHT) * global_oof[valid]
            )
        del x_fit, x_valid

    if (
        not np.isfinite(routed_oof).all()
        or not np.isfinite(mixture_oof).all()
        or np.any(exact_route_oof < 0)
    ):
        raise ValueError(f"OOF coverage or finiteness failed for {experiment.experiment_id}")
    return result_row(
        experiment,
        "COMPLETED_GPU",
        "Completed five-fold GPU OOF; label-free route; R1 hard route and R2 mixture evaluated.",
        global_oof=global_oof,
        routed_oof=routed_oof,
        mixture_oof=mixture_oof,
        route_oof=exact_route_oof,
        y=y,
        fold_ids=fold_ids,
        fit_seconds=timings,
    )


def run_placebo(
    experiment: pd.Series,
    config: dict[str, Any],
    global_oof: np.ndarray,
    context: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, pd.DataFrame],
    ],
    raw: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    route = make_placebo(experiment, raw[ID].to_numpy(), len(raw))
    routed_oof = np.full(len(y), np.nan, dtype=np.float64)
    mixture_oof = np.full(len(y), np.nan, dtype=np.float64)
    timings: list[float] = []
    numeric, categorical, rowwise, keys, extra = context
    for fold, (fit, valid) in enumerate(folds, start=1):
        x_fit, x_valid, cat_columns = feature_pair(
            config,
            numeric,
            categorical,
            rowwise,
            extra,
            keys,
            raw,
            fit,
            valid,
        )
        for group in (False, True):
            fit_rows = np.flatnonzero(route[fit] == group)
            valid_rows = np.flatnonzero(route[valid] == group)
            fit_y = y[fit[fit_rows]]
            class_counts = np.bincount(fit_y, minlength=2)
            if (
                len(fit_rows) < 5000
                or int(class_counts.min()) < MIN_EXPERT_CLASS_ROWS
                or len(valid_rows) < MIN_EXPERT_VALID_ROWS
            ):
                return result_row(
                    experiment,
                    "UNSUPPORTED_GROUP_SIZE",
                    f"Fold {fold} group {group} is below the declared minimum.",
                )
            pred, elapsed = fit_predict(
                config,
                x_fit.iloc[fit_rows].reset_index(drop=True),
                y[fit[fit_rows]],
                x_valid.iloc[valid_rows].reset_index(drop=True),
                cat_columns,
                seed=OUTER_SEED + fold * 100 + int(str(config["id"])[1:]) * 10 + int(group),
            )
            timings.append(elapsed)
            routed_oof[valid[valid_rows]] = pred
        mixture_oof[valid] = (
            GLOBAL_BLEND_WEIGHT * routed_oof[valid]
            + (1.0 - GLOBAL_BLEND_WEIGHT) * global_oof[valid]
        )
        del x_fit, x_valid
    if not np.isfinite(routed_oof).all() or not np.isfinite(mixture_oof).all():
        raise ValueError(f"Placebo OOF coverage failed for {experiment.experiment_id}")
    return result_row(
        experiment,
        "COMPLETED_GPU_PLACEBO",
        "Placebo only; never eligible as a deployable route.",
        global_oof=global_oof,
        routed_oof=routed_oof,
        mixture_oof=mixture_oof,
        route_oof=route,
        y=y,
        fold_ids=fold_ids,
        fit_seconds=timings,
    )


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, lineterminator="\n")


def run_stage1(
    registry: pd.DataFrame,
    registry_hash: str,
    train_path: Path,
    output_dir: Path,
) -> None:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    train = pd.read_csv(train_path, low_memory=False)
    if TARGET not in train or ID not in train:
        raise ValueError("Training file must contain TARGET and SK_ID_CURR")
    y_all = train[TARGET].to_numpy(dtype=np.int8)
    if not set(np.unique(y_all)).issubset({0, 1}):
        raise ValueError("TARGET must be binary")
    discovery_global, confirm_global = train_test_split(
        np.arange(len(train)),
        test_size=1.0 - DISCOVERY_FRACTION,
        random_state=DISCOVERY_SEED,
        stratify=y_all,
    )
    discovery = train.iloc[discovery_global].reset_index(drop=True)
    y = discovery[TARGET].to_numpy(dtype=np.int8)
    raw = discovery.drop(columns=[TARGET]).copy()
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=OUTER_SEED)
    folds = list(cv.split(np.zeros(len(y), dtype=np.int8), y))
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = fold

    context = make_matrix_context(raw)
    configs_by_id = {str(config["id"]): config for config in CONFIGS}
    global_predictions: dict[str, np.ndarray] = {}
    gpu_samples: list[dict[str, float]] = []
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu, args=(gpu_samples, stop_monitor), daemon=True
    )
    monitor.start()
    started = time.perf_counter()

    results_path = output_dir / "PHASE20_STAGE1_RESULTS.csv"
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        previous = pd.read_csv(results_path)
        previous = previous.drop(
            columns=[
                "placebo_delta_q95",
                "passes_placebo_q95",
                "passes_fold_consistency",
                "discovery_qualifies",
            ],
            errors="ignore",
        )
        for record in previous.to_dict(orient="records"):
            if record.get("status") in {
                "COMPLETED_GPU",
                "COMPLETED_GPU_PLACEBO",
                "GPU_BLOCKED",
                "UNSUPPORTED_GROUP_SIZE",
            }:
                completed[str(record["experiment_id"])] = record

    try:
        result_rows: list[dict[str, Any]] = list(completed.values())
        todo = registry[~registry["experiment_id"].astype(str).isin(completed)].copy()
        for config_index, config in enumerate(CONFIGS, start=1):
            config_todo = todo[todo.config_id == config["id"]]
            if config_todo.empty:
                continue
            global_oof, _ = fit_global_predictions(
                config, context, raw, y, folds, config_index
            )
            global_predictions[str(config["id"])] = global_oof
            for item in config_todo.itertuples(index=False):
                experiment = pd.Series(item._asdict())
                if experiment.kind == "placebo":
                    result = run_placebo(
                        experiment,
                        config,
                        global_oof,
                        context,
                        raw,
                        y,
                        folds,
                        fold_ids,
                    )
                else:
                    family = next(
                        family
                        for family in FAMILIES
                        if family["id"] == experiment.family_id
                    )
                    result = train_experiment(
                        experiment,
                        config,
                        family,
                        global_oof,
                        context,
                        raw,
                        y,
                        folds,
                        fold_ids,
                    )
                result_rows.append(result)
                append_results(results_path, result_rows)
                n_done = len(result_rows)
                print(
                    f"EXPERIMENT {n_done}/{len(registry)} "
                    f"id={experiment.experiment_id} status={result['status']} "
                    f"auc={result.get('expert_mixture_pooled_auc', '')} "
                    f"delta={result.get('delta_expert_mixture_vs_global', '')} "
                    f"seconds={result.get('total_gpu_fit_seconds', '')}",
                    flush=True,
                )
                if n_done % 10 == 0:
                    print(
                        f"CHECKPOINT completed={n_done} "
                        f"elapsed_seconds={time.perf_counter()-started:.1f}",
                        flush=True,
                    )

        result_frame = pd.DataFrame(result_rows)
        if result_frame["experiment_id"].nunique() != len(registry):
            raise ValueError("Final result table does not cover every registry item")
        real = result_frame[result_frame.kind == "real"]
        placebos = result_frame[result_frame.kind == "placebo"]
        valid_placebos = pd.to_numeric(
            placebos.loc[
                placebos.status == "COMPLETED_GPU_PLACEBO",
                "delta_expert_mixture_vs_global",
            ],
            errors="coerce",
        ).dropna()
        placebo_q95 = (
            float(np.quantile(valid_placebos.to_numpy(), 0.95))
            if len(valid_placebos)
            else float("nan")
        )
        real["placebo_delta_q95"] = placebo_q95
        real_delta = pd.to_numeric(
            real["delta_expert_mixture_vs_global"], errors="coerce"
        )
        real["passes_placebo_q95"] = real_delta > placebo_q95
        fold_positive = pd.to_numeric(
            real["positive_folds_r2"], errors="coerce"
        ).fillna(0)
        real["passes_fold_consistency"] = fold_positive >= 4
        real["passes_fold_consistency"] = real["passes_fold_consistency"] & (
            real["status"] == "COMPLETED_GPU"
        )
        real["discovery_qualifies"] = (
            real["passes_placebo_q95"] & real["passes_fold_consistency"]
        )
        qualified_ids = set(real.loc[real.discovery_qualifies, "experiment_id"])
        result_frame = result_frame.merge(
            real[
                [
                    "experiment_id",
                    "placebo_delta_q95",
                    "passes_placebo_q95",
                    "passes_fold_consistency",
                    "discovery_qualifies",
                ]
            ],
            on="experiment_id",
            how="left",
        )
        append_results(results_path, result_frame.to_dict(orient="records"))
        eligible = real.loc[real.discovery_qualifies].sort_values(
            "delta_expert_mixture_vs_global", ascending=False
        )
        confirm_results: list[dict[str, Any]] = []
        if not eligible.empty:
            confirm_results = run_confirm_once(
                eligible.head(10),
                discovery,
                train.iloc[confirm_global].reset_index(drop=True),
                output_dir,
            )

        gpu_rows = gpu_samples.copy()
        (output_dir / "PHASE20_GPU_TELEMETRY.json").write_text(
            json.dumps(
                {
                    "device_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
                    "catboost_version": __import__("catboost").__version__,
                    "registry_sha256": registry_hash,
                    "samples": gpu_rows,
                    "sample_count": len(gpu_rows),
                    "max_utilization_percent": max(
                        (row["gpu_utilization_percent"] for row in gpu_rows),
                        default=None,
                    ),
                    "max_memory_used_mib": max(
                        (row["memory_used_mib"] for row in gpu_rows), default=None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        report = [
            "# Phase 20 — segmentation and expert-model research",
            "",
            "**Status:** GPU Stage 1 completed. All results are exploratory because "
            "the labels have been examined in prior phases; the 40% holdback is not "
            "an untouched confirmation set.",
            "",
            "## Preregistered design",
            "",
            f"- Registry SHA-256: `{registry_hash}`.",
            "- Exactly 150 real experiments (15 feature-only segmentation families x 10 distinct GPU CatBoost configurations) and 30 placebo controls.",
            "- Data: supplied train file only; 60% stratified discovery / 40% confirmation partition, seed `20261106`; five discovery folds, seed `20261113`.",
            "- Stage 1 used R2: a fixed 50/50 blend of the global GPU model and hard-routed experts; R1 hard routing is also reported diagnostically.",
            "- The learned-gate configurations use R5: a GPU gate trained on feature-defined membership with cross-fitted training-side gate predictions, mixing two GPU experts; these are not fixed R2 blends.",
            "- Model training was CatBoost `task_type='GPU'`; one-hot categorical threshold 256 prevents target-statistic CTRs on the supplied categorical columns (maximum supported by the installed CatBoost). No early stopping was used.",
            "- Fold-local preprocessing: raw rowwise transforms are deterministic; key-frequency features, income median, and any model gate are fit within the relevant training fold.",
            "- Cluster families are explicitly marked GPU-blocked if no CUDA clustering implementation is available; no CPU clustering fallback is used.",
            "- Placebos: 30 row-random or ID-parity splits, using three controls per model configuration to estimate a pooled null distribution.",
            "",
            "## Stage 1 results",
            "",
            f"- Real experiments: {len(real)}; completed GPU: {int((real.status == 'COMPLETED_GPU').sum())}; GPU-blocked: {int((real.status == 'GPU_BLOCKED').sum())}; unsupported groups: {int((real.status == 'UNSUPPORTED_GROUP_SIZE').sum())}.",
            f"- Placebo controls: {len(placebos)}; completed GPU: {int((placebos.status == 'COMPLETED_GPU_PLACEBO').sum())}.",
            f"- Placebo 95th percentile of R2 pooled-AUC deltas: `{placebo_q95:.9f}`.",
            f"- Discovery candidates passing both the placebo threshold and >=4/5 positive folds: {len(qualified_ids)}.",
            "",
            "The complete 180-row status, experiment definitions, pooled/fold AUCs, paired deltas, subgroup sizes/AUCs, log loss, runtimes, and failures are in `PHASE20_STAGE1_RESULTS.csv`. The exact pre-run definitions are in `PHASE20_EXPERIMENT_REGISTRY.csv`.",
            "",
            "## Confirmation and interpretation",
            "",
            (
                f"Only {len(confirm_results)} discovery-qualified candidates were evaluated once on the predeclared 40% holdback."
                if confirm_results
                else "No candidate met the predeclared discovery-and-placebo threshold; the confirmation labels were not evaluated in Phase 20."
            ),
            "Because all labels were previously examined in historical phases, this holdback cannot establish independent generalization. No champion or submission was changed.",
            "The Phase 15 three-seed OOF score remains `0.768824708`; the retrospective Phase 9 mean-prediction score remains `0.769839127`; neither is an independent champion. No `0.780` claim is made unless it is supported by a valid, previously untouched evaluation.",
            "",
            "## Eligibility",
            "",
            "All routes use supplied feature values and never TARGET. No external data or test data was read. CatBoost target-statistic CTRs are disabled by one-hot encoding the supplied categorical features. Constant-free product-key frequency features are fold-local and unsupervised but remain subject to organizer interpretation. Cluster families are not run on CPU if CUDA clustering is unavailable.",
            "",
        ]
        (output_dir / "PHASE20_EXECUTION_REPORT.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        split_info = {
            "train_rows": len(train),
            "discovery_rows": len(discovery),
            "confirm_rows": len(confirm_global),
            "discovery_positive_rows": int(y.sum()),
            "confirm_positive_rows": int(y_all[confirm_global].sum()),
            "discovery_seed": DISCOVERY_SEED,
            "outer_seed": OUTER_SEED,
            "registry_sha256": registry_hash,
            "train_sha256": sha256_file(train_path),
            "runtime_seconds": time.perf_counter() - started,
            "confirm_evaluated_once": bool(confirm_results),
        }
        (output_dir / "PHASE20_RUN_SUMMARY.json").write_text(
            json.dumps(split_info, indent=2, sort_keys=True), encoding="utf-8"
        )
    finally:
        stop_monitor.set()
        monitor.join(timeout=10)


def run_confirm_once(
    candidates: pd.DataFrame,
    discovery: pd.DataFrame,
    confirm: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, Any]]:
    combined = pd.concat([discovery, confirm], ignore_index=True)
    discovery_count = len(discovery)
    raw = combined.drop(columns=[TARGET]).copy()
    y_discovery = discovery[TARGET].to_numpy(dtype=np.int8)
    y_confirm = confirm[TARGET].to_numpy(dtype=np.int8)
    fit = np.arange(discovery_count)
    apply = np.arange(discovery_count, len(combined))
    context = make_matrix_context(raw)
    numeric, categorical, rowwise, keys, extra = context
    configs_by_id = {str(config["id"]): config for config in CONFIGS}
    results: list[dict[str, Any]] = []

    for candidate in candidates.head(10).itertuples(index=False):
        config = configs_by_id[str(candidate.config_id)]
        family = next(family for family in FAMILIES if family["id"] == candidate.family_id)
        x_fit, x_apply, cat_columns = feature_pair(
            config,
            numeric,
            categorical,
            rowwise,
            extra,
            keys,
            raw,
            fit,
            apply,
        )
        global_prediction, global_seconds = fit_predict(
            config,
            x_fit,
            y_discovery,
            x_apply,
            cat_columns,
            seed=OUTER_SEED + 99,
        )
        route = make_route(family, raw, fit)
        learned_gate = str(candidate.architecture).startswith("R5_")
        gate_timings: list[float] = []
        if learned_gate:
            gate_fit_probability, gate_apply_probability, gate_timings = crossfit_gate(
                x_fit,
                route[fit].astype(np.int8),
                x_apply,
                cat_columns,
                seed=OUTER_SEED + 777,
            )
            fit_membership = gate_fit_probability >= 0.5
            apply_membership = gate_apply_probability >= 0.5
        else:
            fit_membership = route[fit]
            apply_membership = route[apply]
            gate_apply_probability = apply_membership.astype(np.float64)
        expert_predictions: dict[int, np.ndarray] = {}
        expert_timings: list[float] = []
        for group in (False, True):
            group_fit = np.flatnonzero(fit_membership == group)
            if (
                len(group_fit) < 5000
                or int(np.bincount(y_discovery[group_fit], minlength=2).min())
                < MIN_EXPERT_CLASS_ROWS
            ):
                raise ValueError(
                    f"Qualified candidate {candidate.experiment_id} has insufficient confirm-fit expert support"
                )
            pred, elapsed = fit_predict(
                config,
                x_fit.iloc[group_fit].reset_index(drop=True),
                y_discovery[group_fit],
                x_apply,
                cat_columns,
                seed=OUTER_SEED + 1000 + int(group),
            )
            expert_predictions[int(group)] = pred
            expert_timings.append(elapsed)
        routed = np.where(apply_membership, expert_predictions[1], expert_predictions[0])
        if learned_gate:
            mixture = (
                gate_apply_probability * expert_predictions[1]
                + (1.0 - gate_apply_probability) * expert_predictions[0]
            )
        else:
            mixture = (
                GLOBAL_BLEND_WEIGHT * routed
                + (1.0 - GLOBAL_BLEND_WEIGHT) * global_prediction
            )
        if not np.isfinite(mixture).all():
            raise ValueError(
                f"Non-finite confirmation predictions for {candidate.experiment_id}"
            )
        results.append(
            {
                "experiment_id": candidate.experiment_id,
                "status": "CONFIRM_EVALUATED_ONCE_EXPLORATORY",
                "global_auc": float(roc_auc_score(y_confirm, global_prediction)),
                "hard_route_auc": float(roc_auc_score(y_confirm, routed)),
                "mixture_auc": float(roc_auc_score(y_confirm, mixture)),
                "delta_mixture_vs_global": float(
                    roc_auc_score(y_confirm, mixture)
                    - roc_auc_score(y_confirm, global_prediction)
                ),
                "confirm_rows": len(confirm),
                "confirm_positives": int(y_confirm.sum()),
                "global_logloss": float(log_loss(y_confirm, global_prediction)),
                "mixture_logloss": float(log_loss(y_confirm, mixture)),
                "total_gpu_fit_seconds": float(
                    global_seconds + sum(gate_timings) + sum(expert_timings)
                ),
                "independent_confirmation": False,
                "reason": "The labels have been examined in earlier project phases.",
            }
        )
        del x_fit, x_apply
    pd.DataFrame(results).to_csv(
        output_dir / "PHASE20_CONFIRMATION_RESULTS.csv",
        index=False,
        lineterminator="\n",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--registry-only", action="store_true")
    args = parser.parse_args()
    registry = build_registry()
    registry_hash = write_registry(registry, args.output_dir.resolve())
    if args.registry_only:
        return
    run_stage1(registry, registry_hash, args.train.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
