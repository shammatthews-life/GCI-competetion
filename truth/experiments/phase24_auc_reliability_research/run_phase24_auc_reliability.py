from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
import sys

sys.path.insert(0, str(PHASE13))
import phase13_features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
REFERENCE_OOF = (
    REPO
    / "experiments"
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT = ROOT / "runs" / "phase24_forensics_20261106"
SEED = 20261113
FOLDS = 5
ITERATIONS = 600
PRODUCT_KEY = "PRODUCT_KEY_Q4"
MODEL_COLUMNS = (
    "native_catboost_base_all8",
    "native_catboost_phase14_numeric",
    "native_catboost_phase14_plus_time",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_fold_ids(
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    if len(y) != len(fold_ids):
        raise ValueError("Target and fold arrays have different lengths")
    values = np.unique(fold_ids)
    if not np.array_equal(values, np.arange(1, FOLDS + 1)):
        raise ValueError(f"Expected fold IDs 1..{FOLDS}, got {values.tolist()}")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Targets must be finite binary labels")
    return fold_ids.astype(np.int8, copy=False)


def oof_auc_summary(
    y: np.ndarray,
    prediction: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    if len(y) != len(prediction):
        raise ValueError("Target and prediction lengths differ")
    if not np.isfinite(prediction).all():
        raise ValueError("OOF predictions must be finite")
    fold_ids = _validated_fold_ids(y, fold_ids)
    per_fold: list[dict[str, float | int]] = []
    cross_fold_pairs = 0
    total_pairs = int(np.sum(y == 1) * np.sum(y == 0))
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        fold_y = y[mask]
        fold_prediction = prediction[mask]
        positives = int(np.sum(fold_y == 1))
        negatives = int(np.sum(fold_y == 0))
        if positives == 0 or negatives == 0:
            raise ValueError(f"Fold {fold} does not contain both target classes")
        cross_fold_pairs += positives * negatives
        per_fold.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "positives": positives,
                "negatives": negatives,
                "auc": float(roc_auc_score(fold_y, fold_prediction)),
                "prediction_mean": float(fold_prediction.mean()),
                "prediction_std": float(fold_prediction.std()),
                "prediction_q10": float(np.quantile(fold_prediction, 0.10)),
                "prediction_median": float(np.quantile(fold_prediction, 0.50)),
                "prediction_q90": float(np.quantile(fold_prediction, 0.90)),
            }
        )
    fold_aucs = np.asarray([row["auc"] for row in per_fold], dtype=np.float64)
    row_weights = np.asarray([row["rows"] for row in per_fold], dtype=np.float64)
    pair_weights = np.asarray(
        [row["positives"] * row["negatives"] for row in per_fold],
        dtype=np.float64,
    )
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "mean_fold_auc": float(fold_aucs.mean()),
        "row_weighted_fold_auc": float(np.average(fold_aucs, weights=row_weights)),
        "pair_weighted_fold_auc": float(np.average(fold_aucs, weights=pair_weights)),
        "fold_auc_std": float(fold_aucs.std(ddof=1)),
        "pooled_minus_mean_fold_auc": float(roc_auc_score(y, prediction) - fold_aucs.mean()),
        "cross_fold_pair_fraction": float(1.0 - cross_fold_pairs / total_pairs),
        "per_fold": per_fold,
    }


def _group_diagnostics(
    frame: pd.DataFrame,
    target: np.ndarray,
    fold_ids: np.ndarray,
    columns: list[str],
    profile_name: str,
) -> dict[str, Any]:
    group_ids = frame.loc[:, columns].groupby(
        columns, dropna=False, sort=False, observed=True
    ).ngroup().to_numpy(dtype=np.int64)
    group_sizes = np.bincount(group_ids)
    stats = pd.DataFrame(
        {"group": group_ids, "target": target, "fold": fold_ids}
    ).groupby("group", sort=False).agg(
        size=("target", "size"),
        target_count=("target", "nunique"),
        fold_count=("fold", "nunique"),
    )
    duplicate_groups = stats["size"] > 1
    cross_fold_duplicate_groups = duplicate_groups & (stats["fold_count"] > 1)
    return {
        "profile": profile_name,
        "columns": columns,
        "unique_groups": int(len(group_sizes)),
        "duplicate_groups": int(duplicate_groups.sum()),
        "rows_in_duplicate_groups": int(stats.loc[duplicate_groups, "size"].sum()),
        "largest_group": int(group_sizes.max(initial=0)),
        "mixed_target_duplicate_groups": int(
            (duplicate_groups & (stats["target_count"] > 1)).sum()
        ),
        "duplicate_groups_crossing_random_folds": int(
            cross_fold_duplicate_groups.sum()
        ),
        "rows_in_duplicate_groups_crossing_random_folds": int(
            stats.loc[cross_fold_duplicate_groups, "size"].sum()
        ),
    }


def data_audit(
    train: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    id_column = phase13_features.ID_COLUMN
    target_column = phase13_features.TARGET_COLUMN
    y = train[target_column].to_numpy(dtype=np.int8)
    fold_ids = _validated_fold_ids(
        y, reference["fold"].to_numpy(dtype=np.int8)
    )
    if not np.array_equal(train[id_column].to_numpy(), reference[id_column].to_numpy()):
        raise ValueError("Reference OOF IDs do not align with the training data")
    if not np.array_equal(y, reference[target_column].to_numpy(dtype=np.int8)):
        raise ValueError("Reference OOF targets do not align with the training data")

    audits: list[dict[str, Any]] = []
    for column in MODEL_COLUMNS:
        if column not in reference:
            raise ValueError(f"Reference OOF is missing column {column}")
        summary = oof_auc_summary(
            y,
            reference[column].to_numpy(dtype=np.float64),
            fold_ids,
        )
        audits.append(
            {
                "prediction": column,
                **{key: value for key, value in summary.items() if key != "per_fold"},
            }
        )
        pd.DataFrame(summary["per_fold"]).assign(prediction=column).to_csv(
            output_dir / f"PHASE24_FOLD_SCORE_DISTRIBUTIONS_{column}.csv",
            index=False,
            lineterminator="\n",
        )

    predictor = train.drop(columns=[id_column, target_column])
    profile_specs = [
        ("all_predictors", list(predictor.columns)),
        (
            "financial_application",
            [
                "NAME_CONTRACT_TYPE",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
                "AMT_INCOME_TOTAL",
            ],
        ),
        (
            "categorical_applicant",
            [
                "NAME_CONTRACT_TYPE",
                "CODE_GENDER",
                "NAME_INCOME_TYPE",
                "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS",
                "NAME_HOUSING_TYPE",
                "OCCUPATION_TYPE",
                "ORGANIZATION_TYPE",
            ],
        ),
    ]
    profile_rows = [
        _group_diagnostics(
            predictor, y, fold_ids, columns, name
        )
        for name, columns in profile_specs
    ]
    exact_duplicate_rows = int(predictor.duplicated(keep=False).sum())

    decile = pd.qcut(train[id_column], 20, labels=False, duplicates="drop")
    id_bins = (
        pd.DataFrame(
            {
                "id_bin": decile.to_numpy(),
                "id_min": train[id_column].groupby(decile, observed=True).transform("min"),
                "id_max": train[id_column].groupby(decile, observed=True).transform("max"),
                target_column: y,
            }
        )
        .groupby("id_bin", observed=True)
        .agg(
            rows=(target_column, "size"),
            positives=(target_column, "sum"),
            target_rate=(target_column, "mean"),
            id_min=("id_min", "min"),
            id_max=("id_max", "max"),
        )
        .reset_index()
    )
    id_only_auc = float(roc_auc_score(y, train[id_column].to_numpy(dtype=np.float64)))
    profile_report = {
        "rows": int(len(train)),
        "target_rate": float(y.mean()),
        "exact_duplicate_predictor_rows": exact_duplicate_rows,
        "reference_fold_ids_verified": True,
        "oof_score_diagnostics": audits,
        "duplicate_profile_diagnostics": profile_rows,
        "id_only_monotonic_auc": id_only_auc,
        "id_only_orientation_free_auc": float(max(id_only_auc, 1.0 - id_only_auc)),
        "id_decile_count": int(len(id_bins)),
        "id_target_rate_min": float(id_bins["target_rate"].min()),
        "id_target_rate_max": float(id_bins["target_rate"].max()),
        "interpretation": (
            "Descriptive audit of previously exposed labels. OOF cross-fold "
            "comparisons and duplicate-group statistics do not establish "
            "independent generalization or data leakage."
        ),
    }
    (output_dir / "PHASE24_DATA_AUDIT.json").write_text(
        json.dumps(profile_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(profile_rows).to_json(
        output_dir / "PHASE24_DUPLICATE_PROFILE_AUDIT.json",
        orient="records",
        indent=2,
    )
    id_bins.to_csv(
        output_dir / "PHASE24_ID_INTERVALS.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(audits).to_csv(
        output_dir / "PHASE24_OOF_SCORE_AUDIT.csv",
        index=False,
        lineterminator="\n",
    )
    return profile_report


def _model_frame(raw: pd.DataFrame, include_product_key: bool) -> tuple[pd.DataFrame, list[str]]:
    frame = raw.drop(columns=[phase13_features.ID_COLUMN]).copy()
    categories = [
        column
        for column in frame
        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    for column in categories:
        frame[column] = (
            frame[column].astype("string").fillna("MISSING").astype(str)
        )
    if include_product_key:
        key = phase13_features.build_product_keys(raw)[PRODUCT_KEY]
        frame[PRODUCT_KEY] = key.astype(str).to_numpy()
        categories.append(PRODUCT_KEY)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame, categories


def _make_splits(
    train: pd.DataFrame,
    y: np.ndarray,
    product_key: np.ndarray,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    random_cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    random_splits = list(random_cv.split(np.zeros(len(y)), y))
    key_cv = GroupKFold(n_splits=FOLDS)
    key_splits = list(
        key_cv.split(np.zeros(len(y)), y, groups=product_key)
    )
    order = np.argsort(train[phase13_features.ID_COLUMN].to_numpy(), kind="stable")
    id_blocks = np.array_split(order, FOLDS)
    all_rows = np.arange(len(y))
    id_splits = [
        (np.setdiff1d(all_rows, valid, assume_unique=False), valid)
        for valid in id_blocks
    ]
    return {
        "random_stratified": random_splits,
        "unseen_q4_product_key": key_splits,
        "contiguous_id_block": id_splits,
    }


def _gpu_status() -> dict[str, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    values = [float(value.strip()) for value in result.stdout.split(",")]
    if len(values) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout!r}")
    return {
        "utilization_percent": values[0],
        "memory_used_mib": values[1],
        "memory_total_mib": values[2],
        "memory_free_mib": values[2] - values[1],
    }


def _fit_predict(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_valid: pd.DataFrame,
    categories: list[str],
    seed: int,
) -> tuple[np.ndarray, float]:
    if not np.array_equal(x_fit.columns.to_numpy(), x_valid.columns.to_numpy()):
        raise ValueError("Fit and validation feature schemas differ")
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.10,
        border_count=128,
        one_hot_max_size=10,
        max_ctr_complexity=1,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=int(seed),
        allow_writing_files=False,
        verbose=False,
        thread_count=4,
    )
    started = time.perf_counter()
    model.fit(x_fit, y_fit, cat_features=[x_fit.columns.get_loc(c) for c in categories])
    prediction = model.predict_proba(x_valid)[:, 1].astype(np.float64)
    elapsed = time.perf_counter() - started
    if model.get_all_params().get("task_type") != "GPU":
        raise RuntimeError("CatBoost did not execute on GPU")
    if not np.isfinite(prediction).all():
        raise ValueError("GPU model generated non-finite predictions")
    del model
    return prediction, elapsed


def run_gpu_experiment(
    train: pd.DataFrame,
    reference: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    gpu_before = _gpu_status()
    if gpu_before["memory_free_mib"] < 1600:
        raise RuntimeError(
            "GPU preflight requires at least 1600 MiB free; "
            f"observed {gpu_before['memory_free_mib']:.0f} MiB"
        )
    target_column = phase13_features.TARGET_COLUMN
    y = train[target_column].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[target_column]).reset_index(drop=True)
    product_key = phase13_features.build_product_keys(raw)[PRODUCT_KEY].to_numpy()
    matrices = {
        "raw_baseline": _model_frame(raw, include_product_key=False),
        "raw_plus_q4_product_key": _model_frame(raw, include_product_key=True),
    }
    split_sets = _make_splits(train, y, product_key)
    prediction_store: dict[str, np.ndarray] = {}
    fit_log: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for split_name, splits in split_sets.items():
        for candidate, (matrix, categories) in matrices.items():
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for fold, (fit_rows, valid_rows) in enumerate(splits, start=1):
                if len(np.unique(y[fit_rows])) != 2 or len(np.unique(y[valid_rows])) != 2:
                    raise ValueError(f"{split_name} fold {fold} lacks one target class")
                if split_name == "unseen_q4_product_key":
                    overlap = np.intersect1d(
                        product_key[fit_rows], product_key[valid_rows]
                    )
                    if len(overlap):
                        raise ValueError("Q4 key group leaked across validation fold")
                x_fit = matrix.iloc[fit_rows].reset_index(drop=True)
                x_valid = matrix.iloc[valid_rows].reset_index(drop=True)
                prediction, fit_seconds = _fit_predict(
                    x_fit,
                    y[fit_rows],
                    x_valid,
                    categories,
                    SEED + fold * 100 + (0 if candidate == "raw_baseline" else 1),
                )
                oof[valid_rows] = prediction
                fit_log.append(
                    {
                        "split": split_name,
                        "candidate": candidate,
                        "fold": fold,
                        "fit_rows": int(len(fit_rows)),
                        "valid_rows": int(len(valid_rows)),
                        "fit_seconds": fit_seconds,
                        "gpu_task_type": "GPU",
                        "q4_unseen_validation_rows": (
                            int(np.isin(product_key[valid_rows], product_key[fit_rows]).sum())
                            if split_name == "random_stratified"
                            else 0
                        ),
                    }
                )
            if not np.isfinite(oof).all():
                raise ValueError(f"{split_name}/{candidate} OOF is incomplete")
            prediction_store[f"{split_name}__{candidate}"] = oof
            summary = oof_auc_summary(y, oof, _fold_ids_from_splits(splits, len(y)))
            split_rows.append(
                {
                    "split": split_name,
                    "candidate": candidate,
                    "feature_count": int(matrix.shape[1]),
                    "pooled_auc": summary["pooled_auc"],
                    "mean_fold_auc": summary["mean_fold_auc"],
                    "row_weighted_fold_auc": summary["row_weighted_fold_auc"],
                    "fold_auc_std": summary["fold_auc_std"],
                    "cross_fold_pair_fraction": summary["cross_fold_pair_fraction"],
                    "per_fold_auc": json.dumps(summary["per_fold"]),
                }
            )
            print(
                f"{split_name} {candidate}: pooled={summary['pooled_auc']:.6f} "
                f"mean_fold={summary['mean_fold_auc']:.6f}",
                flush=True,
            )

    oof_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            phase13_features.ID_COLUMN: train[phase13_features.ID_COLUMN],
            target_column: y,
            **prediction_store,
        }
    )
    oof_frame.to_csv(
        output_dir / "PHASE24_GPU_OOF_PREDICTIONS.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "PHASE24_GPU_MODEL_RESULTS.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(fit_log).to_csv(
        output_dir / "PHASE24_GPU_FIT_LOG.csv",
        index=False,
        lineterminator="\n",
    )
    return {
        "gpu_before": gpu_before,
        "gpu_after": _gpu_status(),
        "fit_count": len(fit_log),
        "all_fits_gpu": all(row["gpu_task_type"] == "GPU" for row in fit_log),
        "iterations": ITERATIONS,
        "fold_count": FOLDS,
        "seed": SEED,
        "results": split_rows,
    }


def _fold_ids_from_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    row_count: int,
) -> np.ndarray:
    fold_ids = np.zeros(row_count, dtype=np.int8)
    for fold, (_, valid_rows) in enumerate(splits, start=1):
        if np.any(fold_ids[valid_rows] != 0):
            raise ValueError("Validation rows overlap across folds")
        fold_ids[valid_rows] = fold
    if np.any(fold_ids == 0):
        raise ValueError("Some rows have no validation fold")
    return fold_ids


def run(
    train_path: Path,
    reference_path: Path,
    output_dir: Path,
    audit_only: bool,
) -> None:
    if not train_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Training data or reference OOF file is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path, low_memory=False)
    reference = pd.read_csv(reference_path)
    if len(train) != len(reference):
        raise ValueError("Reference OOF row count differs from the training data")
    started = time.perf_counter()
    audit = data_audit(train, reference, output_dir)
    gpu_result = None if audit_only else run_gpu_experiment(train, reference, output_dir)
    summary = {
        "rows": len(train),
        "train_sha256": sha256_file(train_path),
        "reference_oof_sha256": sha256_file(reference_path),
        "source_sha256": sha256_file(Path(__file__)),
        "runtime_seconds": time.perf_counter() - started,
        "test_data_read": False,
        "external_data_read": False,
        "independent_confirmation": False,
        "selection_bias_quantified": False,
        "selection_bias_note": (
            "The complete historical experiment-selection universe and all "
            "candidate scores are not represented in this run; winner's-curse "
            "magnitude cannot be honestly estimated from the selected OOF files."
        ),
        "data_audit": audit,
        "gpu_experiment": gpu_result,
    }
    summary_path = output_dir / "PHASE24_RUN_SUMMARY.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "PHASE24_SHA256_MANIFEST.csv":
            continue
        manifest_rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_rows.extend(
        [
            {
                "path": str(train_path),
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            {
                "path": str(reference_path),
                "bytes": reference_path.stat().st_size,
                "sha256": sha256_file(reference_path),
            },
            {
                "path": str(Path(__file__).resolve()),
                "bytes": Path(__file__).stat().st_size,
                "sha256": sha256_file(Path(__file__)),
            },
        ]
    )
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "PHASE24_SHA256_MANIFEST.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--reference-oof", type=Path, default=REFERENCE_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run(
        args.train.resolve(),
        args.reference_oof.resolve(),
        args.output_dir.resolve(),
        args.audit_only,
    )


if __name__ == "__main__":
    main()
