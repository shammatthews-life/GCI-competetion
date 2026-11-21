from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
CATEGORICAL_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)

SEED = 20261113
FOLDS = 5
ITERATIONS = 700
DEPTH = 6
LEARNING_RATE = 0.05
THREADS = 4
ONE_HOT_MAX_SIZE = 255
BOOTSTRAP_REPLICATES = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def build_raw_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    excluded = [column for column in (ID_COLUMN, TARGET_COLUMN) if column in raw]
    predictors = raw.drop(columns=excluded).copy()
    categorical = [column for column in CATEGORICAL_COLUMNS if column in predictors]

    for column in categorical:
        predictors[column] = (
            predictors[column]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    for column in predictors.columns:
        if column in categorical:
            continue
        values = pd.to_numeric(predictors[column], errors="coerce")
        predictors[column] = values.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )
    return predictors, categorical


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create applicant-local features; no statistics are estimated across rows."""
    predictors, categorical = build_raw_features(raw)
    excluded = [column for column in (ID_COLUMN, TARGET_COLUMN) if column in raw]

    extra: dict[str, pd.Series] = {}
    income = _column(raw, "AMT_INCOME_TOTAL")
    credit = _column(raw, "AMT_CREDIT")
    annuity = _column(raw, "AMT_ANNUITY")
    goods = _column(raw, "AMT_GOODS_PRICE")
    family = _column(raw, "CNT_FAM_MEMBERS")
    age_days = -_column(raw, "DAYS_BIRTH")
    employed_raw = _column(raw, "DAYS_EMPLOYED")
    employed = employed_raw.where(employed_raw != 365243.0)

    extra["MISSING_COUNT"] = raw.drop(
        columns=excluded, errors="ignore"
    ).isna().sum(axis=1).astype("float32")
    extra["EMPLOYED_SENTINEL"] = (employed_raw == 365243.0).astype("float32")
    extra["DAYS_EMPLOYED_CLEAN"] = employed
    extra["AGE_YEARS"] = age_days / 365.25
    extra["EMPLOYED_YEARS"] = -employed / 365.25
    extra["EMPLOYED_TO_AGE"] = _ratio(employed, _column(raw, "DAYS_BIRTH"))
    extra["REGISTRATION_TO_AGE"] = _ratio(
        _column(raw, "DAYS_REGISTRATION"), _column(raw, "DAYS_BIRTH")
    )
    extra["ID_PUBLISH_TO_AGE"] = _ratio(
        _column(raw, "DAYS_ID_PUBLISH"), _column(raw, "DAYS_BIRTH")
    )
    extra["PHONE_CHANGE_TO_AGE"] = _ratio(
        _column(raw, "DAYS_LAST_PHONE_CHANGE"), _column(raw, "DAYS_BIRTH")
    )

    extra["CREDIT_TO_INCOME"] = _ratio(credit, income)
    extra["ANNUITY_TO_INCOME"] = _ratio(annuity, income)
    extra["ANNUITY_TO_CREDIT"] = _ratio(annuity, credit)
    extra["CREDIT_TO_ANNUITY"] = _ratio(credit, annuity)
    extra["GOODS_TO_CREDIT"] = _ratio(goods, credit)
    extra["DOWNPAYMENT_AMOUNT"] = credit - goods
    extra["DOWNPAYMENT_TO_CREDIT"] = _ratio(credit - goods, credit)
    extra["GOODS_TO_INCOME"] = _ratio(goods, income)
    extra["INCOME_PER_FAMILY_MEMBER"] = _ratio(income, family)
    extra["CREDIT_PER_FAMILY_MEMBER"] = _ratio(credit, family)
    extra["ANNUITY_PER_FAMILY_MEMBER"] = _ratio(annuity, family)
    for column, values in (
        ("INCOME", income),
        ("CREDIT", credit),
        ("ANNUITY", annuity),
        ("GOODS_PRICE", goods),
    ):
        extra[f"LOG1P_{column}"] = np.log1p(values.clip(lower=0.0))

    term_proxy = _ratio(credit, annuity)
    for decimals in (1, 2, 3):
        extra[f"CREDIT_ANNUITY_ROUNDED_{decimals}"] = term_proxy.round(decimals)
    extra["GOODS_CREDIT_ROUNDED_3"] = _ratio(goods, credit).round(3)
    extra["ANNUITY_GOODS_ROUNDED_3"] = _ratio(annuity, goods).round(3)

    ext_names = [
        name
        for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if name in raw.columns
    ]
    ext = pd.concat([_column(raw, name) for name in ext_names], axis=1)
    if ext_names:
        extra["EXT_MEAN"] = ext.mean(axis=1)
        extra["EXT_MIN"] = ext.min(axis=1)
        extra["EXT_MAX"] = ext.max(axis=1)
        extra["EXT_STD"] = ext.std(axis=1)
        extra["EXT_RANGE"] = extra["EXT_MAX"] - extra["EXT_MIN"]
        extra["EXT_PRESENT_COUNT"] = ext.notna().sum(axis=1).astype("float32")
        extra["EXT_PRODUCT"] = ext.prod(axis=1, min_count=1)
        extra["EXT_MEAN_X_CREDIT_TO_INCOME"] = (
            extra["EXT_MEAN"] * extra["CREDIT_TO_INCOME"]
        )
        extra["EXT_MEAN_X_ANNUITY_TO_INCOME"] = (
            extra["EXT_MEAN"] * extra["ANNUITY_TO_INCOME"]
        )
        for left_index in range(len(ext_names)):
            for right_index in range(left_index + 1, len(ext_names)):
                left_name, right_name = ext_names[left_index], ext_names[right_index]
                left, right = ext.iloc[:, left_index], ext.iloc[:, right_index]
                pair = f"{left_name[-1]}{right_name[-1]}"
                extra[f"EXT_DIFF_{pair}"] = left - right
                extra[f"EXT_PRODUCT_{pair}"] = left * right
                extra[f"EXT_RATIO_{pair}"] = _ratio(left, right)

    bureau_names = [
        name
        for name in raw.columns
        if name.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if bureau_names:
        bureau = pd.concat([_column(raw, name) for name in bureau_names], axis=1)
        extra["BUREAU_REQUEST_SUM"] = bureau.sum(axis=1, min_count=1)
        extra["BUREAU_REQUEST_NONZERO_COUNT"] = (bureau > 0).sum(axis=1).astype(
            "float32"
        )
    for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE"):
        if name in raw.columns:
            extra[f"{name}_MISSING"] = raw[name].isna().astype("float32")

    engineered = pd.DataFrame(extra, index=raw.index)
    features = pd.concat(
        [predictors.reset_index(drop=True), engineered.reset_index(drop=True)],
        axis=1,
    )
    if features.columns.duplicated().any():
        raise ValueError("Feature engineering created duplicate column names")

    for column in features.columns:
        if column in categorical:
            continue
        values = pd.to_numeric(features[column], errors="coerce")
        features[column] = values.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )

    return features, categorical


def make_model(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=ITERATIONS,
        depth=DEPTH,
        learning_rate=LEARNING_RATE,
        loss_function="Logloss",
        eval_metric="Logloss",
        task_type="CPU",
        thread_count=THREADS,
        one_hot_max_size=ONE_HOT_MAX_SIZE,
        max_ctr_complexity=1,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )


def _auc_record(
    labels: np.ndarray, prediction: np.ndarray, fold_ids: np.ndarray
) -> dict[str, Any]:
    fold_auc = [
        float(roc_auc_score(labels[fold_ids == fold], prediction[fold_ids == fold]))
        for fold in sorted(np.unique(fold_ids))
    ]
    return {
        "pooled_auc": float(roc_auc_score(labels, prediction)),
        "fold_auc": fold_auc,
        "mean_fold_auc": float(np.mean(fold_auc)),
        "std_fold_auc": float(np.std(fold_auc)),
        "log_loss": float(log_loss(labels, prediction)),
    }


def paired_auc_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    repetitions: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 99)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        rows = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(labels[rows], candidate[rows]) - roc_auc_score(
            labels[rows], reference[rows]
        )
    return {
        "replicates": repetitions,
        "delta_mean": float(deltas.mean()),
        "delta_ci_2_5": float(np.quantile(deltas, 0.025)),
        "delta_ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0.0)),
        "interpretation": (
            "Conditional on these OOF vectors; does not include training-fold "
            "variability, earlier research selection, or leaderboard sampling."
        ),
    }


def run_self_tests() -> None:
    sample = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 11],
            "TARGET": [0, 1],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
            "CODE_GENDER": ["F", "M"],
            "AMT_INCOME_TOTAL": [100.0, 200.0],
            "AMT_CREDIT": [1000.0, 1200.0],
            "AMT_ANNUITY": [100.0, 120.0],
            "AMT_GOODS_PRICE": [900.0, 1000.0],
            "CNT_FAM_MEMBERS": [2.0, 3.0],
            "DAYS_BIRTH": [-10000.0, -12000.0],
            "DAYS_EMPLOYED": [365243.0, -1000.0],
            "EXT_SOURCE_1": [0.2, np.nan],
            "EXT_SOURCE_2": [0.4, 0.7],
            "EXT_SOURCE_3": [0.6, 0.8],
        }
    )
    features, categorical = build_features(sample)
    if TARGET_COLUMN in features or ID_COLUMN in features:
        raise AssertionError("ID or target was included in features")
    numeric_features = features.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64
    )
    if np.isinf(numeric_features).any():
        raise AssertionError("Feature construction unexpectedly created infinity")
    single, _ = build_features(sample.iloc[[0]].copy())
    together, _ = build_features(sample.copy())
    pd.testing.assert_frame_equal(
        single.reset_index(drop=True),
        together.iloc[[0]].reset_index(drop=True),
        check_dtype=False,
    )
    if not math.isclose(
        float(features.loc[0, "CREDIT_TO_ANNUITY"]), 10.0, rel_tol=0, abs_tol=1e-6
    ):
        raise AssertionError("Credit/annuity ratio check failed")


def run(train_path: Path, test_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Both --train and --test must be existing CSV files")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_train = {ID_COLUMN, TARGET_COLUMN}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Train file must contain {sorted(required_train)}")
    if ID_COLUMN not in test.columns:
        raise ValueError(f"Test file must contain {ID_COLUMN}")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        raise ValueError("Train and test IDs must each be unique")
    if set(train.columns) - {TARGET_COLUMN} != set(test.columns):
        raise ValueError("Train predictors and test columns do not match")
    if list(train.drop(columns=[TARGET_COLUMN]).columns) != list(test.columns):
        raise ValueError("Train and test columns must have the same order")

    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary labels 0 and 1")
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if np.any(fold_ids < 1):
        raise ValueError("Every training row must receive one OOF fold")

    raw_train = train.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    baseline_train, baseline_categories = build_raw_features(raw_train)
    engineered_train, engineered_categories = build_features(raw_train)
    engineered_test, test_categories = build_features(raw_test)
    if baseline_categories != engineered_categories or engineered_categories != test_categories:
        raise ValueError("Categorical feature lists differ across partitions")
    if not engineered_train.columns.equals(engineered_test.columns):
        raise ValueError("Train and test engineered schemas differ")
    if engineered_categories:
        cardinality = max(
            int(engineered_train[column].nunique(dropna=False))
            for column in engineered_categories
        )
        if cardinality > ONE_HOT_MAX_SIZE:
            raise ValueError(
                "A categorical feature exceeds one-hot limit; refusing to use "
                "CatBoost target statistics in this conservative run"
            )
    else:
        cardinality = 0

    n_rows = len(train)
    baseline_oof = np.full(n_rows, np.nan, dtype=np.float64)
    engineered_oof = np.full(n_rows, np.nan, dtype=np.float64)
    engineered_test_sum = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        model_seed = SEED + fold_number
        baseline_model = make_model(model_seed)
        baseline_model.fit(
            baseline_train.iloc[fit_rows],
            labels[fit_rows],
            cat_features=baseline_categories,
        )
        baseline_prediction = baseline_model.predict_proba(
            baseline_train.iloc[valid_rows]
        )[:, 1]
        baseline_oof[valid_rows] = baseline_prediction
        baseline_fold_auc = float(
            roc_auc_score(labels[valid_rows], baseline_prediction)
        )
        del baseline_model

        model = make_model(model_seed)
        model.fit(
            engineered_train.iloc[fit_rows],
            labels[fit_rows],
            cat_features=engineered_categories,
        )
        prediction = model.predict_proba(engineered_train.iloc[valid_rows])[:, 1]
        test_prediction = model.predict_proba(engineered_test)[:, 1]
        engineered_oof[valid_rows] = prediction
        engineered_test_sum += test_prediction / FOLDS
        engineered_fold_auc = float(roc_auc_score(labels[valid_rows], prediction))
        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "baseline_auc": baseline_fold_auc,
                "engineered_auc": engineered_fold_auc,
                "delta_auc": engineered_fold_auc - baseline_fold_auc,
                "iteration_count": ITERATIONS,
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: baseline={baseline_fold_auc:.6f} "
            f"engineered={engineered_fold_auc:.6f} "
            f"delta={engineered_fold_auc - baseline_fold_auc:+.6f}",
            flush=True,
        )
        del model

    if not np.isfinite(baseline_oof).all() or not np.isfinite(engineered_oof).all():
        raise ValueError("OOF prediction arrays are incomplete or nonfinite")
    if not np.isfinite(engineered_test_sum).all():
        raise ValueError("Test predictions contain nonfinite values")
    baseline_metrics = _auc_record(labels, baseline_oof, fold_ids)
    engineered_metrics = _auc_record(labels, engineered_oof, fold_ids)
    bootstrap = paired_auc_bootstrap(labels, baseline_oof, engineered_oof)

    oof = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: labels,
            "fold": fold_ids,
            "baseline_oof": baseline_oof,
            "engineered_oof": engineered_oof,
        }
    )
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: engineered_test_sum,
        }
    )
    if submission[TARGET_COLUMN].isna().any() or not submission[TARGET_COLUMN].between(
        0.0, 1.0
    ).all():
        raise ValueError("Submission TARGET must be finite and in [0, 1]")
    if not np.array_equal(submission[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy()):
        raise ValueError("Submission IDs do not match test order")

    oof_path = output_dir / "PHASE53_OOF_PREDICTIONS.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE53_SUMMARY.json"
    report_path = output_dir / "PHASE53_REPORT.md"
    oof.to_csv(oof_path, index=False)
    submission.to_csv(submission_path, index=False)

    elapsed = time.perf_counter() - started
    source_path = Path(__file__).resolve()
    summary = {
        "experiment": "Phase 53 independent scratch financial geometry",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "target_rate": float(labels.mean()),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "fold_records": fold_records,
        "baseline_metrics": baseline_metrics,
        "engineered_metrics": engineered_metrics,
        "delta_engineered_minus_baseline": {
            "pooled_auc": engineered_metrics["pooled_auc"]
            - baseline_metrics["pooled_auc"],
            "log_loss": engineered_metrics["log_loss"]
            - baseline_metrics["log_loss"],
            "folds_positive": sum(item["delta_auc"] > 0 for item in fold_records),
            "paired_oof_bootstrap": bootstrap,
        },
        "model": {
            "library": "CatBoost",
            "version": catboost.__version__,
            "device": "CPU",
            "iterations": ITERATIONS,
            "depth": DEPTH,
            "learning_rate": LEARNING_RATE,
            "thread_count": THREADS,
            "one_hot_max_size": ONE_HOT_MAX_SIZE,
            "max_ctr_complexity": 1,
            "early_stopping": False,
            "prediction_aggregation": "arithmetic mean of the five fold-model probabilities",
        },
        "feature_scope": {
            "baseline": "original supplied predictors only",
            "candidate": "same predictors plus applicant-local deterministic formulas",
            "feature_count_baseline": int(baseline_train.shape[1]),
            "feature_count_engineered": int(engineered_train.shape[1]),
            "categorical_columns": engineered_categories,
            "maximum_training_category_cardinality": cardinality,
            "target_encoding": "none; categorical inputs forced to one-hot route",
            "cross_row_features": "none",
            "test_covariate_statistics": "none",
            "external_data": "none",
            "test_labels": "none",
            "manual_prediction_edits": "none",
        },
        "input_sha256": {
            str(train_path.resolve()): sha256_file(train_path),
            str(test_path.resolve()): sha256_file(test_path),
        },
        "source_sha256": sha256_file(source_path),
        "runtime_seconds": elapsed,
        "artifacts": {
            "oof_predictions": oof_path.name,
            "submission": submission_path.name,
        },
        "caveats": [
            "Validation rows were excluded from fitting their fold models.",
            "No outer-fold early stopping or OOF-driven tuning was performed.",
            "The same labeled dataset has prior research exposure; this is exploratory, not independent confirmation.",
            "The paired bootstrap is conditional on saved OOF predictions and does not include training-fold variability.",
            "The competition materials do not fully resolve every modeling eligibility question; this run deliberately avoids pooled test statistics, label propagation, and target encoding.",
            "The submission is a research candidate, not an authorized champion or a guaranteed leaderboard score.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in (
            (oof_path.name, oof_path),
            (submission_path.name, submission_path),
            (summary_path.name, summary_path),
            (report_path.name, report_path),
            (source_path.name, source_path),
        )
    }
    (output_dir / "PHASE53_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={baseline_metrics['pooled_auc']:.9f} "
        f"engineered={engineered_metrics['pooled_auc']:.9f} "
        f"delta={engineered_metrics['pooled_auc'] - baseline_metrics['pooled_auc']:+.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)
    print(f"report: {report_path}", flush=True)


def render_report(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_metrics"]
    engineered = summary["engineered_metrics"]
    delta = summary["delta_engineered_minus_baseline"]
    lines = [
        "# Phase 53 — independent scratch financial-geometry experiment",
        "",
        "## Protocol",
        "",
        f"- Rows: {summary['training_rows']} train / {summary['test_rows']} test.",
        f"- Stratified folds: {summary['fold_count']}, seed `{summary['fold_seed']}`.",
        "- Paired control: original predictors versus the same model with deterministic, row-local features.",
        "- Fixed CatBoost iterations; no outer-fold early stopping or selection on validation labels.",
        "- No target encoding, applicant linking, cross-row aggregates, test-pool statistics, or outside files.",
        "- The test file is used only for fold-model inference and output-ID verification.",
        "",
        "## Results",
        "",
        "| Candidate | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |",
        "|---|---:|---:|---|---:|",
        f"| Raw-feature baseline | {baseline['pooled_auc']:.9f} | {baseline['mean_fold_auc']:.9f} | {', '.join(f'{x:.6f}' for x in baseline['fold_auc'])} | {baseline['log_loss']:.9f} |",
        f"| Financial-geometry features | {engineered['pooled_auc']:.9f} | {engineered['mean_fold_auc']:.9f} | {', '.join(f'{x:.6f}' for x in engineered['fold_auc'])} | {engineered['log_loss']:.9f} |",
        "",
        f"- Pooled AUC delta: `{delta['pooled_auc']:+.9f}`; positive folds: `{delta['folds_positive']}/{summary['fold_count']}`.",
        f"- Log-loss delta (candidate minus baseline): `{delta['log_loss']:+.9f}`.",
        f"- Paired OOF bootstrap interval: `[{delta['paired_oof_bootstrap']['delta_ci_2_5']:+.9f}, {delta['paired_oof_bootstrap']['delta_ci_97_5']:+.9f}]`.",
        "- Bootstrap uncertainty is conditional on these OOF vectors and is not an independent confirmation.",
        "",
        "## Interpretation",
        "",
        "The candidate is a controlled feature-family comparison, not a leaderboard promise. The formulas use only fields from the same application: amounts are converted to ratios/log scales; age and employment durations are expressed in years; and external-score geometry is represented by row-wise moments, pairwise differences, products, and ratios. No loan term or interest-rate assumptions are imposed.",
        "",
        "The five fold models' probabilities are averaged arithmetically to form the single research submission. No CSV-to-CSV rank blend is used. ROC-AUC is invariant to strictly increasing score transformations, so the submission is not rank-normalized or post-calibrated.",
        "",
        "All labels have prior project exposure. Do not describe this result as an untouched validation score or promote the output solely because it wins this comparison.",
        "",
        "## Artifact hashes and runtime",
        "",
        f"- Train SHA-256: `{summary['input_sha256'][next(iter(summary['input_sha256']))]}`.",
        f"- Source SHA-256: `{summary['source_sha256']}`.",
        f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
        "- Full paths and hashes: `PHASE53_SHA256_MANIFEST.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone, rules-conservative scratch AUC experiment"
    )
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument("--test", type=Path, default=Path(r"D:\gci\inpuy\test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "phase53_scratch_financial_geometry_20261117",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        print("self-test passed")
        return
    run(args.train, args.test, args.output_dir)


if __name__ == "__main__":
    main()
