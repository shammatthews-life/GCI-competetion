#!/usr/bin/env python
"""Fold-isolated Home Credit CatBoost ablation using only official CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
ITERATIONS = 1000
BOOTSTRAPS = 1000
ONE_HOT_MAX_SIZE = 256
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_TEST = Path(r"D:\gci\inpuy\test.csv")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "runs"
    / "phase57_rowwise_catboost_20261117"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result.astype(np.float32)


def build_features(raw: pd.DataFrame, include_rowwise: bool) -> tuple[pd.DataFrame, list[str]]:
    if ID not in raw:
        raise ValueError(f"Missing required ID column: {ID}")

    features = raw.drop(columns=[ID]).copy()
    categorical = [
        column
        for column in features.columns
        if pd.api.types.is_object_dtype(features[column])
        or isinstance(features[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(features[column])
    ]
    for column in categorical:
        features[column] = (
            features[column].astype("object").where(features[column].notna(), "__MISSING__").astype(str)
        )

    if include_rowwise:
        def numeric(column: str) -> np.ndarray:
            if column not in raw:
                raise ValueError(f"Required feature source is missing: {column}")
            return pd.to_numeric(raw[column], errors="raise").to_numpy(
                dtype=np.float64, na_value=np.nan
            )

        added: dict[str, np.ndarray] = {}

        def add(name: str, values: np.ndarray) -> None:
            if name in features.columns:
                raise ValueError(f"Derived feature collides with existing column: {name}")
            added[name] = np.asarray(values, dtype=np.float32)

        credit = numeric("AMT_CREDIT")
        annuity = numeric("AMT_ANNUITY")
        goods = numeric("AMT_GOODS_PRICE")
        income = numeric("AMT_INCOME_TOTAL")
        family = numeric("CNT_FAM_MEMBERS")
        birth_days = numeric("DAYS_BIRTH")
        employed_days = numeric("DAYS_EMPLOYED")
        registration_days = numeric("DAYS_REGISTRATION")
        id_publish_days = numeric("DAYS_ID_PUBLISH")
        phone_change_days = numeric("DAYS_LAST_PHONE_CHANGE")

        add("R57_PAYMENT_TO_INCOME", safe_ratio(annuity, income))
        add("R57_CREDIT_TO_INCOME", safe_ratio(credit, income))
        add("R57_GOODS_TO_CREDIT", safe_ratio(goods, credit))
        add("R57_FINANCED_GAP_TO_CREDIT", safe_ratio(credit - goods, credit))
        add("R57_CREDIT_TO_ANNUITY", safe_ratio(credit, annuity))
        add("R57_INCOME_PER_PERSON", safe_ratio(income, family))
        add("R57_CREDIT_PER_PERSON", safe_ratio(credit, family))
        add("R57_AGE_YEARS", -birth_days / 365.25)
        add("R57_EMPLOYMENT_SENTINEL", (employed_days == 365243).astype(np.float32))

        valid_employment = np.isfinite(employed_days) & (employed_days < 0)
        employment_years = np.full(len(raw), np.nan, dtype=np.float64)
        employment_years[valid_employment] = -employed_days[valid_employment] / 365.25
        add("R57_EMPLOYMENT_YEARS_VALID", employment_years)
        add(
            "R57_EMPLOYMENT_TO_AGE",
            safe_ratio(employment_years, -birth_days / 365.25),
        )
        add(
            "R57_REGISTRATION_TO_AGE",
            safe_ratio(registration_days - birth_days, -birth_days),
        )
        add(
            "R57_ID_PUBLICATION_TO_AGE",
            safe_ratio(id_publish_days, birth_days),
        )
        add(
            "R57_PHONE_CHANGE_TO_AGE",
            safe_ratio(phone_change_days, birth_days),
        )

        ext_columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        missing_ext = [column for column in ext_columns if column not in raw]
        if missing_ext:
            raise ValueError(f"Missing external-score columns: {missing_ext}")
        ext = raw[ext_columns].apply(pd.to_numeric, errors="raise")
        ext_values = ext.to_numpy(dtype=np.float64, na_value=np.nan)
        add("R57_EXT_OBSERVED_COUNT", np.isfinite(ext_values).sum(axis=1))
        add("R57_EXT_MEAN", ext.mean(axis=1).to_numpy())
        add("R57_EXT_STD", ext.std(axis=1, ddof=0).to_numpy())
        add("R57_EXT_MIN", ext.min(axis=1).to_numpy())
        add("R57_EXT_MAX", ext.max(axis=1).to_numpy())
        add("R57_EXT_RANGE", (ext.max(axis=1) - ext.min(axis=1)).to_numpy())
        add("R57_EXT_1_MINUS_2", (ext[ext_columns[0]] - ext[ext_columns[1]]).to_numpy())
        add("R57_EXT_1_MINUS_3", (ext[ext_columns[0]] - ext[ext_columns[2]]).to_numpy())
        add("R57_EXT_2_MINUS_3", (ext[ext_columns[1]] - ext[ext_columns[2]]).to_numpy())
        add("R57_EXT_1_TIMES_2", (ext[ext_columns[0]] * ext[ext_columns[1]]).to_numpy())
        add("R57_EXT_1_TIMES_3", (ext[ext_columns[0]] * ext[ext_columns[2]]).to_numpy())
        add("R57_EXT_2_TIMES_3", (ext[ext_columns[1]] * ext[ext_columns[2]]).to_numpy())

        bureau_columns = [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        missing_bureau = [column for column in bureau_columns if column not in raw]
        if missing_bureau:
            raise ValueError(f"Missing bureau inquiry columns: {missing_bureau}")
        bureau = raw[bureau_columns].apply(pd.to_numeric, errors="raise")
        add("R57_BUREAU_TOTAL", bureau.sum(axis=1, min_count=1).to_numpy())
        add("R57_BUREAU_OBSERVED_COUNT", bureau.notna().sum(axis=1).to_numpy())
        add(
            "R57_BUREAU_MONTH_PLUS_YEAR",
            (bureau[bureau_columns[1]] + bureau[bureau_columns[3]]).to_numpy(),
        )
        add(
            "R57_BUREAU_MONTH_TO_YEAR",
            safe_ratio(
                numeric(bureau_columns[1]),
                numeric(bureau_columns[3]) + 1.0,
            ),
        )

        q3 = safe_ratio(annuity, credit)
        add("R57_ANNUITY_CREDIT_RATIO", q3)
        q3_key = np.full(len(raw), "__MISSING__", dtype=object)
        valid_q3 = np.isfinite(q3)
        q3_key[valid_q3] = np.char.mod("%.3f", q3[valid_q3])
        added["R57_Q3_KEY"] = q3_key
        categorical.append("R57_Q3_KEY")

        numeric_raw = raw.select_dtypes(include=[np.number])
        add("R57_RAW_MISSING_COUNT", numeric_raw.isna().sum(axis=1).to_numpy())
        features = pd.concat(
            [features, pd.DataFrame(added, index=raw.index)],
            axis=1,
        )

    features = features.replace([np.inf, -np.inf], np.nan)
    numeric_columns = [
        column for column in features.columns if column not in categorical
    ]
    features[numeric_columns] = features[numeric_columns].astype(np.float32)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature names were generated")
    if features.isna().all(axis=0).any():
        empty_columns = features.columns[features.isna().all(axis=0)].tolist()
        raise ValueError(f"Entirely missing feature columns: {empty_columns}")
    return features, categorical


def make_model(seed: int, device: str) -> CatBoostClassifier:
    params: dict[str, Any] = {
        "iterations": ITERATIONS,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "l2_leaf_reg": 8.0,
        "random_strength": 1.0,
        "one_hot_max_size": ONE_HOT_MAX_SIZE,
        "max_ctr_complexity": 1,
        "border_count": 128,
        "random_seed": seed,
        "task_type": device,
        "thread_count": 6,
        "verbose": False,
        "allow_writing_files": False,
    }
    if device == "GPU":
        params["devices"] = "0"
    return CatBoostClassifier(**params)


def stratified_paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        sample = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(
            labels[sample], candidate[sample]
        ) - roc_auc_score(labels[sample], baseline[sample])
    return {
        "replicates": BOOTSTRAPS,
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "conditional_on_fixed_oof_vectors": True,
    }


def model_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "pooled_auc": float(roc_auc_score(labels, prediction)),
        "mean_fold_auc": float(
            np.mean(
                [
                    roc_auc_score(labels[fold_ids == fold], prediction[fold_ids == fold])
                    for fold in range(1, FOLDS + 1)
                ]
            )
        ),
        "fold_auc": [
            float(
                roc_auc_score(
                    labels[fold_ids == fold], prediction[fold_ids == fold]
                )
            )
            for fold in range(1, FOLDS + 1)
        ],
        "log_loss": float(log_loss(labels, prediction)),
    }


def run_self_tests() -> None:
    n = 80
    raw = pd.DataFrame(
        {
            ID: np.arange(n),
            "NAME_CONTRACT_TYPE": np.where(np.arange(n) % 2, "Cash loans", "Revolving loans"),
            "CODE_GENDER": np.where(np.arange(n) % 3, "F", "M"),
            "AMT_CREDIT": np.full(n, 100000.0),
            "AMT_ANNUITY": np.where(np.arange(n) % 2, 10000.0, 12000.0),
            "AMT_GOODS_PRICE": np.full(n, 90000.0),
            "AMT_INCOME_TOTAL": np.full(n, 50000.0),
            "CNT_FAM_MEMBERS": np.full(n, 2.0),
            "DAYS_BIRTH": -np.linspace(8000, 22000, n),
            "DAYS_EMPLOYED": np.where(np.arange(n) % 7, -1000.0, 365243.0),
            "DAYS_REGISTRATION": -np.linspace(500, 5000, n),
            "DAYS_ID_PUBLISH": -np.linspace(500, 8000, n),
            "DAYS_LAST_PHONE_CHANGE": -np.linspace(100, 3000, n),
            "EXT_SOURCE_1": np.linspace(0.1, 0.9, n),
            "EXT_SOURCE_2": np.linspace(0.2, 0.8, n),
            "EXT_SOURCE_3": np.where(np.arange(n) % 5, 0.6, np.nan),
            "AMT_REQ_CREDIT_BUREAU_HOUR": np.zeros(n),
            "AMT_REQ_CREDIT_BUREAU_MON": np.full(n, 2.0),
            "AMT_REQ_CREDIT_BUREAU_QRT": np.ones(n),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.full(n, 3.0),
        }
    )
    baseline, baseline_cats = build_features(raw, include_rowwise=False)
    candidate, candidate_cats = build_features(raw, include_rowwise=True)
    if ID in baseline or TARGET in baseline or ID in candidate or TARGET in candidate:
        raise AssertionError("ID/target leaked into model features")
    if "R57_Q3_KEY" not in candidate_cats:
        raise AssertionError("Q3 key must be treated as categorical")
    if candidate.loc[0, "R57_Q3_KEY"] == candidate.loc[1, "R57_Q3_KEY"]:
        raise AssertionError("Synthetic Q3 key should distinguish the test rows")
    if len(candidate.columns) <= len(baseline.columns):
        raise AssertionError("Rowwise candidate did not add features")
    print("Phase 57 feature self-test passed.")


def render_report(summary: dict[str, Any]) -> str:
    base = summary["metrics"]["baseline"]
    candidate = summary["metrics"]["candidate"]
    delta = summary["candidate_vs_baseline"]
    if delta["pooled_auc"] > 0 and delta["positive_folds"] >= 4:
        decision = "The candidate improved pooled AUC and was positive in at least four folds; retain it only as an exploratory feature set."
    elif delta["pooled_auc"] > 0:
        decision = "The pooled gain was not consistent across folds; do not promote the candidate based on this run."
    else:
        decision = "The candidate did not improve pooled AUC; reject these added features for now."
    return "\n".join(
        [
            "# Phase 57 — row-local feature CatBoost ablation",
            "",
            "## Research question",
            "",
            "Does a small, predeclared set of row-local financial, timing,",
            "external-score, inquiry, and coarse Q3 features improve a matched",
            "one-hot CatBoost baseline without cross-row or target-derived",
            "feature construction?",
            "",
            "## Protocol",
            "",
            f"- {summary['fold_count']} stratified folds; seed `{summary['fold_seed']}`.",
            f"- CatBoost `{summary['device']}`, `{summary['iterations']}` fixed iterations; no outer-fold early stopping.",
            f"- CatBoost one-hot encodes every categorical column (checked cardinality <= `{ONE_HOT_MAX_SIZE}`); no categorical target statistics are needed.",
            "- Baseline uses raw supplied predictor columns; candidate adds only deterministic row-local formulas.",
            "- All transforms use the current row only. Each fold model sees labels from its training rows only.",
            "- No test/train pooling, explicit target encoding, entity matching, prediction caches, external data, or CSV blending.",
            "",
            "## Results",
            "",
            "| Arm | Pooled AUC | Mean fold AUC | Log loss | Fold AUCs |",
            "|---|---:|---:|---:|---|",
            f"| Raw one-hot baseline | {base['pooled_auc']:.9f} | {base['mean_fold_auc']:.9f} | {base['log_loss']:.9f} | {', '.join(f'{v:.6f}' for v in base['fold_auc'])} |",
            f"| Rowwise + Q3 candidate | {candidate['pooled_auc']:.9f} | {candidate['mean_fold_auc']:.9f} | {candidate['log_loss']:.9f} | {', '.join(f'{v:.6f}' for v in candidate['fold_auc'])} |",
            "",
            f"- Candidate minus baseline pooled AUC: `{delta['pooled_auc']:+.9f}`; positive folds `{delta['positive_folds']}/{summary['fold_count']}`.",
            f"- Log-loss change (candidate minus baseline): `{delta['log_loss']:+.9f}`.",
            f"- Conditional paired-bootstrap interval: `[{delta['bootstrap']['ci_2_5']:+.9f}, {delta['bootstrap']['ci_97_5']:+.9f}]`.",
            "",
            "## Interpretation",
            "",
            decision,
            "Treat this as exploratory: the project labels and folds have prior exposure, and the bootstrap is conditional on fixed OOF predictions. The leaderboard score is unknown until a submission is evaluated; public-score feedback is not a substitute for untouched validation.",
            "",
            f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
            f"- Runner SHA-256: `{summary['source_sha256']}`.",
            "",
        ]
    )


def run(train_path: Path, test_path: Path, output_dir: Path, device: str) -> None:
    started = time.perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Both official train.csv and test.csv must exist")

    train_hash_before = sha256_file(train_path)
    test_hash_before = sha256_file(test_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train or ID not in train or ID not in test:
        raise ValueError("Input files do not contain the required ID/target columns")
    if train[ID].isna().any() or test[ID].isna().any():
        raise ValueError("IDs must not be missing")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("IDs must be unique")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("Train predictor columns and test columns/order differ")

    labels = train[TARGET].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary values 0 and 1")
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    baseline_train, baseline_cats = build_features(raw_train, include_rowwise=False)
    baseline_test, baseline_test_cats = build_features(raw_test, include_rowwise=False)
    candidate_train, candidate_cats = build_features(raw_train, include_rowwise=True)
    candidate_test, candidate_test_cats = build_features(raw_test, include_rowwise=True)
    if baseline_cats != baseline_test_cats or candidate_cats != candidate_test_cats:
        raise ValueError("Train/test categorical schemas differ")
    if not baseline_train.columns.equals(baseline_test.columns):
        raise ValueError("Baseline train/test feature schemas differ")
    if not candidate_train.columns.equals(candidate_test.columns):
        raise ValueError("Candidate train/test feature schemas differ")
    category_cardinality = {
        column: int(candidate_train[column].nunique(dropna=False))
        for column in candidate_cats
    }
    over_limit = {
        column: count
        for column, count in category_cardinality.items()
        if count > ONE_HOT_MAX_SIZE
    }
    if over_limit:
        raise ValueError(
            "A category exceeds the one-hot threshold; refusing target-statistic fallback: "
            f"{over_limit}"
        )

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if np.any(fold_ids < 1):
        raise ValueError("OOF fold assignment is incomplete")

    oof_baseline = np.full(len(labels), np.nan, dtype=np.float64)
    oof_candidate = np.full(len(labels), np.nan, dtype=np.float64)
    test_candidate = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number
        baseline_model = make_model(fold_seed, device)
        baseline_model.fit(
            baseline_train.iloc[fit_rows],
            labels[fit_rows],
            cat_features=baseline_cats,
        )
        baseline_prediction = baseline_model.predict_proba(
            baseline_train.iloc[valid_rows]
        )[:, 1]
        oof_baseline[valid_rows] = baseline_prediction
        del baseline_model

        candidate_model = make_model(fold_seed, device)
        candidate_model.fit(
            candidate_train.iloc[fit_rows],
            labels[fit_rows],
            cat_features=candidate_cats,
        )
        candidate_prediction = candidate_model.predict_proba(
            candidate_train.iloc[valid_rows]
        )[:, 1]
        oof_candidate[valid_rows] = candidate_prediction
        test_candidate += (
            candidate_model.predict_proba(candidate_test)[:, 1] / FOLDS
        )
        del candidate_model

        baseline_auc = float(roc_auc_score(labels[valid_rows], baseline_prediction))
        candidate_auc = float(roc_auc_score(labels[valid_rows], candidate_prediction))
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "baseline_auc": baseline_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": candidate_auc - baseline_auc,
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: "
            f"baseline={baseline_auc:.6f} candidate={candidate_auc:.6f}",
            flush=True,
        )

    if not np.isfinite(oof_baseline).all() or not np.isfinite(oof_candidate).all():
        raise ValueError("OOF predictions are incomplete or nonfinite")
    if not np.isfinite(test_candidate).all() or not np.all(
        (test_candidate >= 0.0) & (test_candidate <= 1.0)
    ):
        raise ValueError("Test predictions must be finite probabilities")

    metrics = {
        "baseline": model_metrics(labels, oof_baseline, fold_ids),
        "candidate": model_metrics(labels, oof_candidate, fold_ids),
    }
    candidate_delta = {
        "pooled_auc": metrics["candidate"]["pooled_auc"]
        - metrics["baseline"]["pooled_auc"],
        "log_loss": metrics["candidate"]["log_loss"]
        - metrics["baseline"]["log_loss"],
        "positive_folds": sum(record["delta_auc"] > 0 for record in fold_records),
        "bootstrap": stratified_paired_bootstrap(
            labels, oof_baseline, oof_candidate, SEED + 57
        ),
    }

    if sha256_file(train_path) != train_hash_before or sha256_file(test_path) != test_hash_before:
        raise ValueError("An input CSV changed during the experiment")
    output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = output_dir / "PHASE57_OOF.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE57_SUMMARY.json"
    report_path = output_dir / "PHASE57_REPORT.md"

    pd.DataFrame(
        {
            ID: train[ID].to_numpy(),
            TARGET: labels,
            "fold": fold_ids,
            "baseline_oof": oof_baseline,
            "candidate_oof": oof_candidate,
        }
    ).to_csv(oof_path, index=False)
    submission = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: test_candidate})
    if not submission[ID].equals(test[ID]) or not submission[ID].is_unique:
        raise ValueError("Submission IDs do not match test order")
    submission.to_csv(submission_path, index=False)

    source_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "experiment": "Phase 57 row-local CatBoost feature ablation",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "iterations": ITERATIONS,
        "one_hot_max_size": ONE_HOT_MAX_SIZE,
        "category_cardinality": category_cardinality,
        "device": device,
        "baseline_feature_count": int(baseline_train.shape[1]),
        "candidate_feature_count": int(candidate_train.shape[1]),
        "candidate_added_feature_names": [
            column
            for column in candidate_train.columns
            if column not in baseline_train.columns
        ],
        "fold_records": fold_records,
        "metrics": metrics,
        "candidate_vs_baseline": candidate_delta,
        "feature_scope": {
            "baseline": "official row predictors, with raw categorical fields passed to CatBoost",
            "candidate": "baseline plus deterministic row-local financial, timing, EXT_SOURCE, inquiry, missingness, and Q3 features",
            "cross_row_features": "none",
            "custom_target_encoding": "none",
            "train_test_pooling": "none",
            "external_data_or_prediction_cache": "none",
            "ID_as_predictor": "no",
            "categorical_target_statistics": "disabled by one-hot threshold; all training-fold category cardinalities are checked against the threshold",
        },
        "input_sha256": {
            str(train_path.resolve()): train_hash_before,
            str(test_path.resolve()): test_hash_before,
        },
        "source_sha256": sha256_file(source_path),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "oof_predictions": oof_path.name,
            "submission": submission_path.name,
        },
        "limitations": [
            "All labeled rows and these folds have prior project exposure.",
            "Bootstrap interval is conditional on saved OOF vectors and does not include model or research-selection uncertainty.",
            "Leaderboard score is unknown until evaluated; no score is guaranteed.",
            "The public/private leaderboard split and target-encoding eligibility are not resolved by local materials.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (oof_path, submission_path, summary_path, report_path, source_path)
    }
    (output_dir / "PHASE57_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={metrics['baseline']['pooled_auc']:.9f} "
        f"candidate={metrics['candidate']['pooled_auc']:.9f} "
        f"delta={candidate_delta['pooled_auc']:+.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["GPU", "CPU"], default="GPU")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    run(args.train, args.test, args.output_dir, args.device)


if __name__ == "__main__":
    main()
