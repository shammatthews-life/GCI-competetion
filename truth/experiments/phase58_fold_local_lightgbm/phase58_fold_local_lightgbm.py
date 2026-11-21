#!/usr/bin/env python
"""Fold-local LightGBM ablation with OOF confusion and ranking diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
ITERATIONS = 1400
BOOTSTRAPS = 1000
TOP_FRACTION = 0.10
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_TEST = Path(r"D:\gci\inpuy\test.csv")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "runs"
    / "phase58_fold_local_lightgbm_20261117"
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
        raise ValueError(f"Missing ID column: {ID}")
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
            features[column]
            .astype("object")
            .where(features[column].notna(), "__MISSING__")
            .astype(str)
        )

    if include_rowwise:
        def values(column: str) -> np.ndarray:
            if column not in raw:
                raise ValueError(f"Missing feature source: {column}")
            return pd.to_numeric(raw[column], errors="raise").to_numpy(
                dtype=np.float64, na_value=np.nan
            )

        derived: dict[str, np.ndarray] = {}

        def add(name: str, array: np.ndarray) -> None:
            if name in features.columns:
                raise ValueError(f"Derived name collides with input feature: {name}")
            derived[name] = np.asarray(array, dtype=np.float32)

        credit = values("AMT_CREDIT")
        annuity = values("AMT_ANNUITY")
        goods = values("AMT_GOODS_PRICE")
        income = values("AMT_INCOME_TOTAL")
        family = values("CNT_FAM_MEMBERS")
        birth = values("DAYS_BIRTH")
        employed = values("DAYS_EMPLOYED")
        registration = values("DAYS_REGISTRATION")
        id_published = values("DAYS_ID_PUBLISH")
        phone_changed = values("DAYS_LAST_PHONE_CHANGE")

        add("S58_PAYMENT_INCOME", safe_ratio(annuity, income))
        add("S58_CREDIT_INCOME", safe_ratio(credit, income))
        add("S58_GOODS_CREDIT", safe_ratio(goods, credit))
        add("S58_FINANCED_SHARE", safe_ratio(credit - goods, credit))
        add("S58_CREDIT_ANNUITY", safe_ratio(credit, annuity))
        add("S58_INCOME_PER_MEMBER", safe_ratio(income, family))
        add("S58_CREDIT_PER_MEMBER", safe_ratio(credit, family))
        add("S58_AGE_YEARS", -birth / 365.25)
        add("S58_EMPLOYMENT_SENTINEL", (employed == 365243).astype(np.float32))

        valid_employment = np.isfinite(employed) & (employed < 0)
        employment_years = np.full(len(raw), np.nan, dtype=np.float64)
        employment_years[valid_employment] = -employed[valid_employment] / 365.25
        add("S58_EMPLOYMENT_YEARS", employment_years)
        add(
            "S58_EMPLOYMENT_AGE_SHARE",
            safe_ratio(employment_years, -birth / 365.25),
        )
        add(
            "S58_REGISTRATION_AGE_SHARE",
            safe_ratio(registration - birth, -birth),
        )
        add(
            "S58_ID_PUBLICATION_AGE_SHARE",
            safe_ratio(id_published, birth),
        )
        add(
            "S58_PHONE_CHANGE_AGE_SHARE",
            safe_ratio(phone_changed, birth),
        )

        ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        missing_ext = [column for column in ext_cols if column not in raw]
        if missing_ext:
            raise ValueError(f"Missing EXT_SOURCE columns: {missing_ext}")
        ext = raw[ext_cols].apply(pd.to_numeric, errors="raise")
        ext_array = ext.to_numpy(dtype=np.float64, na_value=np.nan)
        add("S58_EXT_COUNT", np.isfinite(ext_array).sum(axis=1))
        ext_mean = ext.mean(axis=1)
        ext_min = ext.min(axis=1)
        ext_max = ext.max(axis=1)
        add("S58_EXT_MEAN", ext_mean.to_numpy())
        add("S58_EXT_STD", ext.std(axis=1, ddof=0).to_numpy())
        add("S58_EXT_MIN", ext_min.to_numpy())
        add("S58_EXT_MAX", ext_max.to_numpy())
        add("S58_EXT_RANGE", (ext_max - ext_min).to_numpy())
        add("S58_EXT12_DIFF", (ext[ext_cols[0]] - ext[ext_cols[1]]).to_numpy())
        add("S58_EXT13_DIFF", (ext[ext_cols[0]] - ext[ext_cols[2]]).to_numpy())
        add("S58_EXT23_DIFF", (ext[ext_cols[1]] - ext[ext_cols[2]]).to_numpy())
        add("S58_EXT12_PRODUCT", (ext[ext_cols[0]] * ext[ext_cols[1]]).to_numpy())
        add("S58_EXT13_PRODUCT", (ext[ext_cols[0]] * ext[ext_cols[2]]).to_numpy())
        add("S58_EXT23_PRODUCT", (ext[ext_cols[1]] * ext[ext_cols[2]]).to_numpy())

        bureau_cols = [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        missing_bureau = [column for column in bureau_cols if column not in raw]
        if missing_bureau:
            raise ValueError(f"Missing bureau inquiry columns: {missing_bureau}")
        bureau = raw[bureau_cols].apply(pd.to_numeric, errors="raise")
        add("S58_BUREAU_TOTAL", bureau.sum(axis=1, min_count=1).to_numpy())
        add("S58_BUREAU_COUNT", bureau.notna().sum(axis=1).to_numpy())
        add(
            "S58_BUREAU_MON_YEAR_SUM",
            (bureau[bureau_cols[1]] + bureau[bureau_cols[3]]).to_numpy(),
        )
        add(
            "S58_BUREAU_MON_YEAR_RATIO",
            safe_ratio(values(bureau_cols[1]), values(bureau_cols[3]) + 1.0),
        )

        q3 = safe_ratio(annuity, credit)
        add("S58_ANNUITY_CREDIT", q3)
        key = np.full(len(raw), "__MISSING__", dtype=object)
        valid = np.isfinite(q3)
        key[valid] = np.char.mod("%.3f", q3[valid])
        derived["S58_Q3_KEY"] = key
        categorical.append("S58_Q3_KEY")
        add(
            "S58_RAW_MISSING_COUNT",
            raw.select_dtypes(include=[np.number]).isna().sum(axis=1).to_numpy(),
        )
        features = pd.concat(
            [features, pd.DataFrame(derived, index=raw.index)], axis=1
        )

    features = features.replace([np.inf, -np.inf], np.nan)
    numeric = [column for column in features.columns if column not in categorical]
    features[numeric] = features[numeric].astype(np.float32)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature columns")
    if features.isna().all(axis=0).any():
        raise ValueError("At least one feature is entirely missing")
    return features, categorical


def encode_fold(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
) -> tuple[csr_matrix, csr_matrix, csr_matrix, int]:
    numeric = [column for column in fit.columns if column not in categorical]
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    fit_cat = encoder.fit_transform(fit[categorical])
    validation_cat = encoder.transform(validation[categorical])
    test_cat = encoder.transform(test[categorical])
    fit_matrix = hstack(
        [csr_matrix(fit[numeric].to_numpy(dtype=np.float32)), fit_cat],
        format="csr",
        dtype=np.float32,
    )
    validation_matrix = hstack(
        [csr_matrix(validation[numeric].to_numpy(dtype=np.float32)), validation_cat],
        format="csr",
        dtype=np.float32,
    )
    test_matrix = hstack(
        [csr_matrix(test[numeric].to_numpy(dtype=np.float32)), test_cat],
        format="csr",
        dtype=np.float32,
    )
    if fit_matrix.shape[1] != validation_matrix.shape[1] or fit_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError("Fold one-hot feature dimensions are inconsistent")
    return fit_matrix, validation_matrix, test_matrix, int(fit_matrix.shape[1])


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=ITERATIONS,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=120,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=8.0,
        n_jobs=4,
        random_state=seed,
        verbosity=-1,
    )


def confusion_metrics(
    labels: np.ndarray,
    predicted_positive: np.ndarray,
) -> dict[str, float | int]:
    tp = int(np.sum((labels == 1) & predicted_positive))
    fp = int(np.sum((labels == 0) & predicted_positive))
    fn = int(np.sum((labels == 1) & ~predicted_positive))
    tn = int(np.sum((labels == 0) & ~predicted_positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "predicted_positive_rate": float(predicted_positive.mean()),
    }


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(SEED + 58)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        sample = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(
            labels[sample], candidate[sample]
        ) - roc_auc_score(labels[sample], baseline[sample])
    return {
        "replicates": BOOTSTRAPS,
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "conditional_on_fixed_oof_vectors": True,
    }


def ranking_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    return {
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, predictions)),
        "log_loss": float(log_loss(labels, predictions)),
        "fold_auc": [
            float(
                roc_auc_score(
                    labels[fold_ids == fold], predictions[fold_ids == fold]
                )
            )
            for fold in range(1, FOLDS + 1)
        ],
    }


def run_self_tests() -> None:
    n = 100
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
    raw_features, raw_cats = build_features(raw, include_rowwise=False)
    rich_features, rich_cats = build_features(raw, include_rowwise=True)
    if ID in raw_features or TARGET in rich_features:
        raise AssertionError("ID/target leaked into features")
    fit_x, valid_x, test_x, _ = encode_fold(
        rich_features.iloc[:60],
        rich_features.iloc[60:80],
        rich_features.iloc[80:],
        rich_cats,
    )
    if fit_x.shape[1] != valid_x.shape[1] or fit_x.shape[1] != test_x.shape[1]:
        raise AssertionError("One-hot fold matrices are not aligned")
    labels = np.tile([0, 1], 10)
    diagnostics = confusion_metrics(labels, np.zeros(len(labels), dtype=bool))
    if diagnostics["recall"] != 0.0 or diagnostics["tn"] + diagnostics["fn"] != len(labels):
        raise AssertionError("Confusion diagnostics are inconsistent")
    if len(rich_features.columns) <= len(raw_features.columns) or len(raw_cats) == 0:
        raise AssertionError("Candidate features or categorical handling missing")
    print("Phase 58 feature, fold-encoding, and confusion self-tests passed.")


def render_report(summary: dict[str, Any]) -> str:
    base = summary["metrics"]["baseline"]
    candidate = summary["metrics"]["candidate"]
    delta = summary["candidate_vs_baseline"]
    lines = [
        "# Phase 58 — fold-local LightGBM and rowwise features",
        "",
        "## Protocol",
        "",
        f"- Five stratified folds, seed `{summary['fold_seed']}`; fixed `{summary['iterations']}` rounds.",
        "- One-hot category dictionaries are fit on each outer training fold only; unseen validation/test categories map to all-zero indicators.",
        "- Baseline uses raw supplied predictors; candidate adds deterministic row-local ratios, score summaries, time ratios, bureau summaries, and Q3.",
        "- No target encoding, cross-row aggregates, test pooling, entity matching, external inputs, or CSV blending.",
        "",
        "## Ranking results",
        "",
        "| Model | Pooled AUC | Average precision | Log loss | Fold AUCs |",
        "|---|---:|---:|---:|---|",
        f"| Raw one-hot baseline | {base['roc_auc']:.9f} | {base['average_precision']:.9f} | {base['log_loss']:.9f} | {', '.join(f'{v:.6f}' for v in base['fold_auc'])} |",
        f"| Rowwise + Q3 candidate | {candidate['roc_auc']:.9f} | {candidate['average_precision']:.9f} | {candidate['log_loss']:.9f} | {', '.join(f'{v:.6f}' for v in candidate['fold_auc'])} |",
        "",
        f"- Candidate minus baseline AUC: `{delta['roc_auc']:+.9f}`; positive folds `{delta['positive_folds']}/5`.",
        f"- Paired conditional bootstrap interval: `[{delta['bootstrap']['ci_2_5']:+.9f}, {delta['bootstrap']['ci_97_5']:+.9f}]`.",
        "",
        "## OOF confusion and threshold diagnostics",
        "",
        "Confusion counts are `[TN, FP, FN, TP]`. Precision/recall are evaluated only on OOF rows.",
        "",
        "| Model | Threshold rule | TN | FP | FN | TP | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("baseline", "candidate"):
        for rule_name, stats in summary["classification"][model_name].items():
            lines.append(
                f"| {model_name} | {rule_name} | {stats['tn']} | {stats['fp']} | {stats['fn']} | {stats['tp']} | {stats['precision']:.6f} | {stats['recall']:.6f} |"
            )
    lines.extend(
        [
            "",
            "These threshold metrics diagnose operating points and class imbalance; they do not replace ROC-AUC. The top-decile rule measures how many positives are captured among the highest-scored 10% of each fold, without selecting a threshold from validation labels.",
            "",
            "## Limitations",
            "",
            "All labels/folds have prior project exposure. The bootstrap is conditional on fixed OOF predictions and does not include model/research-selection uncertainty. The public leaderboard score is unknown until evaluated.",
            "",
            f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
            f"- Source SHA-256: `{summary['source_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(train_path: Path, test_path: Path, output_dir: Path) -> None:
    started = time.perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Official train.csv and test.csv are required")

    train_hash = sha256_file(train_path)
    test_hash = sha256_file(test_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train or ID not in train or ID not in test:
        raise ValueError("Required ID/target columns are missing")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("IDs must be unique")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("Train and test predictor schemas differ")
    labels = train[TARGET].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary labels 0 and 1")
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)

    baseline_train, baseline_cats = build_features(raw_train, False)
    baseline_test, baseline_test_cats = build_features(raw_test, False)
    candidate_train, candidate_cats = build_features(raw_train, True)
    candidate_test, candidate_test_cats = build_features(raw_test, True)
    if baseline_cats != baseline_test_cats or candidate_cats != candidate_test_cats:
        raise ValueError("Train/test categorical feature lists differ")
    if not baseline_train.columns.equals(baseline_test.columns):
        raise ValueError("Baseline feature schemas differ")
    if not candidate_train.columns.equals(candidate_test.columns):
        raise ValueError("Candidate feature schemas differ")

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for number, (_, valid) in enumerate(folds, start=1):
        fold_ids[valid] = number
    if np.any(fold_ids < 1):
        raise ValueError("Fold assignment incomplete")

    oof = {
        "baseline": np.full(len(labels), np.nan, dtype=np.float64),
        "candidate": np.full(len(labels), np.nan, dtype=np.float64),
    }
    test_candidate = np.zeros(len(test), dtype=np.float64)
    decisions = {
        name: {
            "fixed_threshold_0.5": np.zeros(len(labels), dtype=bool),
            "outer_train_prevalence": np.zeros(len(labels), dtype=bool),
            "top_10_percent_per_fold": np.zeros(len(labels), dtype=bool),
        }
        for name in oof
    }
    fold_records: list[dict[str, Any]] = []
    encoded_widths: list[dict[str, int]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number
        train_prevalence = float(labels[fit_rows].mean())
        arm_aucs: dict[str, float] = {}
        arm_predictions: dict[str, np.ndarray] = {}
        for arm, x_train, x_test, categorical in (
            ("baseline", baseline_train, baseline_test, baseline_cats),
            ("candidate", candidate_train, candidate_test, candidate_cats),
        ):
            x_fit, x_valid, x_test_fold, width = encode_fold(
                x_train.iloc[fit_rows],
                x_train.iloc[valid_rows],
                x_test,
                categorical,
            )
            encoded_widths.append(
                {"fold": fold_number, "arm": arm, "columns": width}
            )
            model = make_model(fold_seed)
            model.fit(x_fit, labels[fit_rows])
            valid_prediction = model.predict_proba(x_valid)[:, 1]
            if not np.isfinite(valid_prediction).all():
                raise ValueError(f"Nonfinite OOF predictions in {arm}, fold {fold_number}")
            oof[arm][valid_rows] = valid_prediction
            arm_predictions[arm] = valid_prediction
            arm_aucs[arm] = float(
                roc_auc_score(labels[valid_rows], valid_prediction)
            )
            if arm == "candidate":
                test_candidate += model.predict_proba(x_test_fold)[:, 1] / FOLDS
            decisions[arm]["fixed_threshold_0.5"][valid_rows] = (
                valid_prediction >= 0.5
            )
            decisions[arm]["outer_train_prevalence"][valid_rows] = (
                valid_prediction >= train_prevalence
            )
            top_count = int(np.ceil(TOP_FRACTION * len(valid_rows)))
            top_local = np.argsort(valid_prediction, kind="mergesort")[-top_count:]
            top_global = valid_rows[top_local]
            decisions[arm]["top_10_percent_per_fold"][top_global] = True
            del model

        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "train_prevalence": train_prevalence,
                "baseline_auc": arm_aucs["baseline"],
                "candidate_auc": arm_aucs["candidate"],
                "delta_auc": arm_aucs["candidate"] - arm_aucs["baseline"],
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: "
            f"baseline={arm_aucs['baseline']:.6f} "
            f"candidate={arm_aucs['candidate']:.6f}",
            flush=True,
        )

    if any(not np.isfinite(prediction).all() for prediction in oof.values()):
        raise ValueError("Incomplete OOF predictions")
    if not np.isfinite(test_candidate).all() or not np.all(
        (test_candidate >= 0.0) & (test_candidate <= 1.0)
    ):
        raise ValueError("Invalid test probabilities")
    metrics = {
        arm: ranking_metrics(labels, prediction, fold_ids)
        for arm, prediction in oof.items()
    }
    candidate_delta = {
        "roc_auc": metrics["candidate"]["roc_auc"] - metrics["baseline"]["roc_auc"],
        "average_precision": metrics["candidate"]["average_precision"]
        - metrics["baseline"]["average_precision"],
        "log_loss": metrics["candidate"]["log_loss"] - metrics["baseline"]["log_loss"],
        "positive_folds": sum(record["delta_auc"] > 0 for record in fold_records),
        "bootstrap": paired_bootstrap(
            labels, oof["baseline"], oof["candidate"]
        ),
    }
    classification = {
        arm: {
            rule: confusion_metrics(labels, predicted)
            for rule, predicted in arm_decisions.items()
        }
        for arm, arm_decisions in decisions.items()
    }

    if sha256_file(train_path) != train_hash or sha256_file(test_path) != test_hash:
        raise ValueError("Input file changed during the experiment")
    output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = output_dir / "PHASE58_OOF.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE58_SUMMARY.json"
    report_path = output_dir / "PHASE58_REPORT.md"

    pd.DataFrame(
        {
            ID: train[ID].to_numpy(),
            TARGET: labels,
            "fold": fold_ids,
            "baseline_oof": oof["baseline"],
            "candidate_oof": oof["candidate"],
        }
    ).to_csv(oof_path, index=False)
    submission = pd.DataFrame(
        {ID: test[ID].to_numpy(), TARGET: test_candidate}
    )
    if not submission[ID].equals(test[ID]) or not submission[ID].is_unique:
        raise ValueError("Submission ID order mismatch")
    submission.to_csv(submission_path, index=False)

    source_path = Path(__file__).resolve()
    summary: dict[str, Any] = {
        "experiment": "Phase 58 fold-local LightGBM ablation",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "iterations": ITERATIONS,
        "top_fraction": TOP_FRACTION,
        "baseline_features": int(baseline_train.shape[1]),
        "candidate_features": int(candidate_train.shape[1]),
        "onehot_widths": encoded_widths,
        "metrics": metrics,
        "candidate_vs_baseline": candidate_delta,
        "fold_records": fold_records,
        "classification": classification,
        "positive_prevalence": float(labels.mean()),
        "feature_scope": {
            "baseline": "official raw predictors; categorical one-hot vocabularies fit inside each outer training fold",
            "candidate": "baseline plus deterministic row-local finance, timeline, EXT_SOURCE, bureau, missingness, and Q3 features",
            "custom_target_encoding": "none",
            "cross_row_aggregation": "none",
            "train_test_pooling": "none",
            "external_data_or_prior_predictions": "none",
            "ID_as_predictor": "no",
            "thresholds": "0.5; outer-training-fold prevalence; and per-fold top 10% score rank, all diagnostic only",
        },
        "input_sha256": {
            str(train_path.resolve()): train_hash,
            str(test_path.resolve()): test_hash,
        },
        "source_sha256": sha256_file(source_path),
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "oof_predictions": oof_path.name,
            "submission": submission_path.name,
        },
        "limitations": [
            "All training labels and folds have prior project exposure.",
            "Confusion/precision/recall depend on a threshold; ROC-AUC is the competition metric.",
            "The conditional bootstrap does not include model or research-selection uncertainty.",
            "No leaderboard score is claimed or guaranteed.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (oof_path, submission_path, summary_path, report_path, source_path)
    }
    (output_dir / "PHASE58_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={metrics['baseline']['roc_auc']:.9f} "
        f"candidate={metrics['candidate']['roc_auc']:.9f} "
        f"delta={candidate_delta['roc_auc']:+.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    run(args.train, args.test, args.output_dir)


if __name__ == "__main__":
    main()
