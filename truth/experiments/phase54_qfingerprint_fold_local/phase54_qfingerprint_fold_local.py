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
ITERATIONS = 900
THREADS = 4
BOOTSTRAP_REPLICATES = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").astype(np.float64)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _rounded_key(values: pd.Series, decimals: int) -> pd.Series:
    rounded = values.round(decimals)
    return rounded.map(
        lambda value: "__MISSING__"
        if pd.isna(value)
        else format(float(value), f".{decimals}f")
    )


def build_feature_sets(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build raw-plus-rowwise features and a version with rounded product keys."""
    excluded = [name for name in (ID_COLUMN, TARGET_COLUMN) if name in raw]
    original = raw.drop(columns=excluded).copy()
    raw_categoricals = [name for name in CATEGORICAL_COLUMNS if name in original]

    for name in raw_categoricals:
        original[name] = (
            original[name]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )
    for name in original.columns:
        if name in raw_categoricals:
            continue
        original[name] = pd.to_numeric(original[name], errors="raise").astype(
            "float32"
        )

    income = _number(raw, "AMT_INCOME_TOTAL")
    credit = _number(raw, "AMT_CREDIT")
    annuity = _number(raw, "AMT_ANNUITY")
    goods = _number(raw, "AMT_GOODS_PRICE")
    family = _number(raw, "CNT_FAM_MEMBERS")
    birth = _number(raw, "DAYS_BIRTH")
    employed_raw = _number(raw, "DAYS_EMPLOYED")
    employed = employed_raw.where(employed_raw != 365243.0)

    extra: dict[str, pd.Series] = {
        "MISSING_COUNT": raw.drop(columns=excluded, errors="ignore")
        .isna()
        .sum(axis=1)
        .astype("float32"),
        "EMPLOYED_SENTINEL": (employed_raw == 365243.0).astype("float32"),
        "DAYS_EMPLOYED_CLEAN": employed,
        "AGE_YEARS": -birth / 365.25,
        "EMPLOYED_YEARS": -employed / 365.25,
        "EMPLOYED_TO_AGE": _safe_ratio(employed, birth),
        "REGISTRATION_TO_AGE": _safe_ratio(_number(raw, "DAYS_REGISTRATION"), birth),
        "ID_PUBLISH_TO_AGE": _safe_ratio(_number(raw, "DAYS_ID_PUBLISH"), birth),
        "PHONE_CHANGE_TO_AGE": _safe_ratio(
            _number(raw, "DAYS_LAST_PHONE_CHANGE"), birth
        ),
        "CREDIT_TO_INCOME": _safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": _safe_ratio(annuity, income),
        "ANNUITY_TO_CREDIT": _safe_ratio(annuity, credit),
        "CREDIT_TO_ANNUITY": _safe_ratio(credit, annuity),
        "GOODS_TO_CREDIT": _safe_ratio(goods, credit),
        "DOWNPAYMENT_AMOUNT": credit - goods,
        "DOWNPAYMENT_TO_CREDIT": _safe_ratio(credit - goods, credit),
        "GOODS_TO_INCOME": _safe_ratio(goods, income),
        "INCOME_PER_FAMILY_MEMBER": _safe_ratio(income, family),
        "CREDIT_PER_FAMILY_MEMBER": _safe_ratio(credit, family),
        "ANNUITY_PER_FAMILY_MEMBER": _safe_ratio(annuity, family),
    }

    for name, values in (
        ("INCOME", income),
        ("CREDIT", credit),
        ("ANNUITY", annuity),
        ("GOODS_PRICE", goods),
    ):
        extra[f"LOG1P_{name}"] = np.log1p(values.clip(lower=0.0))

    ext_names = [
        name
        for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if name in raw.columns
    ]
    ext = pd.concat([_number(raw, name) for name in ext_names], axis=1)
    if ext_names:
        extra["EXT_MEAN"] = ext.mean(axis=1)
        extra["EXT_MIN"] = ext.min(axis=1)
        extra["EXT_MAX"] = ext.max(axis=1)
        extra["EXT_STD"] = ext.std(axis=1)
        extra["EXT_RANGE"] = extra["EXT_MAX"] - extra["EXT_MIN"]
        extra["EXT_PRESENT_COUNT"] = ext.notna().sum(axis=1).astype("float32")
        extra["EXT_PRODUCT"] = ext.prod(axis=1, min_count=1)
        for left_index in range(len(ext_names)):
            for right_index in range(left_index + 1, len(ext_names)):
                left_name, right_name = ext_names[left_index], ext_names[right_index]
                left, right = ext.iloc[:, left_index], ext.iloc[:, right_index]
                pair = f"{left_name[-1]}{right_name[-1]}"
                extra[f"EXT_DIFF_{pair}"] = left - right
                extra[f"EXT_PRODUCT_{pair}"] = left * right

    bureau_names = [
        name for name in raw.columns if name.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if bureau_names:
        bureau = pd.concat([_number(raw, name) for name in bureau_names], axis=1)
        extra["BUREAU_REQUEST_SUM"] = bureau.sum(axis=1, min_count=1)
        extra["BUREAU_REQUEST_NONZERO_COUNT"] = (bureau > 0).sum(axis=1).astype(
            "float32"
        )
    for name in (
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "OWN_CAR_AGE",
    ):
        if name in raw.columns:
            extra[f"{name}_MISSING"] = raw[name].isna().astype("float32")

    numeric_features = pd.DataFrame(extra, index=raw.index)
    base = pd.concat(
        [original.reset_index(drop=True), numeric_features.reset_index(drop=True)],
        axis=1,
    )
    base_categorical = list(raw_categoricals)
    if base.columns.duplicated().any():
        raise ValueError("Base features contain duplicate columns")

    for name in base.columns:
        if name in base_categorical:
            continue
        values = pd.to_numeric(base[name], errors="coerce")
        base[name] = values.replace([np.inf, -np.inf], np.nan).astype("float32")

    candidate = base.copy()
    annuity_to_credit = _safe_ratio(annuity, credit)
    product_key_columns = []
    for decimals in (3, 4, 5):
        name = f"ANN_CREDIT_KEY_{decimals}"
        candidate[name] = _rounded_key(annuity_to_credit, decimals).to_numpy()
        product_key_columns.append(name)

    candidate_categorical = base_categorical + product_key_columns
    if candidate.columns.duplicated().any():
        raise ValueError("Candidate features contain duplicate columns")
    return base, candidate, base_categorical, candidate_categorical


def encode_fold_categories(
    fit: pd.DataFrame,
    other_frames: list[pd.DataFrame],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, int]]:
    """Encode categories using only observed levels in this model's fit rows."""
    encoded_fit = fit.copy()
    encoded_others = [frame.copy() for frame in other_frames]
    cardinalities: dict[str, int] = {}
    for column in categorical_columns:
        fit_values = (
            encoded_fit[column].astype("string").fillna("__MISSING__").astype(str)
        )
        levels = pd.unique(fit_values)
        mapping = {value: index for index, value in enumerate(levels)}
        cardinalities[column] = len(mapping)
        encoded_fit[column] = fit_values.map(mapping).astype("int32")
        for frame in encoded_others:
            values = frame[column].astype("string").fillna("__MISSING__").astype(str)
            frame[column] = values.map(mapping).fillna(-1).astype("int32")
    return encoded_fit, encoded_others, cardinalities


def make_model(seed: int, iterations: int = ITERATIONS) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=150,
        colsample_bytree=0.5,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=20.0,
        max_bin=127,
        max_cat_threshold=32,
        max_cat_to_onehot=4,
        min_data_per_group=100,
        cat_smooth=20.0,
        cat_l2=20.0,
        n_jobs=THREADS,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def auc_metrics(
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


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
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
        deltas[index] = roc_auc_score(
            labels[rows], candidate[rows]
        ) - roc_auc_score(labels[rows], baseline[rows])
    return {
        "replicates": repetitions,
        "delta_mean": float(deltas.mean()),
        "delta_ci_2_5": float(np.quantile(deltas, 0.025)),
        "delta_ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "interpretation": (
            "Conditional on the OOF vectors; excludes model-fit uncertainty, "
            "prior exposure to these labels, and leaderboard sampling."
        ),
    }


def run_self_tests() -> None:
    rng = np.random.default_rng(411)
    rows = 120
    labels = np.tile([0, 1], rows // 2).astype(np.int8)
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(rows),
            "TARGET": labels,
            "NAME_CONTRACT_TYPE": np.where(np.arange(rows) % 2, "Cash loans", "Revolving loans"),
            "CODE_GENDER": np.where(np.arange(rows) % 3, "F", "M"),
            "NAME_INCOME_TYPE": np.where(np.arange(rows) % 4, "Working", "Pensioner"),
            "NAME_EDUCATION_TYPE": np.where(np.arange(rows) % 3, "Higher", "Secondary"),
            "NAME_FAMILY_STATUS": np.where(np.arange(rows) % 2, "Married", "Single"),
            "NAME_HOUSING_TYPE": np.where(np.arange(rows) % 2, "House", "Rented"),
            "OCCUPATION_TYPE": np.where(np.arange(rows) % 4, "Laborers", "Managers"),
            "ORGANIZATION_TYPE": np.where(np.arange(rows) % 5, "Business", "School"),
            "AMT_INCOME_TOTAL": rng.uniform(10000, 200000, rows),
            "AMT_CREDIT": rng.uniform(10000, 500000, rows),
            "AMT_ANNUITY": rng.uniform(1000, 50000, rows),
            "AMT_GOODS_PRICE": rng.uniform(10000, 400000, rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, rows),
            "DAYS_BIRTH": -rng.integers(7000, 25000, rows),
            "DAYS_EMPLOYED": -rng.integers(1, 10000, rows),
            "EXT_SOURCE_1": rng.uniform(0, 1, rows),
            "EXT_SOURCE_2": rng.uniform(0, 1, rows),
            "EXT_SOURCE_3": rng.uniform(0, 1, rows),
        }
    )
    raw.loc[1, "EXT_SOURCE_1"] = np.nan
    base, candidate, base_cats, candidate_cats = build_feature_sets(raw)
    if ID_COLUMN in candidate or TARGET_COLUMN in candidate:
        raise AssertionError("ID or target was included in features")
    if candidate_cats != base_cats + [
        "ANN_CREDIT_KEY_3",
        "ANN_CREDIT_KEY_4",
        "ANN_CREDIT_KEY_5",
    ]:
        raise AssertionError("Product-key categorical list is incorrect")
    alone = build_feature_sets(raw.iloc[[0]].copy())[1]
    if not alone.iloc[0].equals(candidate.iloc[0]):
        raise AssertionError("Feature values depend on other applicants")

    fit_rows, valid_rows = np.arange(90), np.arange(90, 120)
    fit, others, _ = encode_fold_categories(
        candidate.iloc[fit_rows],
        [candidate.iloc[valid_rows]],
        candidate_cats,
    )
    if any((fit[column] < 0).any() for column in candidate_cats):
        raise AssertionError("Training categories must be nonnegative")
    quick = make_model(SEED, iterations=5)
    quick.fit(
        fit,
        labels[fit_rows],
        categorical_feature=candidate_cats,
    )
    prediction = quick.predict_proba(others[0])[:, 1]
    if not np.isfinite(prediction).all():
        raise AssertionError("LightGBM categorical smoke prediction is nonfinite")


def run(train_path: Path, test_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Both --train and --test must be existing CSV files")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if ID_COLUMN not in train or TARGET_COLUMN not in train or ID_COLUMN not in test:
        raise ValueError("Train/test files are missing the required ID or target")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        raise ValueError("Train and test IDs must each be unique")
    if set(train.columns) - {TARGET_COLUMN} != set(test.columns):
        raise ValueError("Train predictors and test columns do not match")
    if list(train.drop(columns=[TARGET_COLUMN]).columns) != list(test.columns):
        raise ValueError("Train and test columns must have the same order")

    labels = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary labels 0 and 1")
    raw_train = train.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    baseline_train, candidate_train, baseline_cats, candidate_cats = (
        build_feature_sets(raw_train)
    )
    baseline_test, candidate_test, test_base_cats, test_candidate_cats = (
        build_feature_sets(raw_test)
    )
    if baseline_cats != test_base_cats or candidate_cats != test_candidate_cats:
        raise ValueError("Categorical feature lists differ across train and test")
    if not baseline_train.columns.equals(baseline_test.columns):
        raise ValueError("Baseline train/test feature schemas differ")
    if not candidate_train.columns.equals(candidate_test.columns):
        raise ValueError("Candidate train/test feature schemas differ")

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number

    baseline_oof = np.full(len(train), np.nan, dtype=np.float64)
    candidate_oof = np.full(len(train), np.nan, dtype=np.float64)
    candidate_test_prediction = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=False)

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        b_fit, b_others, _ = encode_fold_categories(
            baseline_train.iloc[fit_rows],
            [
                baseline_train.iloc[valid_rows],
                baseline_test,
            ],
            baseline_cats,
        )
        c_fit, c_others, category_cardinalities = encode_fold_categories(
            candidate_train.iloc[fit_rows],
            [
                candidate_train.iloc[valid_rows],
                candidate_test,
            ],
            candidate_cats,
        )
        seed = SEED + fold_number

        baseline_model = make_model(seed)
        baseline_model.fit(
            b_fit,
            labels[fit_rows],
            categorical_feature=baseline_cats,
        )
        baseline_prediction = baseline_model.predict_proba(b_others[0])[:, 1]
        baseline_oof[valid_rows] = baseline_prediction
        del baseline_model

        candidate_model = make_model(seed)
        candidate_model.fit(
            c_fit,
            labels[fit_rows],
            categorical_feature=candidate_cats,
        )
        candidate_prediction = candidate_model.predict_proba(c_others[0])[:, 1]
        candidate_test_prediction += (
            candidate_model.predict_proba(c_others[1])[:, 1] / FOLDS
        )
        candidate_oof[valid_rows] = candidate_prediction
        del candidate_model

        base_auc = float(roc_auc_score(labels[valid_rows], baseline_prediction))
        candidate_auc = float(
            roc_auc_score(labels[valid_rows], candidate_prediction)
        )
        fold_records.append(
            {
                "fold": fold_number,
                "train_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "baseline_auc": base_auc,
                "candidate_auc": candidate_auc,
                "delta_auc": candidate_auc - base_auc,
                "iterations": ITERATIONS,
                "candidate_category_cardinalities_fit_only": category_cardinalities,
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: baseline={base_auc:.6f} "
            f"candidate={candidate_auc:.6f} delta={candidate_auc - base_auc:+.6f}",
            flush=True,
        )

    if not np.isfinite(baseline_oof).all() or not np.isfinite(candidate_oof).all():
        raise ValueError("OOF predictions are incomplete or nonfinite")
    if not np.isfinite(candidate_test_prediction).all():
        raise ValueError("Test predictions contain nonfinite values")
    baseline_metrics = auc_metrics(labels, baseline_oof, fold_ids)
    candidate_metrics = auc_metrics(labels, candidate_oof, fold_ids)
    bootstrap = paired_bootstrap(labels, baseline_oof, candidate_oof)

    oof_path = output_dir / "PHASE54_OOF_PREDICTIONS.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE54_SUMMARY.json"
    report_path = output_dir / "PHASE54_REPORT.md"
    oof = pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: labels,
            "fold": fold_ids,
            "baseline_oof": baseline_oof,
            "candidate_oof": candidate_oof,
        }
    )
    submission = pd.DataFrame(
        {
            ID_COLUMN: test[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: candidate_test_prediction,
        }
    )
    if (
        submission[ID_COLUMN].isna().any()
        or not submission[ID_COLUMN].is_unique
        or not np.array_equal(
            submission[ID_COLUMN].to_numpy(), test[ID_COLUMN].to_numpy()
        )
    ):
        raise ValueError("Submission IDs must match test IDs in original order")
    if not np.isfinite(submission[TARGET_COLUMN]).all() or not submission[
        TARGET_COLUMN
    ].between(0.0, 1.0).all():
        raise ValueError("Submission predictions must be finite values in [0, 1]")
    oof.to_csv(oof_path, index=False)
    submission.to_csv(submission_path, index=False)

    runtime = time.perf_counter() - started
    source_path = Path(__file__).resolve()
    summary = {
        "experiment": "Phase 54 fold-local q-fingerprint categories",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "fold_records": fold_records,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "delta_candidate_minus_baseline": {
            "pooled_auc": candidate_metrics["pooled_auc"]
            - baseline_metrics["pooled_auc"],
            "log_loss": candidate_metrics["log_loss"]
            - baseline_metrics["log_loss"],
            "positive_folds": sum(record["delta_auc"] > 0 for record in fold_records),
            "paired_oof_bootstrap": bootstrap,
        },
        "model": {
            "library": "LightGBM",
            "version": lgb.__version__,
            "device": "CPU",
            "iterations": ITERATIONS,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 150,
            "thread_count": THREADS,
            "early_stopping": False,
            "prediction_aggregation": "arithmetic mean of the five fold-model probabilities",
        },
        "feature_scope": {
            "baseline": "supplied predictors plus deterministic row-local numeric formulas",
            "candidate": "baseline plus q=AMT_ANNUITY/AMT_CREDIT rounded to 3, 4, and 5 decimal category keys",
            "baseline_feature_count": int(baseline_train.shape[1]),
            "candidate_feature_count": int(candidate_train.shape[1]),
            "target_encoding": "none",
            "cross_row_statistics": "none",
            "train_test_pooling": "none",
            "test_rows_used_for": "fold-model inference only",
            "external_data": "none",
            "manual_prediction_edits": "none",
            "excluded_fee_correction": "no 129.5 or other empirical fee constants",
            "category_code_fitting": "separate maps fit only on each model's training fold; unseen values map to missing",
        },
        "input_sha256": {
            str(train_path.resolve()): sha256_file(train_path),
            str(test_path.resolve()): sha256_file(test_path),
        },
        "source_sha256": sha256_file(source_path),
        "runtime_seconds": runtime,
        "artifacts": {
            "oof_predictions": oof_path.name,
            "submission": submission_path.name,
        },
        "caveats": [
            "The same labeled dataset has extensive prior research exposure; this result is exploratory.",
            "The paired bootstrap is conditional on saved OOF vectors and excludes model-fit and research-selection uncertainty.",
            "The reported q-fingerprint source's highest fold is not its pooled OOF AUC; the local audit found pooled notebook OOF claims around 0.77070 for its best member.",
            "This experiment tests only rounded ratio categories, not the source's target encodings, cross-row counts, train/test pooled statistics, or fee correction.",
            "The submission is an experiment output, not an authorized champion or a guaranteed leaderboard score.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (oof_path, submission_path, summary_path, report_path, source_path)
    }
    (output_dir / "PHASE54_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={baseline_metrics['pooled_auc']:.9f} "
        f"candidate={candidate_metrics['pooled_auc']:.9f} "
        f"delta={candidate_metrics['pooled_auc'] - baseline_metrics['pooled_auc']:+.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)


def render_report(summary: dict[str, Any]) -> str:
    baseline = summary["baseline_metrics"]
    candidate = summary["candidate_metrics"]
    delta = summary["delta_candidate_minus_baseline"]
    return "\n".join(
        [
            "# Phase 54 — fold-local q-fingerprint experiment",
            "",
            "## Hypothesis",
            "",
            "The annuity/credit ratio may encode discrete loan-product schedules. A tree can use repeated rounded ratio values as categorical predictors, while a continuous ratio alone encourages monotone threshold splits. This run tests that representation without decoding fees or transferring labels between applicants.",
            "",
            "## Protocol",
            "",
            f"- Train/test rows: {summary['training_rows']} / {summary['test_rows']}.",
            f"- Five stratified folds, seed `{summary['fold_seed']}`; fixed `{summary['model']['iterations']}` rounds and no outer-fold early stopping.",
            "- Paired control and candidate use the same LightGBM settings and folds.",
            "- The only candidate-specific features are row-local `annuity / credit` keys rounded to 3, 4, and 5 decimals.",
            "- Category dictionaries are fit on each training fold only; unseen validation/test keys are missing.",
            "- No custom target encoding, group/value counts, applicant matching, train/test pooling, fee constants, cached predictions, or rank blending.",
            "",
            "## Results",
            "",
            "| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |",
            "|---|---:|---:|---|---:|",
            f"| Baseline | {baseline['pooled_auc']:.9f} | {baseline['mean_fold_auc']:.9f} | {', '.join(f'{v:.6f}' for v in baseline['fold_auc'])} | {baseline['log_loss']:.9f} |",
            f"| q-fingerprint candidate | {candidate['pooled_auc']:.9f} | {candidate['mean_fold_auc']:.9f} | {', '.join(f'{v:.6f}' for v in candidate['fold_auc'])} | {candidate['log_loss']:.9f} |",
            "",
            f"- Paired AUC delta: `{delta['pooled_auc']:+.9f}`; positive folds `{delta['positive_folds']}/5`.",
            f"- Paired OOF bootstrap interval: `[{delta['paired_oof_bootstrap']['delta_ci_2_5']:+.9f}, {delta['paired_oof_bootstrap']['delta_ci_97_5']:+.9f}]` (conditional on these predictions).",
            "",
            "## Interpretation",
            "",
            "This is the clean test of whether quantized ratio categories add value beyond row-local numeric finance features in this learner. A positive OOF delta would support the representation, not the source notebook's target-encoded or pooled-statistic pipeline. A negative/neutral delta rejects these q-key encodings for this setup. All labels have prior project exposure, so this is not an untouched confirmation or a leaderboard score.",
            "",
            f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
            f"- Source SHA-256: `{summary['source_sha256']}`.",
            "- Full artifact hashes: `PHASE54_SHA256_MANIFEST.json`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone fold-local q-fingerprint AUC experiment"
    )
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument("--test", type=Path, default=Path(r"D:\gci\inpuy\test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "phase54_qfingerprint_20261117",
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
