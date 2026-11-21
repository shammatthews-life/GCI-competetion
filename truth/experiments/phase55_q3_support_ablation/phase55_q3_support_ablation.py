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


ID = "SK_ID_CURR"
TARGET = "TARGET"
CATS = (
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
BOOTSTRAPS = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def num(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").astype(np.float64)


def ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.where(b != 0.0)
    return result.replace([np.inf, -np.inf], np.nan)


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded = [c for c in (ID, TARGET) if c in raw]
    base = raw.drop(columns=excluded).copy()
    cats = [c for c in CATS if c in base]
    for column in cats:
        base[column] = (
            base[column].astype("string").fillna("__MISSING__").astype(str)
        )
    for column in base.columns:
        if column not in cats:
            base[column] = pd.to_numeric(base[column], errors="raise").astype(
                "float32"
            )

    income = num(raw, "AMT_INCOME_TOTAL")
    credit = num(raw, "AMT_CREDIT")
    annuity = num(raw, "AMT_ANNUITY")
    goods = num(raw, "AMT_GOODS_PRICE")
    family = num(raw, "CNT_FAM_MEMBERS")
    birth = num(raw, "DAYS_BIRTH")
    employed_raw = num(raw, "DAYS_EMPLOYED")
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
        "EMPLOYED_TO_AGE": ratio(employed, birth),
        "REGISTRATION_TO_AGE": ratio(num(raw, "DAYS_REGISTRATION"), birth),
        "ID_PUBLISH_TO_AGE": ratio(num(raw, "DAYS_ID_PUBLISH"), birth),
        "PHONE_CHANGE_TO_AGE": ratio(num(raw, "DAYS_LAST_PHONE_CHANGE"), birth),
        "CREDIT_TO_INCOME": ratio(credit, income),
        "ANNUITY_TO_INCOME": ratio(annuity, income),
        "ANNUITY_TO_CREDIT": ratio(annuity, credit),
        "CREDIT_TO_ANNUITY": ratio(credit, annuity),
        "GOODS_TO_CREDIT": ratio(goods, credit),
        "DOWNPAYMENT_AMOUNT": credit - goods,
        "DOWNPAYMENT_TO_CREDIT": ratio(credit - goods, credit),
        "GOODS_TO_INCOME": ratio(goods, income),
        "INCOME_PER_FAMILY_MEMBER": ratio(income, family),
        "CREDIT_PER_FAMILY_MEMBER": ratio(credit, family),
        "ANNUITY_PER_FAMILY_MEMBER": ratio(annuity, family),
    }
    for name, values in (
        ("INCOME", income),
        ("CREDIT", credit),
        ("ANNUITY", annuity),
        ("GOODS_PRICE", goods),
    ):
        extra[f"LOG1P_{name}"] = np.log1p(values.clip(lower=0.0))

    ext_names = [
        c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in raw
    ]
    ext = pd.concat([num(raw, c) for c in ext_names], axis=1)
    if ext_names:
        extra["EXT_MEAN"] = ext.mean(axis=1)
        extra["EXT_MIN"] = ext.min(axis=1)
        extra["EXT_MAX"] = ext.max(axis=1)
        extra["EXT_STD"] = ext.std(axis=1)
        extra["EXT_RANGE"] = extra["EXT_MAX"] - extra["EXT_MIN"]
        extra["EXT_PRESENT_COUNT"] = ext.notna().sum(axis=1).astype("float32")
        extra["EXT_PRODUCT"] = ext.prod(axis=1, min_count=1)
        for i in range(len(ext_names)):
            for j in range(i + 1, len(ext_names)):
                a, b = ext.iloc[:, i], ext.iloc[:, j]
                pair = f"{ext_names[i][-1]}{ext_names[j][-1]}"
                extra[f"EXT_DIFF_{pair}"] = a - b
                extra[f"EXT_PRODUCT_{pair}"] = a * b

    bureau_names = [
        c for c in raw.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if bureau_names:
        bureau = pd.concat([num(raw, c) for c in bureau_names], axis=1)
        extra["BUREAU_SUM"] = bureau.sum(axis=1, min_count=1)
        extra["BUREAU_NONZERO_COUNT"] = (bureau > 0).sum(axis=1).astype("float32")
    for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE"):
        if name in raw:
            extra[f"{name}_MISSING"] = raw[name].isna().astype("float32")

    engineered = pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(extra).reset_index(drop=True)],
        axis=1,
    )
    if engineered.columns.duplicated().any():
        raise ValueError("Duplicate engineered feature names")
    for column in engineered.columns:
        if column in cats:
            continue
        values = pd.to_numeric(engineered[column], errors="coerce")
        engineered[column] = values.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )
    q_key = ratio(annuity, credit).round(3).map(
        lambda value: "__MISSING__"
        if pd.isna(value)
        else format(float(value), ".3f")
    )
    return engineered, q_key.reset_index(drop=True), cats


def encode_categories(
    fit: pd.DataFrame,
    other: list[pd.DataFrame],
    categorical: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    fit_out = fit.copy()
    other_out = [frame.copy() for frame in other]
    for column in categorical:
        values = (
            fit_out[column].astype("string").fillna("__MISSING__").astype(str)
        )
        mapping = {value: index for index, value in enumerate(pd.unique(values))}
        fit_out[column] = values.map(mapping).astype("int32")
        for frame in other_out:
            mapped = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
                .map(mapping)
                .fillna(-1)
            )
            frame[column] = mapped.astype("int32")
    return fit_out, other_out


def support_counts(
    fit_keys: pd.Series, query_keys: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-out fit support and fit-only support for query rows."""
    counts = fit_keys.value_counts(dropna=False)
    fit_support = fit_keys.map(counts).to_numpy(dtype=np.float64) - 1.0
    query_support = query_keys.map(counts).fillna(0).to_numpy(dtype=np.float64)
    return np.log1p(np.maximum(fit_support, 0.0)), np.log1p(query_support)


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


def metrics(
    y: np.ndarray, pred: np.ndarray, fold_ids: np.ndarray
) -> dict[str, Any]:
    fold_auc = [
        float(roc_auc_score(y[fold_ids == f], pred[fold_ids == f]))
        for f in sorted(np.unique(fold_ids))
    ]
    return {
        "pooled_auc": float(roc_auc_score(y, pred)),
        "fold_auc": fold_auc,
        "mean_fold_auc": float(np.mean(fold_auc)),
        "std_fold_auc": float(np.std(fold_auc)),
        "log_loss": float(log_loss(y, pred)),
    }


def paired_bootstrap(
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED + seed_offset)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        rows = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(
            y[rows], candidate[rows]
        ) - roc_auc_score(y[rows], reference[rows])
    return {
        "replicates": BOOTSTRAPS,
        "delta_mean": float(deltas.mean()),
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "conditional_only": True,
    }


def run_self_tests() -> None:
    rng = np.random.default_rng(5501)
    n = 120
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": np.tile([0, 1], n // 2),
            "NAME_CONTRACT_TYPE": np.where(np.arange(n) % 2, "Cash loans", "Revolving loans"),
            "CODE_GENDER": np.where(np.arange(n) % 3, "F", "M"),
            "AMT_INCOME_TOTAL": rng.uniform(10000, 200000, n),
            "AMT_CREDIT": np.full(n, 100000.0),
            "AMT_ANNUITY": np.where(np.arange(n) % 2, 10000.0, 12000.0),
            "AMT_GOODS_PRICE": rng.uniform(10000, 90000, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n),
            "DAYS_BIRTH": -rng.integers(7000, 25000, n),
            "DAYS_EMPLOYED": -rng.integers(1, 10000, n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
        }
    )
    base, keys, cats = build_features(raw)
    if ID in base or TARGET in base:
        raise AssertionError("ID/target leaked into features")
    if not (keys.iloc[::2] == "0.120").all() or not (
        keys.iloc[1::2] == "0.100"
    ).all():
        raise AssertionError("Synthetic q keys are not constructed as expected")
    fit, query = keys.iloc[:80], keys.iloc[80:]
    fit_log, query_log = support_counts(fit, query)
    if fit_log[0] != np.log1p((fit == fit.iloc[0]).sum() - 1):
        raise AssertionError("Fit support counts must exclude each row itself")
    if query_log[0] != np.log1p((fit == query.iloc[0]).sum()):
        raise AssertionError("Query support must use fit rows only")

    fit_x, others = encode_categories(
        base.iloc[:90],
        [base.iloc[90:]],
        cats,
    )
    quick = make_model(SEED, iterations=5)
    quick.fit(fit_x, raw[TARGET].to_numpy()[:90], categorical_feature=cats)
    if not np.isfinite(quick.predict_proba(others[0])[:, 1]).all():
        raise AssertionError("Synthetic LightGBM prediction is not finite")


def run(train_path: Path, test_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Both --train and --test must be existing CSV files")

    started = time.perf_counter()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if ID not in train or TARGET not in train or ID not in test:
        raise ValueError("Input schema is missing the required ID or target")
    if not train[ID].is_unique or not test[ID].is_unique:
        raise ValueError("Train and test IDs must each be unique")
    if set(train.columns) - {TARGET} != set(test.columns):
        raise ValueError("Train predictors and test columns do not match")
    if list(train.drop(columns=[TARGET]).columns) != list(test.columns):
        raise ValueError("Train and test columns must have the same order")

    y = train[TARGET].to_numpy(dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary labels 0 and 1")
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    x_train, q_train, cats = build_features(raw_train)
    x_test, q_test, test_cats = build_features(raw_test)
    if cats != test_cats or not x_train.columns.equals(x_test.columns):
        raise ValueError("Train/test feature schemas differ")

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_id = np.full(len(y), -1, dtype=np.int8)
    for number, (_, valid) in enumerate(folds, start=1):
        fold_id[valid] = number
    if np.any(fold_id < 1):
        raise ValueError("Every training row must have exactly one OOF fold")

    oof = {
        "baseline": np.full(len(y), np.nan, dtype=np.float64),
        "q3_only": np.full(len(y), np.nan, dtype=np.float64),
        "q3_support": np.full(len(y), np.nan, dtype=np.float64),
    }
    test_support_prediction = np.zeros(len(test), dtype=np.float64)
    fold_records = []
    output_dir.mkdir(parents=True, exist_ok=False)

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fit_keys = q_train.iloc[fit_rows].reset_index(drop=True)
        valid_keys = q_train.iloc[valid_rows].reset_index(drop=True)
        test_keys = q_test.reset_index(drop=True)
        fit_support, valid_support = support_counts(fit_keys, valid_keys)
        _, test_support = support_counts(fit_keys, test_keys)

        base_fit, base_others = encode_categories(
            x_train.iloc[fit_rows],
            [x_train.iloc[valid_rows]],
            cats,
        )
        q_fit = x_train.iloc[fit_rows].copy().reset_index(drop=True)
        q_valid = x_train.iloc[valid_rows].copy().reset_index(drop=True)
        q_test_frame = x_test.copy().reset_index(drop=True)
        q_fit["ANN_CREDIT_KEY_3"] = fit_keys.to_numpy()
        q_valid["ANN_CREDIT_KEY_3"] = valid_keys.to_numpy()
        q_test_frame["ANN_CREDIT_KEY_3"] = test_keys.to_numpy()
        q_fit["ANN_CREDIT_LOG_SUPPORT"] = fit_support.astype("float32")
        q_valid["ANN_CREDIT_LOG_SUPPORT"] = valid_support.astype("float32")
        q_test_frame["ANN_CREDIT_LOG_SUPPORT"] = test_support.astype("float32")
        q_categories = cats + ["ANN_CREDIT_KEY_3"]
        q_fit_encoded, q_others = encode_categories(
            q_fit,
            [q_valid, q_test_frame],
            q_categories,
        )

        seed = SEED + fold_number
        baseline_model = make_model(seed)
        baseline_model.fit(
            base_fit,
            y[fit_rows],
            categorical_feature=cats,
        )
        oof["baseline"][valid_rows] = baseline_model.predict_proba(
            base_others[0]
        )[:, 1]
        del baseline_model

        q3_model = make_model(seed)
        q3_model.fit(
            q_fit_encoded.drop(columns=["ANN_CREDIT_LOG_SUPPORT"]),
            y[fit_rows],
            categorical_feature=q_categories,
        )
        oof["q3_only"][valid_rows] = q3_model.predict_proba(
            q_others[0].drop(columns=["ANN_CREDIT_LOG_SUPPORT"])
        )[:, 1]
        del q3_model

        support_model = make_model(seed)
        support_model.fit(
            q_fit_encoded,
            y[fit_rows],
            categorical_feature=q_categories,
        )
        support_prediction = support_model.predict_proba(q_others[0])[:, 1]
        oof["q3_support"][valid_rows] = support_prediction
        test_support_prediction += (
            support_model.predict_proba(q_others[1])[:, 1] / FOLDS
        )
        del support_model

        aucs = {
            name: float(roc_auc_score(y[valid_rows], values[valid_rows]))
            for name, values in oof.items()
        }
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "baseline_auc": aucs["baseline"],
                "q3_only_auc": aucs["q3_only"],
                "q3_support_auc": aucs["q3_support"],
                "q3_only_delta": aucs["q3_only"] - aucs["baseline"],
                "q3_support_delta": aucs["q3_support"] - aucs["baseline"],
                "q3_train_category_count": int(fit_keys.nunique()),
                "q3_fit_support_median": float(np.median(np.expm1(fit_support))),
                "q3_fit_rows_with_singleton_support": int((fit_support == 0).sum()),
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: "
            f"base={aucs['baseline']:.6f} q3={aucs['q3_only']:.6f} "
            f"q3+support={aucs['q3_support']:.6f}",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("OOF predictions are incomplete or nonfinite")
    if not np.isfinite(test_support_prediction).all():
        raise ValueError("Test predictions are nonfinite")
    metrics_by_model = {
        name: metrics(y, prediction, fold_id) for name, prediction in oof.items()
    }
    boot_q3 = paired_bootstrap(y, oof["baseline"], oof["q3_only"], 155)
    boot_support = paired_bootstrap(y, oof["baseline"], oof["q3_support"], 156)

    oof_path = output_dir / "PHASE55_OOF_PREDICTIONS.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE55_SUMMARY.json"
    report_path = output_dir / "PHASE55_REPORT.md"
    oof_frame = pd.DataFrame(
        {
            ID: train[ID].to_numpy(),
            TARGET: y,
            "fold": fold_id,
            "baseline_oof": oof["baseline"],
            "q3_only_oof": oof["q3_only"],
            "q3_support_oof": oof["q3_support"],
        }
    )
    submission = pd.DataFrame(
        {ID: test[ID].to_numpy(), TARGET: test_support_prediction}
    )
    if not submission[ID].equals(test[ID]) or not submission[ID].is_unique:
        raise ValueError("Submission IDs do not match test order")
    if not np.isfinite(submission[TARGET]).all() or not submission[TARGET].between(
        0.0, 1.0
    ).all():
        raise ValueError("Submission predictions must be finite values in [0, 1]")
    oof_frame.to_csv(oof_path, index=False)
    submission.to_csv(submission_path, index=False)

    runtime = time.perf_counter() - started
    source_path = Path(__file__).resolve()
    summary = {
        "experiment": "Phase 55 q3 support-count ablation",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "fold_records": fold_records,
        "metrics": metrics_by_model,
        "deltas": {
            "q3_only_vs_baseline": {
                "pooled_auc": metrics_by_model["q3_only"]["pooled_auc"]
                - metrics_by_model["baseline"]["pooled_auc"],
                "log_loss": metrics_by_model["q3_only"]["log_loss"]
                - metrics_by_model["baseline"]["log_loss"],
                "positive_folds": sum(record["q3_only_delta"] > 0 for record in fold_records),
                "paired_bootstrap": boot_q3,
            },
            "q3_support_vs_baseline": {
                "pooled_auc": metrics_by_model["q3_support"]["pooled_auc"]
                - metrics_by_model["baseline"]["pooled_auc"],
                "log_loss": metrics_by_model["q3_support"]["log_loss"]
                - metrics_by_model["baseline"]["log_loss"],
                "positive_folds": sum(record["q3_support_delta"] > 0 for record in fold_records),
                "paired_bootstrap": boot_support,
            },
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
        },
        "feature_scope": {
            "baseline": "supplied predictors plus deterministic row-local numeric formulas",
            "q3_only": "baseline plus q=annuity/credit rounded to 3 decimals as a categorical predictor",
            "q3_support": "q3-only plus log1p number of matching keys in the model-fit fold",
            "fit_support": "leave-one-out count for fit rows; validation/test support counts use only fit rows",
            "target_encoding": "none",
            "external_data": "none",
            "train_test_pooling": "none",
            "cross_row_key_matching": "none; counts are used as scalar model inputs and never transfer labels",
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
            "All labels have prior project exposure; this is exploratory.",
            "The same five folds have been used in prior research; the bootstrap is conditional on OOF predictions.",
            "Q3 support is a training-fold-only, target-free frequency feature; no test rows contribute to its counts.",
            "Submission is generated from the predeclared q3+support candidate, not selected by public leaderboard feedback.",
            "The output is not an organizer-certified champion or a guaranteed leaderboard score.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (oof_path, submission_path, summary_path, report_path, source_path)
    }
    (output_dir / "PHASE55_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={metrics_by_model['baseline']['pooled_auc']:.9f} "
        f"q3={metrics_by_model['q3_only']['pooled_auc']:.9f} "
        f"q3+support={metrics_by_model['q3_support']['pooled_auc']:.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)


def render_report(summary: dict[str, Any]) -> str:
    m = summary["metrics"]
    d = summary["deltas"]
    lines = [
        "# Phase 55 — q3 support-count ablation",
        "",
        "## Protocol",
        "",
        f"- {summary['training_rows']} train / {summary['test_rows']} test rows; five stratified folds, seed `{summary['fold_seed']}`.",
        "- Same fixed LightGBM configuration and folds for the raw-rowwise baseline, q3 categorical key, and q3 plus support count.",
        "- The q3 feature is `AMT_ANNUITY / AMT_CREDIT`, rounded to three decimals.",
        "- Support counts use only each model's fit partition; fit rows use leave-one-out counts, while validation/test rows use fit-partition counts.",
        "- No target encoding, train/test pooling, fee correction, external data, or CSV blending.",
        "",
        "## Results",
        "",
        "| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |",
        "|---|---:|---:|---|---:|",
    ]
    for name in ("baseline", "q3_only", "q3_support"):
        item = m[name]
        lines.append(
            f"| {name} | {item['pooled_auc']:.9f} | {item['mean_fold_auc']:.9f} | "
            f"{', '.join(f'{v:.6f}' for v in item['fold_auc'])} | {item['log_loss']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"- q3-only delta: `{d['q3_only_vs_baseline']['pooled_auc']:+.9f}`; positive folds `{d['q3_only_vs_baseline']['positive_folds']}/5`; conditional bootstrap CI `[{d['q3_only_vs_baseline']['paired_bootstrap']['ci_2_5']:+.9f}, {d['q3_only_vs_baseline']['paired_bootstrap']['ci_97_5']:+.9f}]`.",
            f"- q3+support delta: `{d['q3_support_vs_baseline']['pooled_auc']:+.9f}`; positive folds `{d['q3_support_vs_baseline']['positive_folds']}/5`; conditional bootstrap CI `[{d['q3_support_vs_baseline']['paired_bootstrap']['ci_2_5']:+.9f}, {d['q3_support_vs_baseline']['paired_bootstrap']['ci_97_5']:+.9f}]`.",
            "",
            "## Interpretation",
            "",
            "This separates a lower-cardinality product-key category from its unsupervised training-fold support. It tests a legal conservative alternative to explicit target encoding and pooled train/test counts, not the source notebook's exact pipeline. All labels have prior project exposure; treat the result as exploratory.",
            "",
            f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
            f"- Source SHA-256: `{summary['source_sha256']}`.",
            "- Full hashes: `PHASE55_SHA256_MANIFEST.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone fold-local q3 categorical/support experiment"
    )
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument("--test", type=Path, default=Path(r"D:\gci\inpuy\test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "phase55_q3_support_20261117",
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
