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
BOOTSTRAPS = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").astype(np.float64)


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b.where(b != 0.0)
    return result.replace([np.inf, -np.inf], np.nan)


def key_string(values: pd.Series, decimals: int) -> pd.Series:
    rounded = values.round(decimals)
    return rounded.map(
        lambda value: "__MISSING__"
        if pd.isna(value)
        else format(float(value), f".{decimals}f")
    )


def make_features(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    excluded = [name for name in (ID, TARGET) if name in raw]
    features = raw.drop(columns=excluded).copy()
    categorical = [name for name in CATS if name in features]
    for name in categorical:
        features[name] = (
            features[name].astype("string").fillna("__MISSING__").astype(str)
        )
    for name in features.columns:
        if name not in categorical:
            features[name] = pd.to_numeric(features[name], errors="raise").astype(
                "float32"
            )

    income = number(raw, "AMT_INCOME_TOTAL")
    credit = number(raw, "AMT_CREDIT")
    annuity = number(raw, "AMT_ANNUITY")
    goods = number(raw, "AMT_GOODS_PRICE")
    family = number(raw, "CNT_FAM_MEMBERS")
    birth = number(raw, "DAYS_BIRTH")
    employed_raw = number(raw, "DAYS_EMPLOYED")
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
        "EMPLOYED_TO_AGE": safe_ratio(employed, birth),
        "REGISTRATION_TO_AGE": safe_ratio(number(raw, "DAYS_REGISTRATION"), birth),
        "ID_PUBLISH_TO_AGE": safe_ratio(number(raw, "DAYS_ID_PUBLISH"), birth),
        "PHONE_CHANGE_TO_AGE": safe_ratio(
            number(raw, "DAYS_LAST_PHONE_CHANGE"), birth
        ),
        "CREDIT_TO_INCOME": safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "ANNUITY_TO_CREDIT": safe_ratio(annuity, credit),
        "CREDIT_TO_ANNUITY": safe_ratio(credit, annuity),
        "GOODS_TO_CREDIT": safe_ratio(goods, credit),
        "DOWNPAYMENT_AMOUNT": credit - goods,
        "DOWNPAYMENT_TO_CREDIT": safe_ratio(credit - goods, credit),
        "GOODS_TO_INCOME": safe_ratio(goods, income),
        "INCOME_PER_FAMILY_MEMBER": safe_ratio(income, family),
        "CREDIT_PER_FAMILY_MEMBER": safe_ratio(credit, family),
        "ANNUITY_PER_FAMILY_MEMBER": safe_ratio(annuity, family),
    }
    for name, values in (
        ("INCOME", income),
        ("CREDIT", credit),
        ("ANNUITY", annuity),
        ("GOODS_PRICE", goods),
    ):
        extra[f"LOG1P_{name}"] = np.log1p(values.clip(lower=0.0))

    external = [
        name
        for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if name in raw
    ]
    ext = pd.concat([number(raw, name) for name in external], axis=1)
    if external:
        extra["EXT_MEAN"] = ext.mean(axis=1)
        extra["EXT_MIN"] = ext.min(axis=1)
        extra["EXT_MAX"] = ext.max(axis=1)
        extra["EXT_STD"] = ext.std(axis=1)
        extra["EXT_RANGE"] = extra["EXT_MAX"] - extra["EXT_MIN"]
        extra["EXT_PRESENT_COUNT"] = ext.notna().sum(axis=1).astype("float32")
        extra["EXT_PRODUCT"] = ext.prod(axis=1, min_count=1)
        for i in range(len(external)):
            for j in range(i + 1, len(external)):
                a, b = ext.iloc[:, i], ext.iloc[:, j]
                pair = f"{external[i][-1]}{external[j][-1]}"
                extra[f"EXT_DIFF_{pair}"] = a - b
                extra[f"EXT_PRODUCT_{pair}"] = a * b

    bureau_names = [
        name for name in raw.columns if name.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if bureau_names:
        bureau = pd.concat([number(raw, name) for name in bureau_names], axis=1)
        extra["BUREAU_SUM"] = bureau.sum(axis=1, min_count=1)
        extra["BUREAU_NONZERO_COUNT"] = (bureau > 0).sum(axis=1).astype("float32")
    for name in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OWN_CAR_AGE"):
        if name in raw:
            extra[f"{name}_MISSING"] = raw[name].isna().astype("float32")

    features = pd.concat(
        [features.reset_index(drop=True), pd.DataFrame(extra).reset_index(drop=True)],
        axis=1,
    )
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature columns")
    for name in features.columns:
        if name in categorical:
            continue
        values = pd.to_numeric(features[name], errors="coerce")
        features[name] = values.replace([np.inf, -np.inf], np.nan).astype(
            "float32"
        )

    q_key = key_string(safe_ratio(annuity, credit), 3).reset_index(drop=True)
    if "NAME_CONTRACT_TYPE" in raw:
        contract = (
            raw["NAME_CONTRACT_TYPE"]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
            .reset_index(drop=True)
        )
    else:
        contract = pd.Series("__MISSING__", index=features.index, dtype="string")
    joint_key = (contract.astype(str) + "|" + q_key.astype(str)).rename(
        "Q3_CONTRACT_KEY"
    )
    return features, q_key, joint_key, categorical


def encode_categories(
    fit: pd.DataFrame,
    others: list[pd.DataFrame],
    categorical: list[str],
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, int]]:
    fit_encoded = fit.copy()
    others_encoded = [frame.copy() for frame in others]
    cardinalities: dict[str, int] = {}
    for column in categorical:
        values = (
            fit_encoded[column].astype("string").fillna("__MISSING__").astype(str)
        )
        mapping = {value: index for index, value in enumerate(pd.unique(values))}
        cardinalities[column] = len(mapping)
        fit_encoded[column] = values.map(mapping).astype("int32")
        for frame in others_encoded:
            frame[column] = (
                frame[column]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype("int32")
            )
    return fit_encoded, others_encoded, cardinalities


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


def metric_record(
    labels: np.ndarray, prediction: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    aucs = [
        float(roc_auc_score(labels[folds == fold], prediction[folds == fold]))
        for fold in sorted(np.unique(folds))
    ]
    return {
        "pooled_auc": float(roc_auc_score(labels, prediction)),
        "fold_auc": aucs,
        "mean_fold_auc": float(np.mean(aucs)),
        "std_fold_auc": float(np.std(aucs)),
        "log_loss": float(log_loss(labels, prediction)),
    }


def paired_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED + seed_offset)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        sample = np.concatenate(
            [
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(
            labels[sample], candidate[sample]
        ) - roc_auc_score(labels[sample], baseline[sample])
    return {
        "replicates": BOOTSTRAPS,
        "mean_delta": float(deltas.mean()),
        "ci_2_5": float(np.quantile(deltas, 0.025)),
        "ci_97_5": float(np.quantile(deltas, 0.975)),
        "fraction_positive": float(np.mean(deltas > 0)),
        "conditional_on_oof_vectors": True,
    }


def run_self_tests() -> None:
    rng = np.random.default_rng(5601)
    n = 120
    contract = np.where(np.arange(n) % 2, "Cash loans", "Revolving loans")
    raw = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": np.tile([0, 1], n // 2),
            "NAME_CONTRACT_TYPE": contract,
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
    raw.loc[1, "AMT_ANNUITY"] = 12000.0
    base, q, joint, cats = make_features(raw)
    if ID in base or TARGET in base:
        raise AssertionError("ID or target was included in features")
    if q.iloc[0] != q.iloc[1] or joint.iloc[0] == joint.iloc[1]:
        raise AssertionError("Joint key must distinguish contracts at equal Q3")
    one_row, one_q, one_joint, _ = make_features(raw.iloc[[0]].copy())
    if (
        not one_row.iloc[0].equals(base.iloc[0])
        or one_q.iloc[0] != q.iloc[0]
        or one_joint.iloc[0] != joint.iloc[0]
    ):
        raise AssertionError("Feature construction depends on other applicants")

    q_frame = base.iloc[:90].copy()
    q_frame["Q3_ANN_CREDIT"] = q.iloc[:90].to_numpy()
    q_frame["Q3_CONTRACT_KEY"] = joint.iloc[:90].to_numpy()
    cats_joint = cats + ["Q3_ANN_CREDIT", "Q3_CONTRACT_KEY"]
    encoded, other, _ = encode_categories(
        q_frame,
        [pd.concat([base.iloc[90:].reset_index(drop=True),
                    pd.DataFrame({"Q3_ANN_CREDIT": q.iloc[90:].reset_index(drop=True),
                                  "Q3_CONTRACT_KEY": joint.iloc[90:].reset_index(drop=True)})],
                   axis=1)],
        cats_joint,
    )
    quick = make_model(SEED, iterations=5)
    quick.fit(
        encoded,
        raw[TARGET].to_numpy()[:90],
        categorical_feature=cats_joint,
    )
    if not np.isfinite(quick.predict_proba(other[0])[:, 1]).all():
        raise AssertionError("Synthetic joint-category model prediction failed")


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

    labels = train[TARGET].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("TARGET must contain exactly binary labels 0 and 1")
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)
    x_train, q_train, joint_train, categorical = make_features(raw_train)
    x_test, q_test, joint_test, test_categorical = make_features(raw_test)
    if categorical != test_categorical or not x_train.columns.equals(x_test.columns):
        raise ValueError("Train/test feature schemas differ")

    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if np.any(fold_ids < 1):
        raise ValueError("Every training row must have exactly one OOF fold")

    oof = {
        "baseline": np.full(len(labels), np.nan, dtype=np.float64),
        "q3": np.full(len(labels), np.nan, dtype=np.float64),
        "q3_contract": np.full(len(labels), np.nan, dtype=np.float64),
    }
    test_joint_prediction = np.zeros(len(test), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        seed = SEED + fold_number
        q3_categories = categorical + ["Q3_ANN_CREDIT"]
        joint_categories = q3_categories + ["Q3_CONTRACT_KEY"]

        base_fit, base_others, _ = encode_categories(
            x_train.iloc[fit_rows],
            [x_train.iloc[valid_rows]],
            categorical,
        )

        q_fit = x_train.iloc[fit_rows].copy().reset_index(drop=True)
        q_valid = x_train.iloc[valid_rows].copy().reset_index(drop=True)
        q_test_frame = x_test.copy().reset_index(drop=True)
        q_fit["Q3_ANN_CREDIT"] = q_train.iloc[fit_rows].to_numpy()
        q_valid["Q3_ANN_CREDIT"] = q_train.iloc[valid_rows].to_numpy()
        q_test_frame["Q3_ANN_CREDIT"] = q_test.to_numpy()

        joint_fit = q_fit.copy()
        joint_valid = q_valid.copy()
        joint_test_frame = q_test_frame.copy()
        joint_fit["Q3_CONTRACT_KEY"] = joint_train.iloc[fit_rows].to_numpy()
        joint_valid["Q3_CONTRACT_KEY"] = joint_train.iloc[valid_rows].to_numpy()
        joint_test_frame["Q3_CONTRACT_KEY"] = joint_test.to_numpy()

        q_fit_encoded, q_others, q_cardinality = encode_categories(
            q_fit,
            [q_valid],
            q3_categories,
        )
        joint_fit_encoded, joint_others, joint_cardinality = encode_categories(
            joint_fit,
            [joint_valid, joint_test_frame],
            joint_categories,
        )

        baseline_model = make_model(seed)
        baseline_model.fit(
            base_fit,
            labels[fit_rows],
            categorical_feature=categorical,
        )
        oof["baseline"][valid_rows] = baseline_model.predict_proba(
            base_others[0]
        )[:, 1]
        del baseline_model

        q3_model = make_model(seed)
        q3_model.fit(
            q_fit_encoded,
            labels[fit_rows],
            categorical_feature=q3_categories,
        )
        oof["q3"][valid_rows] = q3_model.predict_proba(q_others[0])[:, 1]
        del q3_model

        joint_model = make_model(seed)
        joint_model.fit(
            joint_fit_encoded,
            labels[fit_rows],
            categorical_feature=joint_categories,
        )
        joint_prediction = joint_model.predict_proba(joint_others[0])[:, 1]
        oof["q3_contract"][valid_rows] = joint_prediction
        test_joint_prediction += (
            joint_model.predict_proba(joint_others[1])[:, 1] / FOLDS
        )
        del joint_model

        fold_auc = {
            name: float(roc_auc_score(labels[valid_rows], pred[valid_rows]))
            for name, pred in oof.items()
        }
        fold_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "baseline_auc": fold_auc["baseline"],
                "q3_auc": fold_auc["q3"],
                "q3_contract_auc": fold_auc["q3_contract"],
                "q3_delta_vs_baseline": fold_auc["q3"] - fold_auc["baseline"],
                "interaction_delta_vs_q3": fold_auc["q3_contract"] - fold_auc["q3"],
                "q3_category_cardinality_fit_only": q_cardinality["Q3_ANN_CREDIT"],
                "joint_category_cardinality_fit_only": joint_cardinality[
                    "Q3_CONTRACT_KEY"
                ],
            }
        )
        print(
            f"fold {fold_number}/{FOLDS}: "
            f"baseline={fold_auc['baseline']:.6f} q3={fold_auc['q3']:.6f} "
            f"q3+contract={fold_auc['q3_contract']:.6f}",
            flush=True,
        )

    if any(not np.isfinite(prediction).all() for prediction in oof.values()):
        raise ValueError("OOF predictions are incomplete or nonfinite")
    if not np.isfinite(test_joint_prediction).all():
        raise ValueError("Test predictions are nonfinite")

    model_metrics = {
        name: metric_record(labels, prediction, fold_ids)
        for name, prediction in oof.items()
    }
    deltas = {
        "q3_vs_baseline": {
            "pooled_auc": model_metrics["q3"]["pooled_auc"]
            - model_metrics["baseline"]["pooled_auc"],
            "log_loss": model_metrics["q3"]["log_loss"]
            - model_metrics["baseline"]["log_loss"],
            "positive_folds": sum(
                item["q3_delta_vs_baseline"] > 0 for item in fold_records
            ),
            "bootstrap": paired_bootstrap(
                labels, oof["baseline"], oof["q3"], 5601
            ),
        },
        "q3_contract_vs_q3": {
            "pooled_auc": model_metrics["q3_contract"]["pooled_auc"]
            - model_metrics["q3"]["pooled_auc"],
            "log_loss": model_metrics["q3_contract"]["log_loss"]
            - model_metrics["q3"]["log_loss"],
            "positive_folds": sum(
                item["interaction_delta_vs_q3"] > 0 for item in fold_records
            ),
            "bootstrap": paired_bootstrap(
                labels, oof["q3"], oof["q3_contract"], 5602
            ),
        },
        "q3_contract_vs_baseline": {
            "pooled_auc": model_metrics["q3_contract"]["pooled_auc"]
            - model_metrics["baseline"]["pooled_auc"],
            "log_loss": model_metrics["q3_contract"]["log_loss"]
            - model_metrics["baseline"]["log_loss"],
            "positive_folds": sum(
                item["q3_contract_auc"] > item["baseline_auc"]
                for item in fold_records
            ),
            "bootstrap": paired_bootstrap(
                labels, oof["baseline"], oof["q3_contract"], 5603
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    oof_path = output_dir / "PHASE56_OOF_PREDICTIONS.csv"
    submission_path = output_dir / "submission.csv"
    summary_path = output_dir / "PHASE56_SUMMARY.json"
    report_path = output_dir / "PHASE56_REPORT.md"
    pd.DataFrame(
        {
            ID: train[ID].to_numpy(),
            TARGET: labels,
            "fold": fold_ids,
            "baseline_oof": oof["baseline"],
            "q3_oof": oof["q3"],
            "q3_contract_oof": oof["q3_contract"],
        }
    ).to_csv(oof_path, index=False)
    submission = pd.DataFrame(
        {ID: test[ID].to_numpy(), TARGET: test_joint_prediction}
    )
    if not submission[ID].equals(test[ID]) or not submission[ID].is_unique:
        raise ValueError("Submission IDs do not match test order")
    if not np.isfinite(submission[TARGET]).all() or not submission[TARGET].between(
        0.0, 1.0
    ).all():
        raise ValueError("Submission predictions must be finite values in [0, 1]")
    submission.to_csv(submission_path, index=False)

    runtime = time.perf_counter() - started
    source_path = Path(__file__).resolve()
    summary = {
        "experiment": "Phase 56 coarse q3 by contract interaction",
        "status": "exploratory; no untouched holdout available",
        "training_rows": len(train),
        "test_rows": len(test),
        "fold_seed": SEED,
        "fold_count": FOLDS,
        "fold_records": fold_records,
        "metrics": model_metrics,
        "deltas": deltas,
        "model": {
            "library": "LightGBM",
            "version": lgb.__version__,
            "device": "CPU",
            "iterations": ITERATIONS,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "min_child_samples": 150,
            "threads": THREADS,
            "early_stopping": False,
        },
        "feature_scope": {
            "baseline": "official predictors plus deterministic row-local ratios/logs/age/external geometry",
            "q3": "baseline plus q=AMT_ANNUITY/AMT_CREDIT rounded to 3 decimals",
            "q3_contract": "q3 model plus row-local joint key NAME_CONTRACT_TYPE|Q3",
            "category_maps": "fit on each model training fold only; unseen validation/test keys map to missing",
            "target_encoding": "none",
            "support_count": "none",
            "train_test_pooling": "none",
            "external_data": "none",
            "prior_predictions_or_cache": "none",
            "manual_prediction_edits": "none",
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
            "All labeled rows and these folds have prior project exposure; this result is exploratory.",
            "The bootstrap intervals are conditional on saved OOF vectors and do not account for model or research-selection uncertainty.",
            "This test asks whether contract type modifies the q3 grouping; it does not reproduce the archived target-encoded pipeline.",
            "The submission is an experiment artifact, not a confirmed champion or a guaranteed leaderboard score.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in (oof_path, submission_path, summary_path, report_path, source_path)
    }
    (output_dir / "PHASE56_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"pooled AUC baseline={model_metrics['baseline']['pooled_auc']:.9f} "
        f"q3={model_metrics['q3']['pooled_auc']:.9f} "
        f"q3+contract={model_metrics['q3_contract']['pooled_auc']:.9f}",
        flush=True,
    )
    print(f"submission: {submission_path}", flush=True)


def render_report(summary: dict[str, Any]) -> str:
    metrics_by_model = summary["metrics"]
    delta = summary["deltas"]
    lines = [
        "# Phase 56 — Q3 × contract-type interaction",
        "",
        "## Hypothesis",
        "",
        "A coarse annuity/credit key may behave differently for Cash versus Revolving loans. A row-local joint category tests that interaction without specialist label transfer or an inferred applicant identity.",
        "",
        "## Protocol",
        "",
        f"- Five stratified folds, seed `{summary['fold_seed']}`; fixed `{summary['model']['iterations']}` rounds.",
        "- Same LightGBM settings/folds for baseline, Q3, and Q3-plus-contract interaction.",
        "- Fold-local category dictionaries; unseen validation/test keys map to missing.",
        "- No target encoding, cross-row label transfer, test pooling, fee constants, cached predictions, or rank blending.",
        "",
        "## Results",
        "",
        "| Model | Pooled AUC | Mean fold AUC | Fold AUCs | Log loss |",
        "|---|---:|---:|---|---:|",
    ]
    for name in ("baseline", "q3", "q3_contract"):
        item = metrics_by_model[name]
        lines.append(
            f"| {name} | {item['pooled_auc']:.9f} | {item['mean_fold_auc']:.9f} | "
            f"{', '.join(f'{v:.6f}' for v in item['fold_auc'])} | {item['log_loss']:.9f} |"
        )
    for name, label in (
        ("q3_vs_baseline", "Q3 vs baseline"),
        ("q3_contract_vs_q3", "Q3×contract incremental vs Q3"),
        ("q3_contract_vs_baseline", "Q3×contract vs baseline"),
    ):
        result = delta[name]
        ci = result["bootstrap"]
        lines.append(
            f"- {label}: `{result['pooled_auc']:+.9f}` AUC; positive folds "
            f"`{result['positive_folds']}/5`; conditional bootstrap CI "
            f"`[{ci['ci_2_5']:+.9f}, {ci['ci_97_5']:+.9f}]`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary question is whether adding the contract-type interaction improves the Q3-only model. A positive interaction effect would justify retaining the row-local interaction for further testing, not prove a generalizable 0.775 result. A negative/neutral effect means the Phase 20 contract-routing direction does not automatically transfer to Q3 products. All labels have prior project exposure.",
            "",
            f"- Runtime seconds: `{summary['runtime_seconds']:.2f}`.",
            f"- Source SHA-256: `{summary['source_sha256']}`.",
            "- Full hashes: `PHASE56_SHA256_MANIFEST.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Q3-by-contract feature-interaction experiment"
    )
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument("--test", type=Path, default=Path(r"D:\gci\inpuy\test.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / "phase56_q3_contract_20261117",
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
