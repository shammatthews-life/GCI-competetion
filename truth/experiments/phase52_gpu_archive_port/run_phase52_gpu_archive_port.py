from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


SEED = 20261113
FOLDS = 5
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


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    denominator = denominator.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = numerator / denominator
    return values.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)


def make_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors = raw.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    base = predictors.copy()
    extra: dict[str, np.ndarray | pd.Series] = {}

    employed = predictors.get("DAYS_EMPLOYED")
    if employed is not None:
        clean_employed = employed.where(employed != 365243)
        extra["DAYS_EMPLOYED_CLEAN"] = clean_employed
        extra["DE_SENT"] = (employed == 365243).astype(np.int8)

    if "ORGANIZATION_TYPE" in predictors:
        extra["ORG_XNA"] = (
            predictors["ORGANIZATION_TYPE"].astype("string") == "XNA"
        ).astype(np.int8)
    if "OWN_CAR_AGE" in predictors:
        extra["CAR_AGE_OUT"] = (predictors["OWN_CAR_AGE"] >= 60).astype(np.int8)
    extra["MISS_CNT"] = predictors.isna().sum(axis=1).astype(np.int16)
    for column in (
        "EXT_SOURCE_1",
        "EXT_SOURCE_3",
        "OWN_CAR_AGE",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ):
        if column in predictors:
            extra[f"{column}_NA"] = predictors[column].isna().astype(np.int8)

    ext_columns = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in predictors]
    if ext_columns:
        ext = predictors[ext_columns]
        extra["EXT_MEAN"] = ext.mean(axis=1)
        extra["EXT_MIN"] = ext.min(axis=1)
        extra["EXT_MAX"] = ext.max(axis=1)
        extra["EXT_STD"] = ext.std(axis=1)
        extra["EXT_RANGE"] = extra["EXT_MAX"] - extra["EXT_MIN"]
        extra["EXT_NOBS"] = ext.notna().sum(axis=1)
        extra["EXT_PRODUCT"] = ext.prod(axis=1, min_count=1)
        if len(ext_columns) == 3:
            e1, e2, e3 = (predictors[c] for c in ext_columns)
            extra["EXT_WEIGHTED"] = (2 * e1 + 3 * e2 + 4 * e3) / 9
            for name, value in {
                "EXT_12_PRODUCT": e1 * e2,
                "EXT_13_PRODUCT": e1 * e3,
                "EXT_23_PRODUCT": e2 * e3,
                "EXT_12_DIFF": e1 - e2,
                "EXT_13_DIFF": e1 - e3,
                "EXT_23_DIFF": e2 - e3,
            }.items():
                extra[name] = value

    required_finance = (
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
    )
    if all(column in predictors for column in required_finance):
        income = predictors["AMT_INCOME_TOTAL"]
        credit = predictors["AMT_CREDIT"]
        annuity = predictors["AMT_ANNUITY"]
        goods = predictors["AMT_GOODS_PRICE"]
        extra["CREDIT_INC"] = _safe_ratio(credit, income)
        extra["ANN_INC"] = _safe_ratio(annuity, income)
        extra["ANN_CREDIT"] = _safe_ratio(annuity, credit)
        extra["TERM"] = _safe_ratio(credit, annuity)
        extra["GOODS_CREDIT"] = _safe_ratio(goods, credit)
        extra["CREDIT_MINUS_GOODS"] = (credit - goods).to_numpy(dtype=np.float32)
        extra["GOODS_INC"] = _safe_ratio(goods, income)
        for column in required_finance:
            extra[f"LOG_{column}"] = np.log1p(
                predictors[column].clip(lower=0)
            ).to_numpy(dtype=np.float32)
        if "CNT_FAM_MEMBERS" in predictors:
            family = predictors["CNT_FAM_MEMBERS"]
            extra["INC_PER_FAM"] = _safe_ratio(income, family)
            extra["CREDIT_PER_FAM"] = _safe_ratio(credit, family)
            extra["ANN_PER_FAM"] = _safe_ratio(annuity, family)

    if "DAYS_BIRTH" in predictors:
        age_days = -predictors["DAYS_BIRTH"]
        extra["AGE_YEARS"] = age_days / 365.25
        if employed is not None:
            de = employed.where(employed != 365243)
            extra["EMP_AGE_RATIO"] = _safe_ratio(de, predictors["DAYS_BIRTH"])
            extra["EMP_YEARS"] = -de / 365.25
        for source, name in (
            ("DAYS_REGISTRATION", "REG_AGE_RATIO"),
            ("DAYS_ID_PUBLISH", "ID_AGE_RATIO"),
            ("DAYS_LAST_PHONE_CHANGE", "PHONE_AGE_RATIO"),
        ):
            if source in predictors:
                extra[name] = _safe_ratio(predictors[source], predictors["DAYS_BIRTH"])
        if "OWN_CAR_AGE" in predictors:
            extra["CAR_AGE_RATIO"] = _safe_ratio(
                predictors["OWN_CAR_AGE"], extra["AGE_YEARS"]
            )
        if "AMT_CREDIT" in predictors:
            extra["CREDIT_PER_AGE"] = _safe_ratio(
                predictors["AMT_CREDIT"], extra["AGE_YEARS"]
            )
        if "AMT_INCOME_TOTAL" in predictors:
            extra["INC_PER_AGE"] = _safe_ratio(
                predictors["AMT_INCOME_TOTAL"], extra["AGE_YEARS"]
            )

    bureau = [c for c in predictors.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau:
        extra["BUR_SUM"] = predictors[bureau].sum(axis=1, min_count=1)
        recent = [
            c
            for c in ("AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON")
            if c in predictors
        ]
        if recent:
            extra["BUR_RECENT"] = predictors[recent].sum(axis=1, min_count=1)
        month, year = (
            predictors.get("AMT_REQ_CREDIT_BUREAU_MON"),
            predictors.get("AMT_REQ_CREDIT_BUREAU_YEAR"),
        )
        if month is not None and year is not None:
            extra["BUR_MON_YEAR"] = _safe_ratio(month, year + 1)

    expanded = pd.concat(
        [base.reset_index(drop=True), pd.DataFrame(extra).reset_index(drop=True)],
        axis=1,
    )
    expanded = expanded.loc[:, ~expanded.columns.duplicated()].copy()
    for frame in (base, expanded):
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        frame.loc[:, numeric_columns] = frame[numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
    return base, expanded


def feature_groups(
    base: pd.DataFrame, expanded: pd.DataFrame
) -> dict[str, list[str]]:
    added = set(expanded.columns) - set(base.columns)
    clean_columns = {
        "DAYS_EMPLOYED_CLEAN",
        "DE_SENT",
        "ORG_XNA",
        "CAR_AGE_OUT",
        "MISS_CNT",
        "EXT_SOURCE_1_NA",
        "EXT_SOURCE_3_NA",
        "OWN_CAR_AGE_NA",
        "AMT_REQ_CREDIT_BUREAU_YEAR_NA",
    } & added
    groups = {
        "clean_missingness": clean_columns,
        "external_score_geometry": {
            name
            for name in added
            if name.startswith("EXT_") and name not in clean_columns
        },
        "financial_ratios": {
            "CREDIT_INC",
            "ANN_INC",
            "ANN_CREDIT",
            "TERM",
            "GOODS_CREDIT",
            "CREDIT_MINUS_GOODS",
            "GOODS_INC",
            "INC_PER_FAM",
            "CREDIT_PER_FAM",
            "ANN_PER_FAM",
            *(
                name
                for name in added
                if name.startswith("LOG_AMT_")
            ),
        } & added,
        "age_employment": {
            "AGE_YEARS",
            "EMP_AGE_RATIO",
            "EMP_YEARS",
            "REG_AGE_RATIO",
            "ID_AGE_RATIO",
            "PHONE_AGE_RATIO",
            "CAR_AGE_RATIO",
            "CREDIT_PER_AGE",
            "INC_PER_AGE",
        } & added,
        "bureau_activity": {
            name for name in added if name.startswith("BUR_")
        },
    }
    if set.union(*groups.values()) != added:
        missing = sorted(added - set.union(*groups.values()))
        duplicate = sorted(
            name
            for name in added
            if sum(name in members for members in groups.values()) > 1
        )
        raise ValueError(
            f"Feature groups must partition engineered columns; "
            f"unassigned={missing}, duplicated={duplicate}"
        )
    return {name: sorted(columns) for name, columns in groups.items()}


def _matrix(
    frame: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    categorical_columns: list[str],
) -> tuple[csr_matrix, csr_matrix]:
    present_cats = [c for c in categorical_columns if c in frame]
    numeric = frame.drop(columns=present_cats).select_dtypes(include=[np.number])
    fit_numeric = csr_matrix(numeric.iloc[fit_rows].to_numpy(dtype=np.float32))
    apply_numeric = csr_matrix(numeric.iloc[apply_rows].to_numpy(dtype=np.float32))
    if not present_cats:
        return fit_numeric, apply_numeric
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        dtype=np.float32,
        sparse_output=True,
    )
    fit_cat_values = (
        frame.iloc[fit_rows][present_cats].astype("string").fillna("__MISSING__")
    )
    apply_cat_values = (
        frame.iloc[apply_rows][present_cats].astype("string").fillna("__MISSING__")
    )
    fit_cats = encoder.fit_transform(fit_cat_values).tocsr()
    apply_cats = encoder.transform(apply_cat_values).tocsr()
    return (
        hstack([fit_numeric, fit_cats], format="csr", dtype=np.float32),
        hstack([apply_numeric, apply_cats], format="csr", dtype=np.float32),
    )


def _model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        max_bin=256,
        n_jobs=8,
        random_state=SEED,
        verbosity=0,
    )


def run(train_path: Path, output_dir: Path) -> dict:
    from catboost.utils import get_gpu_device_count

    if get_gpu_device_count() < 1:
        raise RuntimeError("No CatBoost-visible CUDA GPU was detected")
    train = pd.read_csv(train_path)
    if "TARGET" not in train or "SK_ID_CURR" not in train:
        raise ValueError("Training CSV must contain TARGET and SK_ID_CURR")
    labels = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    base, engineered = make_features(train)
    groups = feature_groups(base, engineered)
    variants = {"raw_baseline": base}
    for group_name, columns in groups.items():
        variants[f"only_{group_name}"] = pd.concat(
            [base, engineered[columns]], axis=1
        )
    variants["v7_label_free_features"] = engineered
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(labels)), labels
        )
    )
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(labels), np.nan, dtype=np.float64)
        for name in variants
    }
    fold_results = []
    start = time.perf_counter()
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        for name, frame in variants.items():
            x_fit, x_valid = _matrix(
                frame, fit_rows, valid_rows, list(CATEGORICAL_COLUMNS)
            )
            model = _model()
            model.fit(x_fit, labels[fit_rows], verbose=False)
            fold_prediction = model.predict_proba(x_valid)[:, 1]
            predictions[name][valid_rows] = fold_prediction
            fold_auc = float(roc_auc_score(labels[valid_rows], fold_prediction))
            fold_results.append(
                {
                    "fold": fold_number,
                    "model": name,
                    "auc": fold_auc,
                    "train_rows": int(len(fit_rows)),
                    "valid_rows": int(len(valid_rows)),
                    "feature_count": int(x_fit.shape[1]),
                    "device": "cuda",
                }
            )
            print(
                f"{name} fold {fold_number}/{FOLDS}: "
                f"AUC={fold_auc:.6f}, features={x_fit.shape[1]}, "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "Phase 52 GPU translation of v7-style label-free features",
        "train_path": str(train_path),
        "train_sha256": _sha256(train_path),
        "seed": SEED,
        "folds": FOLDS,
        "xgboost_version": xgb.__version__,
        "device": "CUDA (XGBoost hist)",
        "feature_scope": (
            "raw predictors versus raw predictors plus deterministic v7-style "
            "feature families; no 129.5/0.0132 constants, target encoding, "
            "train+test pooling, or external data"
        ),
        "feature_groups": groups,
        "fold_results": fold_results,
        "models": {},
        "elapsed_seconds": time.perf_counter() - start,
    }
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Incomplete or non-finite OOF predictions: {name}")
        summary["models"][name] = {
            "pooled_auc": float(roc_auc_score(labels, prediction)),
            "log_loss": float(log_loss(labels, prediction, labels=[0, 1])),
            "fold_auc": [
                row["auc"] for row in fold_results if row["model"] == name
            ],
        }
    summary["delta_engineered_minus_baseline_auc"] = (
        summary["models"]["v7_label_free_features"]["pooled_auc"]
        - summary["models"]["raw_baseline"]["pooled_auc"]
    )
    summary["diagnostic_note"] = (
        "Single-family variants are standalone additions to the same raw "
        "baseline. Their gains are not additive or causal, and comparing "
        "multiple variants is exploratory."
    )
    pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": labels,
            **predictions,
        }
    ).to_csv(output_dir / "phase52_oof.csv", index=False)
    (output_dir / "phase52_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path(r"D:\gci\inpuy\train.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "phase52_gpu",
    )
    args = parser.parse_args()
    summary = run(args.train, args.output_dir)
    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
