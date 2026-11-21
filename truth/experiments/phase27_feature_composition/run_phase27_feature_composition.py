from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
PHASE14 = REPO / "experiments" / "phase14_label_free_model_development"
PHASE15 = REPO / "experiments" / "phase15_autonomous_research"
PHASE23 = REPO / "experiments" / "phase23_auc_feature_research"
PHASE25 = REPO / "experiments" / "phase25_exp02_compliance"
sys.path[:0] = [str(PHASE25), str(PHASE23), str(PHASE15), str(PHASE14), str(PHASE13)]

import phase13_features as features13
import phase14_features as features14
import phase15_features as features15
import run_phase23_auc_feature_research as phase23
import run_phase25_exp02_compliant_features as phase25


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
TARGET = features13.TARGET_COLUMN
ID = features13.ID_COLUMN
SEED = 20261113
FOLDS = 5
MODEL_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 80,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "n_jobs": 8,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

CANDIDATES = (
    "baseline",
    "application_process",
    "external_disagreement",
    "financial_residual",
    "applicant_consistency",
    "formula_peer",
    "context_interactions",
    "origin_formula_geometry",
    "origin_copula",
    "composed_all_blocks",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(raw: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(raw[column], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def context_interaction_features(raw: pd.DataFrame) -> pd.DataFrame:
    income = _numeric(raw, "AMT_INCOME_TOTAL")
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    family = _numeric(raw, "CNT_FAM_MEMBERS")
    age_days = np.abs(_numeric(raw, "DAYS_BIRTH"))
    employment_days = np.abs(_numeric(raw, "DAYS_EMPLOYED"))
    employment_days[employment_days == 365243.0] = np.nan

    scores = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(scores)
    count = observed.sum(axis=1)
    score_mean = np.divide(
        np.nansum(scores, axis=1),
        count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=count > 0,
    )
    minimum = np.where(observed, scores, np.inf).min(axis=1)
    maximum = np.where(observed, scores, -np.inf).max(axis=1)
    spread = maximum - minimum
    spread[count < 2] = np.nan
    credit_to_income = _safe_ratio(credit, income)
    annuity_to_income = _safe_ratio(annuity, income)
    annuity_to_credit = _safe_ratio(annuity, credit)
    credit_to_goods = _safe_ratio(credit, goods)
    employment_to_age = _safe_ratio(employment_days, age_days)

    return pd.DataFrame(
        {
            "P27_EXT_SPREAD_X_CREDIT_TO_INCOME": spread * credit_to_income,
            "P27_EXT_SPREAD_X_ANNUITY_TO_INCOME": spread * annuity_to_income,
            "P27_EXT_SPREAD_X_ANNUITY_TO_CREDIT": spread * annuity_to_credit,
            "P27_EXT_SPREAD_X_CREDIT_TO_GOODS": spread * credit_to_goods,
            "P27_EXT_MISSING_X_CREDIT_TO_INCOME": (3 - count)
            * credit_to_income,
            "P27_EXT_MISSING_X_EMPLOYMENT_TO_AGE": (3 - count)
            * employment_to_age,
            "P27_EXT_MEAN_X_CREDIT_TO_INCOME": score_mean
            * credit_to_income,
            "P27_EXT_MEAN_X_ANNUITY_TO_CREDIT": score_mean
            * annuity_to_credit,
            "P27_FAMILY_ADJUSTED_ANNUITY_BURDEN": _safe_ratio(
                annuity * family, income
            ),
        }
    ).replace([np.inf, -np.inf], np.nan)


def origin_formula_geometry_features(raw: pd.DataFrame) -> pd.DataFrame:
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(ext)
    count = observed.sum(axis=1)
    nonnegative_scores = np.where(observed, np.maximum(ext, 0.0), 0.0)
    score_sum = nonnegative_scores.sum(axis=1)
    proportions = np.divide(
        nonnegative_scores,
        score_sum[:, None],
        out=np.zeros_like(nonnegative_scores),
        where=score_sum[:, None] > 0,
    )
    entropy_terms = np.zeros_like(proportions)
    valid_terms = proportions > 0
    rows, columns = np.nonzero(valid_terms)
    entropy_terms[rows, columns] = proportions[rows, columns] * np.log(
        count[rows] * proportions[rows, columns]
    )
    entropy_kl = entropy_terms.sum(axis=1)
    entropy_kl[(count == 0) | (score_sum == 0)] = np.nan

    income = _numeric(raw, "AMT_INCOME_TOTAL")
    credit = _numeric(raw, "AMT_CREDIT")
    credit_to_income = _safe_ratio(credit, income)
    e2 = ext[:, 1]
    e3 = ext[:, 2]
    return pd.DataFrame(
        {
            "P27_EXT_KL_TO_UNIFORM": entropy_kl,
            "P27_EXT_E2_E3_DIAGONAL_DISTANCE": np.abs(e2 - e3)
            / np.sqrt(2.0),
            "P27_CREDIT_INCOME_QUADRATIC_X_1_MINUS_E2": (
                np.square(credit_to_income) * (1.0 - e2)
            ),
            "P27_INV_REG_BIRTH": _numeric(raw, "DAYS_REGISTRATION")
            - _numeric(raw, "DAYS_BIRTH"),
            "P27_INV_ID_PUBLISH_BIRTH": _numeric(raw, "DAYS_ID_PUBLISH")
            - _numeric(raw, "DAYS_BIRTH"),
        }
    ).replace([np.inf, -np.inf], np.nan)


def origin_copula_features(raw: pd.DataFrame) -> pd.DataFrame:
    ext = raw[["EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(dtype=np.float64)
    u1 = np.clip(np.nan_to_num(ext[:, 0], nan=0.5), 1e-4, 1.0 - 1e-4)
    u2 = np.clip(np.nan_to_num(ext[:, 1], nan=0.5), 1e-4, 1.0 - 1e-4)
    theta_12 = 1.2
    theta_2 = 2.0
    theta_gumbel = 1.5
    return pd.DataFrame(
        {
            "P27_COPULA_CLAYTON_12": np.maximum(
                np.power(u1, -theta_12) + np.power(u2, -theta_12) - 1.0,
                1e-6,
            )
            ** (-1.0 / theta_12),
            "P27_COPULA_CLAYTON_2": np.maximum(
                np.power(u1, -theta_2) + np.power(u2, -theta_2) - 1.0,
                1e-6,
            )
            ** (-1.0 / theta_2),
            "P27_COPULA_GUMBEL_15": np.exp(
                -np.power(
                    np.power(-np.log(u1), theta_gumbel)
                    + np.power(-np.log(u2), theta_gumbel),
                    1.0 / theta_gumbel,
                )
            ),
        }
    ).replace([np.inf, -np.inf], np.nan)


def _append(
    left: pd.DataFrame, right: pd.DataFrame
) -> pd.DataFrame:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError("Feature blocks have inconsistent row counts")
    collisions = left.columns.intersection(right.columns)
    if len(collisions):
        raise ValueError(f"Feature names collide: {collisions.tolist()}")
    return pd.concat([left, right], axis=1)


def _make_base_features(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric, categorical, rowwise = features13.build_raw_feature_blocks(raw)
    decompositions = features13.build_label_free_decompositions(raw)
    enhancements = features14.build_numeric_enhancements(raw)
    enhancements = enhancements.drop(
        columns=["BUREAU_REQUEST_SUM"], errors="ignore"
    )
    time_features = features15.build_time_lifecycle_features(raw)
    numeric_features = _append(
        _append(numeric, rowwise),
        _append(_append(decompositions, enhancements), time_features),
    )

    x_fit = numeric_features.iloc[fit_rows].reset_index(drop=True)
    x_valid = numeric_features.iloc[valid_rows].reset_index(drop=True)
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )
    fit_categories = categorical.iloc[fit_rows].astype(str)
    valid_categories = categorical.iloc[valid_rows].astype(str)
    encoded_fit = encoder.fit_transform(fit_categories)
    encoded_valid = encoder.transform(valid_categories)
    names = encoder.get_feature_names_out(categorical.columns)
    cat_fit = pd.DataFrame(encoded_fit, columns=names)
    cat_valid = pd.DataFrame(encoded_valid, columns=names)
    return _append(x_fit, cat_fit), _append(x_valid, cat_valid)


def _fold_feature_blocks(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
    seed: int,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    process_fit, process_valid = phase23.application_process_features(
        raw, fit_rows, valid_rows
    )
    external = phase23.external_disagreement_features(raw)
    financial_fit, financial_valid = phase23.financial_residual_features(
        raw, fit_rows, valid_rows, seed
    )
    consistency_fit, consistency_valid = phase23.applicant_consistency_features(
        raw, fit_rows, valid_rows, seed
    )
    peer_fit, peer_valid = phase25.formula_peer_features(
        raw, fit_rows, valid_rows
    )
    interactions = context_interaction_features(raw)
    origin_geometry = origin_formula_geometry_features(raw)
    origin_copula = origin_copula_features(raw)

    return {
        "application_process": (process_fit, process_valid),
        "external_disagreement": (
            external.iloc[fit_rows].reset_index(drop=True),
            external.iloc[valid_rows].reset_index(drop=True),
        ),
        "financial_residual": (financial_fit, financial_valid),
        "applicant_consistency": (consistency_fit, consistency_valid),
        "formula_peer": (peer_fit, peer_valid),
        "context_interactions": (
            interactions.iloc[fit_rows].reset_index(drop=True),
            interactions.iloc[valid_rows].reset_index(drop=True),
        ),
        "origin_formula_geometry": (
            origin_geometry.iloc[fit_rows].reset_index(drop=True),
            origin_geometry.iloc[valid_rows].reset_index(drop=True),
        ),
        "origin_copula": (
            origin_copula.iloc[fit_rows].reset_index(drop=True),
            origin_copula.iloc[valid_rows].reset_index(drop=True),
        ),
    }


def _candidate_features(
    base: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
    candidate: str,
) -> pd.DataFrame:
    if candidate == "baseline":
        return base
    if candidate == "composed_all_blocks":
        selected = [
            "application_process",
            "external_disagreement",
            "financial_residual",
            "applicant_consistency",
            "formula_peer",
            "context_interactions",
            "origin_formula_geometry",
            "origin_copula",
        ]
    else:
        selected = [candidate]
    result = base
    for name in selected:
        result = _append(result, blocks[name])
    return result


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    started = time.perf_counter()
    data = pd.read_csv(train_path)
    if TARGET not in data or ID not in data:
        raise ValueError(f"Training data must contain {ID!r} and {TARGET!r}")
    y = pd.to_numeric(data[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
        or set(np.unique(y)) != {0, 1}
    ):
        raise ValueError("TARGET must contain both binary classes and only 0/1")
    ids = data[ID].copy()
    raw = data.drop(columns=[TARGET]).reset_index(drop=True)
    y = y.astype(np.int8)
    folds = list(
        StratifiedKFold(
            n_splits=FOLDS, shuffle=True, random_state=SEED
        ).split(np.zeros(len(y)), y)
    )

    oof = {
        candidate: np.full(len(y), np.nan, dtype=np.float32)
        for candidate in CANDIDATES
    }
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    fold_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        print(
            f"FOLD {fold_number}/{FOLDS}: "
            f"fit={len(fit_rows)} valid={len(valid_rows)}",
            flush=True,
        )
        base_fit, base_valid = _make_base_features(raw, fit_rows, valid_rows)
        fold_blocks = _fold_feature_blocks(
            raw, fit_rows, valid_rows, SEED + fold_number * 100
        )
        fit_blocks = {name: pair[0] for name, pair in fold_blocks.items()}
        valid_blocks = {name: pair[1] for name, pair in fold_blocks.items()}
        fold_ids[valid_rows] = fold_number

        for candidate in CANDIDATES:
            x_fit = _candidate_features(base_fit, fit_blocks, candidate)
            x_valid = _candidate_features(base_valid, valid_blocks, candidate)
            if not x_fit.columns.equals(x_valid.columns):
                raise ValueError(
                    f"{candidate} feature schema differs in fold {fold_number}"
                )
            if TARGET in x_fit.columns or ID in x_fit.columns:
                raise ValueError("Target or row ID entered the feature matrix")
            model = lgb.LGBMClassifier(
                **MODEL_PARAMS,
                random_state=SEED + fold_number * 100,
            )
            model.fit(x_fit.to_numpy(dtype=np.float32), y[fit_rows])
            prediction = model.predict_proba(
                x_valid.to_numpy(dtype=np.float32)
            )[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"{candidate} produced non-finite predictions in fold "
                    f"{fold_number}"
                )
            oof[candidate][valid_rows] = prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], prediction))
            fold_records.append(
                {
                    "candidate": candidate,
                    "fold": fold_number,
                    "auc": auc,
                    "fit_rows": int(len(fit_rows)),
                    "valid_rows": int(len(valid_rows)),
                    "feature_count": int(x_fit.shape[1]),
                }
            )
            print(
                f"  {candidate}: AUC={auc:.6f}, "
                f"features={x_fit.shape[1]}",
                flush=True,
            )
            del model, x_fit, x_valid, prediction

        del base_fit, base_valid, fold_blocks, fit_blocks, valid_blocks

    if np.any(fold_ids < 1):
        raise ValueError("Some rows did not receive an outer-fold assignment")
    for candidate, prediction in oof.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"OOF predictions are incomplete for {candidate}")

    output_dir.mkdir(parents=True, exist_ok=False)
    fold_frame = pd.DataFrame(fold_records)
    summary_rows = []
    baseline_auc = float(roc_auc_score(y, oof["baseline"]))
    for candidate, prediction in oof.items():
        fold_aucs = fold_frame.loc[
            fold_frame["candidate"] == candidate, "auc"
        ].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "candidate": candidate,
                "pooled_auc": float(roc_auc_score(y, prediction)),
                "mean_fold_auc": float(fold_aucs.mean()),
                "std_fold_auc": float(fold_aucs.std(ddof=0)),
                "delta_vs_baseline": float(
                    roc_auc_score(y, prediction) - baseline_auc
                ),
                "pooled_log_loss": float(log_loss(y, prediction)),
                "positive_folds_vs_baseline": int(
                    np.sum(
                        fold_aucs
                        > fold_frame.loc[
                            fold_frame["candidate"] == "baseline", "auc"
                        ].to_numpy(dtype=np.float64)
                    )
                ),
            }
        )

    fold_frame.to_csv(output_dir / "PHASE27_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "PHASE27_SUMMARY.csv", index=False
    )
    oof_frame = pd.DataFrame(
        {
            ID: ids,
            TARGET: y,
            "outer_fold": fold_ids,
            **{
                f"oof_{candidate}": prediction
                for candidate, prediction in oof.items()
            },
        }
    )
    oof_frame.to_csv(output_dir / "PHASE27_OOF_PREDICTIONS.csv", index=False)

    files = [
        Path(__file__),
        train_path,
        output_dir / "PHASE27_FOLD_RESULTS.csv",
        output_dir / "PHASE27_SUMMARY.csv",
        output_dir / "PHASE27_OOF_PREDICTIONS.csv",
    ]
    manifest = {
        "phase": "Phase 27 feature composition",
        "status": "exploratory; repeated-label selection risk applies",
        "data_scope": "training CSV only; no test rows or external data",
        "target_encoding": "none; categorical variables use fit-fold one-hot encoding",
        "pooled_train_test_statistics": "none",
        "empirical_pricing_constants": "excluded",
        "folds": FOLDS,
        "seed": SEED,
        "model": {"library": "LightGBM", **MODEL_PARAMS},
        "candidates": list(CANDIDATES),
        "baseline_pooled_auc": baseline_auc,
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in files
        },
        "input_rows": int(len(data)),
        "feature_protocol": (
            "Raw numeric columns, fold-local one-hot categorical columns, "
            "and deterministic rowwise financial/time formulas; feature-family "
            "statistics are fit on outer training rows only."
        ),
        "included_origin_ledger_findings": [
            "application-process missingness patterns",
            "external-score disagreement and geometry",
            "fold-local applicant profile frequencies",
            "occupation-relative external-score percentile",
            "temporal invariant fingerprints",
            "credit/income nonlinear interaction",
            "Archimedean score-copula transforms",
        ],
        "excluded_prior_hypotheses": [
            "target encoding and default-label-derived statistics",
            "train/test identity pooling or score coupling",
            "130.5 fee-offset and 0.0066/0.0132 tariff-step product keys",
            "12-period income/annuity assumptions and Merton debt proxy",
            "Alexander repayment horizons with unverified period units",
            "bureau totals that sum counts across different time windows",
        ],
        "exploratory_formula_constants": {
            "clayton_theta": [1.2, 2.0],
            "gumbel_theta": 1.5,
            "status": "fixed prior hypotheses; tested only as a separate block",
        },
    }
    (output_dir / "PHASE27_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"RUN_COMPLETE {output_dir}", flush=True)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs"
        / f"phase27_composition_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}",
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)


if __name__ == "__main__":
    main()
