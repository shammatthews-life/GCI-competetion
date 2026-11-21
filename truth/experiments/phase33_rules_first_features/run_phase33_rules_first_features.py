from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import sys

PHASE13 = Path(__file__).resolve().parents[1] / "phase13_matched_baseline"
sys.path.insert(0, str(PHASE13))
from phase13_features import build_raw_feature_blocks


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
MAX_ITERATIONS = 1600
STOPPING_ROUNDS = 100
KEY_COLUMNS = (
    "CLEAN_KEY_ANNUITY_CREDIT_3",
    "CLEAN_KEY_ANNUITY_CREDIT_4",
    "CLEAN_KEY_ANNUITY_CREDIT_5",
    "CLEAN_KEY_ANNUITY_GOODS_3",
)
VARIANTS = (
    "base",
    "financial",
    "peer",
    "keys",
    "keys_frequency",
    "all_safe",
)
CONFIG = {
    "num_leaves": 63,
    "min_child_samples": 100,
    "learning_rate": 0.03,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "max_bin": 127,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    values = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    np.divide(numerator, denominator, out=values, where=valid)
    return values


def _integer_key(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(values * (10**decimals))
    result = np.full(len(values), "__MISSING__", dtype=object)
    finite = np.isfinite(scaled)
    bounds = np.iinfo(np.int64)
    if np.any(finite & ((scaled < bounds.min) | (scaled > bounds.max))):
        raise ValueError("Rounded ratio key is outside int64 range")
    result[finite] = scaled[finite].astype(np.int64).astype(str)
    return result.astype(str)


def build_safe_blocks(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if TARGET in raw or ID not in raw:
        raise ValueError("Feature input must include ID and exclude TARGET")
    numeric, categorical, rowwise = build_raw_feature_blocks(raw)
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    income = _numeric(raw, "AMT_INCOME_TOTAL")
    family = _numeric(raw, "CNT_FAM_MEMBERS")
    financial = pd.DataFrame(
        {
            "SAFE_FIN_CREDIT_TO_INCOME": _ratio(credit, income),
            "SAFE_FIN_ANNUITY_TO_INCOME": _ratio(annuity, income),
            "SAFE_FIN_GOODS_TO_CREDIT": _ratio(goods, credit),
            "SAFE_FIN_CREDIT_TO_ANNUITY": _ratio(credit, annuity),
            "SAFE_FIN_INCOME_PER_FAMILY_MEMBER": _ratio(income, family),
            "SAFE_FIN_CREDIT_PER_FAMILY_MEMBER": _ratio(credit, family),
            "SAFE_FIN_ANNUITY_PER_FAMILY_MEMBER": _ratio(annuity, family),
        }
    )
    keys = pd.DataFrame(
        {
            KEY_COLUMNS[0]: _integer_key(_ratio(annuity, credit), 3),
            KEY_COLUMNS[1]: _integer_key(_ratio(annuity, credit), 4),
            KEY_COLUMNS[2]: _integer_key(_ratio(annuity, credit), 5),
            KEY_COLUMNS[3]: _integer_key(_ratio(annuity, goods), 3),
        }
    )
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(ext)
    count = observed.sum(axis=1)
    ext_mean = np.divide(
        np.nansum(ext, axis=1),
        count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=count > 0,
    )
    payment_rate = _ratio(annuity, credit)
    peer_rowwise = pd.DataFrame(
        {
            "SAFE_PAYMENT_RATE_X_EXT_MEAN": payment_rate * ext_mean,
        }
    )
    return (
        numeric.reset_index(drop=True),
        categorical.reset_index(drop=True),
        rowwise.reset_index(drop=True),
        financial,
        keys,
        peer_rowwise,
    )


def _peer_percentile(
    fit_values: np.ndarray,
    fit_groups: np.ndarray,
    apply_values: np.ndarray,
    apply_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_result = np.full(len(fit_values), 0.5, dtype=np.float32)
    apply_result = np.full(len(apply_values), 0.5, dtype=np.float32)
    fit_frame = pd.DataFrame(
        {"group": fit_groups.astype(str), "value": fit_values}
    )
    apply_group_values = apply_groups.astype(str)
    for group, positions in fit_frame.groupby("group", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        values = fit_values[positions]
        valid = np.isfinite(values)
        if not valid.any():
            continue
        valid_positions = positions[valid]
        valid_values = values[valid]
        fit_result[valid_positions] = (
            pd.Series(valid_values).rank(method="average", pct=True).to_numpy(
                dtype=np.float32
            )
        )
        reference = np.sort(valid_values)
        apply_positions = np.flatnonzero(apply_group_values == group)
        candidates = apply_values[apply_positions]
        apply_valid = np.isfinite(candidates)
        if apply_valid.any():
            left = np.searchsorted(reference, candidates[apply_valid], side="left")
            right = np.searchsorted(reference, candidates[apply_valid], side="right")
            apply_result[apply_positions[apply_valid]] = (
                (left + right) / (2.0 * len(reference))
            ).astype(np.float32)
    return fit_result, apply_result


def _peer_feature_pair(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(ext)
    count = observed.sum(axis=1)
    ext_mean = np.divide(
        np.nansum(ext, axis=1),
        count,
        out=np.full(len(ext), np.nan, dtype=np.float64),
        where=count > 0,
    )
    groups = (
        raw["OCCUPATION_TYPE"].astype("string").fillna("__MISSING__").astype(str)
    ).to_numpy()
    fit, apply = _peer_percentile(
        ext_mean[fit_rows],
        groups[fit_rows],
        ext_mean[apply_rows],
        groups[apply_rows],
    )
    return fit, apply


def _numeric_matrix(frame: pd.DataFrame) -> csr_matrix:
    values = frame.to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return csr_matrix(values)


def make_matrices(
    raw: pd.DataFrame,
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    variant: str,
) -> tuple[csr_matrix, csr_matrix]:
    numeric, categorical, rowwise, financial, keys, peer_rowwise = blocks
    use_financial = variant in {"financial", "all_safe"}
    use_peer = variant in {"peer", "all_safe"}
    use_keys = variant in {"keys", "keys_frequency", "all_safe"}
    use_frequency = variant in {"keys_frequency", "all_safe"}
    fit_numeric = pd.concat(
        [
            numeric.iloc[fit_rows].reset_index(drop=True),
            rowwise.iloc[fit_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    apply_numeric = pd.concat(
        [
            numeric.iloc[apply_rows].reset_index(drop=True),
            rowwise.iloc[apply_rows].reset_index(drop=True),
        ],
        axis=1,
    )
    if use_financial:
        fit_numeric = pd.concat(
            [fit_numeric, financial.iloc[fit_rows].reset_index(drop=True)], axis=1
        )
        apply_numeric = pd.concat(
            [apply_numeric, financial.iloc[apply_rows].reset_index(drop=True)],
            axis=1,
        )
    if use_peer:
        fit_peer = peer_rowwise.iloc[fit_rows].reset_index(drop=True).copy()
        apply_peer = peer_rowwise.iloc[apply_rows].reset_index(drop=True).copy()
        fit_rank, apply_rank = _peer_feature_pair(raw, fit_rows, apply_rows)
        fit_peer["SAFE_EXT_SOURCE_PEER_PERCENTILE"] = fit_rank
        apply_peer["SAFE_EXT_SOURCE_PEER_PERCENTILE"] = apply_rank
        fit_numeric = pd.concat([fit_numeric, fit_peer], axis=1)
        apply_numeric = pd.concat([apply_numeric, apply_peer], axis=1)

    fit_blocks = [_numeric_matrix(fit_numeric)]
    apply_blocks = [_numeric_matrix(apply_numeric)]
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        dtype=np.float32,
        sparse_output=True,
    )
    fit_cat = encoder.fit_transform(categorical.iloc[fit_rows].astype(str)).tocsr()
    apply_cat = encoder.transform(categorical.iloc[apply_rows].astype(str)).tocsr()
    fit_blocks.append(fit_cat)
    apply_blocks.append(apply_cat)

    if use_keys:
        key_encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2 if use_frequency else None,
            dtype=np.float32,
            sparse_output=True,
        )
        fit_key_values = keys.iloc[fit_rows].astype(str)
        apply_key_values = keys.iloc[apply_rows].astype(str)
        fit_key = key_encoder.fit_transform(fit_key_values).tocsr()
        apply_key = key_encoder.transform(apply_key_values).tocsr()
        fit_blocks.append(fit_key)
        apply_blocks.append(apply_key)
        if use_frequency:
            fit_counts: list[np.ndarray] = []
            apply_counts: list[np.ndarray] = []
            for column in KEY_COLUMNS:
                counts = fit_key_values[column].value_counts()
                fit_counts.append(
                    fit_key_values[column]
                    .map(counts)
                    .to_numpy(dtype=np.float32)
                )
                apply_counts.append(
                    apply_key_values[column]
                    .map(counts)
                    .fillna(0)
                    .to_numpy(dtype=np.float32)
                )
            fit_blocks.append(
                csr_matrix(np.column_stack(fit_counts), dtype=np.float32)
            )
            apply_blocks.append(
                csr_matrix(np.column_stack(apply_counts), dtype=np.float32)
            )
    return (
        hstack(fit_blocks, format="csr", dtype=np.float32),
        hstack(apply_blocks, format="csr", dtype=np.float32),
    )


def _model(seed: int, iterations: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        num_leaves=CONFIG["num_leaves"],
        min_child_samples=CONFIG["min_child_samples"],
        learning_rate=CONFIG["learning_rate"],
        colsample_bytree=CONFIG["colsample_bytree"],
        subsample=CONFIG["subsample"],
        subsample_freq=CONFIG["subsample_freq"],
        reg_lambda=CONFIG["reg_lambda"],
        max_bin=CONFIG["max_bin"],
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must have unique IDs and TARGET")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, (0, 1)).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    predictions = {
        variant: np.full(len(y), np.nan, dtype=np.float32)
        for variant in VARIANTS
    }
    records: list[dict[str, Any]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_seed = SEED + fold_number * 100
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=0.12, random_state=fold_seed
        )
        inner_fit_idx, inner_valid_idx = next(
            inner.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_idx]
        inner_valid = fit_rows[inner_valid_idx]
        for variant in VARIANTS:
            x_inner_fit, x_inner_valid = make_matrices(
                raw, blocks, inner_fit, inner_valid, variant
            )
            selector = _model(fold_seed, MAX_ITERATIONS)
            selector.fit(
                x_inner_fit,
                y[inner_fit],
                eval_X=x_inner_valid,
                eval_y=y[inner_valid],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            iterations = max(
                100, int(selector.best_iteration_ or MAX_ITERATIONS)
            )
            del selector, x_inner_fit, x_inner_valid
            x_fit, x_valid = make_matrices(
                raw, blocks, fit_rows, valid_rows, variant
            )
            model = _model(fold_seed, iterations)
            model.fit(x_fit, y[fit_rows])
            prediction = model.predict_proba(x_valid)[:, 1]
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"Nonfinite prediction for {variant}, fold {fold_number}"
                )
            predictions[variant][valid_rows] = prediction.astype(np.float32)
            auc = float(roc_auc_score(y[valid_rows], prediction))
            records.append(
                {
                    "variant": variant,
                    "fold": fold_number,
                    "auc": auc,
                    "iterations": iterations,
                    "fit_rows": int(len(fit_rows)),
                    "validation_rows": int(len(valid_rows)),
                }
            )
            print(
                f"variant={variant} fold={fold_number}/{FOLDS} "
                f"AUC={auc:.6f} iterations={iterations}",
                flush=True,
            )
    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE33_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE33_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE33_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    oof = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": np.full(len(y), -1, dtype=np.int8),
            **{f"oof_{name}": values for name, values in predictions.items()},
        }
    )
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        oof.loc[valid_rows, "fold"] = fold_number
    if oof.isna().any().any():
        raise ValueError("OOF artifact contains missing values")
    oof.to_csv(oof_path, index=False)
    result_metrics: dict[str, dict[str, float]] = {}
    for variant, values in predictions.items():
        result_metrics[variant] = {
            "pooled_auc": float(roc_auc_score(y, values)),
            "mean_fold_auc": float(
                pd.DataFrame(records).query("variant == @variant")["auc"].mean()
            ),
            "log_loss": float(log_loss(y, values)),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 33 rules-first feature screen",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "model": {
            "family": "LightGBM binary classifier",
            "categorical_representation": (
                "one-hot encoders fitted separately on each model-fit partition"
            ),
            "parameters": CONFIG,
            "max_iterations": MAX_ITERATIONS,
            "early_stopping_rounds": STOPPING_ROUNDS,
            "inner_validation_fraction": 0.12,
        },
        "metrics": result_metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
            "outer_validation_labels": (
                "used only for final fold scoring; not feature fitting, stopping, "
                "or model selection"
            ),
            "label_free_peer_feature": (
                "fit-fold occupation percentile of rowwise external-score mean"
            ),
        },
        "feature_hypotheses": {
            "financial": "seven guarded EXP02 formulas from supplied fields",
            "keys": "rounded annuity/credit and annuity/goods ratios at fixed decimal precision",
            "key_frequency": (
                "fit-partition-only counts and one-hot rare grouping; no target values"
            ),
        },
        "metrics_by_variant": result_metrics,
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                PHASE13 / "phase13_features.py",
                fold_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase33_rules_first_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
