from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "phase10_catboost_product_robustness"
    ),
)
from phase10_independent_feature_builder import (
    build_product_keys,
    build_rowwise_features,
    categorical_columns,
)
from phase11_independent_runner import make_model, sha256_file


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_CONFIGS = (
    ("depth5_lr005", 5, 0.05),
    ("depth6_lr003", 6, 0.03),
    ("depth6_lr005", 6, 0.05),
)


def _one_hot_matrix(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[csr_matrix, csr_matrix]:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=2,
        dtype=np.float32,
    )
    fit_categorical = encoder.fit_transform(fit[categorical])
    apply_categorical = encoder.transform(apply[categorical])
    fit_numeric = fit[numeric].to_numpy(dtype=np.float32, copy=True)
    apply_numeric = apply[numeric].to_numpy(dtype=np.float32, copy=True)
    fit_numeric[~np.isfinite(fit_numeric)] = np.nan
    apply_numeric[~np.isfinite(apply_numeric)] = np.nan
    fit_matrix = hstack(
        [csr_matrix(fit_numeric), fit_categorical], format="csr"
    )
    apply_matrix = hstack(
        [csr_matrix(apply_numeric), apply_categorical], format="csr"
    )
    return fit_matrix, apply_matrix


def _fit_lightgbm_predict(
    frame: pd.DataFrame,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    y: np.ndarray,
    seed: int,
    iterations: int,
) -> np.ndarray:
    categorical = categorical_columns(frame)
    numeric = [column for column in frame.columns if column not in categorical]
    fit_matrix, apply_matrix = _one_hot_matrix(
        frame.iloc[fit_idx],
        frame.iloc[apply_idx],
        categorical,
        numeric,
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        device_type="cpu",
        n_jobs=4,
        verbosity=-1,
        random_state=seed,
    )
    model.fit(fit_matrix, y[fit_idx])
    return model.predict_proba(apply_matrix)[:, 1]


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def run(
    train_path: Path,
    output_dir: Path,
    outer_folds: int,
    inner_folds: int,
    iterations: int,
    seed: int,
    smoke_rows: int | None,
) -> None:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    train = pd.read_csv(train_path, low_memory=False)
    if smoke_rows is not None:
        if smoke_rows < outer_folds * inner_folds:
            raise ValueError("smoke_rows is too small for the requested folds")
        train, _ = train_test_split(
            train,
            train_size=smoke_rows,
            stratify=train["TARGET"],
            random_state=seed,
        )
        train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    y = train["TARGET"].to_numpy(dtype=np.int8)
    raw = train.drop(columns=["TARGET"])
    row_features = build_rowwise_features(raw)
    keys = build_product_keys(raw)
    product_features = pd.concat([row_features, keys], axis=1)
    predictions: dict[str, np.ndarray] = {
        name: np.full(len(train), np.nan, dtype=np.float32)
        for name in (
            "catboost_nested_tuned",
            "lightgbm_onehot_raw",
            "lightgbm_onehot_keys",
            "fixed_blend_05",
            "fixed_blend_10",
            "fixed_blend_20",
            "nested_logistic_stack",
        )
    }
    fold_assignments = np.full(len(train), -1, dtype=np.int16)
    selection_rows: list[dict[str, Any]] = []
    effective_ctr_config: dict[str, Any] = {}
    outer = StratifiedKFold(
        n_splits=outer_folds, shuffle=True, random_state=seed
    )

    for outer_fold, (outer_fit, outer_apply) in enumerate(
        outer.split(row_features, y), start=1
    ):
        fold_assignments[outer_apply] = outer_fold
        inner = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        inner_catboost = {
            name: np.full(len(outer_fit), np.nan, dtype=np.float32)
            for name, _, _ in DEFAULT_CONFIGS
        }
        inner_lightgbm = np.full(len(outer_fit), np.nan, dtype=np.float32)

        for inner_fold, (inner_fit_local, inner_apply_local) in enumerate(
            inner.split(outer_fit, y[outer_fit]), start=1
        ):
            inner_fit = outer_fit[inner_fit_local]
            inner_apply = outer_fit[inner_apply_local]
            for name, depth, learning_rate in DEFAULT_CONFIGS:
                model = make_model(
                    seed + outer_fold * 100 + inner_fold * 10,
                    iterations,
                    depth=depth,
                    learning_rate=learning_rate,
                )
                cat_columns = categorical_columns(product_features)
                model.fit(
                    product_features.iloc[inner_fit],
                    y[inner_fit],
                    cat_features=cat_columns,
                )
                inner_catboost[name][inner_apply_local] = (
                    model.predict_proba(
                        product_features.iloc[inner_apply]
                    )[:, 1]
                )
            inner_lightgbm[inner_apply_local] = _fit_lightgbm_predict(
                product_features,
                inner_fit,
                inner_apply,
                y,
                seed + outer_fold * 100 + inner_fold,
                iterations,
            )
            print(
                f"outer={outer_fold}/{outer_folds} "
                f"inner={inner_fold}/{inner_folds} completed",
                flush=True,
            )

        config_auc = {
            name: float(roc_auc_score(y[outer_fit], oof))
            for name, oof in inner_catboost.items()
        }
        selected_name = max(config_auc, key=config_auc.get)
        selected_config = next(
            config for config in DEFAULT_CONFIGS if config[0] == selected_name
        )
        selected_oof = inner_catboost[selected_name]
        if not np.isfinite(selected_oof).all() or not np.isfinite(
            inner_lightgbm
        ).all():
            raise AssertionError("Inner OOF predictions are incomplete")

        name, depth, learning_rate = selected_config
        final_catboost = make_model(
            seed + outer_fold * 1000,
            iterations,
            depth=depth,
            learning_rate=learning_rate,
        )
        cat_columns = categorical_columns(product_features)
        final_catboost.fit(
            product_features.iloc[outer_fit],
            y[outer_fit],
            cat_features=cat_columns,
        )
        if outer_fold == 1:
            effective_ctr_config = {
                key: value
                for key, value in final_catboost.get_all_params().items()
                if "ctr" in key.lower() or "one_hot" in key.lower()
            }
        cat_apply = final_catboost.predict_proba(
            product_features.iloc[outer_apply]
        )[:, 1]
        lgb_apply = _fit_lightgbm_predict(
            product_features,
            outer_fit,
            outer_apply,
            y,
            seed + outer_fold * 1000,
            iterations,
        )
        lgb_raw_apply = _fit_lightgbm_predict(
            row_features,
            outer_fit,
            outer_apply,
            y,
            seed + outer_fold * 1000,
            iterations,
        )
        stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        stacker.fit(
            np.column_stack((_logit(selected_oof), _logit(inner_lightgbm))),
            y[outer_fit],
        )
        stack_apply = stacker.predict_proba(
            np.column_stack((_logit(cat_apply), _logit(lgb_apply)))
        )[:, 1]

        predictions["catboost_nested_tuned"][outer_apply] = cat_apply
        predictions["lightgbm_onehot_raw"][outer_apply] = lgb_raw_apply
        predictions["lightgbm_onehot_keys"][outer_apply] = lgb_apply
        for weight in (0.05, 0.10, 0.20):
            predictions[f"fixed_blend_{int(weight * 100):02d}"][outer_apply] = (
                (1 - weight) * cat_apply + weight * lgb_apply
            )
        predictions["nested_logistic_stack"][outer_apply] = stack_apply
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "selected_config": selected_name,
                "inner_oof_config_auc": json.dumps(config_auc, sort_keys=True),
                "inner_lightgbm_auc": float(
                    roc_auc_score(y[outer_fit], inner_lightgbm)
                ),
                "outer_catboost_auc": float(
                    roc_auc_score(y[outer_apply], cat_apply)
                ),
                "outer_lightgbm_raw_auc": float(
                    roc_auc_score(y[outer_apply], lgb_raw_apply)
                ),
                "outer_lightgbm_auc": float(
                    roc_auc_score(y[outer_apply], lgb_apply)
                ),
            }
        )
        print(
            f"outer={outer_fold}/{outer_folds} selected={selected_name} "
            f"catboost_auc={selection_rows[-1]['outer_catboost_auc']:.6f} "
            f"lightgbm_auc={selection_rows[-1]['outer_lightgbm_auc']:.6f}",
            flush=True,
        )

    if np.any(fold_assignments < 0):
        raise AssertionError("Some rows have no outer OOF assignment")
    output = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int32),
            "fold": fold_assignments,
            "target": y,
            **predictions,
        }
    )
    prediction_path = output_dir / "nested_oof_predictions.csv"
    output.to_csv(prediction_path, index=False)
    fold_rows = []
    summary: dict[str, Any] = {}
    for name, values in predictions.items():
        pooled_auc = float(roc_auc_score(y, values))
        summary[f"{name}_pooled_oof_auc"] = pooled_auc
        summary[f"{name}_correlation_vs_catboost"] = float(
            pd.Series(values).corr(
                pd.Series(predictions["catboost_nested_tuned"]),
                method="spearman",
            )
        )
    for fold in range(1, outer_folds + 1):
        mask = fold_assignments == fold
        row: dict[str, Any] = {"fold": fold}
        for name, values in predictions.items():
            row[f"{name}_auc"] = float(roc_auc_score(y[mask], values[mask]))
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "inner_model_selection.csv", index=False
    )
    manifest = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "prediction_sha256": sha256_file(prediction_path),
        "rows": len(train),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "iterations": iterations,
        "smoke_run_only": smoke_rows is not None,
        "catboost_gpu": {
            "device": 0,
            "candidate_configs": DEFAULT_CONFIGS,
            "selected_per_outer_fold_using_inner_oof_only": True,
            "categorical_feature_cardinality": {
                column: int(product_features[column].nunique(dropna=False))
                for column in categorical_columns(product_features)
            },
            "effective_ctr_parameters_from_first_outer_model": effective_ctr_config,
            "uses_internal_target_statistics_for_high_cardinality_categories": True,
            "categorical_target_statistics_eligibility": "unresolved",
        },
        "lightgbm": {
            "device_type": "cpu",
            "categoricals": "fold-fitted OneHotEncoder; no target-rate encoding",
            "min_frequency": 2,
            "inner_only_preprocessing": True,
            "raw_and_eight_key_models_use_identical_outer_folds": True,
        },
        "nested_stacker": "LogisticRegression(C=1) fit on inner OOF logits only",
        "test_data_opened": False,
        "locked_holdout_used": False,
        "submission_created": False,
        "legacy_exp85_imported": False,
        "versions": {
            "catboost": __import__("catboost").__version__,
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("phase11_independent_runner.py"),
                Path(__file__).parents[1]
                / "phase10_catboost_product_robustness"
                / "phase10_independent_feature_builder.py",
            )
        },
        "results": summary,
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.iterations < 1:
        parser.error("fold counts must be >= 2 and iterations must be positive")
    run(
        args.train,
        args.output_dir,
        args.outer_folds,
        args.inner_folds,
        args.iterations,
        args.seed,
        args.smoke_rows,
    )


if __name__ == "__main__":
    main()
