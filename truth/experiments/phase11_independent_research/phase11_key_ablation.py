from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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


def _variant_frames(
    row_features: pd.DataFrame, keys: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    variants = {"raw": row_features}
    variants["all_keys"] = pd.concat([row_features, keys], axis=1)
    for key in keys.columns:
        variants[f"single__{key}"] = pd.concat(
            [row_features, keys[[key]]], axis=1
        )
        remaining = keys.drop(columns=[key])
        variants[f"leave_out__{key}"] = pd.concat(
            [row_features, remaining], axis=1
        )
    return variants


def run(
    train_path: Path,
    output_dir: Path,
    folds_count: int,
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
        if smoke_rows < folds_count * 2:
            raise ValueError("smoke_rows must be at least twice the fold count")
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
    variants = _variant_frames(row_features, keys)
    predictions = {
        name: np.full(len(train), np.nan, dtype=np.float32)
        for name in variants
    }
    fold_assignments = np.full(len(train), -1, dtype=np.int16)
    splitter = StratifiedKFold(
        n_splits=folds_count, shuffle=True, random_state=seed
    )
    fold_rows: list[dict[str, Any]] = []
    fold_sizes: dict[str, dict[str, int]] = {}

    for fold, (fit_idx, apply_idx) in enumerate(splitter.split(row_features, y), 1):
        fold_assignments[apply_idx] = fold
        row: dict[str, Any] = {"fold": fold}
        for variant, frame in variants.items():
            model = make_model(
                seed + fold * 100,
                iterations,
            )
            model.fit(
                frame.iloc[fit_idx],
                y[fit_idx],
                cat_features=categorical_columns(frame),
            )
            prediction = model.predict_proba(frame.iloc[apply_idx])[:, 1]
            predictions[variant][apply_idx] = prediction
            row[f"{variant}_auc"] = float(
                roc_auc_score(y[apply_idx], prediction)
            )
            print(
                f"fold={fold}/{folds_count} variant={variant} "
                f"task_type=GPU",
                flush=True,
            )
        fold_rows.append(row)
        fold_frame = pd.DataFrame(
            {
                "row_index": apply_idx,
                "fold": fold,
                "target": y[apply_idx],
                **{
                    name: values[apply_idx]
                    for name, values in predictions.items()
                },
            }
        )
        fold_frame.to_csv(
            output_dir / f"fold_{fold}_oof_predictions.csv", index=False
        )
        fold_sizes[str(fold)] = {
            column: int(keys.iloc[fit_idx][column].nunique())
            for column in keys.columns
        }

    if np.any(fold_assignments < 0):
        raise AssertionError("Some rows have no outer OOF assignment")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise AssertionError("Some variants have missing OOF predictions")
    pooled = {
        name: float(roc_auc_score(y, values))
        for name, values in predictions.items()
    }
    baseline_auc = pooled["raw"]
    summary = {
        "rows": len(train),
        "variant_count": len(variants),
        "iterations": iterations,
        "smoke_run_only": smoke_rows is not None,
        "folds": folds_count,
        "seed": seed,
        "pooled_oof_auc": pooled,
        "delta_vs_raw_baseline": {
            name: score - baseline_auc for name, score in pooled.items()
        },
        "key_unique_counts_full_train": {
            column: int(keys[column].nunique()) for column in keys.columns
        },
        "key_unique_counts_by_fold_fit": fold_sizes,
        "adaptive_ablation_selection_is_exploratory": True,
    }
    all_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(train), dtype=np.int32),
            "fold": fold_assignments,
            "target": y,
            **predictions,
        }
    )
    prediction_path = output_dir / "oof_predictions.csv"
    all_predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    manifest = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "oof_sha256": sha256_file(prediction_path),
        "rows": len(train),
        "folds": folds_count,
        "seed": seed,
        "iterations": iterations,
        "model": {
            "task_type": "GPU",
            "device": 0,
            "depth": 6,
            "learning_rate": 0.05,
        },
        "features": {
            "independent_builder": True,
            "product_keys": keys.columns.tolist(),
            "target_excluded_from_feature_construction": True,
            "identifier_excluded_from_model_features": True,
        },
        "test_data_opened": False,
        "locked_holdout_used": False,
        "submission_created": False,
        "legacy_exp85_imported": False,
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
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-rows", type=int)
    args = parser.parse_args()
    if args.folds < 2 or args.iterations < 1:
        parser.error("folds must be >= 2 and iterations must be positive")
    run(
        args.train,
        args.output_dir,
        args.folds,
        args.iterations,
        args.seed,
        args.smoke_rows,
    )


if __name__ == "__main__":
    main()
