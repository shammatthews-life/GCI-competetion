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

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "phase10_catboost_product_robustness"
    ),
)
from phase10_independent_feature_builder import build_product_keys
from phase11_candidate_search import (
    Candidate,
    _candidate,
    crossfit_smoothed_target_rate,
    significant_integer_codes,
)
from phase11_independent_runner import sha256_file


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")


def _ratio_candidate(
    raw: pd.DataFrame, numerator: str, denominator: str, digits: int
) -> Candidate:
    top = pd.to_numeric(raw[numerator], errors="raise").to_numpy(
        dtype=np.float64
    )
    bottom = pd.to_numeric(raw[denominator], errors="raise").to_numpy(
        dtype=np.float64
    )
    ratio = np.full(len(raw), np.nan, dtype=np.float64)
    valid = np.isfinite(top) & np.isfinite(bottom) & (bottom != 0)
    np.divide(top, bottom, out=ratio, where=valid)
    return _candidate(
        f"ratio:{numerator}:{denominator}:sig{digits}",
        (significant_integer_codes(ratio, digits),),
        "predeclared_numeric_ratio",
    )


def run(
    train_path: Path,
    predictions_path: Path,
    output_dir: Path,
    smoothing: float,
    null_repetitions: int,
    seed: int,
) -> None:
    if not train_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError("Training CSV or OOF predictions are missing")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc)
    train = pd.read_csv(train_path, low_memory=False)
    predictions = pd.read_csv(predictions_path)
    if len(train) != len(predictions):
        raise ValueError("Training rows and OOF prediction rows differ")
    if not np.array_equal(
        predictions["row_index"].to_numpy(), np.arange(len(train))
    ):
        raise ValueError("OOF predictions are not aligned to training row order")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    if not np.array_equal(y, predictions["target"].to_numpy(dtype=np.int8)):
        raise ValueError("OOF targets do not match the source training labels")
    baseline = predictions["product_keys_bagged"].to_numpy(dtype=np.float64)
    folds = tuple(
        (
            np.flatnonzero(predictions["fold"].to_numpy() != fold),
            np.flatnonzero(predictions["fold"].to_numpy() == fold),
        )
        for fold in sorted(predictions["fold"].unique())
    )
    raw = train.drop(columns=["TARGET"])
    keys = build_product_keys(raw)
    candidates: list[Candidate] = []
    for column in keys.columns:
        codes, _ = pd.factorize(
            keys[column].astype("string").fillna("MISSING"),
            sort=True,
            use_na_sentinel=False,
        )
        candidates.append(
            Candidate(
                name=f"product_key:{column}",
                category_codes=codes.astype(np.int32),
                family="predeclared_product_key",
                unique_categories=int(np.unique(codes).size),
            )
        )
    for numerator, denominator in (
        ("AMT_CREDIT", "AMT_ANNUITY"),
        ("AMT_ANNUITY", "AMT_GOODS_PRICE"),
    ):
        for digits in (3, 4, 5):
            candidates.append(
                _ratio_candidate(raw, numerator, denominator, digits)
            )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        encoded = crossfit_smoothed_target_rate(
            candidate.category_codes, y, folds, smoothing
        )
        row: dict[str, Any] = {
            "candidate": candidate.name,
            "family": candidate.family,
            "unique_categories": candidate.unique_categories,
            "target_rate_auc": float(roc_auc_score(y, encoded)),
            "target_rate_spearman_vs_product_model": float(
                pd.Series(encoded).corr(
                    pd.Series(baseline), method="spearman"
                )
            ),
        }
        for weight in (0.05, 0.10, 0.20):
            blend = (1 - weight) * baseline + weight * encoded
            row[f"blend_{weight:.2f}_auc"] = float(
                roc_auc_score(y, blend)
            )
            row[f"blend_{weight:.2f}_delta_vs_product_model"] = (
                row[f"blend_{weight:.2f}_auc"]
                - roc_auc_score(y, baseline)
            )
        results.append(row)

    rng = np.random.default_rng(seed)
    null_rows: list[dict[str, Any]] = []
    for key_name in (
        "PRODUCT_KEY_Q4",
        "PRODUCT_KEY_KQ4",
        "PRODUCT_KEY_P4",
        "PRODUCT_KEY_P5",
    ):
        source = next(
            candidate
            for candidate in candidates
            if candidate.name == f"product_key:{key_name}"
        )
        for replicate in range(null_repetitions):
            null = Candidate(
                name=f"shuffled:{key_name}:rep{replicate + 1}",
                category_codes=rng.permutation(source.category_codes),
                family="frequency_preserving_shuffled_key",
                unique_categories=source.unique_categories,
            )
            encoded = crossfit_smoothed_target_rate(
                null.category_codes, y, folds, smoothing
            )
            blend = 0.9 * baseline + 0.1 * encoded
            null_rows.append(
                {
                    "candidate": null.name,
                    "source_key": key_name,
                    "blend_0.10_auc": float(roc_auc_score(y, blend)),
                    "delta_vs_product_model": float(
                        roc_auc_score(y, blend) - roc_auc_score(y, baseline)
                    ),
                }
            )

    results_frame = pd.DataFrame(results).sort_values(
        "blend_0.10_auc", ascending=False
    )
    results_frame.to_csv(output_dir / "product_baseline_target_rates.csv", index=False)
    null_frame = pd.DataFrame(null_rows)
    null_frame.to_csv(
        output_dir / "frequency_preserving_nulls_vs_product.csv", index=False
    )
    null_summary = null_frame.groupby("source_key")["delta_vs_product_model"].agg(
        ["mean", "max", "count"]
    )
    baseline_auc = float(roc_auc_score(y, baseline))
    manifest = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "oof_predictions_path": str(predictions_path),
        "oof_predictions_sha256": sha256_file(predictions_path),
        "rows": len(train),
        "baseline_column": "product_keys_bagged",
        "baseline_oof_auc": baseline_auc,
        "candidate_count": len(candidates),
        "smoothing": smoothing,
        "folds": len(folds),
        "null_repetitions_per_key": null_repetitions,
        "adaptive_selection_is_exploratory": True,
        "target_encoding_eligibility": "unresolved; research-only",
        "test_data_opened": False,
        "locked_holdout_used": False,
        "source_sha256": {
            str(path): sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("phase11_candidate_search.py"),
                Path(__file__).with_name("phase11_independent_runner.py"),
            )
        },
        "best_fixed_blend_10": results_frame.iloc[0].to_dict(),
        "frequency_preserving_null_summary": json.loads(
            null_summary.reset_index().to_json(orient="records")
        ),
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--null-repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()
    if args.smoothing <= 0 or args.null_repetitions < 1:
        parser.error("smoothing and null repetitions must be positive")
    run(
        args.train,
        args.predictions,
        args.output_dir,
        args.smoothing,
        args.null_repetitions,
        args.seed,
    )


if __name__ == "__main__":
    main()
