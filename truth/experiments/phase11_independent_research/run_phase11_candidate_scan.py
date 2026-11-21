from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    crossfit_smoothed_target_rate,
    generate_candidates,
    score_candidate,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frequency_rate(
    codes: np.ndarray, folds: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> np.ndarray:
    output = np.full(len(codes), np.nan, dtype=np.float32)
    n_categories = int(codes.max()) + 1
    for fit_idx, apply_idx in folds:
        counts = np.bincount(codes[fit_idx], minlength=n_categories)
        output[apply_idx] = np.log1p(counts[codes[apply_idx]])
    if not np.isfinite(output).all():
        raise AssertionError("Frequency feature has incomplete fold coverage")
    return output


def _key_candidates(raw: pd.DataFrame) -> list[Candidate]:
    keys = build_product_keys(raw)
    output = []
    for column in keys.columns:
        codes, _ = pd.factorize(
            keys[column].astype("string").fillna("MISSING"),
            sort=True,
            use_na_sentinel=False,
        )
        output.append(
            Candidate(
                name=f"product_key:{column}",
                category_codes=codes.astype(np.int32),
                family="predeclared_product_key",
                unique_categories=int(np.unique(codes).size),
            )
        )
    return output


def run(
    train_path: Path,
    predictions_path: Path,
    output_dir: Path,
    smoothing: float,
    top_blends: int,
    null_repetitions: int,
    seed: int,
) -> None:
    started_at = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=False)
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
    baseline = predictions["raw_bagged"].to_numpy(dtype=np.float64)
    folds = tuple(
        (
            np.flatnonzero(predictions["fold"].to_numpy() != fold),
            np.flatnonzero(predictions["fold"].to_numpy() == fold),
        )
        for fold in sorted(predictions["fold"].unique())
    )
    raw = train.drop(columns=["TARGET"])
    results: dict[str, dict[str, object]] = {}
    processed: set[str] = set()

    def record(candidate: Candidate, blend_weights: tuple[float, ...]) -> None:
        row = score_candidate(
            candidate,
            y,
            baseline,
            folds,
            smoothing=smoothing,
            blend_weights=blend_weights,
        )
        results.setdefault(candidate.name, {}).update(row)
        processed.add(candidate.name)

    for index, candidate in enumerate(generate_candidates(raw), start=1):
        record(candidate, (0.10,))
        if index % 50 == 0:
            pd.DataFrame(results.values()).to_csv(
                output_dir / "candidate_scan_checkpoint.csv", index=False
            )
            print(f"screened {index} candidates", flush=True)

    explicit_keys = _key_candidates(raw)
    for candidate in explicit_keys:
        if candidate.name not in processed:
            record(candidate, (0.10,))

    ranked = pd.DataFrame(results.values()).sort_values(
        "target_rate_oof_auc", ascending=False
    )
    selected_names = set(ranked.head(top_blends)["candidate"].astype(str))
    selected_names.update(candidate.name for candidate in explicit_keys)
    for candidate in generate_candidates(raw):
        if candidate.name in selected_names:
            record(candidate, (0.05, 0.20))

    frequency_rows = _label_free_frequency_rows(
        explicit_keys, y, baseline, folds
    )

    rng = np.random.default_rng(seed)
    controls: list[dict[str, object]] = []
    positive_controls = [
        candidate
        for candidate in explicit_keys
        if candidate.name in {
            "product_key:PRODUCT_KEY_Q4",
            "product_key:PRODUCT_KEY_KQ4",
            "product_key:PRODUCT_KEY_P4",
            "product_key:PRODUCT_KEY_AG3",
        }
    ]
    for candidate in positive_controls:
        for replicate in range(null_repetitions):
            shuffled_codes = rng.permutation(candidate.category_codes)
            null_candidate = Candidate(
                name=f"shuffled:{candidate.name}:rep{replicate + 1}",
                category_codes=shuffled_codes,
                family="frequency_preserving_shuffled_key",
                unique_categories=candidate.unique_categories,
            )
            controls.append(
                score_candidate(
                    null_candidate,
                    y,
                    baseline,
                    folds,
                    smoothing=smoothing,
                    blend_weights=(0.10,),
                )
            )
        print(f"completed shuffled-key controls for {candidate.name}", flush=True)

    all_results = pd.DataFrame(results.values())
    if "fixed_blend_0.05_auc" not in all_results:
        all_results["fixed_blend_0.05_auc"] = np.nan
        all_results["fixed_blend_0.05_delta"] = np.nan
    if "fixed_blend_0.20_auc" not in all_results:
        all_results["fixed_blend_0.20_auc"] = np.nan
        all_results["fixed_blend_0.20_delta"] = np.nan
    all_results.to_csv(output_dir / "candidate_scan.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(
        output_dir / "label_free_frequency.csv", index=False
    )
    pd.DataFrame(controls).to_csv(
        output_dir / "frequency_preserving_controls.csv", index=False
    )
    summary = {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "train_sha256": _sha256_file(train_path),
        "oof_predictions_path": str(predictions_path),
        "oof_predictions_sha256": _sha256_file(predictions_path),
        "source_sha256": {
            str(path): _sha256_file(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("phase11_candidate_search.py"),
                Path(__file__).parents[1]
                / "phase10_catboost_product_robustness"
                / "phase10_independent_feature_builder.py",
            )
        },
        "train_rows": len(train),
        "baseline_column": "raw_bagged",
        "baseline_oof_auc": float(roc_auc_score(y, baseline)),
        "candidate_count": len(all_results),
        "target_encoding_smoothing": smoothing,
        "crossfit_fold_count": len(folds),
        "frequency_preserving_null_repetitions_per_product_key": null_repetitions,
        "all_data_source_is_train_csv": True,
        "test_data_opened": False,
        "locked_holdout_used": False,
        "adaptive_search_is_exploratory_not_confirmatory": True,
        "target_encoding_eligibility": "unresolved; do not promote for submission",
        "best_target_rate_candidates": all_results.sort_values(
            "fixed_blend_0.10_auc", ascending=False
        )
        .head(20)[
            [
                "candidate",
                "family",
                "unique_categories",
                "target_rate_oof_auc",
                "fixed_blend_0.10_auc",
                "fixed_blend_0.10_delta",
            ]
        ]
        .to_dict(orient="records"),
    }
    (output_dir / "SCAN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def _label_free_frequency_rows(
    candidates: list[Candidate],
    y: np.ndarray,
    baseline: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        frequency = _frequency_rate(candidate.category_codes, folds)
        rows.append(
            {
                "candidate": candidate.name,
                "family": "label_free_fold_frequency",
                "unique_categories": candidate.unique_categories,
                "frequency_auc": float(roc_auc_score(y, frequency)),
                "frequency_spearman_vs_raw_baseline": float(
                    pd.Series(frequency).corr(
                        pd.Series(baseline), method="spearman"
                    )
                ),
                "frequency_score_min": float(frequency.min()),
                "frequency_score_max": float(frequency.max()),
            }
        )
    return rows


def recompute_label_free_frequency(
    train_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> None:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Existing scan output directory not found: {output_dir}")
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
    folds = tuple(
        (
            np.flatnonzero(predictions["fold"].to_numpy() != fold),
            np.flatnonzero(predictions["fold"].to_numpy() == fold),
        )
        for fold in sorted(predictions["fold"].unique())
    )
    keys = _key_candidates(train.drop(columns=["TARGET"]))
    rows = _label_free_frequency_rows(
        keys,
        y,
        predictions["raw_bagged"].to_numpy(dtype=np.float64),
        folds,
    )
    path = output_dir / "label_free_frequency_recomputed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(
        json.dumps(
            {
                "rows": len(train),
                "keys": len(rows),
                "output": str(path),
                "interpretation": "standalone label-free frequency diagnostic; not a probability blend",
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoothing", type=float, default=30.0)
    parser.add_argument("--top-blends", type=int, default=100)
    parser.add_argument("--null-repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--frequency-only", action="store_true")
    args = parser.parse_args()
    if args.frequency_only:
        recompute_label_free_frequency(
            args.train, args.predictions, args.output_dir
        )
        return
    if args.top_blends < 1 or args.null_repetitions < 1:
        parser.error("top-blends and null-repetitions must be positive")
    run(
        args.train,
        args.predictions,
        args.output_dir,
        args.smoothing,
        args.top_blends,
        args.null_repetitions,
        args.seed,
    )


if __name__ == "__main__":
    main()
