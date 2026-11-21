from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase13_features import PRODUCT_KEYS, build_product_keys


ROOT = Path(__file__).resolve().parent
PHASE11 = ROOT.parent / "phase11_independent_research"
sys.path.insert(0, str(PHASE11))
from phase11_candidate_search import (  # noqa: E402
    Candidate,
    NUMERIC_FIELDS,
    crossfit_smoothed_target_rate,
    generate_candidates,
    significant_integer_codes,
    score_candidate,
)


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_PHASE13_RUNS = ROOT / "runs"
SMOOTHING = 30.0
BLEND_WEIGHTS = (0.05, 0.10, 0.20)
NULL_REPETITIONS = 10
NULL_SEED = 20261127


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_candidates(raw: pd.DataFrame) -> list[Candidate]:
    keys = build_product_keys(raw)
    candidates = []
    for column in PRODUCT_KEYS:
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
    return candidates


def screen_candidates(raw: pd.DataFrame):
    yield from generate_candidates(raw)
    numeric = {
        column: pd.to_numeric(raw[column], errors="raise").to_numpy(dtype=np.float64)
        for column in NUMERIC_FIELDS
    }
    for left_name, right_name in itertools.combinations(NUMERIC_FIELDS, 2):
        numerator = numeric[right_name]
        denominator = numeric[left_name]
        values = np.full(len(numerator), np.nan, dtype=np.float64)
        valid = (
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator != 0)
        )
        np.divide(numerator, denominator, out=values, where=valid)
        for digits in (3, 4, 5):
            rounded = significant_integer_codes(values, digits)
            codes, _ = pd.factorize(rounded, sort=False, use_na_sentinel=False)
            yield Candidate(
                name=f"pair:{right_name}:ratio:{left_name}:sig{digits}",
                category_codes=codes.astype(np.int32),
                family="numeric_pair_ratio",
                unique_categories=int(np.unique(codes).size),
            )


def run(train_path: Path, phase13_oof_path: Path, output_root: Path) -> Path:
    start = datetime.now(timezone.utc)
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    if not phase13_oof_path.is_file():
        raise FileNotFoundError(f"Phase 13 OOF file not found: {phase13_oof_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"hidden_key_screen_{start.strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(exist_ok=False)
    (output / "EXECUTION_LOG.txt").write_text(
        f"Started UTC: {start.isoformat()}\n", encoding="utf-8"
    )
    input_hash = sha256_file(train_path)
    oof_hash = sha256_file(phase13_oof_path)
    train = pd.read_csv(train_path, low_memory=False)
    oof = pd.read_csv(phase13_oof_path)
    expected_index = np.arange(len(train))
    if len(train) != len(oof) or not np.array_equal(
        oof["row_index"].to_numpy(), expected_index
    ):
        raise ValueError("Phase 13 OOF rows do not align with training row order")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    if not np.array_equal(y, oof["TARGET"].to_numpy(dtype=np.int8)):
        raise ValueError("Phase 13 OOF targets differ from the training labels")
    if oof["primary_fold"].isna().any() or set(oof["primary_fold"].unique()) != {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError("Phase 13 OOF must contain all five primary folds")
    baseline = oof["C_complete_raw_rowwise"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("Phase 13 C baseline OOF predictions are incomplete")
    fold_ids = oof["primary_fold"].to_numpy(dtype=np.int8)
    folds = tuple(
        (
            np.flatnonzero(fold_ids != fold),
            np.flatnonzero(fold_ids == fold),
        )
        for fold in range(1, 6)
    )
    raw = train.drop(columns=["TARGET"])
    results: dict[str, dict[str, object]] = {}
    processed: set[str] = set()
    candidate_count = 0

    def record(candidate: Candidate, weights: tuple[float, ...]) -> None:
        row = score_candidate(
            candidate,
            y,
            baseline,
            folds,
            smoothing=SMOOTHING,
            blend_weights=weights,
        )
        results.setdefault(candidate.name, {}).update(row)
        processed.add(candidate.name)

    started_search = time.perf_counter()
    for candidate_count, candidate in enumerate(screen_candidates(raw), start=1):
        record(candidate, (0.10,))
        if candidate_count % 50 == 0:
            pd.DataFrame(results.values()).to_csv(
                output / "candidate_scan_checkpoint.csv", index=False
            )
            with (output / "EXECUTION_LOG.txt").open("a", encoding="utf-8") as log:
                log.write(f"Screened {candidate_count} candidates\n")

    explicit_keys = key_candidates(raw)
    for candidate in explicit_keys:
        if candidate.name not in processed:
            record(candidate, (0.10,))

    ranked = pd.DataFrame(results.values()).sort_values(
        "fixed_blend_0.10_auc", ascending=False
    )
    selected_names = set(ranked.head(20)["candidate"].astype(str))
    selected_names.update(candidate.name for candidate in explicit_keys)
    for candidate in screen_candidates(raw):
        if candidate.name in selected_names:
            record(candidate, (0.05, 0.20))
    for candidate in explicit_keys:
        if candidate.name in selected_names:
            record(candidate, (0.05, 0.20))

    rng = np.random.default_rng(NULL_SEED)
    controls = []
    positive_controls = [
        candidate
        for candidate in explicit_keys
        if candidate.name
        in {
            "product_key:PRODUCT_KEY_Q4",
            "product_key:PRODUCT_KEY_KQ4",
            "product_key:PRODUCT_KEY_P4",
            "product_key:PRODUCT_KEY_AG3",
        }
    ]
    for candidate in positive_controls:
        for repetition in range(1, NULL_REPETITIONS + 1):
            null_candidate = Candidate(
                name=f"shuffled:{candidate.name}:rep{repetition}",
                category_codes=rng.permutation(candidate.category_codes),
                family="frequency_preserving_shuffled_key",
                unique_categories=candidate.unique_categories,
            )
            controls.append(
                score_candidate(
                    null_candidate,
                    y,
                    baseline,
                    folds,
                    smoothing=SMOOTHING,
                    blend_weights=(0.10,),
                )
            )

    all_results = pd.DataFrame(results.values()).sort_values(
        "fixed_blend_0.10_auc", ascending=False
    )
    all_results.to_csv(output / "HIDDEN_KEY_CANDIDATE_RESULTS.csv", index=False)
    pd.DataFrame(controls).to_csv(
        output / "FREQUENCY_PRESERVING_NULLS.csv", index=False
    )
    key_oof = pd.DataFrame(
        {
            "row_index": expected_index,
            "SK_ID_CURR": train["SK_ID_CURR"],
            "TARGET": y,
            "primary_fold": fold_ids,
            "baseline_C_oof": baseline,
        }
    )
    for candidate in explicit_keys:
        key_oof[candidate.name] = crossfit_smoothed_target_rate(
            candidate.category_codes, y, folds, SMOOTHING
        )
    key_oof.to_csv(output / "PRODUCT_KEY_CROSSFIT_RATE_OOF.csv", index=False)

    finished = datetime.now(timezone.utc)
    if sha256_file(train_path) != input_hash or sha256_file(phase13_oof_path) != oof_hash:
        raise AssertionError("An input file changed during the screen")
    if len(all_results) != len(results) or all_results["candidate"].duplicated().any():
        raise AssertionError("Candidate results contain missing or duplicate entries")
    baseline_auc = float(roc_auc_score(y, baseline))
    best_blend = all_results.iloc[0]
    top_numeric_pairs = all_results[
        all_results["family"].astype(str).str.startswith("numeric_pair_")
    ].head(10)
    log_path = output / "EXECUTION_LOG.txt"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"Completed UTC: {finished.isoformat()}\n"
            f"Generated candidates: {candidate_count}; including explicit keys: {len(all_results)}\n"
            f"Elapsed candidate screen seconds: {time.perf_counter() - started_search:.3f}\n"
            f"Baseline C pooled OOF AUC: {baseline_auc:.12f}\n"
            "All candidate target rates were cross-fitted on the five fixed outer folds.\n"
        )
    report = [
        "# Exploratory hidden-key screen",
        "",
        f"- Started/completed UTC: {start.isoformat()} / {finished.isoformat()}",
        f"- Train SHA-256: `{input_hash}`; Phase 13 OOF SHA-256: `{oof_hash}`.",
        f"- Rows: {len(train):,}; fixed folds: 5; candidate count: {len(all_results):,}.",
        f"- Baseline: Phase 13 complete raw+rowwise C; pooled OOF AUC `{baseline_auc:.6f}`.",
        f"- Cross-fitted category-rate smoothing: {SMOOTHING:g}; blend weights screened: {', '.join(f'{w:.2f}' for w in BLEND_WEIGHTS)}.",
        "- Candidate generation uses the prior Phase 11 search helper over numeric singles, pair sums/differences and both ratio orientations at 3/4/5 significant digits, smooth-bin controls, low-cardinality crosses, and selected numeric-bin/category crosses. Eight product keys were added as positive controls.",
        "- For each candidate, fold-held-out category-rate estimates were computed from labels in the other four folds only. This is target-dependent exploratory research; its rule eligibility is unresolved and it is not part of the Phase 13 label-free benchmark.",
        "- Quantile-based smooth/numeric-bin candidate boundaries are descriptive screen definitions estimated from train predictors before cross-fitting, including held-out-fold predictor values. They use no labels or test data, but are not fold-local preprocessing and cannot be promoted without a fold-local rerun.",
        "- The OOF rows have already been examined in prior phases and thousands of candidates were screened. The winning blend is adaptively selected and is not confirmatory evidence.",
        "- No test file, external data, holdout, submission, or champion was used.",
        "",
        "## Strongest result",
        "",
        f"- `{best_blend['candidate']}` ({best_blend['family']}): category-rate OOF AUC {best_blend['target_rate_oof_auc']:.6f}; fixed 10% blend AUC {best_blend['fixed_blend_0.10_auc']:.6f} (delta {best_blend['fixed_blend_0.10_delta']:+.6f}); distinct categories {int(best_blend['unique_categories']):,}.",
        "",
        "## Highest numeric-pair results",
        "",
        "| Candidate | Categories | Target-rate AUC | 10% blend delta vs C |",
        "|---|---:|---:|---:|",
    ]
    for row in top_numeric_pairs.to_dict(orient="records"):
        report.append(
            f"| {row['candidate']} | {int(row['unique_categories']):,} | "
            f"{row['target_rate_oof_auc']:.6f} | {row['fixed_blend_0.10_delta']:+.6f} |"
        )
    report.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "This is a machine-scale hypothesis screen, not proof of a latent identifier, contractual product tier, or competition eligibility. High-cardinality candidate labels are derived from supplied training predictors; their target-rate values use only other folds' labels. Adaptive comparison across many candidates inflates the best observed OOF result. Treat this as a prioritization signal only; any candidate would need a preregistered nested evaluation and explicit rules review.",
            "",
            f"See `HIDDEN_KEY_CANDIDATE_RESULTS.csv` for every candidate and `FREQUENCY_PRESERVING_NULLS.csv` for {len(controls)} shuffled controls. The exact Phase 13 key rate vectors are in `PRODUCT_KEY_CROSSFIT_RATE_OOF.csv`.",
            "",
        ]
    )
    (output / "HIDDEN_KEY_SCREEN_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    manifest_sources = (
        Path(__file__),
        ROOT / "phase13_features.py",
        PHASE11 / "phase11_candidate_search.py",
    )
    files = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "SHA256_MANIFEST.csv"
    )
    manifest = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    manifest.extend(
        {
            "file": f"source:{path.name}",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in manifest_sources
    )
    pd.DataFrame(manifest).to_csv(output / "SHA256_MANIFEST.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an exploratory train-only key screen.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--phase13-oof", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PHASE13_RUNS)
    arguments = parser.parse_args()
    result = run(arguments.train, arguments.phase13_oof, arguments.output_root)
    print(result)


if __name__ == "__main__":
    main()
