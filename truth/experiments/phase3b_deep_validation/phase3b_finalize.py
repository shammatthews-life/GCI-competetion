from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
BASELINE = ROOT / "experiments" / "phase2_advanced_feature_engineering"
OUT = ROOT / "experiments" / "phase3b_deep_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(paths: list[Path]) -> None:
    absent = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if absent:
        raise FileNotFoundError("Required Phase 3B outputs absent/empty:\n" + "\n".join(absent))


def main() -> None:
    now = datetime.now(timezone.utc)
    required = [
        "PHASE3B_EXECUTION_REPORT.md",
        "CURRENT_SCORE_REPRODUCTION.csv",
        "PRODUCT_FEATURE_DIAGNOSTICS.csv",
        "PRODUCT_COMPONENT_ABLATIONS.csv",
        "PRODUCT_RECONSTRUCTION_STABILITY.csv",
        "FEE_HYPOTHESIS_ANALYSIS.csv",
        "FINANCING_LATTICE_ANALYSIS.csv",
        "PRODUCT_GROUP_STABILITY.csv",
        "NEGATIVE_CONTROL_RESULTS.csv",
        "NEW_FORMULA_HYPOTHESES.csv",
        "NEW_FORMULA_VALIDATION.csv",
        "FEATURE_REDUNDANCY_REPORT.csv",
        "FOLD_LEVEL_RESULTS.csv",
        "VALIDATION_GOVERNANCE.md",
        "PHASE3B_SCOPE_STATUS.csv",
        "EXECUTION_LOG.txt",
        "OUTPUT_MANIFEST.csv",
    ]
    require_files(
        [
            OUT / name
            for name in required
            if name not in ("PHASE3B_EXECUTION_REPORT.md", "PHASE3B_SCOPE_STATUS.csv", "OUTPUT_MANIFEST.csv")
        ]
    )
    scores = pd.read_csv(OUT / "CURRENT_SCORE_REPRODUCTION.csv")
    ablations = pd.read_csv(OUT / "PRODUCT_COMPONENT_ABLATIONS.csv")
    fold_results = pd.read_csv(OUT / "FOLD_LEVEL_RESULTS.csv")
    fees = pd.read_csv(OUT / "FEE_HYPOTHESIS_ANALYSIS.csv")
    lattice = pd.read_csv(OUT / "FINANCING_LATTICE_ANALYSIS.csv")
    reconstruction = pd.read_csv(OUT / "PRODUCT_RECONSTRUCTION_STABILITY.csv")
    groups = pd.read_csv(OUT / "PRODUCT_GROUP_STABILITY.csv")
    controls = pd.read_csv(OUT / "NEGATIVE_CONTROL_RESULTS.csv")
    formula_results = pd.read_csv(OUT / "NEW_FORMULA_VALIDATION.csv")
    individual_path = OUT / "NEW_FORMULA_INDIVIDUAL_ABLATIONS.csv"
    if individual_path.exists():
        individual = pd.read_csv(individual_path)
    else:
        individual = pd.DataFrame()

    baseline_score = float(
        scores.loc[scores.variant == "corrected_baseline", "pooled_oof_auc"].iloc[0]
    )
    historical_product_score = float(
        scores.loc[scores.variant == "product_decoder_all", "pooled_oof_auc"].iloc[0]
    )
    candidate_rows = ablations.loc[
        ~ablations.variant.isin(
            [
                "saved_baseline_reference",
                "raw_baseline_refit",
                "random_permuted",
                "random_keys",
                "random_fee_P",
                "random_step_K",
            ]
        )
    ].sort_values("pooled_oof_auc", ascending=False)
    best_candidate = candidate_rows.iloc[0] if len(candidate_rows) else None
    random_rows = ablations.loc[
        ablations.variant.isin(["random_permuted", "random_keys", "random_fee_P", "random_step_K"])
    ].sort_values("pooled_oof_auc", ascending=False)
    product_all = ablations.loc[ablations.variant == "all_product"].iloc[0]
    refit_row = ablations.loc[ablations.variant == "raw_baseline_refit"].iloc[0]
    fee_fit = fees.loc[fees.method == "fit_mode_fee"]
    fee_subsample = fees.loc[fees.method == "subsample_median_fee"]
    lattice_summary = (
        lattice.groupby(lattice.candidate_step.round(4))
        .agg(
            mean_fit_support=("fit_within_0_1_pct", "mean"),
            mean_validation_support=("validation_within_0_1_pct", "mean"),
            mean_validation_residual=("validation_residual_mean", "mean"),
            mean_credit_reconstruction_mae=("credit_reconstruction_mae", "mean"),
            mean_credit_reconstruction_p95=("credit_reconstruction_p95_abs_error", "mean"),
            mean_odd_integer_share=("validation_odd_integer_share_among_supported_pct", "mean"),
        )
    )
    lattice_0066 = lattice_summary.loc[0.0066] if 0.0066 in lattice_summary.index else None
    lattice_0132 = lattice_summary.loc[0.0132] if 0.0132 in lattice_summary.index else None
    rec_group = reconstruction.groupby("component").agg(
        valid=("n", "sum"),
        median_abs_error=("median_abs_error", "median"),
        mae=("mae", "mean"),
        p95_abs_error=("p95_abs_error", "mean"),
        exact_pct=("exact_pct", "mean"),
        within_cent_pct=("pct_abs_error_le_0_01", "mean"),
    )
    control_summary = controls.groupby("control_type").agg(
        settings=("setting", "count"),
        mean_repeat_share=("validation_repeated_in_fit_pct", "mean"),
        mean_mae=("mae", "mean"),
    )
    group_summary = groups.groupby("grouping").agg(
        validation_rows=("validation_group_n", "sum"),
        q5_repeated_pct=("q5_repeated_fit_pct", "mean"),
        p5_repeated_pct=("p5_repeated_fit_pct", "mean"),
        k_repeated_pct=("k_integer_repeated_fit_pct", "mean"),
    )
    score_table = scores[
        ["variant", "pooled_oof_auc", "mean_fold_auc", "std_fold_auc"]
    ].to_string(index=False)
    ablation_columns = [
        "variant",
        "pooled_oof_auc",
        "delta_vs_baseline",
        "mean_fold_auc",
        "std_fold_auc",
        "folds_improved",
    ]
    ablation_table = ablations[ablation_columns].sort_values(
        "pooled_oof_auc", ascending=False
    ).to_string(index=False)
    best_text = (
        f"{best_candidate.variant}: {best_candidate.pooled_oof_auc:.6f} "
        f"(delta vs in-run raw baseline {best_candidate.delta_vs_baseline:+.6f}; "
        f"{int(best_candidate.folds_improved)} of 5 folds improved)"
        if best_candidate is not None
        else "No candidate result was available."
    )
    random_text = (
        random_rows[["variant", "pooled_oof_auc", "delta_vs_baseline", "folds_improved"]]
        .to_string(index=False)
        if len(random_rows)
        else "No model negative-control results are available."
    )
    formula_text = (
        individual[
            [
                "feature_name",
                "pooled_oof_auc",
                "pooled_delta_vs_refit_baseline",
                "folds_improved",
            ]
        ].to_string(index=False)
        if len(individual)
        else "Individual new-formula ablations were not completed."
    )
    fee_interval = (
        f"fit-mode fee mean {fee_fit.parameter_estimate.mean():.2f}, "
        f"range {fee_fit.parameter_estimate.min():.2f}–{fee_fit.parameter_estimate.max():.2f}; "
        f"60%-subsample median estimates range "
        f"{fee_subsample.parameter_estimate.min():.2f}–{fee_subsample.parameter_estimate.max():.2f}."
        if len(fee_fit) and len(fee_subsample)
        else "Fee estimates are not available."
    )
    lattice_text = (
        f"0.0066 fit/validation support {lattice_0066.mean_fit_support:.2f}%/"
        f"{lattice_0066.mean_validation_support:.2f}%, mean reconstruction MAE "
        f"{lattice_0066.mean_credit_reconstruction_mae:.2f}, mean p95 error "
        f"{lattice_0066.mean_credit_reconstruction_p95:.2f}, odd-coordinate share "
        f"{lattice_0066.mean_odd_integer_share:.2f}%; versus 0.0132 support "
        f"{lattice_0132.mean_fit_support:.2f}%/"
        f"{lattice_0132.mean_validation_support:.2f}%."
        if lattice_0066 is not None and lattice_0132 is not None
        else "Candidate-step summary unavailable."
    )

    report = f"""# Phase 3B execution report — deep product-structure validation

**Status:** completed for score reproduction, fold-local structural diagnostics, the predeclared product ablation/control matrix, and selected formula tests. This is not a claim that the entire competition research program is complete or that any feature is promoted.

## Reproduced OOF references

Train rows: 171,202; target prevalence: 8.073%; supplied test rows were not used for fitting or validation. All saved OOF sets were joined by applicant ID and checked for exact ID coverage, TARGET alignment, fold alignment, and finite predictions.

{score_table}

The corrected saved raw baseline reproduces at **{baseline_score:.12f}**. The previously reported product-decoder OOF reproduces at **{historical_product_score:.12f}**, a difference of **{historical_product_score-baseline_score:+.6f}** from that saved baseline. The latter is a reused-fold discovery score, not a public leaderboard result or independent confirmation.

## Phase 3B controlled model comparison

The in-run raw-only baseline was refit on the same saved outer folds. Each feature family used the same fold-specific boosting-round count selected on one inner 10% split using raw baseline features only. The identical count was applied to every family in that fold. Candidate feature maps and unsupervised parameters were estimated on the current outer-training rows only.

{ablation_table}

Best non-control Phase 3B candidate: **{best_text}**. In-run raw baseline AUC: **{refit_row.pooled_oof_auc:.6f}** (saved baseline reference: {baseline_score:.6f}). Product-only family AUC: **{product_all.pooled_oof_auc:.6f}** (delta vs in-run baseline {product_all.delta_vs_baseline:+.6f}).

### Negative controls

{random_text}

The target-free reconstruction controls are recorded separately in `NEGATIVE_CONTROL_RESULTS.csv`. Arbitrary fee constants have near-baseline repeated-key coverage, so key repetition alone does not identify a literal fee. Permuted/random keys and arbitrary parameter controls are not all equivalent to a valid financial mechanism; their model results are a check against generic added-column effects, not a causal test.

## Structural interpretation

- **Fee hypothesis:** {fee_interval} Multiple contract/credit regimes and robust subsample estimates are documented in `FEE_HYPOTHESIS_ANALYSIS.csv`. Held-out repeated-key coverage differences are small relative to their high base level. Random fee constants often produce similar coverage. This is not credible evidence for a fixed contractual fee; at most P-family reparameterizations may be predictive.
- **K financing lattice:** {lattice_text} The coarser 0.0066 grid is a better numerical description than 0.0132 on the predeclared fit/validation diagnostics, but exact support is not universal and extreme reconstruction errors remain. Since 0.0132 is twice 0.0066, its coordinate grid admits only a subset of the finer-grid points. This does not establish a primitive contract increment, business meaning, or causal repayment mechanism.
- **Reconstruction:** {rec_group.to_string()} The lattice's high near-integer-coordinate share must not be confused with exact money reconstruction; the complete error quantiles and fold estimates are in `PRODUCT_RECONSTRUCTION_STABILITY.csv`.
- **Applicant strata:** group-conditioned sample sizes, key coverage, and errors are recorded in `PRODUCT_GROUP_STABILITY.csv`. Coverage is lowest in high-cardinality organization/occupation groups and remains high for low-cardinality K keys; raw key coverage is therefore not evidence of group-specific contracts.
- **Redundancy and interpretation:** product features largely re-express supplied credit, goods-price, and annuity fields. In particular, fee-adjusted annuity is a constant translation of the raw annuity, K-decimal is an affine transform of credit/goods, and Q/P precisions are quantized ratio variants. A tree model may use these representations differently, but they do not prove a new underlying loan product.
- **Univariate associations and mutual information in `PRODUCT_FEATURE_DIAGNOSTICS.csv` are descriptive only** and were not used to select formulas. They must not be interpreted as incremental feature value. Fold importance was collected only for the all-product model.

## Additional formula research

Candidate formula evidence, definitions, missingness, infinities, redundancy, and semantic decisions are in `NEW_FORMULA_HYPOTHESES.csv` and `NEW_FORMULA_VALIDATION.csv`. Exact duplicates of baseline features are not treated as new discoveries. Bureau inquiry windows overlap; summing them double-counts events, and month/year ratios are not rates. These are excluded from incremental modeling. Annuity frequency remains unspecified by the local dictionary.

{formula_text}

The four individually re-tested ratios overlap with earlier Phase 3 financial-formula work; the refreshed fixed-round ablations are not independent replications. No feature is promoted.

## Compliance, validation limits, and decision

- The saved folds have been used repeatedly in earlier research. Nested outer/inner processing prevents outer-label use during transformations and stopping-round selection, but researcher exposure remains. A new random split of this same dataset would not erase that exposure.
- This is a fixed five-fold discovery evaluation with inner raw-baseline early stopping, not a fresh independent test. A genuinely independent confirmation requires previously unseen labelled applicants.
- The competition test set is unlabeled and was not used. No pooled train/test statistics, target encoding, external data, hand-created labels, inferred applicant identity, ensemble tuning, or submission generation was performed.
- The local rule audit says pooled test-covariate statistics and target encoding are unresolved; both remain disabled. No live organizer amendment was verified during this work.
- The requested **0.790 AUC target is not verified or reached**. The Phase 3B best candidate is {best_text}; its gap to 0.790 is {0.790-float(best_candidate.pooled_oof_auc):.6f} AUC points. The existing Phase 3 discovery remains {historical_product_score:.6f}.

**Recommendation:** do not change the champion. Freeze the best predeclared candidate family and obtain genuinely new labelled confirmation (or organizer-approved unseen validation data) before making a promotion decision. Do not start ensemble/hyperparameter tuning based on these reused folds.

## Reproducibility and artifacts

Execution logs, commands, timestamps, software versions, source/data hashes, model outputs, fold-level results, and errors are recorded in `EXECUTION_LOG.txt`, `INPUT_FILES_SHA256.txt`, and the `environment.json` files. The manifest hashes every Phase 3B file except itself. The initial ablation attempt failed due to a boolean-index alignment bug in the floor-key transform; it produced no final ablation artifacts. The index handling was corrected and the complete matrix was rerun. No historical project output was modified.
"""
    (OUT / "PHASE3B_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")

    scope_rows = [
        ("score_reproduction", "completed", "Nine prediction vectors aligned by ID, TARGET, and fold; pooled AUC recalculated."),
        ("product_feature_diagnostics", "completed", "Missingness/cardinality/distributions, descriptive AUC/MI, raw-feature correlation, and all-product gain importance."),
        ("product_component_ablation_matrix", "completed", "K/Q/P, combinations, all, redundancy-pruned, stable, rare-key, and fixed-round model variants."),
        ("product_reconstruction_and_lattice", "completed", "Predeclared step grid, fit/validation support, odd coordinates, money reconstruction errors, and fold stability."),
        ("fee_hypothesis_analysis", "completed", "No/fixed/fitted/median/trimmed/regime/proportional candidates; training-size/outlier sensitivity."),
        ("negative_controls", "completed", "Rounding, fee, step, perturbation, key-frequency, permutation and random-key model controls."),
        ("applicant_group_stability", "completed", "Training-cutpoint numerical bands plus contract, occupation, and organization groups."),
        ("new_formula_hypotheses", "completed", "All candidate formulas and semantic/redundancy decisions documented."),
        ("new_formula_individual_cv", "completed" if len(individual) else "not_started", "Four non-duplicate candidate ratios evaluated individually; exact baseline duplicates and invalid overlapping bureau ratios excluded."),
        ("full_nested_confirmation", "not_possible", "Prior researcher exposure of the entire labelled table; new labels required."),
        ("test_set_auc", "not_possible", "Test has no labels and was not used."),
        ("champion_promotion", "not_done", "No feature promoted; independent confirmation not available."),
        ("ensemble_or_hyperparameter_tuning", "not_started_by_design", "Explicitly deferred until product evidence is confirmed."),
    ]
    scope = pd.DataFrame(scope_rows, columns=["workstream", "status", "evidence_or_reason"])
    scope.to_csv(OUT / "PHASE3B_SCOPE_STATUS.csv", index=False)

    inputs = [
        DATA / "train.csv",
        DATA / "test.csv",
        BASELINE / "OOF_BASELINE_PREDICTIONS.csv",
        BASELINE / "FOLD_ASSIGNMENTS.csv",
        PHASE3 / "run_phase3_hidden_pattern_discovery.py",
        PHASE3 / "run_phase3_batch1_family_ablation.py",
        PHASE3 / "run_phase3_product_component_ablation.py",
        PHASE3 / "run_phase3_fee_constant_ablation.py",
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
    ]
    with (OUT / "INPUT_FILES_SHA256.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"recorded_utc={now.isoformat()}\n")
        stream.write("test.csv is hashed for provenance only; Phase 3B did not load it.\n")
        for path in inputs:
            stream.write(f"{path} sha256={sha256(path)} size_bytes={path.stat().st_size}\n")

    env = {
        "recorded_utc": now.isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scikit_learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
        "validation": "saved 5 folds; inner raw-only stratified 10% stopping split",
        "outer_validation_used_for_early_stopping": False,
        "target_encoding": False,
        "pooled_train_test_statistics": False,
        "test_data_used": False,
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    (OUT / "requirements.txt").write_text(
        "\n".join(
            f"{name}=={env[key]}"
            for name, key in [
                ("numpy", "numpy"),
                ("pandas", "pandas"),
                ("scikit-learn", "scikit_learn"),
                ("lightgbm", "lightgbm"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    log_entry = [
        f"finalized_utc={now.isoformat()}",
        "finalization_status=SUCCESS",
        "ablation_first_attempt=FAILED_AND_FIXED",
        "ablation_first_error=Keyed boolean Series index alignment in q_floor transform; aborted before output files were written.",
        "ablation_rerun=completed; see PHASE3B_ABLATION_LOG.txt",
        "overall_claim=partial_phase3b_milestone_only; no independent confirmation or promotion",
    ]
    with (OUT / "EXECUTION_LOG.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n" + "\n".join(log_entry) + "\n")

    # Create a deterministic manifest after all deliverables have been written.
    manifest_path = OUT / "OUTPUT_MANIFEST.csv"
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        if "__pycache__" in path.parts:
            continue
        entries.append(
            {
                "relative_path": str(path.relative_to(OUT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(entries)
    require_files([OUT / name for name in required])
    print(f"finalized_utc={now.isoformat()}")
    print(f"required_outputs={len(required)}")
    print(f"manifest_entries={len(entries)}")
    print(f"report={OUT / 'PHASE3B_EXECUTION_REPORT.md'}")
    print(f"best_candidate={best_text}")
    print(f"baseline={baseline_score:.12f}")


if __name__ == "__main__":
    main()
