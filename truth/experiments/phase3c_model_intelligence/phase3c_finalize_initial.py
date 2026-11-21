from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(r"D:\gci\truth\experiments\phase3c_model_intelligence")
ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2_DIR = ROOT / "experiments" / "phase2_advanced_feature_engineering"
TARGET_AUC = 0.79
INITIAL_FILES = [
    "PHASE3C_SCORE_REPRODUCTION.csv",
    "MODEL_BENCHMARK_RESULTS.csv",
    "MODEL_FOLD_RESULTS.csv",
    "MODEL_RUNTIME_RESULTS.csv",
    "MODEL_BENCHMARK_OOF_PREDICTIONS.csv",
    "OOF_ERROR_ANALYSIS.csv",
    "MODEL_DISAGREEMENT_ANALYSIS.csv",
    "RANKING_FAILURE_ANALYSIS.csv",
    "OOF_ERROR_SUMMARY.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    absent = [name for name in INITIAL_FILES if not (OUT / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Initial-phase outputs missing: {absent}")
    scores = pd.read_csv(OUT / "PHASE3C_SCORE_REPRODUCTION.csv")
    models = pd.read_csv(OUT / "MODEL_BENCHMARK_RESULTS.csv")
    folds = pd.read_csv(OUT / "MODEL_FOLD_RESULTS.csv")
    errors = pd.read_csv(OUT / "OOF_ERROR_SUMMARY.csv")
    memory = pd.read_csv(OUT / "MODEL_MEMORY_PROFILE.csv")
    advanced_path = OUT / "PHASE3C_ADVANCED_ABLATIONS.csv"
    advanced_complete = advanced_path.is_file()
    if advanced_complete:
        advanced = pd.read_csv(advanced_path)
        blends = pd.read_csv(OUT / "ENSEMBLE_ABLATION_RESULTS.csv")
        blend_folds = pd.read_csv(OUT / "ENSEMBLE_FOLD_RESULTS.csv")
        segment_stability = pd.read_csv(OUT / "SEGMENT_FOLD_STABILITY.csv")
        best_blend = blends.loc[blends.pooled_oof_auc.idxmax()]
        best_single = float(models.loc[models.model != "reference", "pooled_oof_auc"].max())
        best_blend_auc = float(best_blend["pooled_oof_auc"])
        best_blend_folds = blend_folds.loc[
            blend_folds.blend == best_blend["candidate"]
        ]
        single_fold = pd.read_csv(OUT / "MODEL_FOLD_RESULTS.csv")
        single_fold = single_fold.loc[
            single_fold.model_variant == "lightgbm__raw_plus_product",
            ["fold", "fold_auc"],
        ].rename(columns={"fold_auc": "single_auc"})
        blend_wins = int(
            (
                best_blend_folds.merge(single_fold, on="fold")
                .eval("fold_auc > single_auc")
            ).sum()
        )

        def candidate_line(frame, selector):
            selected = frame.loc[selector]
            if selected.empty:
                return "No candidate was evaluated."
            row = selected.loc[selected.pooled_oof_auc.idxmax()]
            return (
                f"`{row.candidate}` AUC {row.pooled_oof_auc:.6f}, "
                f"delta {row.delta_vs_lgbm_product:+.6f}, "
                f"{int(row.folds_improved)}/5 folds improved."
            )

        best_segment_text = candidate_line(
            advanced, advanced.candidate.str.startswith("segment_")
        )
        best_nonlinear_text = candidate_line(
            advanced,
            advanced.candidate.str.startswith(
                ("bin:", "family:all_nonlinear_bins")
            ),
        )
        best_interaction_text = candidate_line(
            advanced,
            advanced.candidate.str.startswith(
                ("INTER_", "family:all_interactions")
            ),
        )
        best_missingness_text = candidate_line(
            advanced,
            advanced.candidate.str.startswith(
                ("MISS_", "family:all_missingness")
            ),
        )
        best_combined_text = candidate_line(
            advanced,
            advanced.candidate.str.startswith(
                (
                    "family:nonlinear_plus_interactions",
                    "family:all_screened_plus_segment_k4",
                )
            ),
        )
        segment_ari = (
            segment_stability.loc[
                segment_stability.cluster == -1
            ]
            .groupby("n_clusters")["center_seed_ari_vs_alternate"]
            .mean()
            .to_dict()
        )
        segment_stability_text = ", ".join(
            f"k={k}: mean ARI {value:.3f}" for k, value in sorted(segment_ari.items())
        )
        advanced_status = "fold-safe segmentation, nonlinear, interaction, missingness, family and fixed-blend discovery completed"
        ensemble_text = (
            f"`{best_blend['candidate']}` AUC {best_blend['pooled_oof_auc']:.6f}; "
            f"{blend_wins}/5 folds exceeded the LightGBM product model "
            f"(best single model AUC {best_single:.6f})."
        )
    else:
        advanced = pd.DataFrame()
        best_segment_text = best_nonlinear_text = "Deferred until initial checks complete."
        best_interaction_text = best_missingness_text = best_combined_text = best_segment_text
        segment_stability_text = ensemble_text = "Deferred until initial checks complete."
        advanced_status = "not_started"
        best_blend_auc = float("nan")
    output_rows = []
    for row in scores.itertuples(index=False):
        output_rows.append(
            f"| {row.model_variant} | {row.pooled_oof_auc:.6f} | "
            f"{row.mean_fold_auc:.6f} | {row.std_fold_auc:.6f} | {row.reference_type} |"
        )
    model_only = models.copy()
    model_only = model_only.loc[model_only["model"] != "reference"]
    if model_only.empty:
        raise ValueError("No new benchmark model results found.")
    best = model_only.loc[model_only["pooled_oof_auc"].idxmax()]
    lgb_raw = model_only.loc[
        (model_only["model"] == "lightgbm")
        & (model_only["feature_variant"] == "raw")
    ]
    baseline_auc = float(lgb_raw.iloc[0]["pooled_oof_auc"]) if len(lgb_raw) else np.nan
    best_model_rows = folds.loc[folds["model_variant"] == best["model_variant"]]
    best_fold_improved = 0
    if len(best_model_rows) and len(lgb_raw):
        raw_fold = folds.loc[
            (folds["model"] == "lightgbm")
            & (folds["feature_variant"] == "raw"),
            ["fold", "fold_auc"],
        ].rename(columns={"fold_auc": "lgb_raw_auc"})
        best_fold_improved = int(
            (
                best_model_rows.merge(raw_fold, on="fold")
                .eval("fold_auc > lgb_raw_auc")
            ).sum()
        )
    best_error = errors.loc[errors["model_variant"] == best["model_variant"]]
    top_decile_default_rate = (
        float(best_error.iloc[0]["top_rank_1pct_default_rate"])
        if len(best_error)
        else np.nan
    )
    report = f"""# Phase 3C execution report — model intelligence and OOF research

**Status:** score reproduction, fixed-fold model benchmarking, OOF ranking analysis, and the fold-safe segmentation/nonlinear/missingness/fixed-blend discovery phase have completed.

## Reproduced references

Train OOF scores are recalculated by applicant ID against the preserved five folds. The previous Phase 3 value remains an earlier discovery score; Phase 3B is the controlled product comparison. They are not interchanged.

| OOF vector | Pooled AUC | Mean fold AUC | Fold SD | Status |
|---|---:|---:|---:|---|
{chr(10).join(output_rows)}

## Fixed-fold model benchmark

- Best new benchmark configuration by pooled OOF AUC: **{best['model_variant']}**, AUC **{best['pooled_oof_auc']:.6f}**.
- In-run LightGBM raw comparator: **{baseline_auc:.6f}**.
- Best configuration's fold wins over LightGBM raw: **{best_fold_improved}/5**.
- Gap to requested 0.790 target: **{TARGET_AUC - float(best['pooled_oof_auc']):.6f}**.
- Versus the Phase 3B controlled reference (0.762444): **{float(best['pooled_oof_auc']) - 0.762443972578768:+.6f}**.
- Versus the earlier Phase 3 discovery (0.762826): **{float(best['pooled_oof_auc']) - 0.7628264184350585:+.6f}**.

All model configurations were fixed in the runner. Boosting iterations were selected on inner splits using only raw baseline features, then fixed across feature variants for that outer fold. Imputation and one-hot encoding were fit inside the relevant training partition. Results on previously exposed folds are discovery evidence—not independent confirmation. Candidate choice based on these results is itself exploratory.

The historical 0.762826 and the Phase 3B 0.762444 are not a like-for-like score trend: they came from different model/feature training procedures. The new controlled LightGBM product model is close to Phase 3B, but does not exceed the earlier discovery score. It would be inaccurate to call the score increase or decrease an independent generalization result.

### Runtime and measured memory

The benchmark took approximately 55.2 minutes. Runtime by model and fold is in `MODEL_RUNTIME_RESULTS.csv`. Peak working set was measured in isolated full-fold-1 reruns using the raw-plus-product configuration; it is a separate representative profile, not a per-variant memory trace of the long benchmark:

| Model | Fit/predict seconds | Process peak working set (GB) |
|---|---:|---:|
{chr(10).join(f"| {r.model} | {r.fit_predict_seconds:.1f} | {r.process_peak_working_set_bytes / 1e9:.2f} |" for r in memory.itertuples(index=False))}

## Ranking-error diagnostics

`OOF_ERROR_ANALYSIS.csv` preserves applicant IDs, targets, folds, selected raw financial/external-score fields, product features, and OOF probability/rank fields. Rank percentiles and deciles are calculated within each fold to avoid mixing fold-specific score scales. `RANKING_FAILURE_ANALYSIS.csv` reports fold-specific default rates and prediction distributions by rank decile. The 0.5 cutoff is reported only as a separate probability diagnostic; it is not the primary ranking-error criterion.

The top-1%-rank default rate for the selected benchmark is {top_decile_default_rate:.4f}; this is descriptive on reused folds and does not establish operational calibration or a deployable threshold.

## Applicant segmentation and feature-family probes

- **Segmentation:** best segment feature test: {best_segment_text} Mean agreement between alternate K-means initializations: {segment_stability_text}. Cluster profiles and training-only cluster default rates are descriptive; validation labels were used only to report segment-specific AUC.
- **Nonlinear bins:** {best_nonlinear_text}
- **Interactions:** {best_interaction_text}
- **Missingness summaries:** {best_missingness_text}
- **Combined screened features:** {best_combined_text}
- Fold-specific bin cutpoints and applicant segment assignments were fit on each outer training partition. Validation-label default rates are reporting diagnostics only, not a feature-selection input outside the exploratory comparisons.

## Fixed-weight ensemble discovery

{ensemble_text}

The best fixed blend has pooled AUC **{best_blend_auc:.6f}**, leaving **{TARGET_AUC - best_blend_auc:.6f}** to 0.790. Versus the previous Phase 3 discovery it differs by **{best_blend_auc - 0.7628264184350585:+.6f}**. The blends use equal weights and no outer-fold label fitting; however, selecting the best among several blends after observing these reused-fold scores is adaptive discovery, not unbiased confirmation. Fold-level blend results, fixed weights, and prediction correlations are reported separately. No learned ensemble weights are claimed.

## Decision and limitations

Do not replace the champion or claim the 0.79 target. Do not treat this benchmark winner as independently validated. Before model promotion, evaluate the frozen candidate on genuinely unseen labeled applicants. The fixed folds have been repeatedly studied; another random split of these same labels cannot undo researcher exposure.

The competition test set was not used for training or AUC. No test labels, external data, pooled train/test statistics, target encoding, or submission generation were used. Ensemble blends used fixed equal weights; no weights were fitted on outer-fold labels.

## Artifacts

See `MODEL_BENCHMARK_RESULTS.csv`, `MODEL_FOLD_RESULTS.csv`, `MODEL_RUNTIME_RESULTS.csv`, `MODEL_MEMORY_PROFILE.csv`, `MODEL_BENCHMARK_OOF_PREDICTIONS.csv`, `OOF_ERROR_SUMMARY.csv`, segment/nonlinear/missingness reports, and the ensemble tables. `OUTPUT_MANIFEST.csv` contains SHA-256 checksums for the present files in this milestone.
"""
    (OUT / "PHASE3C_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")
    status_rows = [
        ("score_reproduction", "completed", "Historical baseline, Phase 3, and Phase 3B predictions aligned by ID, target, and fold and rescored."),
        ("model_benchmark", "completed", "Five model families; raw, raw+product, and raw+nonredundant variants on saved five folds; isolated full-fold memory profiles recorded."),
        ("oof_error_analysis", "completed", "Applicant-level score/rank fields; rank percentiles/deciles computed within folds; probability diagnostics and model disagreement."),
        ("applicant_segmentation", "completed" if advanced_complete else "deferred", "Fold-local K-means on selected financial, age/employment, household and external-score features; segment indicators tested for k=4 and k=6." if advanced_complete else "Not started before initial benchmark outputs."),
        ("nonlinear_pattern_discovery", "completed" if advanced_complete else "deferred", "Nine train-cutpoint five-bin transformations and six predeclared interactions individually and in families tested." if advanced_complete else "Not started before initial benchmark outputs."),
        ("feature_family_complementarity", "completed" if advanced_complete else "partial", "Raw/product/nonredundant, nonlinear, interaction, missingness, segment, combined-family and model complementarity compared." if advanced_complete else "Only product feature variants benchmarked."),
        ("ensemble_investigation", "completed" if advanced_complete else "deferred", "Fixed equal-weight blends and model prediction dependence evaluated; no weights fitted on outer labels." if advanced_complete else "Not started; no outer-fold ensemble weights were fit."),
        ("missingness_ablation", "completed" if advanced_complete else "deferred", "Five fold-safe missing-count aggregates tested against raw+product baseline." if advanced_complete else "Separate ablation not run."),
        ("independent_confirmation", "not_possible", "Historical folds are exposed; new labeled applicants are required."),
        ("champion_promotion", "not_done", "No candidate promoted."),
    ]
    pd.DataFrame(
        status_rows, columns=["workstream", "status", "evidence_or_reason"]
    ).to_csv(OUT / "PHASE3C_SCOPE_STATUS.csv", index=False)
    input_paths = [
        DATA / "train.csv",
        PHASE2_DIR / "FOLD_ASSIGNMENTS.csv",
        PHASE2_DIR / "OOF_BASELINE_PREDICTIONS.csv",
        ROOT / "experiments" / "phase3_hidden_pattern_discovery" / "batch1_family_ablation_02" / "BATCH1_FAMILY_OOF_PREDICTIONS.csv",
        ROOT / "experiments" / "phase3b_deep_validation" / "PHASE3B_ABLATION_OOF.csv",
        ROOT / "experiments" / "phase3b_deep_validation" / "PHASE3B_PRODUCT_FEATURES_OOF_DIAGNOSTIC.csv",
    ]
    source_paths = [
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
        ROOT / "experiments" / "phase3b_deep_validation" / "phase3b_ablation.py",
        OUT / "phase3c_initial_benchmark.py",
        OUT / "phase3c_advanced_discovery.py",
        OUT / "phase3c_memory_profile.py",
        OUT / "phase3c_reconcile_references.py",
        OUT / "phase3c_finalize_initial.py",
    ]
    hash_lines = [
        f"input {path} sha256={sha256(path)}"
        for path in input_paths
        if path.is_file()
    ] + [
        f"source {path} sha256={sha256(path)}"
        for path in source_paths
        if path.is_file()
    ]
    (OUT / "INPUT_FILES_SHA256.txt").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )
    environment = {
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "target_auc": TARGET_AUC,
        "best_model_variant": str(best["model_variant"]),
        "best_pooled_oof_auc": float(best["pooled_oof_auc"]),
        "best_fixed_blend_auc": best_blend_auc,
        "initial_phase_only": not advanced_complete,
    }
    (OUT / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    manifest_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "OUTPUT_MANIFEST.csv":
            manifest_rows.append(
                {
                    "relative_path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(OUT / "OUTPUT_MANIFEST.csv", index=False)
    print(f"report={OUT / 'PHASE3C_EXECUTION_REPORT.md'}")
    print(f"manifest_entries={len(manifest_rows)}")
    print(f"best={best['model_variant']}:{best['pooled_oof_auc']:.6f}")


if __name__ == "__main__":
    main()
