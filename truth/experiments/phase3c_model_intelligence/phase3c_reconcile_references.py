from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(r"D:\gci\truth\experiments\phase3c_model_intelligence")
ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
PHASE3B = ROOT / "experiments" / "phase3b_deep_validation"
PHASE3_PRODUCT_REFERENCE = (
    PHASE3 / "batch1_family_ablation_02" / "BATCH1_FAMILY_OOF_PREDICTIONS.csv"
)

spec = importlib.util.spec_from_file_location(
    "phase3c_initial_benchmark_module", OUT / "phase3c_initial_benchmark.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load Phase 3C benchmark definitions.")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def main():
    train = pd.read_csv(DATA / "train.csv")
    ids = train[benchmark.ID].to_numpy()
    y = train[benchmark.TARGET].to_numpy(dtype=np.int8)
    folds = (
        pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv")
        .set_index(benchmark.ID)
        .loc[ids, "fold"]
        .to_numpy(dtype=np.int8)
    )
    saved_oof = pd.read_csv(OUT / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv")
    if saved_oof[benchmark.ID].duplicated().any():
        raise ValueError("Benchmark OOF file has duplicate applicant IDs.")
    aligned_oof = saved_oof.set_index(benchmark.ID).reindex(ids)
    if not np.array_equal(aligned_oof[benchmark.TARGET].to_numpy(), y):
        raise ValueError("Benchmark OOF target alignment failed.")
    if not np.array_equal(aligned_oof["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("Benchmark OOF fold alignment failed.")
    historical = pd.read_csv(PHASE3_PRODUCT_REFERENCE)
    phase3_prediction = benchmark.validate_reference(
        historical,
        ids,
        y,
        folds,
        "OOF_product_decoder_only",
        "phase3_product_decoder_discovery",
    )
    aligned_oof["reference__phase3_discovery"] = phase3_prediction
    aligned_oof = aligned_oof.reset_index(names=benchmark.ID)
    aligned_oof.to_csv(OUT / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv", index=False)
    predictions = {
        column: aligned_oof[column].to_numpy(dtype=np.float64)
        for column in aligned_oof.columns
        if column.startswith(("reference__", "lightgbm__", "catboost__", "xgboost__", "hist_gradient_boosting__", "random_forest__"))
    }
    if len(predictions) != len(
        [column for column in aligned_oof if column not in (benchmark.ID, benchmark.TARGET, "fold")]
    ):
        raise ValueError("Unexpected columns in model OOF prediction matrix.")
    summary, fold_summary = benchmark.metric_tables(predictions, y, folds)
    summary["reference_type"] = np.where(
        summary["model"] == "reference",
        "historical_reference",
        "new_fixed_capacity_discovery",
    )
    summary.to_csv(OUT / "PHASE3C_SCORE_REPRODUCTION.csv", index=False)
    summary.drop(columns=["reference_type"]).to_csv(
        OUT / "MODEL_BENCHMARK_RESULTS.csv", index=False
    )
    fold_summary.to_csv(OUT / "MODEL_FOLD_RESULTS.csv", index=False)
    benchmark.make_error_outputs(train, folds, predictions)
    log_path = OUT / "EXECUTION_LOG.txt"
    old_lines = log_path.read_text(encoding="utf-8").splitlines()
    retained = []
    skip_reconciliation = False
    for line in old_lines:
        if line.startswith("reference_reconciliation_utc="):
            skip_reconciliation = True
            continue
        if skip_reconciliation:
            if line == "oof_error_outputs_regenerated=True":
                skip_reconciliation = False
            continue
        retained.append(line)
    retained.extend(
        [
            "",
            "reference_reconciliation_utc="
            + datetime.now(timezone.utc).isoformat(),
            "reference_correction=Used the exact Phase 3 product_decoder_only OOF vector identified by the Phase 3B score audit; a separate later PHASE3_PRODUCT_OOF_PREDICTIONS.csv was not the 0.762826 reference.",
            "references_revalidated_by_id_target_fold=True",
            "oof_error_outputs_regenerated=True",
        ]
    )
    log_path.write_text("\n".join(retained) + "\n", encoding="utf-8")
    print(
        summary[
            ["model_variant", "pooled_oof_auc", "mean_fold_auc", "std_fold_auc"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
