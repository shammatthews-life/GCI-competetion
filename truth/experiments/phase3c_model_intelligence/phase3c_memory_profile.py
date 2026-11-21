from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(r"D:\gci\truth\experiments\phase3c_model_intelligence")
ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"

spec = importlib.util.spec_from_file_location(
    "phase3c_benchmark_memory_module", OUT / "phase3c_initial_benchmark.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Unable to load benchmark implementation.")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "lightgbm",
            "catboost",
            "xgboost",
            "hist_gradient_boosting",
            "random_forest",
        ],
    )
    args = parser.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    y = train[benchmark.TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[benchmark.TARGET]).reset_index(drop=True)
    ids = train[benchmark.ID].to_numpy()
    folds = (
        pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv")
        .set_index(benchmark.ID)
        .loc[ids, "fold"]
        .to_numpy(dtype=np.int8)
    )
    fold = 1
    valid_mask = folds == fold
    fit_idx, valid_idx = np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)
    product_params = benchmark.shared.fit_fee_and_step(raw.iloc[fit_idx])
    fit_frame, valid_frame = benchmark.model_feature_frames(
        raw.iloc[fit_idx],
        raw.iloc[valid_idx],
        "raw_plus_product",
        product_params=product_params,
    )
    runtime = pd.read_csv(OUT / "MODEL_RUNTIME_RESULTS.csv")
    selected = runtime.loc[
        (runtime["model"] == args.model)
        & (runtime["fold"] == fold)
        & (runtime["stage"] == "inner_round_selection"),
        "selected_rounds_or_trees",
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected capacity for {args.model}, fold {fold}.")
    rounds = int(selected.iloc[0])
    before = benchmark.mem_snapshot()
    started = time.perf_counter()
    prediction = benchmark.fit_predict(
        args.model,
        fit_frame,
        valid_frame,
        y[fit_idx],
        rounds,
        fold,
    )
    elapsed = time.perf_counter() - started
    after = benchmark.mem_snapshot()
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite memory-profile prediction for {args.model}.")
    row = {
        "model": args.model,
        "feature_variant": "raw_plus_product",
        "fold": fold,
        "n_train": len(fit_idx),
        "n_valid": len(valid_idx),
        "rounds_or_trees": rounds,
        "fit_predict_seconds": elapsed,
        "working_set_before_bytes": before["working_set_bytes"],
        "working_set_after_bytes": after["working_set_bytes"],
        "process_peak_working_set_bytes": after["peak_working_set_bytes"],
        "memory_profile_scope": "isolated process; full fold-1 outer training set; diagnostic rerun",
    }
    profile_path = OUT / "MODEL_MEMORY_PROFILE.csv"
    prior = (
        pd.read_csv(profile_path)
        if profile_path.exists()
        else pd.DataFrame(columns=row.keys())
    )
    prior = prior.loc[prior["model"] != args.model]
    pd.concat([prior, pd.DataFrame([row])], ignore_index=True).to_csv(
        profile_path, index=False
    )
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
