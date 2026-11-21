from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
EXPERIMENT = ROOT / "experiments" / "phase8_nested_product_target_encoding"
RUNS = EXPERIMENT / "runs"
TRAIN_PATH = DATA / "train.csv"
LEDGER_PATH = ROOT / "GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md"
RULES_PATH = ROOT / "reports" / "final_competition_audit" / "OFFICIAL_RULES_AUDIT.md"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE3_SOURCE = (
    ROOT
    / "experiments"
    / "phase3_hidden_pattern_discovery"
    / "run_phase3_hidden_pattern_discovery.py"
)
PHASE7_RUNNER = (
    ROOT
    / "experiments"
    / "phase7_product_feature_discovery"
    / "run_phase7_product_discovery.py"
)
PHASE7_RESULT = (
    ROOT
    / "experiments"
    / "phase7_product_feature_discovery"
    / "runs"
    / "phase7_20261104T073000Z"
)
PHASE7_METRICS = PHASE7_RESULT / "PHASE7_MODEL_COMPARISON.csv"

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD_COUNT = 5
FOLD_SEEDS = (42, 7, 2026)
INNER_SPLITS = 5
SMOOTHING = 30.0
FIXED_FEE_HYPOTHESIS = 130.5
FIXED_STEP_HYPOTHESIS = 0.0066
MODEL_SEED_BASE = 20261150

Q4_ONLY = ("Q4",)
KEYS_8 = ("Q3", "Q4", "Q5", "KQ3", "KQ4", "P4", "P5", "AG3")
KEYS_10 = KEYS_8 + ("P3", "KQ5")
VARIANTS = (
    "raw_baseline",
    "raw_plus_phase3_product",
    "product_plus_te_q4",
    "product_plus_te8",
    "product_plus_te8_and_counts",
    "product_plus_te10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    output[valid] = numerator[valid] / denominator[valid]
    return output


def quantized_key(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(values * (10**decimals))
    result = np.full(len(values), "MISSING", dtype=object)
    valid = np.isfinite(scaled)
    result[valid] = scaled[valid].astype(np.int64).astype(str)
    return result.astype(str)


def make_product_keys(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = raw["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)

    q = safe_ratio(annuity, credit)
    p = safe_ratio(annuity - FIXED_FEE_HYPOTHESIS, credit)
    ag = safe_ratio(annuity, goods)
    excess = safe_ratio(credit, goods) - 1.0
    k_coordinate = excess / FIXED_STEP_HYPOTHESIS
    k_integer = np.rint(k_coordinate)
    k_valid = (
        (contract == "Cash loans")
        & np.isfinite(k_coordinate)
        & (credit > goods)
        & (goods > 0)
    )

    keys: dict[str, np.ndarray] = {}
    for decimals in (3, 4, 5):
        keys[f"Q{decimals}"] = quantized_key(q, decimals)
        keys[f"P{decimals}"] = quantized_key(p, decimals)
    keys["AG3"] = quantized_key(ag, 3)
    for decimals in (3, 4, 5):
        q_key = keys[f"Q{decimals}"]
        composite = np.full(len(raw), "NONELIGIBLE", dtype=object)
        valid = k_valid & (q_key != "MISSING")
        composite[valid] = (
            k_integer[valid].astype(np.int64).astype(str)
            + "|"
            + q_key[valid]
        )
        keys[f"KQ{decimals}"] = composite.astype(str)
    return keys


def encode_keys(
    keys: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
    smoothing: float = SMOOTHING,
) -> tuple[np.ndarray, np.ndarray]:
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    if np.intersect1d(fit_indices, apply_indices).size:
        raise ValueError("Target-encoding fit/apply indices overlap")
    prior = float(labels[fit_indices].mean())
    frame = pd.DataFrame(
        {"key": keys[fit_indices], "target": labels[fit_indices].astype(np.float64)}
    )
    grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + smoothing * prior) / (
        grouped["count"] + smoothing
    )
    apply_keys = pd.Series(keys[apply_indices])
    encoded = apply_keys.map(smoothed).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(encoded).all() or not np.isfinite(counts).all():
        raise ValueError("Target encoding created non-finite values")
    return encoded, counts


def nested_training_encodings(
    keys: dict[str, np.ndarray],
    labels: np.ndarray,
    outer_fit_indices: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    names = list(KEYS_10)
    values = np.full((len(outer_fit_indices), len(names) * 2), np.nan, dtype=np.float32)
    local_y = labels[outer_fit_indices]
    inner = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    diagnostics = []
    for inner_fold, (fit_local, apply_local) in enumerate(
        inner.split(np.zeros(len(outer_fit_indices)), local_y), start=1
    ):
        fit_global = outer_fit_indices[fit_local]
        apply_global = outer_fit_indices[apply_local]
        if np.intersect1d(fit_global, apply_global).size:
            raise AssertionError("Inner target-encoding fold overlaps")
        for key_idx, name in enumerate(names):
            encoded, count = encode_keys(
                keys[name], labels, fit_global, apply_global
            )
            values[apply_local, 2 * key_idx] = encoded
            values[apply_local, 2 * key_idx + 1] = np.log1p(count)
        diagnostics.append(
            {
                "inner_fold": inner_fold,
                "map_fit_rows": len(fit_global),
                "encoding_rows": len(apply_global),
                "map_fit_encoding_rows_disjoint": True,
            }
        )
    if not np.isfinite(values).all():
        raise ValueError("Nested training encodings are incomplete")
    columns = [
        column
        for name in names
        for column in (f"TE_{name}", f"LOG_COUNT_{name}")
    ]
    return pd.DataFrame(values, columns=columns), {
        "inner_folds": diagnostics,
        "outer_fit_rows": len(outer_fit_indices),
        "nested_training_rows": len(outer_fit_indices),
    }


def apply_training_encodings(
    keys: dict[str, np.ndarray],
    labels: np.ndarray,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> pd.DataFrame:
    columns: dict[str, np.ndarray] = {}
    for name in KEYS_10:
        encoded, count = encode_keys(keys[name], labels, fit_indices, apply_indices)
        columns[f"TE_{name}"] = encoded
        columns[f"LOG_COUNT_{name}"] = np.log1p(count)
    return pd.DataFrame(columns)


def auc(y: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")
    return float(roc_auc_score(y, prediction))


def main(run_id: str) -> int:
    started = datetime.now(timezone.utc)
    wall = time.perf_counter()
    out = RUNS / f"phase8_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    log_path = out / "EXECUTION_LOG.txt"

    source_paths = [
        Path(__file__).resolve(),
        TRAIN_PATH,
        LEDGER_PATH,
        RULES_PATH,
        BASELINE_SOURCE,
        PHASE3_SOURCE,
        PHASE7_RUNNER,
        PHASE7_METRICS,
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 8 input missing: {path}")
    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    phase7_metrics = pd.read_csv(PHASE7_METRICS)
    if "raw_plus_all_product" not in set(phase7_metrics.variant):
        raise ValueError("Phase 7 primary comparator is absent")

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    log("status=RUNNING")
    log(f"run_id={run_id}")
    log(f"started_utc={started.isoformat()}")

    plan = f"""# Phase 8 preregistered experiment plan

## Question

Does correctly nested, smoothed target encoding of the hypothesized loan-product keys add signal beyond the fold-safe raw-plus-Phase-3 product model? Claude reported a large lift on a weaker HistGradientBoosting baseline. This run tests the claim on LightGBM and repeats five-fold CV across seeds `{FOLD_SEEDS}`.

## Data and rule boundary

- Read only the provided `train.csv`; do not open or use test predictors, test IDs, or test labels.
- No external data or hand labels.
- The supplied rule audit says target encoding is not expressly prohibited but is **unresolved in scope**. This experiment is research evidence only and does not authorize submission.
- Every target map is fit on labels from a strict subset of the current outer-training fold. Outer-validation labels are never used for feature construction, stopping, or model fitting.

## Frozen model and candidate definitions

- Five stratified folds per seed, seeds `{FOLD_SEEDS}`.
- Raw baseline and raw + exact Phase 3 K/Q/P block are matched controls.
- Primary augmentation: raw + Phase 3 block + smoothed target means for the eight predeclared keys `Q3,Q4,Q5,KQ3,KQ4,P4,P5,AG3`.
- Nested training encodings: inner 5-fold OOF within each outer-fit partition, smoothing alpha `{SMOOTHING}`. Outer-validation encodings use maps fit on the complete outer-fit partition.
- Comparisons: `Q4` only; 8 keys; 8 means plus their log-count controls; 10 keys (adds `P3,KQ5`).
- `P` keys use fixed offset `{FIXED_FEE_HYPOTHESIS}` and `KQ` keys use fixed step `{FIXED_STEP_HYPOTHESIS}` solely as hypotheses from the supplied discussion. They are not verified fees/terms. Key rounding uses integerized decimal ratios; no target-dependent key construction.
- Shared boosting rounds are selected on an inner split of outer-fit rows using raw baseline features only, then held constant across variants. No hyperparameter search, seed bagging, test-time inference, or holdout scoring is performed.

## Prespecified outputs

Report per-seed pooled OOF AUC, five fold AUCs and paired deltas vs raw and raw-plus-Phase-3, key support/cardinality, prediction hashes, encoding overlap checks, and exact inputs/source hashes. Repeated folds reuse the same labeled dataset and do not constitute independent confirmation.
"""
    (out / "PHASE8_EXECUTION_PLAN.md").write_text(plan, encoding="utf-8")

    provenance = pd.DataFrame(
        [
            {
                "path": str(path),
                "sha256": source_hashes[str(path)],
                "bytes": path.stat().st_size,
            }
            for path in source_paths
        ]
    )
    provenance.to_csv(out / "PHASE8_SOURCE_PROVENANCE.csv", index=False)

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    if len(train) != 171202 or train[ID].isna().any() or train[ID].duplicated().any():
        raise ValueError("Train row count / ID integrity check failed")
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("Training target integrity check failed")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    keys = make_product_keys(raw)
    if any(len(values) != len(train) for values in keys.values()):
        raise AssertionError("Product key arrays do not align with train rows")

    baseline = load_module("phase8_preserved_baseline", BASELINE_SOURCE)
    phase3 = load_module("phase8_phase3_features", PHASE3_SOURCE)
    phase7 = load_module("phase8_phase7_helpers", PHASE7_RUNNER)
    if not set(phase7.ALL_PRODUCT).issubset(phase3.MODEL_FEATURES):
        raise AssertionError("Phase 7 model features are not present in Phase 3 source")
    if list(phase7.ALL_PRODUCT) != list(
        load_module("phase8_phase3b_blocks", phase7.PHASE3B_SOURCE).ALL_PRODUCT
    ):
        raise AssertionError("Phase 7 and Phase 3B product blocks differ")

    base_features = baseline.build_rowwise_features(raw)
    if "DAYS_EMPLOYED_SENTINEL" not in base_features:
        raise AssertionError("Baseline is missing its employment sentinel feature")
    if "RAW_MISSING_COUNT" not in base_features or "EXT_COUNT" not in base_features:
        raise AssertionError("Baseline is missing its existing missingness summaries")
    raw_feature_columns = list(base_features.columns)
    raw_missing_columns = [c for c in raw_feature_columns if c.startswith("NA_")]

    prediction_columns: dict[str, np.ndarray] = {}
    fold_rows: list[dict[str, object]] = []
    encoding_diagnostics: list[dict[str, object]] = []
    cardinality_rows: list[dict[str, object]] = []

    for seed in FOLD_SEEDS:
        folds = list(
            StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y
            )
        )
        seed_predictions = {
            variant: np.full(len(y), np.nan, dtype=np.float64)
            for variant in VARIANTS
        }
        for fold, (fit_indices, valid_indices) in enumerate(folds, start=1):
            if np.intersect1d(fit_indices, valid_indices).size:
                raise AssertionError("Outer fit/validation folds overlap")
            seed_fold = seed * 100 + fold
            log(f"seed={seed}; fold={fold}/{FOLD_COUNT}; selecting_shared_rounds")
            raw_fit = raw.iloc[fit_indices].reset_index(drop=True)
            raw_valid = raw.iloc[valid_indices].reset_index(drop=True)
            y_fit, y_valid = y[fit_indices], y[valid_indices]
            rounds = phase7.select_rounds(
                baseline,
                raw_fit,
                y_fit,
                seed=seed_fold + 100000,
            )

            params, maps, product_fit, product_valid = phase7.prepare_product(
                phase3, raw_fit, raw_valid
            )
            product_fit = product_fit[phase7.ALL_PRODUCT].reset_index(drop=True)
            product_valid = product_valid[phase7.ALL_PRODUCT].reset_index(drop=True)
            raw_fit_features = base_features.iloc[fit_indices].reset_index(drop=True)
            raw_valid_features = base_features.iloc[valid_indices].reset_index(drop=True)

            fit_te, nested_meta = nested_training_encodings(
                keys,
                y,
                fit_indices,
                seed=seed_fold + 200000,
            )
            valid_te = apply_training_encodings(keys, y, fit_indices, valid_indices)
            fit_te = fit_te.reset_index(drop=True)
            valid_te = valid_te.reset_index(drop=True)
            if len(fit_te) != len(fit_indices) or len(valid_te) != len(valid_indices):
                raise AssertionError("Nested target encodings do not align to fold rows")

            for name in KEYS_10:
                fit_vals = keys[name][fit_indices]
                valid_vals = keys[name][valid_indices]
                fit_counts = pd.Series(fit_vals).value_counts(dropna=False)
                cardinality_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "key": name,
                        "fit_rows": len(fit_indices),
                        "fit_unique_keys": int(fit_counts.size),
                        "fit_singleton_row_share": float(
                            fit_counts[fit_counts == 1].sum() / len(fit_indices)
                        ),
                        "validation_rows": len(valid_indices),
                        "validation_unseen_key_share": float(
                            (~pd.Series(valid_vals).isin(fit_counts.index)).mean()
                        ),
                    }
                )
            encoding_diagnostics.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "outer_fit_rows": len(fit_indices),
                    "outer_validation_rows": len(valid_indices),
                    "inner_splits": INNER_SPLITS,
                    "smoothing_alpha": SMOOTHING,
                    "outer_fit_validation_disjoint": True,
                    "inner_map_apply_disjoint": all(
                        row["map_fit_encoding_rows_disjoint"]
                        for row in nested_meta["inner_folds"]
                    ),
                    "inner_fold_metadata_json": json.dumps(
                        nested_meta["inner_folds"], sort_keys=True
                    ),
                    "fee_hypothesis": FIXED_FEE_HYPOTHESIS,
                    "step_hypothesis": FIXED_STEP_HYPOTHESIS,
                    "fit_only_product_fee": params["fee_estimate_fit_only"],
                    "fit_only_product_step": params["step_estimate_fit_only"],
                    "outer_validation_labels_used_for_encoding": False,
                }
            )

            fit_frames: dict[str, pd.DataFrame] = {
                "raw_baseline": raw_fit_features,
                "raw_plus_phase3_product": pd.concat(
                    [raw_fit_features, product_fit], axis=1
                ),
            }
            valid_frames: dict[str, pd.DataFrame] = {
                "raw_baseline": raw_valid_features,
                "raw_plus_phase3_product": pd.concat(
                    [raw_valid_features, product_valid], axis=1
                ),
            }

            def add_te_variant(
                name: str,
                key_names: tuple[str, ...],
                include_counts: bool,
            ) -> None:
                te_columns = []
                for key_name in key_names:
                    te_columns.append(f"TE_{key_name}")
                    if include_counts:
                        te_columns.append(f"LOG_COUNT_{key_name}")
                fit_frames[name] = pd.concat(
                    [
                        raw_fit_features,
                        product_fit,
                        fit_te[te_columns],
                    ],
                    axis=1,
                )
                valid_frames[name] = pd.concat(
                    [
                        raw_valid_features,
                        product_valid,
                        valid_te[te_columns],
                    ],
                    axis=1,
                )

            add_te_variant("product_plus_te_q4", Q4_ONLY, False)
            add_te_variant("product_plus_te8", KEYS_8, False)
            add_te_variant("product_plus_te8_and_counts", KEYS_8, True)
            add_te_variant("product_plus_te10", KEYS_10, False)

            for variant in VARIANTS:
                fit_frame = fit_frames[variant]
                valid_frame = valid_frames[variant]
                if fit_frame.columns.duplicated().any():
                    raise AssertionError(f"Duplicate feature columns: {variant}")
                fit_matrix, valid_matrix = baseline.make_lightgbm_frames(
                    fit_frame, valid_frame
                )
                model_params = phase7.get_model_params(
                    baseline, MODEL_SEED_BASE + seed + fold
                )
                model = lgb.train(
                    model_params,
                    lgb.Dataset(fit_matrix, label=y_fit),
                    num_boost_round=rounds,
                    callbacks=[lgb.log_evaluation(0)],
                )
                pred = model.predict(valid_matrix, num_iteration=rounds)
                if not np.isfinite(pred).all():
                    raise AssertionError(f"Nonfinite OOF for {variant}, seed={seed}, fold={fold}")
                seed_predictions[variant][valid_indices] = pred
                fold_rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "fit_rows": len(fit_indices),
                        "validation_rows": len(valid_indices),
                        "validation_positive_n": int(y_valid.sum()),
                        "shared_rounds_selected_from_outer_fit_only": rounds,
                        "feature_columns": fit_frame.shape[1],
                        "fold_auc": auc(y_valid, pred),
                        "outer_validation_labels_used_for_training_or_encoding": False,
                    }
                )
                log(
                    f"seed={seed}; fold={fold}/{FOLD_COUNT}; variant={variant}; "
                    f"auc={fold_rows[-1]['fold_auc']:.8f}; rounds={rounds}"
                )
                del model, fit_matrix, valid_matrix

        for variant, prediction in seed_predictions.items():
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Incomplete OOF: {variant}, seed={seed}")
            prediction_columns[f"seed{seed}__{variant}"] = prediction

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(out / "FOLD_RESULTS.csv", index=False)
    pd.DataFrame(encoding_diagnostics).to_csv(
        out / "TARGET_ENCODING_INTEGRITY.csv", index=False
    )
    pd.DataFrame(cardinality_rows).to_csv(
        out / "PRODUCT_KEY_SUPPORT.csv", index=False
    )
    oof = pd.DataFrame({ID: ids, TARGET: y})
    for seed in FOLD_SEEDS:
        folds = list(
            StratifiedKFold(n_splits=FOLD_COUNT, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y
            )
        )
        fold_ids = np.empty(len(y), dtype=np.int8)
        for fold, (_, valid_idx) in enumerate(folds, start=1):
            fold_ids[valid_idx] = fold
        oof[f"fold_seed{seed}"] = fold_ids
    for name, prediction in prediction_columns.items():
        oof[name] = prediction
    oof.to_csv(out / "OOF_PREDICTIONS.csv", index=False)

    metrics_rows = []
    for seed in FOLD_SEEDS:
        seed_fold = fold_results[fold_results.seed == seed]
        pooled = {
            variant: auc(y, prediction_columns[f"seed{seed}__{variant}"])
            for variant in VARIANTS
        }
        base_rows = seed_fold[seed_fold.variant == "raw_baseline"].set_index("fold")
        product_rows = seed_fold[
            seed_fold.variant == "raw_plus_phase3_product"
        ].set_index("fold")
        for variant in VARIANTS:
            rows = seed_fold[seed_fold.variant == variant].set_index("fold")
            delta_product = rows.fold_auc - product_rows.fold_auc
            delta_raw = rows.fold_auc - base_rows.fold_auc
            metrics_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "pooled_oof_auc": pooled[variant],
                    "pooled_delta_vs_raw_baseline": pooled[variant] - pooled["raw_baseline"],
                    "pooled_delta_vs_phase3_product": pooled[variant]
                    - pooled["raw_plus_phase3_product"],
                    "mean_fold_auc": float(rows.fold_auc.mean()),
                    "fold_sd": float(rows.fold_auc.std(ddof=1)),
                    "folds_positive_vs_raw": int((delta_raw > 0).sum()),
                    "folds_positive_vs_phase3_product": int((delta_product > 0).sum()),
                    "fold_deltas_vs_phase3_product_json": json.dumps(
                        {str(k): float(v) for k, v in delta_product.items()},
                        sort_keys=True,
                    ),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "SEED_METRICS.csv", index=False)

    aggregate_rows = []
    for variant in VARIANTS:
        part = metrics[metrics.variant == variant]
        fold_variant = fold_results[fold_results.variant == variant]
        fold_product = fold_results[
            fold_results.variant == "raw_plus_phase3_product"
        ].set_index(["seed", "fold"])["fold_auc"]
        paired = fold_variant.set_index(["seed", "fold"])["fold_auc"] - fold_product
        aggregate_rows.append(
            {
                "variant": variant,
                "mean_seed_pooled_auc": float(part.pooled_oof_auc.mean()),
                "sd_seed_pooled_auc": float(part.pooled_oof_auc.std(ddof=1)),
                "min_seed_pooled_auc": float(part.pooled_oof_auc.min()),
                "max_seed_pooled_auc": float(part.pooled_oof_auc.max()),
                "mean_pooled_delta_vs_phase3_product": float(
                    part.pooled_delta_vs_phase3_product.mean()
                ),
                "positive_folds_vs_phase3_product_out_of_15": int((paired > 0).sum()),
                "positive_seed_pooled_deltas_vs_phase3_product_out_of_3": int(
                    (part.pooled_delta_vs_phase3_product > 0).sum()
                ),
                "interpretation": "Repeated-seed discovery on previously exposed labels; not independent confirmation.",
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out / "AGGREGATE_COMPARISON.csv", index=False)

    baseline_match = (
        "DAYS_EMPLOYED_SENTINEL, DAYS_EMPLOYED_CLEAN, per-column NA_* flags, "
        "RAW_MISSING_COUNT, and EXT_COUNT (count of available EXT_SOURCE_1..3) "
        "are already present in the raw baseline feature builder. The candidate "
        "does not duplicate these features."
    )
    (out / "MISSINGNESS_SENTINEL_AUDIT.md").write_text(
        "# Missingness and employment-sentinel audit\n\n"
        "The executable baseline `run_independent_baseline.py` creates the employment sentinel "
        "indicator, replaces the 365243 sentinel with missing in clean employment-duration "
        "features, emits an `NA_` indicator for every numeric source column, computes "
        "`RAW_MISSING_COUNT`, and includes `EXT_COUNT` for the number of available external "
        "scores. Therefore Claude's proposed sentinel and external-score missing-count ideas "
        "are already represented in the matched baseline. The four bureau inquiry fields also "
        "have individual NA flags plus the global missing count. Adding duplicate versions "
        "would not test new information. No separate missingness model ablation was run.\n\n"
        "This conclusion is a source audit, not a claim that the existing flags are optimal. "
        "Prior Phase 6 screening reported little or no incremental missingness lift; the Phase 8 "
        "target-encoding comparison keeps the existing raw baseline unchanged.\n"
        f"\nSource confirmation: {baseline_match}\n",
        encoding="utf-8",
    )

    primary = aggregate.loc[aggregate.variant == "product_plus_te8"].iloc[0]
    te10 = aggregate.loc[aggregate.variant == "product_plus_te10"].iloc[0]
    count_variant = aggregate.loc[
        aggregate.variant == "product_plus_te8_and_counts"
    ].iloc[0]
    report = f"""# Phase 8 nested product target-encoding results

## Outcome

The study completed on `train.csv` only. It repeated five-fold stratified OOF modeling at seeds `{FOLD_SEEDS}` with the same LightGBM family and shared, inner-selected boosting rounds. It trained {len(VARIANTS) * len(FOLD_SEEDS) * FOLD_COUNT} outer-fold models (plus {len(FOLD_SEEDS) * FOLD_COUNT} inner stopping models). No test rows, submission, new holdout, external data, or model-family/hyperparameter sweep was used.

## Main claim test

- Mean seed-pooled AUC for raw + Phase 3 product + nested 8-key target means: `{primary.mean_seed_pooled_auc:.9f}` (seed SD `{primary.sd_seed_pooled_auc:.9f}`).
- Mean paired pooled delta vs raw + Phase 3 product: `{primary.mean_pooled_delta_vs_phase3_product:+.9f}`.
- Fold improvements vs raw + Phase 3 product: `{int(primary.positive_folds_vs_phase3_product_out_of_15)}/15`; positive seed-pooled deltas: `{int(primary.positive_seed_pooled_deltas_vs_phase3_product_out_of_3)}/3`.
- 10-key means: mean pooled delta `{te10.mean_pooled_delta_vs_phase3_product:+.9f}`, `{int(te10.positive_folds_vs_phase3_product_out_of_15)}/15` positive folds.
- Eight target means plus log-count features: mean pooled delta `{count_variant.mean_pooled_delta_vs_phase3_product:+.9f}`, `{int(count_variant.positive_folds_vs_phase3_product_out_of_15)}/15` positive folds.

See `SEED_METRICS.csv` for every seed and `FOLD_RESULTS.csv` for all 90 outer-fold fits. The primary metric is the mean of three seed-specific pooled OOF AUCs, not an aggregate over three duplicate predictions per applicant.

## Encoding integrity

Training-row encodings are inner-fold OOF within each outer fit fold. An inner row's target is excluded from its target map. Outer-validation encodings use target maps built only from the complete outer-fit labels. Smoothing alpha is fixed at `{SMOOTHING}`; unseen keys fall back to that fit partition's target prior. No key definition uses TARGET, and no fit/apply indices overlap. Details and key cardinality/singleton/unseen rates are in `TARGET_ENCODING_INTEGRITY.csv` and `PRODUCT_KEY_SUPPORT.csv`.

The product key catalog is a reconstruction of Claude's text, **not an exact source reproduction**: raw A/C ratios are rounded to 3/4/5 decimals; KQ keys combine the rounded A/C representation with nearest integer `(C/G - 1)/0.0066` on cash loans; P keys use the unverified fixed offset 130.5; AG3 is rounded A/G. Eight keys are `Q3,Q4,Q5,KQ3,KQ4,P4,P5,AG3`; the 10-key candidate adds `P3,KQ5`. These constants are hypotheses only, not verified loan terms, fees, or product codes.

## Missingness and sentinel proposals

The matched raw baseline already contains per-numeric-column missing indicators, a raw missing count, external-score availability count, and employment-sentinel handling. These proposals were audited as already present rather than redundantly re-added. See `MISSINGNESS_SENTINEL_AUDIT.md`.

## Rule and evidence status

The local official rule audit finds target encoding **unresolved in rule scope** (not expressly prohibited in the supplied text), while the former implementation failed validation nesting. The current encodings are nested correctly for research, but this run does not resolve eligibility. Do not submit a target-encoded model without organizer clarification.

All three seeds reuse the same labeled dataset and prior project research has exposed these labels. Repeated folds assess sensitivity to partition seeds but are not independent confirmations; the many candidate keys also create selection risk. No 0.77 claim or champion promotion follows from this run. Claude's HistGradientBoosting AUCs and these LightGBM AUCs are not direct comparisons because the learner, feature set, baseline, and exact key code differ.

## Final status

Interpret `AGGREGATE_COMPARISON.csv` as evidence about whether supervised product-key signal survives proper nested encoding on repeated CV seeds. Any observed gain remains exploratory and conditional on rule clarification. Preserve all candidates and the per-row OOF vectors; do not tune a final blend on these same predictions.
"""
    (out / "PHASE8_RESULTS_REPORT.md").write_text(report, encoding="utf-8")

    eligibility = pd.DataFrame(
        [
            {
                "method": "Fold-local nested target encoding from supplied training labels",
                "rule_status": "UNRESOLVED; not expressly prohibited in local text",
                "validation_status": "NESTED CORRECTLY FOR THIS RESEARCH RUN",
                "submission_status": "DO NOT SUBMIT WITHOUT ORGANIZER CLARIFICATION",
            },
            {
                "method": "Row-wise amount-ratio product keys",
                "rule_status": "ALLOWED AS TRANSFORMS OF SUPPLIED FIELDS; semantics unverified",
                "validation_status": "TARGET MAPS NESTED; formula hypotheses fixed before training",
                "submission_status": "REQUIRES review of target-encoding rule status",
            },
            {
                "method": "Test-covariate fitting / external data / manual labels",
                "rule_status": "NOT USED",
                "validation_status": "NOT USED",
                "submission_status": "NOT APPLICABLE",
            },
        ]
    )
    eligibility.to_csv(out / "COMPETITION_ELIGIBILITY.csv", index=False)

    finished = datetime.now(timezone.utc)
    log("status=COMPLETED")
    log(f"finished_utc={finished.isoformat()}")
    log(f"runtime_seconds={time.perf_counter() - wall:.3f}")
    log(f"python={sys.version}")
    log(f"platform={platform.platform()}")
    log(f"train_sha256={source_hashes[str(TRAIN_PATH)]}")
    log(f"phase3_source_sha256={source_hashes[str(PHASE3_SOURCE)]}")
    log(f"phase7_runner_sha256={source_hashes[str(PHASE7_RUNNER)]}")
    log(f"variants={len(VARIANTS)}; seeds={FOLD_SEEDS}; folds={FOLD_COUNT}")
    log("test_file_opened=NO; external_data=NO; target_maps_outer_validation_labels=NO")
    log("submission_created=NO; champion_modified=NO; holdout_scored=NO")
    log(f"output_directory={out}")
    manifest_rows = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(out / "SHA256_MANIFEST.csv", index=False)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    value = parser.parse_args().run_id
    if not value.replace("T", "").replace("Z", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only alphanumeric characters, T, Z, and underscore")
    raise SystemExit(main(value))
