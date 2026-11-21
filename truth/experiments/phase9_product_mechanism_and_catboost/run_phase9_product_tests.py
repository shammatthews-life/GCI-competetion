from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from importlib.metadata import version
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
EXPERIMENT = ROOT / "experiments" / "phase9_product_mechanism_and_catboost"
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
PHASE3B_SOURCE = (
    ROOT
    / "experiments"
    / "phase3b_deep_validation"
    / "phase3b_ablation.py"
)
PHASE7_RUNNER = (
    ROOT
    / "experiments"
    / "phase7_product_feature_discovery"
    / "run_phase7_product_discovery.py"
)
PHASE8_RUNNER = (
    ROOT
    / "experiments"
    / "phase8_nested_product_target_encoding"
    / "run_phase8_nested_product_te.py"
)
PHASE8_AUDIT = (
    ROOT
    / "experiments"
    / "phase8_nested_product_target_encoding"
    / "audit_phase8_results.py"
)
PHASE8_RUN = (
    ROOT
    / "experiments"
    / "phase8_nested_product_target_encoding"
    / "runs"
    / "phase8_20261104T080000Z"
)

ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD_SEEDS = (42, 7, 2026)
N_SPLITS = 5
INNER_SPLITS = 5
TE_SMOOTHING = 30.0
FEE_HYPOTHESIS = 130.5
STEP_HYPOTHESIS = 0.0066
LGBM_SEED_BASE = 20261170
CAT_SEED_BASE = 20261270
CAT_THREADS = min(8, os.cpu_count() or 1)
CAT_TASK_TYPE = "GPU"
CAT_GPU_DEVICE = "0"
CAT_GPU_RAM_PART = 0.80

LGBM_VARIANTS = (
    "product_baseline",
    "te_q3",
    "te_q4",
    "te_q5",
    "te_q4_cash",
    "te_amount_ca",
    "te_amount_gc",
    "te_exact_gca",
    "te_k_only",
    "te_q4_plus_k",
    "ordinal_q4",
)

CAT_NATIVE_KEYS = ("Q3", "Q4", "Q5", "KQ3", "KQ4", "P4", "P5", "AG3")
CAT_VARIANTS = (
    "cat_product_baseline",
    "cat_native_q4",
    "cat_native_8_keys",
    "cat_native_8_keys_plus_crosses",
)
CATEGORICAL_CROSSES = (
    "CAT_ORGANIZATION_X_OCCUPATION",
    "CAT_INCOME_TYPE_X_EDUCATION",
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
        raise ImportError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[valid] = a[valid] / b[valid]
    return out


def int_key(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(values * (10**decimals))
    out = np.full(len(values), "MISSING", dtype=object)
    finite = np.isfinite(scaled)
    out[finite] = scaled[finite].astype(np.int64).astype(str)
    return out.astype(str)


def exact_component_key(raw: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    parts = []
    for column in columns:
        values = raw[column].to_numpy(dtype=np.float64)
        encoded = np.full(len(values), "MISSING", dtype=object)
        finite = np.isfinite(values)
        encoded[finite] = np.fromiter(
            (
                np.format_float_positional(value, unique=True, trim="-")
                for value in values[finite]
            ),
            dtype=object,
            count=int(finite.sum()),
        )
        parts.append(encoded)
    stacked = np.column_stack(parts)
    return np.asarray(
        ["|".join(row) for row in stacked],
        dtype=str,
    )


def product_key_catalog(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    goods = raw["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
    q = safe_ratio(annuity, credit)
    q3, q4, q5 = (int_key(q, d) for d in (3, 4, 5))
    p = safe_ratio(annuity - FEE_HYPOTHESIS, credit)
    p4, p5 = int_key(p, 4), int_key(p, 5)
    ag3 = int_key(safe_ratio(annuity, goods), 3)
    excess = safe_ratio(credit, goods) - 1.0
    k = excess / STEP_HYPOTHESIS
    k_valid = (
        (contract == "Cash loans")
        & np.isfinite(k)
        & (credit > goods)
        & (goods > 0)
    )
    k_int = np.full(len(raw), "NONCASH_OR_INVALID", dtype=object)
    k_int[k_valid] = np.rint(k[k_valid]).astype(np.int64).astype(str)

    def cross(a: np.ndarray, b: np.ndarray, eligible: np.ndarray | None = None):
        values = np.asarray([f"{x}|{z}" for x, z in zip(a, b)], dtype=object)
        if eligible is not None:
            values[~eligible] = "NONCASH_OR_INVALID"
        return values.astype(str)

    q4_cash = q4.copy()
    q4_cash[contract != "Cash loans"] = "NONCASH_OR_INVALID"
    keys = {
        "Q3": q3,
        "Q4": q4,
        "Q5": q5,
        "Q4_CASH_ONLY": q4_cash,
        "AMOUNT_CA": cross(
            contract,
            exact_component_key(raw, ("AMT_CREDIT", "AMT_ANNUITY")),
        ),
        "AMOUNT_GC": cross(
            contract,
            exact_component_key(raw, ("AMT_GOODS_PRICE", "AMT_CREDIT")),
        ),
        "AMOUNT_GCA": cross(
            contract,
            exact_component_key(
                raw, ("AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY")
            ),
        ),
        "K_ONLY": k_int.astype(str),
        "Q4_PLUS_K": cross(
            k_int, q4, eligible=k_valid & (q4 != "MISSING")
        ),
        "P4": p4,
        "P5": p5,
        "AG3": ag3,
        "KQ3": cross(k_int, q3, eligible=k_valid & (q3 != "MISSING")),
        "KQ4": cross(k_int, q4, eligible=k_valid & (q4 != "MISSING")),
    }
    return keys


def encode_map(
    keys: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    smoothing: float = TE_SMOOTHING,
) -> tuple[np.ndarray, np.ndarray]:
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    apply_idx = np.asarray(apply_idx, dtype=np.int64)
    if np.intersect1d(fit_idx, apply_idx).size:
        raise ValueError("Target-map fit/apply indices overlap")
    prior = float(labels[fit_idx].mean())
    frame = pd.DataFrame(
        {"key": keys[fit_idx], "target": labels[fit_idx].astype(np.float64)}
    )
    grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
    means = (grouped["sum"] + smoothing * prior) / (
        grouped["count"] + smoothing
    )
    apply_keys = pd.Series(keys[apply_idx])
    encoded = apply_keys.map(means).fillna(prior).to_numpy(dtype=np.float32)
    counts = apply_keys.map(grouped["count"]).fillna(0).to_numpy(dtype=np.float32)
    return encoded, counts


def nested_single_key_encode(
    keys: np.ndarray,
    labels: np.ndarray,
    outer_fit_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int | bool]]]:
    output = np.full(len(outer_fit_idx), np.nan, dtype=np.float32)
    local_y = labels[outer_fit_idx]
    splitter = StratifiedKFold(
        n_splits=INNER_SPLITS, shuffle=True, random_state=seed
    )
    rows = []
    for inner_fold, (fit_local, apply_local) in enumerate(
        splitter.split(np.zeros(len(outer_fit_idx)), local_y), start=1
    ):
        fit_idx = outer_fit_idx[fit_local]
        apply_idx = outer_fit_idx[apply_local]
        encoded, _ = encode_map(keys, labels, fit_idx, apply_idx)
        output[apply_local] = encoded
        rows.append(
            {
                "inner_fold": inner_fold,
                "fit_rows": len(fit_idx),
                "apply_rows": len(apply_idx),
                "fit_apply_disjoint": not bool(
                    np.intersect1d(fit_idx, apply_idx).size
                ),
            }
        )
    if not np.isfinite(output).all():
        raise AssertionError("Single-key nested encodings are incomplete")
    return output, rows


def train_lgbm_fold(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_valid: pd.DataFrame,
    rounds: int,
    baseline,
    seed: int,
) -> np.ndarray:
    fit_matrix, valid_matrix = baseline.make_lightgbm_frames(x_fit, x_valid)
    params = {
        key: value
        for key, value in baseline.MODEL_CONFIG["lightgbm"].items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = seed
    model = lgb.train(
        params,
        lgb.Dataset(fit_matrix, label=y_fit),
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    prediction = model.predict(valid_matrix, num_iteration=rounds)
    del model, fit_matrix, valid_matrix
    return np.asarray(prediction, dtype=np.float64)


def build_cat_keys(raw: pd.DataFrame, keys: dict[str, np.ndarray]) -> pd.DataFrame:
    data = {f"PRODUCT_KEY_{name}": keys[name] for name in CAT_NATIVE_KEYS}
    income_type = raw["NAME_INCOME_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
    education = raw["NAME_EDUCATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
    organization = raw["ORGANIZATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
    occupation = raw["OCCUPATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
    data[CATEGORICAL_CROSSES[0]] = np.asarray(
        [f"{a}|{b}" for a, b in zip(organization, occupation)], dtype=str
    )
    data[CATEGORICAL_CROSSES[1]] = np.asarray(
        [f"{a}|{b}" for a, b in zip(income_type, education)], dtype=str
    )
    return pd.DataFrame(data)


def cat_rounds(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    cat_cols: list[str],
    seed: int,
) -> int:
    fit_local, stop_local = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.1,
            random_state=seed,
        ).split(np.zeros(len(y_fit)), y_fit)
    )
    model = CatBoostClassifier(
        iterations=2500,
        depth=7,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        border_count=128,
        task_type=CAT_TASK_TYPE,
        devices=CAT_GPU_DEVICE,
        gpu_ram_part=CAT_GPU_RAM_PART,
        thread_count=CAT_THREADS,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        x_fit.iloc[fit_local],
        y_fit[fit_local],
        cat_features=cat_cols,
        eval_set=(x_fit.iloc[stop_local], y_fit[stop_local]),
        early_stopping_rounds=150,
        verbose=False,
    )
    rounds = int(model.get_best_iteration() + 1)
    del model
    return max(1, rounds)


def train_catboost_fold(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_valid: pd.DataFrame,
    cat_cols: list[str],
    rounds: int,
    seed: int,
) -> np.ndarray:
    model = CatBoostClassifier(
        iterations=rounds,
        depth=7,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        border_count=128,
        task_type=CAT_TASK_TYPE,
        devices=CAT_GPU_DEVICE,
        gpu_ram_part=CAT_GPU_RAM_PART,
        thread_count=CAT_THREADS,
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x_fit, y_fit, cat_features=cat_cols, verbose=False)
    prediction = model.predict_proba(x_valid)[:, 1]
    del model
    return np.asarray(prediction, dtype=np.float64)


def main(run_id: str) -> int:
    out = RUNS / f"phase9_{run_id}"
    out.mkdir(parents=True, exist_ok=False)
    log_path = out / "EXECUTION_LOG.txt"
    started = datetime.now(timezone.utc)
    wall = time.perf_counter()

    source_paths = [
        Path(__file__).resolve(),
        TRAIN_PATH,
        LEDGER_PATH,
        RULES_PATH,
        BASELINE_SOURCE,
        PHASE3_SOURCE,
        PHASE3B_SOURCE,
        PHASE7_RUNNER,
        PHASE8_RUNNER,
        PHASE8_AUDIT,
        PHASE8_RUN / "OOF_PREDICTIONS.csv",
        PHASE8_RUN / "SEED_METRICS.csv",
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required source missing: {path}")
    hashes = {str(path): sha256_file(path) for path in source_paths}
    pd.DataFrame(
        [
            {"path": str(path), "sha256": hashes[str(path)], "bytes": path.stat().st_size}
            for path in source_paths
        ]
    ).to_csv(out / "PHASE9_SOURCE_PROVENANCE.csv", index=False)

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    log("status=RUNNING")
    log(f"run_id={run_id}")
    log(f"started_utc={started.isoformat()}")
    gpu_info = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (out / "GPU_DEVICE_INFO.txt").write_text(gpu_info + "\n", encoding="utf-8")
    log(f"gpu_device_info={gpu_info}")

    plan = f"""# Phase 9 pre-registered experiment plan

## Workstreams

**A. Product-mechanism discrimination.** Compare annuity/credit identity keys at 3/4/5 decimals against exact loan amount keys (C,A), (G,C), (G,C,A), financing multiplier K, cash-only ratio identity, and Q4+K. All target means are nested inner-OOF within each outer-fit fold; smoothing alpha={TE_SMOOTHING}. Three five-fold seeds: {FOLD_SEEDS}. The baseline is the exact raw+Phase-3 product feature block.

**B. Model-family/encoding.** On the same seeds/folds, compare GPU CatBoost raw+product baseline against native categorical Q4, eight product keys, and eight keys plus the predeclared organization×occupation and income-type×education crosses. CatBoost iterations are selected on an inner split of outer-fit rows on the baseline, shared across CatBoost candidates; outer validation never selects iterations. LightGBM remains CPU because the installed build has no GPU tree learner.

## Hypotheses

- H1 product identity: ratio-key encodings should outperform exact amount-pair/triple and K-only encodings. Predictive gain does not establish risk-based pricing or causation.
- H2 smooth term vs discrete identity: compare Q3/Q4/Q5 key resolution. Finer-key gain is consistent with higher-resolution identity, but not proof that keys are contractual products.
- H3 histogram resolution: compare high-resolution unsupervised ordinal Q4 against supervised target means and CatBoost native categorical keys. A target-mean advantage indicates useful supervised group-risk estimates beyond simple ordered precision; it cannot rule out every histogram/representation explanation.
- Segment claims: use saved cross-fit Phase 8 OOF predictions for cash/revolving, age quartiles, and pensioner subgroup diagnostics. These are exploratory subgroup summaries with no causal interpretation.
- Annuity fee claim: verify repeated exact (goods,credit) cells' adjacent annuity gaps and compare the 130.5 gap to nearby placebo widths; do not infer multiplicity or a true fee.

## Restrictions and reporting

Only the supplied training CSV is opened. No test data, external data, or test-to-test coupling. Target encoding is nested correctly but remains unresolved by the local competition rules; results are research-only pending written organizer guidance. No holdout is scored, no submission is generated, and no champion is replaced. All seeds reuse historically examined labels and are sensitivity checks, not independent confirmation.
"""
    (out / "PHASE9_EXECUTION_PLAN.md").write_text(plan, encoding="utf-8")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    if len(train) != 171202 or train[ID].isna().any() or train[ID].duplicated().any():
        raise ValueError("Train row/ID validation failed")
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("TARGET is not complete binary labels")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)

    baseline = load_module("phase9_baseline", BASELINE_SOURCE)
    phase3 = load_module("phase9_phase3", PHASE3_SOURCE)
    phase3b = load_module("phase9_phase3b", PHASE3B_SOURCE)
    phase7 = load_module("phase9_phase7", PHASE7_RUNNER)
    if list(phase3b.ALL_PRODUCT) != phase7.ALL_PRODUCT:
        raise AssertionError("Phase 3B and Phase 7 product feature blocks differ")

    keys = product_key_catalog(raw)
    base_features = baseline.build_rowwise_features(raw)
    if not {
        "DAYS_EMPLOYED_SENTINEL",
        "DAYS_EMPLOYED_CLEAN",
        "RAW_MISSING_COUNT",
        "EXT_COUNT",
    }.issubset(base_features.columns):
        raise AssertionError("Expected missingness/sentinel features already in baseline")

    lgbm_predictions: dict[str, np.ndarray] = {}
    lgbm_fold_rows: list[dict[str, object]] = []
    map_integrity: list[dict[str, object]] = []
    key_support: list[dict[str, object]] = []
    for seed in FOLD_SEEDS:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        folds = list(cv.split(np.zeros(len(y)), y))
        seed_oof = {
            variant: np.full(len(y), np.nan, dtype=np.float64)
            for variant in LGBM_VARIANTS
        }
        for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
            fold_start = time.perf_counter()
            seed_fold = seed * 100 + fold
            log(f"LGBM seed={seed}; fold={fold}/{N_SPLITS}; prepare")
            raw_fit = raw.iloc[fit_idx].reset_index(drop=True)
            raw_valid = raw.iloc[valid_idx].reset_index(drop=True)
            y_fit, y_valid = y[fit_idx], y[valid_idx]
            rounds = phase7.select_rounds(
                baseline,
                raw_fit,
                y_fit,
                seed=seed_fold + 100000,
            )
            raw_fit_features = base_features.iloc[fit_idx].reset_index(drop=True)
            raw_valid_features = base_features.iloc[valid_idx].reset_index(drop=True)
            _, _, fit_product, valid_product = phase7.prepare_product(
                phase3, raw_fit, raw_valid
            )
            fit_product = fit_product[phase7.ALL_PRODUCT].reset_index(drop=True)
            valid_product = valid_product[phase7.ALL_PRODUCT].reset_index(drop=True)
            fit_base_product = pd.concat(
                [raw_fit_features, fit_product], axis=1
            )
            valid_base_product = pd.concat(
                [raw_valid_features, valid_product], axis=1
            )
            seed_map = {}
            for key_name in (
                "Q3",
                "Q4",
                "Q5",
                "Q4_CASH_ONLY",
                "AMOUNT_CA",
                "AMOUNT_GC",
                "AMOUNT_GCA",
                "K_ONLY",
                "Q4_PLUS_K",
            ):
                encoded, inner_rows = nested_single_key_encode(
                    keys[key_name], y, fit_idx, seed=seed_fold + 200000
                )
                valid_encoded, valid_count = encode_map(
                    keys[key_name], y, fit_idx, valid_idx
                )
                if not np.isfinite(encoded).all() or not np.isfinite(valid_encoded).all():
                    raise AssertionError(f"Invalid target encoding for {key_name}")
                if len(inner_rows) != INNER_SPLITS:
                    raise AssertionError("Incorrect nested-fold count")
                seed_map[key_name] = (
                    encoded,
                    valid_encoded,
                    np.log1p(valid_count),
                )
                fit_counts = pd.Series(keys[key_name][fit_idx]).value_counts()
                key_support.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "key": key_name,
                        "fit_unique_keys": int(fit_counts.size),
                        "fit_singleton_share": float(
                            fit_counts[fit_counts == 1].sum() / len(fit_idx)
                        ),
                        "validation_unseen_share": float(
                            (~pd.Series(keys[key_name][valid_idx]).isin(fit_counts.index)).mean()
                        ),
                    }
                )
                map_integrity.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "key": key_name,
                        "outer_fit_validation_disjoint": True,
                        "inner_map_apply_disjoint": all(
                            row["fit_apply_disjoint"] for row in inner_rows
                        ),
                        "outer_validation_labels_used_for_map": False,
                        "inner_folds": len(inner_rows),
                    }
                )
            for variant in LGBM_VARIANTS:
                if variant == "product_baseline":
                    fit_x, valid_x = fit_base_product, valid_base_product
                elif variant == "ordinal_q4":
                    fit_ord = np.fromiter(
                        (int(v) if v != "MISSING" else np.nan for v in keys["Q4"][fit_idx]),
                        dtype=np.float64,
                        count=len(fit_idx),
                    )
                    valid_ord = np.fromiter(
                        (int(v) if v != "MISSING" else np.nan for v in keys["Q4"][valid_idx]),
                        dtype=np.float64,
                        count=len(valid_idx),
                    )
                    fit_x = pd.concat(
                        [fit_base_product, pd.DataFrame({"Q4_ORDINAL": fit_ord})],
                        axis=1,
                    )
                    valid_x = pd.concat(
                        [valid_base_product, pd.DataFrame({"Q4_ORDINAL": valid_ord})],
                        axis=1,
                    )
                else:
                    key_names = {
                        "te_q3": ("Q3",),
                        "te_q4": ("Q4",),
                        "te_q5": ("Q5",),
                        "te_q4_cash": ("Q4_CASH_ONLY",),
                        "te_amount_ca": ("AMOUNT_CA",),
                        "te_amount_gc": ("AMOUNT_GC",),
                        "te_exact_gca": ("AMOUNT_GCA",),
                        "te_k_only": ("K_ONLY",),
                        "te_q4_plus_k": ("Q4_PLUS_K",),
                    }[variant]
                    fit_extra = {
                        f"TE_{name}": seed_map[name][0] for name in key_names
                    }
                    valid_extra = {
                        f"TE_{name}": seed_map[name][1] for name in key_names
                    }
                    fit_x = pd.concat(
                        [fit_base_product, pd.DataFrame(fit_extra)],
                        axis=1,
                    )
                    valid_x = pd.concat(
                        [valid_base_product, pd.DataFrame(valid_extra)],
                        axis=1,
                    )
                prediction = train_lgbm_fold(
                    fit_x,
                    y_fit,
                    valid_x,
                    rounds,
                    baseline,
                    LGBM_SEED_BASE + seed + fold,
                )
                if not np.isfinite(prediction).all():
                    raise AssertionError(f"Nonfinite LightGBM prediction: {variant}")
                seed_oof[variant][valid_idx] = prediction
                lgbm_fold_rows.append(
                    {
                        "model_family": "LightGBM",
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "fold_auc": float(roc_auc_score(y_valid, prediction)),
                        "fit_rows": len(fit_idx),
                        "validation_rows": len(valid_idx),
                        "shared_rounds": rounds,
                        "fold_runtime_seconds": time.perf_counter() - fold_start,
                    }
                )
                log(
                    f"LGBM seed={seed}; fold={fold}; variant={variant}; "
                    f"auc={lgbm_fold_rows[-1]['fold_auc']:.8f}"
                )
        for variant, prediction in seed_oof.items():
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Incomplete OOF: seed={seed}; {variant}")
            lgbm_predictions[f"seed{seed}__{variant}"] = prediction

    pd.DataFrame(lgbm_fold_rows).to_csv(out / "LIGHTGBM_FOLD_RESULTS.csv", index=False)
    pd.DataFrame(map_integrity).to_csv(out / "TARGET_MAP_INTEGRITY.csv", index=False)
    pd.DataFrame(key_support).to_csv(out / "KEY_SUPPORT.csv", index=False)

    cat_fold_rows: list[dict[str, object]] = []
    cat_predictions: dict[str, np.ndarray] = {}
    raw_cat = base_features.reset_index(drop=True).copy()
    cat_key_frame = build_cat_keys(raw, keys)
    if raw_cat[list(baseline.CATEGORICAL_COLUMNS)].isna().any().any():
        raise AssertionError("Baseline categorical strings must already be filled")

    for seed in FOLD_SEEDS:
        folds = list(
            StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y
            )
        )
        seed_oof = {
            variant: np.full(len(y), np.nan, dtype=np.float64)
            for variant in CAT_VARIANTS
        }
        for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
            seed_fold = seed * 100 + fold
            raw_fit = raw.iloc[fit_idx].reset_index(drop=True)
            raw_valid = raw.iloc[valid_idx].reset_index(drop=True)
            y_fit, y_valid = y[fit_idx], y[valid_idx]
            _, _, product_fit, product_valid = phase7.prepare_product(
                phase3, raw_fit, raw_valid
            )
            product_fit = product_fit[phase7.ALL_PRODUCT].reset_index(drop=True)
            product_valid = product_valid[phase7.ALL_PRODUCT].reset_index(drop=True)
            fit_base = raw_cat.iloc[fit_idx].reset_index(drop=True)
            valid_base = raw_cat.iloc[valid_idx].reset_index(drop=True)
            common_fit = pd.concat([fit_base, product_fit], axis=1)
            common_valid = pd.concat([valid_base, product_valid], axis=1)
            fit_keys = cat_key_frame.iloc[fit_idx].reset_index(drop=True)
            valid_keys = cat_key_frame.iloc[valid_idx].reset_index(drop=True)
            inner_rounds = cat_rounds(
                common_fit,
                y_fit,
                list(baseline.CATEGORICAL_COLUMNS),
                CAT_SEED_BASE + seed_fold + 900000,
            )
            for variant in CAT_VARIANTS:
                if variant == "cat_product_baseline":
                    fit_x, valid_x = common_fit, common_valid
                    cat_cols = list(baseline.CATEGORICAL_COLUMNS)
                elif variant == "cat_native_q4":
                    fit_x = pd.concat(
                        [common_fit, fit_keys[["PRODUCT_KEY_Q4"]]], axis=1
                    )
                    valid_x = pd.concat(
                        [common_valid, valid_keys[["PRODUCT_KEY_Q4"]]], axis=1
                    )
                    cat_cols = list(baseline.CATEGORICAL_COLUMNS) + [
                        "PRODUCT_KEY_Q4"
                    ]
                else:
                    extra_columns = [f"PRODUCT_KEY_{key}" for key in CAT_NATIVE_KEYS]
                    if variant == "cat_native_8_keys_plus_crosses":
                        extra_columns += list(CATEGORICAL_CROSSES)
                        extra_fit = pd.concat(
                            [
                                fit_keys[[f"PRODUCT_KEY_{key}" for key in CAT_NATIVE_KEYS]],
                                raw.iloc[fit_idx][["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE"]]
                                .reset_index(drop=True)
                                .pipe(
                                    lambda d: pd.DataFrame(
                                        {
                                            CATEGORICAL_CROSSES[0]: d["ORGANIZATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
                                            + "|"
                                            + d["OCCUPATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str),
                                            CATEGORICAL_CROSSES[1]: d["NAME_INCOME_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
                                            + "|"
                                            + d["NAME_EDUCATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str),
                                        }
                                    )
                                ),
                            ],
                            axis=1,
                        )
                        extra_valid = pd.concat(
                            [
                                valid_keys[[f"PRODUCT_KEY_{key}" for key in CAT_NATIVE_KEYS]],
                                raw.iloc[valid_idx][["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "ORGANIZATION_TYPE", "OCCUPATION_TYPE"]]
                                .reset_index(drop=True)
                                .pipe(
                                    lambda d: pd.DataFrame(
                                        {
                                            CATEGORICAL_CROSSES[0]: d["ORGANIZATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
                                            + "|"
                                            + d["OCCUPATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str),
                                            CATEGORICAL_CROSSES[1]: d["NAME_INCOME_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str)
                                            + "|"
                                            + d["NAME_EDUCATION_TYPE"].astype("string").fillna("MISSING").to_numpy(dtype=str),
                                        }
                                    )
                                ),
                            ],
                            axis=1,
                        )
                    else:
                        extra_fit = fit_keys[[f"PRODUCT_KEY_{key}" for key in CAT_NATIVE_KEYS]]
                        extra_valid = valid_keys[[f"PRODUCT_KEY_{key}" for key in CAT_NATIVE_KEYS]]
                    fit_x = pd.concat([common_fit, extra_fit], axis=1)
                    valid_x = pd.concat([common_valid, extra_valid], axis=1)
                    cat_cols = list(baseline.CATEGORICAL_COLUMNS) + extra_columns
                if fit_x.columns.duplicated().any() or valid_x.columns.duplicated().any():
                    raise AssertionError(f"Duplicate CatBoost feature columns: {variant}")
                prediction = train_catboost_fold(
                    fit_x,
                    y_fit,
                    valid_x,
                    cat_cols,
                    inner_rounds,
                    CAT_SEED_BASE + seed + fold,
                )
                seed_oof[variant][valid_idx] = prediction
                cat_fold_rows.append(
                    {
                        "model_family": "CatBoost_GPU",
                        "seed": seed,
                        "fold": fold,
                        "variant": variant,
                        "fold_auc": float(roc_auc_score(y_valid, prediction)),
                        "fit_rows": len(fit_idx),
                        "validation_rows": len(valid_idx),
                        "iterations_selected_inner_fit_only": inner_rounds,
                        "categorical_columns": len(cat_cols),
                    }
                )
                log(
                    f"CatBoost seed={seed}; fold={fold}; variant={variant}; "
                    f"auc={cat_fold_rows[-1]['fold_auc']:.8f}; iterations={inner_rounds}"
                )
        for variant, prediction in seed_oof.items():
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Incomplete CatBoost OOF: {variant}")
            cat_predictions[f"seed{seed}__{variant}"] = prediction

    pd.DataFrame(cat_fold_rows).to_csv(out / "CATBOOST_FOLD_RESULTS.csv", index=False)

    metric_rows = []
    for seed in FOLD_SEEDS:
        fold_ids = np.empty(len(y), dtype=np.int8)
        for fold, (_, valid_idx) in enumerate(
            StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(
                np.zeros(len(y)), y
            ),
            start=1,
        ):
            fold_ids[valid_idx] = fold
        for family, predictions, variants in (
            ("LightGBM", lgbm_predictions, LGBM_VARIANTS),
            ("CatBoost_GPU", cat_predictions, CAT_VARIANTS),
        ):
            reference = (
                "product_baseline"
                if family == "LightGBM"
                else "cat_product_baseline"
            )
            reference_pred = predictions[f"seed{seed}__{reference}"]
            ref_auc = float(roc_auc_score(y, reference_pred))
            for variant in variants:
                pred = predictions[f"seed{seed}__{variant}"]
                fold_auc_values = [
                    float(roc_auc_score(y[fold_ids == fold], pred[fold_ids == fold]))
                    for fold in range(1, N_SPLITS + 1)
                ]
                ref_fold_auc = [
                    float(
                        roc_auc_score(
                            y[fold_ids == fold],
                            reference_pred[fold_ids == fold],
                        )
                    )
                    for fold in range(1, N_SPLITS + 1)
                ]
                deltas = np.asarray(fold_auc_values) - np.asarray(ref_fold_auc)
                metric_rows.append(
                    {
                        "model_family": family,
                        "seed": seed,
                        "variant": variant,
                        "pooled_oof_auc": float(roc_auc_score(y, pred)),
                        "delta_vs_family_product_baseline": float(
                            roc_auc_score(y, pred) - ref_auc
                        ),
                        "positive_folds_vs_family_baseline": int((deltas > 0).sum()),
                        "fold_auc_json": json.dumps(fold_auc_values),
                        "fold_delta_json": json.dumps(deltas.tolist()),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "SEED_METRICS.csv", index=False)

    aggregate_rows = []
    for (family, variant), part in metrics.groupby(
        ["model_family", "variant"], sort=False
    ):
        aggregate_rows.append(
            {
                "model_family": family,
                "variant": variant,
                "mean_seed_pooled_auc": float(part.pooled_oof_auc.mean()),
                "seed_pooled_auc_sd": float(part.pooled_oof_auc.std(ddof=1)),
                "mean_delta_vs_family_product_baseline": float(
                    part.delta_vs_family_product_baseline.mean()
                ),
                "positive_folds_across_seeds": int(
                    sum(
                        int(value)
                        for value in part.positive_folds_vs_family_baseline
                    )
                ),
                "positive_seed_pooled_deltas": int(
                    (part.delta_vs_family_product_baseline > 0).sum()
                ),
                "evidence_status": "Repeated-seed exploratory OOF on historically exposed labels",
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out / "AGGREGATE_COMPARISON.csv", index=False)

    prediction_frame = pd.DataFrame({ID: ids, TARGET: y})
    for name, pred in {**lgbm_predictions, **cat_predictions}.items():
        prediction_frame[name] = pred
    prediction_frame.to_csv(out / "OOF_PREDICTIONS.csv", index=False)

    subgroup_rows = []
    phase8_oof = pd.read_csv(PHASE8_RUN / "OOF_PREDICTIONS.csv")
    aligned = train[[ID, TARGET, "NAME_CONTRACT_TYPE", "DAYS_BIRTH", "NAME_INCOME_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].merge(
        phase8_oof,
        on=[ID, TARGET],
        how="left",
        validate="one_to_one",
    )
    if aligned["seed42__product_plus_te8"].isna().any():
        raise AssertionError("Phase 8 subgroup OOF did not align")
    age = -aligned["DAYS_BIRTH"].to_numpy(dtype=np.float64) / 365.25
    age_cuts = np.quantile(age[np.isfinite(age)], [0.25, 0.5, 0.75])
    groups = {
        "all": np.ones(len(aligned), dtype=bool),
        "cash_loans": aligned["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy(),
        "revolving_loans": aligned["NAME_CONTRACT_TYPE"].eq("Revolving loans").to_numpy(),
        "pensioner": aligned["NAME_INCOME_TYPE"].eq("Pensioner").to_numpy(),
        "age_q1_youngest": age <= age_cuts[0],
        "age_q2": (age > age_cuts[0]) & (age <= age_cuts[1]),
        "age_q3": (age > age_cuts[1]) & (age <= age_cuts[2]),
        "age_q4_oldest": age > age_cuts[2],
    }
    ext = aligned[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(dtype=np.float64)
    ext_count = np.isfinite(ext).sum(axis=1)
    groups["ext_available_0"] = ext_count == 0
    groups["ext_available_1"] = ext_count == 1
    groups["ext_available_2plus"] = ext_count >= 2
    for seed in FOLD_SEEDS:
        ref = aligned[f"seed{seed}__raw_plus_phase3_product"].to_numpy(dtype=np.float64)
        cand = aligned[f"seed{seed}__product_plus_te8"].to_numpy(dtype=np.float64)
        for group_name, mask in groups.items():
            if mask.sum() < 200 or np.unique(aligned.loc[mask, TARGET]).size < 2:
                continue
            subgroup_rows.append(
                {
                    "seed": seed,
                    "group": group_name,
                    "n": int(mask.sum()),
                    "positive_n": int(aligned.loc[mask, TARGET].sum()),
                    "default_rate": float(aligned.loc[mask, TARGET].mean()),
                    "phase3_product_auc": float(
                        roc_auc_score(aligned.loc[mask, TARGET], ref[mask])
                    ),
                    "nested_te8_auc": float(
                        roc_auc_score(aligned.loc[mask, TARGET], cand[mask])
                    ),
                    "delta_auc": float(
                        roc_auc_score(aligned.loc[mask, TARGET], cand[mask])
                        - roc_auc_score(aligned.loc[mask, TARGET], ref[mask])
                    ),
                    "age_quartile_cutpoints_years": json.dumps(age_cuts.tolist()),
                    "interpretation": "Post-hoc subgroup diagnostic; exploratory, not causal or independent confirmation.",
                }
            )
    pd.DataFrame(subgroup_rows).to_csv(out / "SUBGROUP_OOF_DIAGNOSTICS.csv", index=False)

    residual_rows = []
    q4_keys = keys["Q4"]
    for seed in FOLD_SEEDS:
        pred = lgbm_predictions[f"seed{seed}__product_baseline"]
        expected = np.clip(pred, 1e-8, 1 - 1e-8)
        residual = y - expected
        stats = pd.DataFrame(
            {
                "key": q4_keys,
                "residual": residual,
                "target": y.astype(np.float64),
                "variance": expected * (1 - expected),
                "age": age,
                "pensioner": raw["NAME_INCOME_TYPE"].eq("Pensioner").to_numpy(dtype=np.float64),
            }
        )
        grouped = stats.groupby("key", sort=False).agg(
            n=("residual", "size"),
            residual_sum=("residual", "sum"),
            variance_sum=("variance", "sum"),
            mean_age=("age", "mean"),
            pensioner_share=("pensioner", "mean"),
            observed_rate=("target", "mean"),
        )
        grouped = grouped[grouped.n >= 300].copy()
        grouped["pearson_residual_z"] = grouped.residual_sum / np.sqrt(
            grouped.variance_sum
        )
        if len(grouped) > 2:
            corr_age = float(grouped["pearson_residual_z"].corr(grouped["mean_age"]))
            corr_pension = float(
                grouped["pearson_residual_z"].corr(grouped["pensioner_share"])
            )
        else:
            corr_age = np.nan
            corr_pension = np.nan
        residual_rows.append(
            {
                "seed": seed,
                "q4_keys_with_at_least_300_rows": len(grouped),
                "share_abs_pearson_z_gt_3": float(
                    (grouped["pearson_residual_z"].abs() > 3).mean()
                ),
                "pearson_z_age_correlation": corr_age,
                "pearson_z_pensioner_share_correlation": corr_pension,
                "interpretation": "Descriptive residual heterogeneity by key; clustered rows and label reuse invalidate naive p-values.",
            }
        )
        grouped.reset_index().to_csv(
            out / f"Q4_PRODUCT_RESIDUALS_SEED_{seed}.csv", index=False
        )
    pd.DataFrame(residual_rows).to_csv(out / "PRODUCT_RESIDUAL_HETEROGENEITY.csv", index=False)

    fee_rows = []
    cash = raw["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    amount = raw.loc[:, ["AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"]].copy()
    amount = amount.loc[cash].dropna()
    pair_groups = amount.groupby(["AMT_GOODS_PRICE", "AMT_CREDIT"])["AMT_ANNUITY"]
    adjacent = []
    for _, values in pair_groups:
        vals = np.sort(values.unique().astype(np.float64))
        if len(vals) > 1:
            adjacent.extend(np.diff(vals).tolist())
    adjacent = np.asarray(adjacent, dtype=np.float64)
    for gap in (130.5, 129.5, 126.5, 120.0, 125.0, 135.0, 140.0):
        exact = np.isclose(adjacent, gap, rtol=0.0, atol=1e-6)
        fee_rows.append(
            {
                "adjacent_annuity_gap": gap,
                "adjacent_gap_count": int(exact.sum()),
                "adjacent_gap_share": float(exact.mean()) if len(exact) else np.nan,
                "n_adjacent_gaps": len(adjacent),
                "cash_loans_only": True,
                "grouped_by_exact_goods_and_credit": True,
                "interpretation": "Observed adjacent amount gap only; not proof of a fee or assignment mechanism.",
            }
        )
    pd.DataFrame(fee_rows).to_csv(out / "FEE_GAP_PLACEBO_AUDIT.csv", index=False)

    report = f"""# Phase 9 product-mechanism and model-family report

## Completion

Train-only work completed with CPU LightGBM (the installed build has no GPU tree learner) and GPU CatBoost, across three five-fold seeds, with no test-file access. Full results are in `AGGREGATE_COMPARISON.csv`, `SEED_METRICS.csv`, `LIGHTGBM_FOLD_RESULTS.csv`, and `CATBOOST_FOLD_RESULTS.csv`.

## H1/H2/H3 product-key discrimination

LightGBM candidates compare the exact raw+Phase-3 product baseline against nested target means for Q3/Q4/Q5; cash-only Q4; exact (credit,annuity), (goods,credit), and (goods,credit,annuity) keys; K-only; Q4+K; and an unsupervised ordinal Q4 feature. All map features are nested inside each outer training fold; alpha={TE_SMOOTHING}; no outer validation TARGET is used for maps or stopping.

Interpret results as prediction evidence only. Q4/Q5 lifting more than Q3 would be consistent with greater product-key resolution, but will not prove discrete contract products or rule out every histogram effect. Ratio keys outperforming exact amount keys would support portable ratio identity rather than row/entity duplication or loan-size buckets. K-only and Q4+K test whether financing multiplier adds distinct signal. CatBoost native categorical results test whether native ordered category statistics recover similar information without hand-built target maps.

## CatBoost

CatBoost was run on the NVIDIA GPU with inner-only iteration selection. It compares raw+product baseline, native Q4, native eight product keys, and eight keys plus organization×occupation and income-type×education crosses. No CatBoost metric is directly comparable to the prior HistGradientBoosting claim; use within-family paired deltas.

## Mechanism diagnostics

- `SUBGROUP_OOF_DIAGNOSTICS.csv`: Phase 8 held-out predictions compared within cash/revolving, pensioner, age quartiles, and external-score availability groups. This checks the claimed subgroup pattern; these are post-hoc diagnostics.
- `PRODUCT_RESIDUAL_HETEROGENEITY.csv` and per-seed Q4 tables: groups with at least 300 rows and OOF residual heterogeneity. These do not support causal pricing claims and have no naive p-values.
- `FEE_GAP_PLACEBO_AUDIT.csv`: exact (goods,credit)-cell adjacent annuity gaps compared with prespecified nearby alternatives. This audits the narrow 130.5 gap claim, not fee multiplicity.

## Rules and confirmation

The official local materials leave target encoding unresolved; nested validation is necessary but does not establish competition permission. All results reuse the same historically exposed training labels. No independent confirmation, 0.77 guarantee, model promotion, or submission is claimed.
"""
    (out / "PHASE9_RESULTS_REPORT.md").write_text(report, encoding="utf-8")

    eligibility = pd.DataFrame(
        [
            ("Nested supervised product-key target maps", "Rule scope unresolved", "Research only; nested correctly"),
            ("CatBoost native product-key categories", "Training-label-derived categorical statistics; rule scope unresolved", "Research only"),
            ("Organization/occupation and income/education crosses", "Rowwise supplied columns", "No external or test data; exploratory"),
            ("Test-to-test coupling, external data, manual labels", "Not used", "Excluded"),
        ],
        columns=["method", "rule_status", "phase9_status"],
    )
    eligibility.to_csv(out / "COMPETITION_ELIGIBILITY.csv", index=False)
    log("status=COMPLETED")
    log(f"finished_utc={datetime.now(timezone.utc).isoformat()}")
    log(f"runtime_seconds={time.perf_counter() - wall:.3f}")
    log(f"python={sys.version}")
    log(f"platform={platform.platform()}")
    log(
        "packages="
        + json.dumps(
            {
                name: version(name)
                for name in ("lightgbm", "catboost", "scikit-learn", "pandas", "numpy")
            },
            sort_keys=True,
        )
    )
    log(f"train_sha256={hashes[str(TRAIN_PATH)]}")
    log(f"test_file_opened=NO; external_data=NO; submission_created=NO")
    log("holdout_scored=NO; champion_modified=NO")
    log(f"lgbm_variants={len(LGBM_VARIANTS)}; lgbm_device=CPU (installed build lacks GPU learner); catboost_variants={len(CAT_VARIANTS)}; catboost_device={CAT_TASK_TYPE}:{CAT_GPU_DEVICE}; seeds={FOLD_SEEDS}; folds={N_SPLITS}")
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
    run_id = parser.parse_args().run_id
    if not run_id.replace("T", "").replace("Z", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only alphanumeric characters, T, Z, and underscore")
    try:
        exit_code = main(run_id)
    except Exception as error:
        run_dir = RUNS / f"phase9_{run_id}"
        log_path = run_dir / "EXECUTION_LOG.txt"
        if log_path.is_file():
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"status=FAILED\nfailure_type={type(error).__name__}\n")
        raise
    raise SystemExit(exit_code)
