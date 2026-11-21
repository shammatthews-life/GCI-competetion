from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE7 = ROOT / "experiments" / "phase7_product_feature_discovery"
RUNS = PHASE7 / "runs"
PHASE6_RUN = ROOT / "experiments" / "phase6_clean_room_research" / "runs" / "phase6_20261104T063447Z"
TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
BASELINE_OOF_PATH = PHASE6_RUN / "RAW_BASELINE_OOF.csv"
PHASE6_MANIFEST = PHASE6_RUN / "SHA256_MANIFEST.csv"
RULES_PATH = ROOT / "reports" / "final_competition_audit" / "OFFICIAL_RULES_AUDIT.md"
DICT_PATH = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition\HomeCredit_columns_description.xlsx"
)
LEDGER_PATH = ROOT / "GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md"
PHASE2_RUNNER = ROOT / "experiments" / "phase2_advanced_feature_engineering" / "run_corrected_validation.py"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE3_SOURCE = ROOT / "experiments" / "phase3_hidden_pattern_discovery" / "run_phase3_hidden_pattern_discovery.py"
PHASE3B_SOURCE = ROOT / "experiments" / "phase3b_deep_validation" / "phase3b_ablation.py"
PHASE3C_REPORT = ROOT / "experiments" / "phase3c_model_intelligence" / "PHASE3C_EXECUTION_REPORT.md"
PHASE6_REPORT = PHASE6_RUN / "PHASE6_EXECUTION_REPORT.md"
PHASE6_AUDIT = PHASE6_RUN / "PHASE2_PHASE3_CODE_AUDIT.md"
PHASE6_HIDDEN = PHASE6_RUN / "HIDDEN_PATTERN_REPORT.md"
PHASE6_ELIGIBILITY = PHASE6_RUN / "COMPETITION_ELIGIBILITY_REGISTER.csv"
PHASE6_VALIDATION = PHASE6_RUN / "VALIDATION_FRAMEWORK_SPEC.md"

ID = "SK_ID_CURR"
TARGET = "TARGET"
HOLDOUT_SEED = 20261112
DEV_FOLD_SEED = 20261113
INNER_SEED_BASE = 20261120
FINAL_INNER_SEED = 20261201
MODEL_SEED_BASE = 20261130
HOLDOUT_FRACTION = 0.20
N_DEV_FOLDS = 5
BOOTSTRAP_REPLICATES = 1000
PAIR_SAMPLES = 500_000

K_COLS = [
    "PROD_K_DECIMAL",
    "PROD_K_NEAREST_INTEGER",
    "PROD_K_LATTICE_RESIDUAL",
    "PROD_K_VALID",
    "PROD_K_FIT_FREQUENCY_LOG",
]
Q_COLS = [
    "PROD_Q3",
    "PROD_Q4",
    "PROD_Q5",
    "PROD_Q5_FIT_FREQUENCY_LOG",
]
P_COLS = [
    "PROD_P3",
    "PROD_P4",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_ADJUSTED_ANNUITY",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_P5_FIT_FREQUENCY_LOG",
]
ALL_PRODUCT = K_COLS + Q_COLS + P_COLS
NO_REDUNDANT = list(dict.fromkeys([
    "PROD_K_NEAREST_INTEGER",
    "PROD_K_LATTICE_RESIDUAL",
    "PROD_K_VALID",
    "PROD_K_FIT_FREQUENCY_LOG",
    "PROD_Q5",
    "PROD_Q5_FIT_FREQUENCY_LOG",
    "PROD_P5",
    "PROD_PF4_FLAG",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL",
    "PROD_P5_FIT_FREQUENCY_LOG",
]))
BLOCKS = {
    "raw_plus_K": K_COLS,
    "raw_plus_Q": Q_COLS,
    "raw_plus_P": P_COLS,
    "raw_plus_KQ": K_COLS + Q_COLS,
    "raw_plus_KP": K_COLS + P_COLS,
    "raw_plus_QP": Q_COLS + P_COLS,
    "raw_plus_all_product": ALL_PRODUCT,
    "raw_plus_no_redundant": NO_REDUNDANT,
}
CONTROLS = [
    "random_permuted",
    "random_keys",
    "random_fee_P",
    "random_step_K",
]

PHASE3_DEFINITIONS = {
    "PROD_K_DECIMAL": "(AMT_CREDIT / AMT_GOODS_PRICE - 1) / step_fit; cash-loan restriction is applied",
    "PROD_K_NEAREST_INTEGER": "round(PROD_K_DECIMAL) when cash, AMT_CREDIT > AMT_GOODS_PRICE, and lattice residual < 0.1; otherwise missing",
    "PROD_K_LATTICE_RESIDUAL": "abs(PROD_K_DECIMAL - round(PROD_K_DECIMAL))",
    "PROD_K_VALID": "1 when cash and AMT_CREDIT > AMT_GOODS_PRICE and lattice residual < 0.1, else 0",
    "PROD_K_FIT_FREQUENCY_LOG": "log1p(fit-row count of rounded PROD_K_DECIMAL, cash-loan fit rows only)",
    "PROD_Q3": "round(AMT_ANNUITY / AMT_CREDIT, 3)",
    "PROD_Q4": "round(AMT_ANNUITY / AMT_CREDIT, 4)",
    "PROD_Q5": "round(AMT_ANNUITY / AMT_CREDIT, 5)",
    "PROD_Q5_FIT_FREQUENCY_LOG": "log1p(fit-row count of rounded Q5, cash-loan fit rows only)",
    "PROD_P3": "round((AMT_ANNUITY - fee_fit) / AMT_CREDIT, 3)",
    "PROD_P4": "round((AMT_ANNUITY - fee_fit) / AMT_CREDIT, 4)",
    "PROD_P5": "round((AMT_ANNUITY - fee_fit) / AMT_CREDIT, 5)",
    "PROD_PF4_FLAG": "1 if fit-frequency(P4) > fit-frequency(Q5) for cash loan, else 0",
    "PROD_FEE_ADJUSTED_ANNUITY": "AMT_ANNUITY - fee_fit",
    "PROD_FEE_RECONSTRUCTION_RESIDUAL": "AMT_ANNUITY - (fee_fit + AMT_CREDIT * round((AMT_ANNUITY-fee_fit)/AMT_CREDIT, 5))",
    "PROD_P5_FIT_FREQUENCY_LOG": "log1p(fit-row count of rounded P5, cash-loan fit rows only)",
}

for _single in ALL_PRODUCT:
    if _single not in PHASE3_DEFINITIONS:
        raise RuntimeError(f"Missing preregistered formula for {_single}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_model_params(baseline, seed: int) -> dict[str, object]:
    config = baseline.MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = seed
    return params


def select_rounds(baseline, raw: pd.DataFrame, y: np.ndarray, seed: int) -> int:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    fit_idx, stop_idx = next(split.split(np.zeros(len(y)), y))
    fit_features = baseline.build_rowwise_features(raw.iloc[fit_idx])
    stop_features = baseline.build_rowwise_features(raw.iloc[stop_idx])
    fit_matrix, stop_matrix = baseline.make_lightgbm_frames(fit_features, stop_features)
    train_set = lgb.Dataset(fit_matrix, label=y[fit_idx])
    stop_set = lgb.Dataset(stop_matrix, label=y[stop_idx], reference=train_set)
    config = baseline.MODEL_CONFIG["lightgbm"]
    model = lgb.train(
        get_model_params(baseline, seed),
        train_set,
        num_boost_round=config["num_boost_round"],
        valid_sets=[stop_set],
        callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)],
    )
    rounds = int(model.best_iteration or model.current_iteration())
    if rounds < 1:
        raise RuntimeError("Inner stopping did not select a positive iteration count")
    return rounds


def prepare_product(phase3, raw_fit: pd.DataFrame, raw_valid: pd.DataFrame):
    params = phase3.fit_fee_and_step(raw_fit)
    maps = phase3.fit_product_maps(raw_fit, params)
    fit_frame = phase3.make_product_features(raw_fit, params, maps)
    valid_frame = phase3.make_product_features(raw_valid, params, maps)
    if list(fit_frame.columns) != list(valid_frame.columns):
        raise AssertionError("Product feature order differs between fit and validation")
    if fit_frame.columns.duplicated().any():
        raise AssertionError("Duplicate Phase 3 product feature names")
    for frame in (fit_frame, valid_frame):
        array = frame.to_numpy(dtype=np.float64)
        if np.isinf(array).any():
            raise AssertionError("Infinite product feature values")
    return params, maps, fit_frame, valid_frame


def make_control(phase3b, raw_fit, raw_valid, family: str):
    return phase3b.make_family(
        raw_fit,
        raw_valid,
        feature_params=None,
        family_name=family,
    )


def bootstrap_auc_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        sample = np.concatenate(
            [
                rng.choice(pos, size=len(pos), replace=True),
                rng.choice(neg, size=len(neg), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[sample], candidate[sample]) - roc_auc_score(
            y[sample], base[sample]
        )
    return {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "delta_mean": float(np.mean(deltas)),
        "delta_ci_2_5": float(np.quantile(deltas, 0.025)),
        "delta_ci_97_5": float(np.quantile(deltas, 0.975)),
        "prob_delta_gt_0": float(np.mean(deltas > 0)),
        "interpretation": "Descriptive paired row bootstrap on exposed development labels; not independent confirmation.",
    }


def rank_disagreement(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[float, float]:
    rho = float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(a), size=PAIR_SAMPLES)
    j = rng.integers(0, len(a), size=PAIR_SAMPLES)
    valid = i != j
    da = a[i[valid]] - a[j[valid]]
    db = b[i[valid]] - b[j[valid]]
    informative = (da != 0) & (db != 0)
    disagreement = float(np.mean(np.sign(da[informative]) != np.sign(db[informative])))
    return rho, disagreement


def lattice_audit(phase3, train: pd.DataFrame, dev_indices: np.ndarray, folds: np.ndarray):
    records = []
    fixed_steps = (0.0066, 0.0132)
    placebo_steps = (0.0057, 0.0097, 0.0173, 0.0231)
    tolerances = (0.01, 0.05, 0.1)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    y = train[TARGET].to_numpy(dtype=np.int8)

    def residuals(frame: pd.DataFrame, step: float):
        cash = frame["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        credit = frame["AMT_CREDIT"].to_numpy(dtype=np.float64)
        goods = frame["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
        valid = cash & np.isfinite(credit) & np.isfinite(goods) & (credit > goods) & (goods > 0)
        coord = np.full(len(frame), np.nan)
        coord[valid] = (credit[valid] / goods[valid] - 1.0) / step
        return valid, np.abs(coord - np.rint(coord))

    if len(dev_indices) != len(folds):
        raise ValueError("Development indices and local fold assignments differ in length")
    for fold in sorted(np.unique(folds)):
        valid_idx = dev_indices[folds == fold]
        fit_idx = dev_indices[folds != fold]
        params = phase3.fit_fee_and_step(raw.iloc[fit_idx])
        estimated = float(params["step_estimate_fit_only"])
        for step, kind in [
            *((s, "prespecified_hypothesis") for s in fixed_steps),
            *((s, "negative_control_placebo") for s in placebo_steps),
            (estimated, "fit_only_selected_step"),
        ]:
            mask, residual = residuals(raw.iloc[valid_idx], step)
            valid_resid = residual[mask]
            row = {
                "fold": int(fold),
                "step": float(step),
                "step_kind": kind,
                "n_cash_credit_gt_goods": int(mask.sum()),
                "mean_absolute_integer_residual": float(np.mean(valid_resid)),
                "median_absolute_integer_residual": float(np.median(valid_resid)),
                "p95_absolute_integer_residual": float(np.quantile(valid_resid, 0.95)),
                "validation_labels_used_for_step_or_features": False,
                "validation_default_rate_descriptive_only": float(y[valid_idx][mask].mean())
                if mask.any()
                else np.nan,
            }
            for tolerance in tolerances:
                row[f"share_residual_lt_{tolerance:g}"] = float(
                    np.mean(valid_resid < tolerance)
                )
            records.append(row)
    return pd.DataFrame(records)


def main(run_id: str) -> int:
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    out = RUNS / f"phase7_{run_id}"
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [
        Path(__file__).resolve(),
        TRAIN_PATH, TEST_PATH, BASELINE_OOF_PATH, PHASE6_MANIFEST, PHASE6_REPORT,
        PHASE6_AUDIT, PHASE6_HIDDEN, PHASE6_ELIGIBILITY, PHASE6_VALIDATION,
        RULES_PATH, DICT_PATH, LEDGER_PATH, PHASE2_RUNNER, BASELINE_SOURCE, PHASE3_SOURCE,
        PHASE3B_SOURCE, PHASE3C_REPORT,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required Phase 7 source missing: {path}")
    source_hashes = {str(path): sha256_file(path) for path in input_paths}
    phase6_manifest = pd.read_csv(PHASE6_MANIFEST)
    expected_baseline_hash = phase6_manifest.loc[
        phase6_manifest.file == "RAW_BASELINE_OOF.csv", "sha256"
    ].item()
    if sha256_file(BASELINE_OOF_PATH) != expected_baseline_hash:
        raise ValueError("Phase 6 baseline OOF SHA-256 does not match its manifest")

    # This is the preregistered primary candidate. Secondary feature ablations
    # are exploratory and cannot replace the primary based on confirmation data.
    primary_candidate = "raw_plus_all_product"
    model_order = [
        "raw_baseline",
        *[f"single:{name}" for name in ALL_PRODUCT],
        *BLOCKS.keys(),
        *CONTROLS,
    ]
    if len(model_order) != len(set(model_order)):
        raise AssertionError("Duplicate model variants in preregistration")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    if len(train) != 171202 or train[ID].isna().any() or train[ID].duplicated().any():
        raise ValueError("Training data row/ID validation failed")
    y = train[TARGET].to_numpy(dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training target is not binary")
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)

    # Read only the test header and identifier integrity fields.
    test_header = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    if test_header != [c for c in train.columns if c != TARGET]:
        raise ValueError("Test schema mismatch")
    if len(test_ids) != 61500 or test_ids.isna().any() or test_ids.duplicated().any():
        raise ValueError("Test IDs failed count/uniqueness checks")
    del test_ids

    phase6_oof = pd.read_csv(BASELINE_OOF_PATH)
    aligned_phase6 = phase6_oof.set_index(ID).reindex(ids)
    if (
        len(phase6_oof) != len(train)
        or phase6_oof[ID].duplicated().any()
        or aligned_phase6[TARGET].isna().any()
        or not np.array_equal(aligned_phase6[TARGET].to_numpy(dtype=np.int8), y)
        or not np.isfinite(aligned_phase6["oof_raw_baseline"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("Phase 6 baseline OOF ID/target/prediction integrity failed")
    del phase6_oof, aligned_phase6

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=HOLDOUT_FRACTION, random_state=HOLDOUT_SEED
    )
    dev_idx, holdout_idx = next(splitter.split(np.zeros(len(y)), y))
    if np.intersect1d(dev_idx, holdout_idx).size or len(dev_idx) + len(holdout_idx) != len(y):
        raise AssertionError("Development/holdout partition overlap or coverage failure")
    dev_fold_ids = np.empty(len(dev_idx), dtype=np.int8)
    cv = StratifiedKFold(n_splits=N_DEV_FOLDS, shuffle=True, random_state=DEV_FOLD_SEED)
    dev_folds = list(cv.split(np.zeros(len(dev_idx)), y[dev_idx]))
    for fold, (_, valid_positions) in enumerate(dev_folds, start=1):
        dev_fold_ids[valid_positions] = fold
    partition = np.full(len(train), "locked_confirmation_holdout", dtype=object)
    partition[dev_idx] = "development"
    partition_folds = np.full(len(train), 0, dtype=np.int8)
    partition_folds[dev_idx] = dev_fold_ids
    partition_manifest = pd.DataFrame(
        {
            ID: ids,
            "partition": partition,
            "development_fold": partition_folds,
        }
    )
    partition_manifest.to_csv(out / "VALIDATION_PARTITION_MANIFEST.csv", index=False)
    partition_hash = sha256_file(out / "VALIDATION_PARTITION_MANIFEST.csv")

    # Lock protocol before training. The manifest contains no labels.
    plan = f"""# Phase 7 execution plan (pre-run lock)

Run ID: `{run_id}`. Primary model comparison and confirmation rule were fixed before this run's Phase 7 model results were computed.

## Phase 6 input verification

- Phase 6 report, audit, hidden-pattern report, eligibility register, validation framework, baseline OOF, and manifest were read.
- Baseline OOF SHA-256: `{source_hashes[str(BASELINE_OOF_PATH)]}`; matches Phase 6 manifest: yes. IDs are unique, all 171,202 rows align to current train TARGET, predictions are finite.
- Phase 3 product implementation and Phase 3B feature selection code were read. Feature formulas, fit-only parameter/maps, rounding, cash-loan masks, and random-control mechanisms are recorded in the formula register and reproduction report.
- Source/input SHA-256 values are in `PHASE7_SOURCE_PROVENANCE.csv`.

## Locked data protocol

- Confirmation partition: `StratifiedShuffleSplit(test_size={HOLDOUT_FRACTION}, random_state={HOLDOUT_SEED})`.
- Development folds: `StratifiedKFold(n_splits={N_DEV_FOLDS}, shuffle=True, random_state={DEV_FOLD_SEED})`, applied only to development rows.
- Development/holdout ID manifest SHA-256: `{partition_hash}`. The manifest records IDs and roles, not labels.
- Important contamination limitation: all supplied training labels have already participated in earlier project OOF analyses. This partition is locked against *new Phase 7* feature/model decisions, but it is **not an untouched independent confirmation set**. Its score is reported once as a secondary holdout diagnostic only; it cannot satisfy promotion confirmation.

## Frozen model and feature comparisons

- Primary candidate: `{primary_candidate}` (raw baseline plus the exact 16-column Phase 3B all-product block).
- Matched baseline: `raw_baseline`.
- Secondary discovery: each of the 16 individual product columns; K, Q, P, KQ, KP, QP blocks; existing `no_redundant` block; four negative controls.
- All variants use the same fold-specific boosting rounds selected on an inner split using only outer development-fit raw baseline features. No candidate-specific stopping or parameter tuning.
- Product parameters and key frequencies are refit from outer development-fit rows only. No TARGET is used in decoder fitting or feature maps.
- Primary holdout comparison, evaluated once after development CV is complete: raw baseline vs frozen `raw_plus_all_product`. No feature/weight/threshold changes after seeing holdout results.
- Main metric ROC-AUC. Report pooled development OOF, fold AUCs/deltas, paired bootstrap interval (descriptive; repeated-label exposure), rank correlation/disagreement, runtime, and one-time holdout AUC.

## Negative controls and promotion

- `random_permuted`, `random_keys`, `random_fee_P` (fixed false offset 177.1), `random_step_K` (fixed false step 0.0097).
- An apparent control gain is treated as noise/selection warning, not useful signal.
- No score target is promised. Promotion requires eligible methods, correct OOF, gains over raw and existing product baseline, stable folds, controls beaten, and genuinely unseen labeled confirmation. This data set cannot provide the last requirement.
"""
    (out / "PHASE7_EXECUTION_PLAN.md").write_text(plan, encoding="utf-8")

    provenance_rows = [
        {
            "path": str(path),
            "sha256": source_hashes[str(path)],
            "bytes": path.stat().st_size,
            "role": "input_or_source",
        }
        for path in input_paths
    ]
    pd.DataFrame(provenance_rows).to_csv(out / "PHASE7_SOURCE_PROVENANCE.csv", index=False)

    formula_rows = []
    for name in ALL_PRODUCT:
        group = "K" if name in K_COLS else "Q" if name in Q_COLS else "P"
        alias = {
            "PROD_Q3": "Rounded representation of annuity/credit; raw ratio already available.",
            "PROD_Q4": "Rounded representation of annuity/credit; raw ratio already available.",
            "PROD_Q5": "Rounded representation of annuity/credit; raw ratio already available.",
            "PROD_P3": "Fold-estimated offset ratio; fee interpretation is unverified.",
            "PROD_P4": "Fold-estimated offset ratio; fee interpretation is unverified.",
            "PROD_P5": "Fold-estimated offset ratio; fee interpretation is unverified.",
            "PROD_FEE_ADJUSTED_ANNUITY": "Annuity minus fit-only offset; offset not a proven fee.",
            "PROD_K_DECIMAL": "Fit-only lattice coordinate hypothesis; not a loan tier claim.",
            "PROD_K_NEAREST_INTEGER": "Thresholded rounded lattice coordinate; hypothesis only.",
            "PROD_K_LATTICE_RESIDUAL": "Distance to hypothesized integer grid.",
            "PROD_K_VALID": "Rule indicator for hypothesized grid support.",
            "PROD_K_FIT_FREQUENCY_LOG": "Fit-only frequency statistic over cash-loan lattice keys.",
            "PROD_Q5_FIT_FREQUENCY_LOG": "Fit-only frequency over cash-loan rounded Q keys.",
            "PROD_P5_FIT_FREQUENCY_LOG": "Fit-only frequency over cash-loan offset-ratio keys.",
            "PROD_PF4_FLAG": "Fit-only relative frequency flag; no fee meaning asserted.",
            "PROD_FEE_RECONSTRUCTION_RESIDUAL": "Rounding reconstruction residual; offset meaning unverified.",
        }[name]
        formula_rows.append(
            {
                "feature": name,
                "group": group,
                "exact_source_formula": PHASE3_DEFINITIONS[name],
                "implementation_source": str(PHASE3_SOURCE),
                "implementation_source_sha256": source_hashes[str(PHASE3_SOURCE)],
                "parameter_or_map_fit_scope": (
                    "outer-fit rows only" if ("FIT_FREQUENCY" in name or group in {"K", "P"}) else "rowwise using supplied row; map not needed"
                ),
                "eligibility": "eligible for this training-only fold-local experiment; source semantics remain hypotheses",
                "interpretation_caution": alias,
                "alias_or_duplicate_warning": (
                    "Potentially redundant with raw or another product column; model ablation required."
                ),
            }
        )
    pd.DataFrame(formula_rows).to_csv(out / "PRODUCT_FEATURE_FORMULA_REGISTER.csv", index=False)

    eligibility_update = pd.DataFrame(
        [
            ("Raw row-wise supplied columns", "ELIGIBLE", "No external/test-derived feature fitting."),
            ("K lattice product features", "ELIGIBLE WITH RESTRICTIONS", "Step fitted on outer fit rows; all claims numerical, not actual contract tiers."),
            ("Q rounded annuity/credit", "ELIGIBLE WITH RESTRICTIONS", "Rowwise supplied values; annuity period is unspecified."),
            ("P offset-adjusted ratio features", "ELIGIBLE WITH RESTRICTIONS", "Offset fitted only on outer fit rows; do not call it an actual fee."),
            ("Fit-only product key frequencies", "ELIGIBLE WITH RESTRICTIONS", "Counts use outer fit rows only, unseen keys map to zero."),
            ("Randomized negative controls", "ELIGIBLE FOR VALIDATION ONLY", "Never candidate features; used to assess generic feature/noise gain."),
            ("Pooled train/test maps", "EXCLUDED", "Unresolved under local competition rules."),
            ("Target encoding", "EXCLUDED", "Unresolved under local rules; not required for this feature family."),
            ("Test-to-test coupling / entity matching", "EXCLUDED", "Ambiguous/high risk and outside this experiment."),
            ("External data/manual labels", "PROHIBITED", "Not used."),
        ],
        columns=["method", "status", "control"],
    )
    eligibility_update.to_csv(out / "COMPETITION_ELIGIBILITY_UPDATE.csv", index=False)

    phase3 = load_module("phase7_phase3_features", PHASE3_SOURCE)
    phase3b = load_module("phase7_phase3b_ablation", PHASE3B_SOURCE)
    baseline = load_module("phase7_baseline_source", BASELINE_SOURCE)
    if not set(ALL_PRODUCT).issubset(phase3.MODEL_FEATURES):
        raise AssertionError("Phase 7 product columns are not a subset of Phase 3 model features")
    if list(phase3b.ALL_PRODUCT) != ALL_PRODUCT:
        raise AssertionError("Phase 7 K/Q/P blocks differ from Phase 3B block definitions")
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    fold_predictions = {
        name: np.full(len(dev_idx), np.nan, dtype=np.float64) for name in model_order
    }
    fold_records = []
    feature_diag_rows = []
    runtime_by_variant = {name: 0.0 for name in model_order}
    best_rounds: dict[int, int] = {}

    for fold, (local_fit_idx, local_valid_idx) in enumerate(dev_folds, start=1):
        fit_global = dev_idx[local_fit_idx]
        valid_global = dev_idx[local_valid_idx]
        y_fit, y_valid = y[fit_global], y[valid_global]
        raw_fit, raw_valid = raw_train.iloc[fit_global], raw_train.iloc[valid_global]

        rounds = select_rounds(
            baseline,
            raw_fit.reset_index(drop=True),
            y_fit,
            INNER_SEED_BASE + fold,
        )
        best_rounds[fold] = rounds
        base_fit = baseline.build_rowwise_features(raw_fit)
        base_valid = baseline.build_rowwise_features(raw_valid)
        params, maps, prod_fit, prod_valid = prepare_product(phase3, raw_fit, raw_valid)
        if list(prod_fit.columns) != list(prod_valid.columns):
            raise AssertionError("Product feature columns differ by fold")
        for feature in ALL_PRODUCT:
            values = prod_fit[feature].to_numpy(dtype=np.float64)
            validation_values = prod_valid[feature].to_numpy(dtype=np.float64)
            feature_diag_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "fit_missing_rate": float(np.isnan(values).mean()),
                    "validation_missing_rate": float(np.isnan(validation_values).mean()),
                    "fit_unique_nonmissing": int(pd.Series(values).nunique(dropna=True)),
                    "validation_unique_nonmissing": int(pd.Series(validation_values).nunique(dropna=True)),
                    "fit_fee_estimate": float(params["fee_estimate_fit_only"]),
                    "fit_step_estimate": float(params["step_estimate_fit_only"]),
                    "fit_cash_fee_n": int(params["n_cash_fee_fit"]),
                    "fit_cash_step_n": int(params["n_cash_step_fit"]),
                    "feature_fit_excluded_validation": True,
                }
            )
        # Manual formula consistency checks are required on every fold.
        credit = raw_valid["AMT_CREDIT"].to_numpy(dtype=np.float64)
        goods = raw_valid["AMT_GOODS_PRICE"].to_numpy(dtype=np.float64)
        annuity = raw_valid["AMT_ANNUITY"].to_numpy(dtype=np.float64)
        cash = raw_valid["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
        k_manual = (credit / goods - 1.0) / params["step_estimate_fit_only"]
        k_manual[~cash] = np.nan
        q5_manual = np.round(annuity / credit, 5)
        p5_manual = np.round((annuity - params["fee_estimate_fit_only"]) / credit, 5)
        for name, observed, expected in (
            ("PROD_K_DECIMAL", prod_valid["PROD_K_DECIMAL"].to_numpy(), k_manual),
            ("PROD_Q5", prod_valid["PROD_Q5"].to_numpy(), q5_manual),
            ("PROD_P5", prod_valid["PROD_P5"].to_numpy(), p5_manual),
        ):
            mask = np.isfinite(observed) & np.isfinite(expected)
            error = float(np.max(np.abs(observed[mask] - expected[mask]))) if mask.any() else 0.0
            if error > 1e-10:
                raise AssertionError(f"Phase 3 formula mismatch {name} fold={fold}: {error}")
        manual_q5_freq = np.fromiter(
            (float(maps["Q5"].get(value, 0)) if np.isfinite(value) else np.nan for value in np.round(annuity / credit, 5)),
            dtype=np.float64,
            count=len(raw_valid),
        )
        manual_p5_freq = np.fromiter(
            (
                float(maps["P5"].get(value, 0))
                if np.isfinite(value)
                else np.nan
                for value in np.round(
                    (annuity - params["fee_estimate_fit_only"]) / credit, 5
                )
            ),
            dtype=np.float64,
            count=len(raw_valid),
        )
        for name, observed, expected in (
            ("PROD_Q5_FIT_FREQUENCY_LOG", prod_valid["PROD_Q5_FIT_FREQUENCY_LOG"].to_numpy(), np.where(cash, np.log1p(manual_q5_freq), np.nan)),
            ("PROD_P5_FIT_FREQUENCY_LOG", prod_valid["PROD_P5_FIT_FREQUENCY_LOG"].to_numpy(), np.where(cash, np.log1p(manual_p5_freq), np.nan)),
        ):
            mask = np.isfinite(observed) & np.isfinite(expected)
            error = float(np.max(np.abs(observed[mask] - expected[mask]))) if mask.any() else 0.0
            if error > 1e-10:
                raise AssertionError(f"Fold-frequency map mismatch {name} fold={fold}: {error}")
        expected_product_columns = len(phase3.FEATURE_DEFINITIONS)
        if len(prod_fit.columns) != expected_product_columns:
            raise AssertionError(
                f"Expected {expected_product_columns} Phase 3 derived columns; "
                f"got {len(prod_fit.columns)}"
            )

        variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
            "raw_baseline": (base_fit, base_valid),
        }
        for feature in ALL_PRODUCT:
            variants[f"single:{feature}"] = (
                prod_fit[[feature]],
                prod_valid[[feature]],
            )
        for name, cols in BLOCKS.items():
            variants[name] = (prod_fit[cols], prod_valid[cols])
        for control in CONTROLS:
            c_fit, c_valid, c_cols = make_control(
                phase3b, raw_fit, raw_valid, control
            )
            if control == "random_permuted":
                # This historical control permutes rows independently in fit and validation.
                variants[control] = (c_fit[c_cols], c_valid[c_cols])
            else:
                variants[control] = (c_fit[c_cols], c_valid[c_cols])
        if set(variants) != set(model_order):
            missing = set(model_order) - set(variants)
            extra = set(variants) - set(model_order)
            raise AssertionError(f"Variant set mismatch: missing={missing}, extra={extra}")

        for variant in model_order:
            fit_extra, valid_extra = variants[variant]
            if variant == "raw_baseline":
                fit_x, valid_x = fit_extra, valid_extra
            else:
                fit_x = pd.concat([base_fit.reset_index(drop=True), fit_extra.reset_index(drop=True)], axis=1)
                valid_x = pd.concat([base_valid.reset_index(drop=True), valid_extra.reset_index(drop=True)], axis=1)
            if fit_x.columns.duplicated().any():
                raise AssertionError(f"Duplicate model feature names for {variant}")
            fit_matrix, valid_matrix = baseline.make_lightgbm_frames(fit_x, valid_x)
            fold_model_params = get_model_params(baseline, MODEL_SEED_BASE + fold)
            start = time.perf_counter()
            model = lgb.train(
                fold_model_params,
                lgb.Dataset(fit_matrix, label=y_fit),
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            prediction = model.predict(valid_matrix, num_iteration=rounds)
            runtime_by_variant[variant] += time.perf_counter() - start
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Nonfinite OOF for {variant}, fold {fold}")
            fold_predictions[variant][local_valid_idx] = prediction
            auc = float(roc_auc_score(y_valid, prediction))
            fold_records.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "n_fit": len(fit_global),
                    "n_validation": len(valid_global),
                    "positive_validation": int(y_valid.sum()),
                    "boosting_rounds_from_inner_raw_only": rounds,
                    "auc": auc,
                    "feature_count_added": int(fit_extra.shape[1]) if variant != "raw_baseline" else 0,
                    "outer_validation_used_for_stopping_or_feature_fit": False,
                }
            )
            del model, fit_matrix, valid_matrix

    for name, pred in fold_predictions.items():
        if len(pred) != len(dev_idx) or not np.isfinite(pred).all():
            raise AssertionError(f"OOF incomplete for {name}")

    y_dev = y[dev_idx]
    primary_pred = fold_predictions[primary_candidate]
    raw_pred = fold_predictions["raw_baseline"]
    comparison_rows = []
    fold_result = pd.DataFrame(fold_records)
    base_fold = fold_result[fold_result.variant == "raw_baseline"].set_index("fold")["auc"]
    for variant, prediction in fold_predictions.items():
        auc = float(roc_auc_score(y_dev, prediction))
        fold_rows = fold_result[fold_result.variant == variant].sort_values("fold")
        fold_aucs = fold_rows["auc"].to_numpy(dtype=np.float64)
        deltas = fold_aucs - base_fold.reindex(fold_rows.fold).to_numpy(dtype=np.float64)
        rho, disagreement = (1.0, 0.0) if variant == "raw_baseline" else rank_disagreement(
            raw_pred, prediction, seed=HOLDOUT_SEED + len(variant)
        )
        comparison_rows.append(
            {
                "variant": variant,
                "pooled_development_oof_auc": auc,
                "delta_vs_raw_baseline": auc - float(roc_auc_score(y_dev, raw_pred)),
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
                "positive_paired_fold_deltas": int(np.sum(deltas > 0)),
                "fold_deltas_json": json.dumps(
                    {str(int(row.fold)): float(row.auc - base_fold.loc[row.fold]) for row in fold_rows.itertuples()},
                    sort_keys=True,
                ),
                "prediction_spearman_vs_raw": rho,
                "sampled_pairwise_rank_disagreement_vs_raw": disagreement,
                "model_fit_runtime_seconds": runtime_by_variant[variant],
                "interpretation": "Development discovery only; folds and all labels have prior project exposure.",
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    product_baseline_auc = float(
        comparison.loc[
            comparison.variant == primary_candidate, "pooled_development_oof_auc"
        ].iloc[0]
    )
    product_fold_auc = fold_result[fold_result.variant == primary_candidate].set_index("fold")["auc"]
    comparison["delta_vs_existing_product_block"] = (
        comparison["pooled_development_oof_auc"] - product_baseline_auc
    )
    comparison["positive_folds_vs_existing_product_block"] = comparison["variant"].map(
        lambda variant: int(
            np.sum(
                fold_result[fold_result.variant == variant]
                .sort_values("fold")["auc"].to_numpy(dtype=np.float64)
                - product_fold_auc.reindex(
                    fold_result[fold_result.variant == variant].sort_values("fold")["fold"]
                ).to_numpy(dtype=np.float64)
                > 0
            )
        )
    )
    comparison.to_csv(out / "PHASE7_MODEL_COMPARISON.csv", index=False)
    fold_result.to_csv(out / "DEVELOPMENT_FOLD_RESULTS.csv", index=False)

    oof_frame = pd.DataFrame({ID: ids[dev_idx], TARGET: y_dev, "fold": dev_fold_ids})
    for variant, prediction in fold_predictions.items():
        col = variant.replace(":", "__")
        oof_frame[col] = prediction
    oof_frame.to_csv(out / "DEV_OOF_PREDICTIONS.csv", index=False)

    primary_row = comparison.loc[comparison.variant == primary_candidate].iloc[0]
    primary_ci = bootstrap_auc_delta(
        y_dev, raw_pred, primary_pred, seed=HOLDOUT_SEED + 99
    )
    pd.DataFrame([primary_ci]).to_csv(out / "PAIRED_BOOTSTRAP_UNCERTAINTY.csv", index=False)

    control_rows = comparison[comparison.variant.isin(CONTROLS)].copy()
    control_rows["control_type"] = control_rows.variant.map(
        {
            "random_permuted": "within-partition row-permuted true product features",
            "random_keys": "random integer keys with approximate observed cardinality",
            "random_fee_P": "false fixed fee offset 177.1, fit-only maps",
            "random_step_K": "false fixed lattice step 0.0097, fit-only maps",
        }
    )
    control_rows.to_csv(out / "NEGATIVE_CONTROL_RESULTS.csv", index=False)

    base_comparison = comparison.set_index("variant")
    individual_rows = []
    for feature in ALL_PRODUCT:
        row = base_comparison.loc[f"single:{feature}"].to_dict()
        row["feature"] = feature
        row["decision_scope"] = "individual exploratory ablation; not a promotion test"
        individual_rows.append(row)
    pd.DataFrame(individual_rows).to_csv(out / "PRODUCT_INDIVIDUAL_ABLATION.csv", index=False)
    pd.DataFrame(
        [
            {
                **base_comparison.loc[name].to_dict(),
                "block": name,
                "feature_columns": "|".join(cols),
                "decision_scope": (
                    "primary predeclared comparison" if name == primary_candidate
                    else "secondary exploratory block ablation"
                ),
            }
            for name, cols in BLOCKS.items()
        ]
    ).to_csv(out / "PRODUCT_BLOCK_ABLATION.csv", index=False)
    pd.DataFrame(feature_diag_rows).to_csv(out / "PRODUCT_FEATURE_FOLD_DIAGNOSTICS.csv", index=False)

    # Fold-wise lattice analysis uses only development-fit rows to fit a candidate
    # step, and untouched development-validation rows to evaluate numerical support.
    lattice = lattice_audit(phase3, train, dev_idx, dev_fold_ids)
    lattice.to_csv(out / "LATTICE_FOLD_STABILITY.csv", index=False)
    tolerance_columns = [c for c in lattice.columns if c.startswith("share_residual_lt_")]
    lattice_summary_rows = []
    for (kind, step), part in lattice.groupby(["step_kind", "step"], sort=False):
        row = {
            "step_kind": kind,
            "step": float(step),
            "folds": int(part.fold.nunique()),
            "validation_n_total": int(part.n_cash_credit_gt_goods.sum()),
            "mean_absolute_integer_residual": float(part.mean_absolute_integer_residual.mean()),
            "mean_median_residual": float(part.median_absolute_integer_residual.mean()),
            "mean_p95_residual": float(part.p95_absolute_integer_residual.mean()),
        }
        for col in tolerance_columns:
            row[f"mean_{col}"] = float(part[col].mean())
            row[f"fold_sd_{col}"] = float(part[col].std(ddof=1))
        lattice_summary_rows.append(row)
    lattice_summary = pd.DataFrame(lattice_summary_rows)
    lattice_summary.to_csv(out / "LATTICE_STEP_COMPARISON.csv", index=False)

    # The locked holdout is scored once, after freezing the primary comparison and
    # final fit procedure. Labels are not written to the manifest or used in fit.
    holdout_raw = raw_train.iloc[holdout_idx]
    dev_raw_all = raw_train.iloc[dev_idx]
    dev_y_all = y[dev_idx]
    final_rounds = select_rounds(
        baseline,
        dev_raw_all.reset_index(drop=True),
        dev_y_all,
        FINAL_INNER_SEED,
    )
    dev_base = baseline.build_rowwise_features(dev_raw_all)
    holdout_base = baseline.build_rowwise_features(holdout_raw)
    final_params, final_maps, product_dev, product_holdout = prepare_product(
        phase3, dev_raw_all, holdout_raw
    )
    fit_frames = {
        "raw_baseline": (dev_base, holdout_base),
        primary_candidate: (
            pd.concat([dev_base.reset_index(drop=True), product_dev[ALL_PRODUCT].reset_index(drop=True)], axis=1),
            pd.concat([holdout_base.reset_index(drop=True), product_holdout[ALL_PRODUCT].reset_index(drop=True)], axis=1),
        ),
    }
    holdout_pred = {}
    holdout_runtime = {}
    for variant, (fit_x, valid_x) in fit_frames.items():
        fit_matrix, valid_matrix = baseline.make_lightgbm_frames(fit_x, valid_x)
        start = time.perf_counter()
        final_model = lgb.train(
            get_model_params(baseline, MODEL_SEED_BASE + 100),
            lgb.Dataset(fit_matrix, label=dev_y_all),
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(0)],
        )
        holdout_pred[variant] = final_model.predict(valid_matrix, num_iteration=final_rounds)
        holdout_runtime[variant] = time.perf_counter() - start
        if not np.isfinite(holdout_pred[variant]).all():
            raise AssertionError(f"Nonfinite holdout prediction for {variant}")
    # First and only access for holdout model scoring in this Phase 7 runner.
    holdout_y = y[holdout_idx]
    holdout_rows = []
    for variant in ("raw_baseline", primary_candidate):
        holdout_rows.append(
            {
                "variant": variant,
                "holdout_n": len(holdout_idx),
                "holdout_positive_n": int(holdout_y.sum()),
                "holdout_auc_one_time": float(roc_auc_score(holdout_y, holdout_pred[variant])),
                "delta_vs_raw_holdout": (
                    float(roc_auc_score(holdout_y, holdout_pred[variant]) - roc_auc_score(holdout_y, holdout_pred["raw_baseline"]))
                    if variant != "raw_baseline"
                    else 0.0
                ),
                "final_rounds_selected_from_development_only": final_rounds,
                "fit_rows": len(dev_idx),
                "runtime_seconds": holdout_runtime[variant],
                "evaluated_once_after_primary_freeze": True,
                "independent_confirmation": False,
                "reason_not_independent": "Training labels were used by prior project OOF analyses before this holdout was locked.",
            }
        )
    holdout_frame = pd.DataFrame(
        {
            ID: ids[holdout_idx],
            "partition": "locked_confirmation_holdout",
            "prediction_raw_baseline": holdout_pred["raw_baseline"],
            "prediction_primary_all_product": holdout_pred[primary_candidate],
        }
    )
    holdout_frame.to_csv(out / "LOCKED_HOLDOUT_PREDICTIONS.csv", index=False)
    holdout_frame["TARGET_for_one_time_score"] = holdout_y
    holdout_frame.to_csv(out / "LOCKED_HOLDOUT_ONE_TIME_SCORE_INPUT.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(out / "LOCKED_HOLDOUT_ONE_TIME_METRICS.csv", index=False)

    decision = "NOT PROMOTED"
    delta_dev = float(primary_row["delta_vs_raw_baseline"])
    delta_hold = float(holdout_rows[1]["delta_vs_raw_holdout"])
    controls_best = float(control_rows["pooled_development_oof_auc"].max())
    if (
        delta_dev > 0
        and delta_hold > 0
        and int(primary_row["positive_paired_fold_deltas"]) >= 4
        and float(primary_row["pooled_development_oof_auc"]) > controls_best
    ):
        decision = "RESEARCH CANDIDATE ONLY — independent confirmation unavailable"

    formula_report = f"""# Product feature reproduction report

## Exact source and fold-safe implementation

The Phase 3 feature source is `{PHASE3_SOURCE}` (SHA-256 `{source_hashes[str(PHASE3_SOURCE)]}`); Phase 3B ablation source is `{PHASE3B_SOURCE}` (SHA-256 `{source_hashes[str(PHASE3B_SOURCE)]}`). The direct implementation uses `fit_fee_and_step(raw_fit)`, `fit_product_maps(raw_fit, params)`, and `make_product_features(raw, params, maps)`. Parameters and frequency maps are fit only on each outer fit fold and then applied unchanged to its validation fold. No target is passed into these functions and test data is not used.

- Formula columns reproduced: `{len(ALL_PRODUCT)}` model columns (`K={len(K_COLS)}`, `Q={len(Q_COLS)}`, `P={len(P_COLS)}`) from a `{len(phase3.FEATURE_DEFINITIONS)}`-definition Phase 3 catalog.
- Cash-loan masks, fit-only frequency tables, Q/P rounding precision, fee offset, K step, and K validity threshold are unchanged from source.
- Per-fold fee/step values, feature missingness/cardinality, and parameter-fit row counts are in `PRODUCT_FEATURE_FOLD_DIAGNOSTICS.csv`.
- Independent equation checks for Q5, P5, and K coordinate passed at max absolute error <= 1e-10 in every development fold.
- Formula definitions, aliases, and eligibility status are in `PRODUCT_FEATURE_FORMULA_REGISTER.csv`.
- Existing Phase 3 `NO_REDUNDANT` feature list has been de-duplicated only to prevent duplicate column insertion; feature definitions themselves are unchanged.

## Interpretation

The product decoder remains a numerical feature hypothesis. Its estimated offset is not an established fee; its K coordinate is not an actual financing tier; Q ratios do not establish installment count or loan duration. Features based on `AMT_ANNUITY` retain the dictionary limitation that its payment period is unspecified.
"""
    (out / "PRODUCT_FEATURE_REPRODUCTION_REPORT.md").write_text(formula_report, encoding="utf-8")

    lattice_report = f"""# Numerical lattice hypothesis report

## Formula tested

For cash-loan rows with positive goods price and credit greater than goods price:

`coordinate = (AMT_CREDIT / AMT_GOODS_PRICE - 1) / step`

Prespecified candidate steps were `0.0066` and `0.0132`. Negative-control placebo steps were `0.0057`, `0.0097`, `0.0173`, and `0.0231`. A fit-only selected step from the original Phase 3 decoder was also evaluated on each held-out development fold. No target labels enter step fitting or residual calculations.

## Results

See `LATTICE_FOLD_STABILITY.csv` for every fold, sample size, residual mean/median/p95, and shares below absolute residual tolerances 0.01, 0.05, and 0.1; `LATTICE_STEP_COMPARISON.csv` aggregates stability. Each fitted step was estimated only from its fold's development-fit rows; evaluation used that fold's held-out rows. The confirmation holdout was not used for lattice discovery.

## Controls and interpretation

Placebo step sizes are deliberately not selected on validation labels. Compare their held-out support to the prespecified hypotheses, and report all tested values. Any fit-only grid search is descriptive and its selected step must be evaluated on held-out rows; do not quote fit support as generalization. Integer residuals demonstrate only a mathematical pattern in these variables, not a contract increment, fee, interest rate, repayment term, or causal mechanism.
"""
    (out / "LATTICE_HYPOTHESIS_REPORT.md").write_text(lattice_report, encoding="utf-8")

    dev_integrity = {
        "dev_rows": len(dev_idx),
        "dev_oof_rows": len(oof_frame),
        "unique_dev_oof_ids": int(oof_frame[ID].nunique()),
        "oof_id_exact_partition_alignment": bool(np.array_equal(oof_frame[ID].to_numpy(), ids[dev_idx])),
        "oof_target_exact_alignment": bool(np.array_equal(oof_frame[TARGET].to_numpy(), y_dev)),
        "oof_fold_exact_alignment": bool(np.array_equal(oof_frame["fold"].to_numpy(), dev_fold_ids)),
        "all_variant_oof_finite": bool(np.isfinite(oof_frame.drop(columns=[ID, TARGET, "fold"]).to_numpy()).all()),
        "one_prediction_per_dev_row": True,
        "validation_labels_used_in_fit_or_stopping": False,
        "product_parameters_fit_outer_training_only": True,
        "holdout_disjoint_by_id": bool(set(ids[dev_idx]).isdisjoint(set(ids[holdout_idx]))),
        "holdout_prediction_ids_unique": bool(pd.Series(ids[holdout_idx]).is_unique),
        "holdout_evaluated_once": True,
        "holdout_independent": False,
        "holdout_independence_limitation": "All supplied training labels appeared in prior project OOF analyses before this split was created.",
        "phase6_baseline_oof_sha256_verified": expected_baseline_hash,
        "phase7_dev_oof_sha256": sha256_file(out / "DEV_OOF_PREDICTIONS.csv"),
        "phase7_partition_manifest_sha256": partition_hash,
        "primary_candidate": primary_candidate,
        "promotion_decision": decision,
    }
    oof_integrity = f"""# OOF integrity report

## Development OOF

- Development rows: {len(dev_idx)}; expected five validation folds.
- Unique OOF IDs: {oof_integrity_value(dev_integrity, 'unique_dev_oof_ids')}.
- Exact ID / TARGET / fold alignment: {dev_integrity['oof_id_exact_partition_alignment']} / {dev_integrity['oof_target_exact_alignment']} / {dev_integrity['oof_fold_exact_alignment']}.
- One finite prediction per development row for all variants: {dev_integrity['one_prediction_per_dev_row']} / {dev_integrity['all_variant_oof_finite']}.
- Outer validation labels used for model fit, stopping, feature selection, or feature map fitting: **No**.
- Product parameters and maps use outer-fit rows only: **Yes**.
- OOF file SHA-256: `{dev_integrity['phase7_dev_oof_sha256']}`.

## Locked holdout

- Manifest SHA-256: `{partition_hash}`; development and holdout applicant IDs are disjoint.
- The raw and primary product models were fit on all development rows. The shared iteration count was selected using only an inner development split; product parameters/maps were fit on development rows and applied to the holdout.
- The two preregistered predictions were evaluated once after the primary candidate/procedure was fixed; no retuning occurred.
- **This is not independent confirmation.** Every label in the training table had already appeared in earlier Phase 2/3 OOF research before the new holdout was created. Therefore the holdout metric is a secondary one-time diagnostic only; it cannot satisfy an untouched confirmation criterion or justify model promotion.

## Files and limitations

`DEV_OOF_PREDICTIONS.csv` contains the development OOF predictions; `LOCKED_HOLDOUT_PREDICTIONS.csv` contains IDs and predictions only; `LOCKED_HOLDOUT_ONE_TIME_SCORE_INPUT.csv` adds labels for the one-time audit record. Hashes are in the manifest. The same OOF labels have been examined historically, so development fold scores are not an unbiased independent estimate.
"""
    (out / "OOF_INTEGRITY_REPORT.md").write_text(oof_integrity, encoding="utf-8")

    comparison["primary_candidate"] = comparison["variant"].eq(primary_candidate)
    comparison["promotion_status"] = comparison["variant"].map(
        lambda value: decision if value == primary_candidate else "EXPLORATORY ONLY"
    )
    comparison.to_csv(out / "PHASE7_MODEL_COMPARISON.csv", index=False)
    block_ablation = pd.read_csv(out / "PRODUCT_BLOCK_ABLATION.csv")
    block_ablation["promotion_status"] = block_ablation["block"].map(
        lambda value: decision if value == primary_candidate else "EXPLORATORY ONLY"
    )
    block_ablation.to_csv(out / "PRODUCT_BLOCK_ABLATION.csv", index=False)
    individual = pd.read_csv(out / "PRODUCT_INDIVIDUAL_ABLATION.csv")
    individual["promotion_status"] = "EXPLORATORY ONLY; not eligible for selection from confirmation"
    individual.to_csv(out / "PRODUCT_INDIVIDUAL_ABLATION.csv", index=False)

    delta_summary = comparison.set_index("variant")
    top_individual = delta_summary.loc[
        [f"single:{name}" for name in ALL_PRODUCT]
    ].sort_values("delta_vs_raw_baseline", ascending=False).head(3)
    top_blocks = delta_summary.loc[list(BLOCKS)].sort_values(
        "delta_vs_raw_baseline", ascending=False
    )
    best_control = delta_summary.loc[CONTROLS].sort_values(
        "pooled_development_oof_auc", ascending=False
    )
    decision_report = f"""# Experiment decision report

## Primary comparison

- Frozen primary: `{primary_candidate}`.
- Development raw baseline pooled OOF AUC: `{float(roc_auc_score(y_dev, raw_pred)):.9f}`.
- Development primary product-block pooled OOF AUC: `{float(primary_row['pooled_development_oof_auc']):.9f}`.
- Paired development delta: `{delta_dev:+.9f}`; positive fold deltas: `{int(primary_row['positive_paired_fold_deltas'])}/5`.
- Paired row-bootstrap 95% interval: `[{primary_ci['delta_ci_2_5']:+.9f}, {primary_ci['delta_ci_97_5']:+.9f}]` (descriptive on previously exposed labels).
- One-time locked-holdout raw AUC: `{holdout_rows[0]['holdout_auc_one_time']:.9f}`.
- One-time locked-holdout primary AUC: `{holdout_rows[1]['holdout_auc_one_time']:.9f}`; delta `{delta_hold:+.9f}`.
- Best control development AUC: `{controls_best:.9f}` (`{best_control.index[0]}`).

## Secondary results

Top individual single-feature discovery ablations:

{top_individual[['pooled_development_oof_auc','delta_vs_raw_baseline','positive_paired_fold_deltas']].to_string()}

Product block ablations:

{top_blocks[['pooled_development_oof_auc','delta_vs_raw_baseline','positive_paired_fold_deltas']].to_string()}

Negative controls:

{best_control[['pooled_development_oof_auc','delta_vs_raw_baseline','positive_paired_fold_deltas']].to_string()}

See the corresponding CSVs for all candidates, folds, runtimes, ranking differences, and controls.

## Decision

**{decision}.** The holdout's labels were already used in previous project OOF analyses, so it is not independent confirmation. Even if the primary block improves on this one-time partition, the required untouched labeled confirmation is unavailable. No candidate is promoted, no 0.77 target is claimed, and no submission or champion change is made.

## Statistical caution

The individual/block/control matrix contains many comparisons; secondary winners are selection-biased. The paired bootstrap interval is not a substitute for a genuinely independent sample and does not adjust for feature-family search. Fold-average differences alone are not a significance test.
"""
    (out / "EXPERIMENT_DECISION_REPORT.md").write_text(decision_report, encoding="utf-8")

    recommendation = f"""# Next phase recommendation

## Current outcome

The Phase 7 primary candidate `{primary_candidate}` has development delta `{delta_dev:+.6f}` and the one-time holdout delta `{delta_hold:+.6f}`. However, the holdout is not independent because its labels were included in prior project cross-validation outputs. Therefore no promotion criteria are fully satisfied.

## Single best next step

Obtain a genuinely new, organizer-approved labeled confirmation cohort (or a truly untouched course-provided validation set, if one exists and was not previously inspected). Freeze the Phase 7 primary code, exact formula block, data contract, and model configuration before exposing those labels. If no such cohort is available, retain the Phase 7 result as exploratory and do not claim independent improvement.

Do not continue broad feature search or tune on the locked holdout. If new labels cannot be obtained, the next development-only study should be a narrow predeclared external-score disagreement ablation against the raw-plus-product development baseline, with the same folds and no holdout use; this still cannot authorize promotion.

## Compliance blockers

Pooled train/test transforms, target encoding, and test-to-test coupling remain excluded pending organizer clarification. Product decoder fee/step features are permitted here only as fit-fold numerical hypotheses; they are not validated contract semantics.
"""
    (out / "NEXT_PHASE_RECOMMENDATION.md").write_text(recommendation, encoding="utf-8")

    # Summary reports use exact final values and list what ran.
    finished = datetime.now(timezone.utc)
    elapsed = time.perf_counter() - wall_start
    peak_bytes = None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if ctypes.WinDLL("Psapi.dll").GetProcessMemoryInfo(
                ctypes.WinDLL("Kernel32.dll").GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                peak_bytes = int(counters.PeakWorkingSetSize)
        except Exception:
            peak_bytes = None

    log_lines = [
        "status=COMPLETED",
        f"run_id={run_id}",
        f"started_utc={started.isoformat()}",
        f"finished_utc={finished.isoformat()}",
        f"runtime_seconds={elapsed:.3f}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"train_sha256={source_hashes[str(TRAIN_PATH)]}",
        f"test_sha256={source_hashes[str(TEST_PATH)]}",
        f"phase6_baseline_oof_sha256={source_hashes[str(BASELINE_OOF_PATH)]}",
        f"baseline_oof_matches_manifest={source_hashes[str(BASELINE_OOF_PATH)] == expected_baseline_hash}",
        f"partition_manifest_sha256={partition_hash}",
        f"development_rows={len(dev_idx)}; holdout_rows={len(holdout_idx)}",
        f"development_seed={DEV_FOLD_SEED}; holdout_seed={HOLDOUT_SEED}; holdout_fraction={HOLDOUT_FRACTION}",
        f"primary_candidate={primary_candidate}",
        f"variant_count={len(model_order)}; development_fold_model_fits={len(model_order) * N_DEV_FOLDS}",
        f"primary_dev_auc={float(primary_row['pooled_development_oof_auc']):.9f}; primary_dev_delta={delta_dev:+.9f}",
        f"primary_holdout_auc={holdout_rows[1]['holdout_auc_one_time']:.9f}; primary_holdout_delta={delta_hold:+.9f}",
        "holdout_is_independent_confirmation=NO; all labels appeared in earlier project OOF work",
        f"process_peak_working_set_bytes={peak_bytes}",
        "test_predictors_read=NO; test_TARGET_read=NO; external_data=NO; pooled_train_test_fit=NO",
        "target_encoding=NO; test_to_test_coupling=NO; broad_model_sweeps=NO",
        "submission_created=NO; champion_modified=NO; historical_files_modified=NO",
        f"decision={decision}",
        f"outputs={out}",
    ]
    (out / "EXECUTION_LOG.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    report = f"""# Phase 7 execution report

Run ID `{run_id}` completed. The Phase 6 baseline provenance and OOF rows were verified before experiments. All primary and secondary feature construction used training data only, with feature parameters and maps fit within the relevant development training fold.

## Results

- Primary predeclared raw-only development pooled OOF AUC: `{float(roc_auc_score(y_dev, raw_pred)):.9f}`.
- Primary `raw + all Phase 3 product` development pooled OOF AUC: `{float(primary_row['pooled_development_oof_auc']):.9f}` (delta `{delta_dev:+.9f}`).
- One-time holdout primary delta: `{delta_hold:+.9f}`; this holdout is **not independent confirmation** because its labels appeared in prior project OOF research.
- Detailed individual/block/control results: `PHASE7_MODEL_COMPARISON.csv`, `PRODUCT_INDIVIDUAL_ABLATION.csv`, `PRODUCT_BLOCK_ABLATION.csv`, `NEGATIVE_CONTROL_RESULTS.csv`.
- Decision: **{decision}**.

## Scope

Exact Phase 3 product formulas were reproduced, 16 individual columns tested, eight product blocks tested, and four negative controls run. Lattice stability was assessed across five development folds and fixed placebo steps. One primary raw-vs-product holdout comparison was scored once. No V16/EXP85 reconstruction, test prediction pipeline, submission, champion change, or broad model sweep was performed.

All labeled rows and historical folds have prior researcher exposure. Therefore results are development evidence only; no independent 0.77 claim or model promotion is justified.
"""
    (out / "PHASE7_EXECUTION_REPORT.md").write_text(report, encoding="utf-8")

    # Manifest includes every artifact written so far; it is regenerated last.
    manifest_rows = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    pd.DataFrame(manifest_rows).to_csv(out / "SHA256_MANIFEST.csv", index=False)
    print("\n".join(log_lines))
    print(f"OUTPUT_DIRECTORY={out}")
    return 0


def oof_integrity_value(values: dict, key: str) -> int:
    return int(values[key])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    run_id = parser.parse_args().run_id
    if not run_id.replace("T", "").replace("Z", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only alphanumeric characters, T, Z, and underscore")
    raise SystemExit(main(run_id))
