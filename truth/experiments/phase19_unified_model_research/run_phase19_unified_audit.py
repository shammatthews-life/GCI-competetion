from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXP = REPO / "experiments"
sys.path.insert(0, str(EXP / "phase13_matched_baseline"))
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE9 = (
    EXP
    / "phase9_product_mechanism_and_catboost"
    / "runs"
    / "phase9_20261104T181500LOCAL"
    / "OOF_PREDICTIONS.csv"
)
PHASE11 = (
    EXP
    / "phase11_independent_research"
    / "runs"
    / "phase11_gpu_cv_20261105"
    / "oof_predictions.csv"
)
PHASE13 = (
    EXP
    / "phase13_matched_baseline"
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE15_MULTI = (
    EXP
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_multiseed_20261105T125826Z"
    / "MULTISEED_OOF_PREDICTIONS.csv"
)
PHASE16 = (
    EXP
    / "phase16_nested_catboost_research"
    / "runs"
    / "phase16_nested_catboost_20261105T132953Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE17 = (
    EXP
    / "phase17_nested_constant_free_target_encoding"
    / "runs"
    / "phase17_nested_te_20261105T140620Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE18 = (
    EXP
    / "phase18_nested_oof_ensemble"
    / "runs"
    / "phase18_nested_blend_20261105T142048Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE9_SEEDS = (42, 7, 2026)
PHASE15_SEEDS = (20261113, 20261114, 20261115)
PRIMARY_SEED = 20261113

MODEL_SPECS: dict[str, dict[str, str]] = {
    "phase9_native8_seed_average": {
        "phase": "Phase 9",
        "configuration": "GPU CatBoost; depth 6; LR 0.05; 700 iterations; native categories; 8 keys",
        "features": "Raw/rowwise Phase 9 builder + eight product keys",
        "eligibility": "Track B: key formula constants and CatBoost CTR eligibility unresolved",
        "reproducibility": "Three saved seed-specific OOF vectors; historical feature builder differs from Phase 15",
        "fold_scheme": "Mixed seed-specific folds; aggregate has no single native fold partition",
    },
    "phase11_catboost_keys": {
        "phase": "Phase 11",
        "configuration": "GPU CatBoost; depth 6; LR 0.05; 700 iterations; 2 replicas per fold",
        "features": "Independent raw/rowwise builder + eight product keys",
        "eligibility": "Track B: key formula constants and CatBoost CTR eligibility unresolved",
        "reproducibility": "Saved full five-fold OOF; fold seed 2026",
        "fold_scheme": "StratifiedKFold(seed=2026)",
    },
    "phase13_E_frequency_ohe": {
        "phase": "Phase 13",
        "configuration": "CPU LightGBM; 5 folds; frequency-supported OHE; matched baseline design",
        "features": "Complete raw/rowwise baseline + eight product keys",
        "eligibility": "Track B: all-eight key set includes unresolved 130.5 and 0.0066 hypotheses",
        "reproducibility": "Saved full five-fold OOF; primary fold seed 20261113",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
    "phase13_C_raw_rowwise": {
        "phase": "Phase 13",
        "configuration": "CPU LightGBM; matched complete raw/rowwise baseline",
        "features": "Raw numeric + raw categories + approved rowwise features; no product keys",
        "eligibility": "Conservative Track A candidate; no custom product keys or target statistics",
        "reproducibility": "Saved full five-fold OOF; primary fold seed 20261113",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
    "phase15_three_seed_average": {
        "phase": "Phase 15",
        "configuration": "GPU CatBoost; depth 6; LR 0.05; 700 iterations; three seed OOF vectors averaged rowwise",
        "features": "Phase 14 raw/rowwise + numeric enhancements + seven time features + eight keys",
        "eligibility": "Track B: key constants and native CTR status unresolved",
        "reproducibility": "Verified three complete aligned OOF vectors; each seed has its own fold partition",
        "fold_scheme": "Mixed seeds 20261113/14/15; aggregate has no single native fold partition",
    },
    "phase16_nested_tuned_catboost": {
        "phase": "Phase 16",
        "configuration": "Nested selection among four GPU CatBoost parameter sets; 3 inner folds / 5 outer folds",
        "features": "Phase 15 features; all eight keys",
        "eligibility": "Track B: key constants and native CTR status unresolved",
        "reproducibility": "Verified complete OOF and nested configuration choices; outer seed 20261113",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
    "phase17_nested_te_lightgbm": {
        "phase": "Phase 17",
        "configuration": "CPU LightGBM + nested smoothed rates/log counts; constant-free key subset",
        "features": "Complete raw/rowwise + Q3/Q4/Q5/AG3; target stats are nested",
        "eligibility": "Track B: target-derived statistics eligibility unresolved",
        "reproducibility": "Verified complete OOF and nested maps; outer seed 20261113",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
    "phase18_nested_blend": {
        "phase": "Phase 18",
        "configuration": "CatBoost/Phase 17 blend; weight selected on 3-fold inner OOF from a fixed 5-point grid",
        "features": "Phase 15 single-seed CatBoost + Phase 17 nested target-stat LightGBM",
        "eligibility": "Track B: unresolved product constants/CTR and label-derived target statistics",
        "reproducibility": "Verified weights and outputs; outer seed 20261113",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
    "phase19_mlp_constant_free": {
        "phase": "Phase 19",
        "configuration": "CPU PyTorch embedding MLP; fold-local early stopping and refit",
        "features": "Raw/rowwise/decomposed numerics + raw categories + constant-free Q3/Q4/Q5/AG3",
        "eligibility": "Track A candidate; no target encodings or external/test data; deterministic keys still need rule interpretation",
        "reproducibility": "Five-fold OOF; outer seed 20261113; fold-local category vocabularies and scaling",
        "fold_scheme": "StratifiedKFold(seed=20261113)",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fold_assignment(y: np.ndarray, seed: int) -> np.ndarray:
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        fold_ids[valid] = fold
    return fold_ids


def auc(y: np.ndarray, prediction: np.ndarray) -> float:
    ensure(np.isfinite(prediction).all(), "A prediction vector has non-finite values")
    return float(roc_auc_score(y, prediction))


def optional_float(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def add_model(
    models: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    model_id: str,
    prediction: np.ndarray,
    y: np.ndarray,
    actual_folds: np.ndarray | None,
    source: str,
    seed_info: str,
) -> None:
    spec = MODEL_SPECS[model_id]
    score = auc(y, prediction)
    models[model_id] = prediction
    rows.append(
        {
            "model_id": model_id,
            "source_phase": spec["phase"],
            "pooled_oof_auc": score,
            "seed_auc_mean": "",
            "prediction_average_auc": score,
            "seed_info": seed_info,
            "fold_scheme": spec["fold_scheme"],
            "native_fold_scheme_available": actual_folds is not None,
            "model_configuration": spec["configuration"],
            "feature_groups": spec["features"],
            "eligibility": spec["eligibility"],
            "reproducibility": spec["reproducibility"],
            "source_file": source,
        }
    )


def check_row_alignment(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    name: str,
    id_column: str,
    target_column: str,
    row_index_column: str | None = None,
    fold_column: str | None = None,
    expected_fold: np.ndarray | None = None,
) -> np.ndarray | None:
    ensure(len(frame) == len(train), f"{name}: row count mismatch")
    ensure(
        np.array_equal(frame[id_column].to_numpy(), train["SK_ID_CURR"].to_numpy()),
        f"{name}: applicant ID/order mismatch",
    )
    ensure(
        np.array_equal(frame[target_column].to_numpy(), train["TARGET"].to_numpy()),
        f"{name}: target/order mismatch",
    )
    if row_index_column is not None:
        ensure(
            np.array_equal(frame[row_index_column].to_numpy(), np.arange(len(train))),
            f"{name}: row_index mismatch",
        )
    if fold_column is None:
        return None
    actual = frame[fold_column].to_numpy(dtype=np.int8)
    if expected_fold is not None:
        ensure(np.array_equal(actual, expected_fold), f"{name}: expected fold mismatch")
    return actual


def main(output_dir: Path, mlp_run: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    y = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    primary_folds = fold_assignment(y, PRIMARY_SEED)
    models: dict[str, np.ndarray] = {}
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    p9_required_columns = ["SK_ID_CURR", "TARGET"] + [
        f"seed{seed}__{variant}"
        for seed in PHASE9_SEEDS
        for variant in (
            "cat_product_baseline",
            "cat_native_q4",
            "cat_native_8_keys",
            "cat_native_8_keys_plus_crosses",
        )
    ]
    p9 = pd.read_csv(PHASE9, usecols=p9_required_columns)
    ensure(
        np.array_equal(p9["SK_ID_CURR"].to_numpy(), ids)
        and np.array_equal(p9["TARGET"].to_numpy(), y),
        "Phase 9 ID/target alignment failed",
    )
    p9_columns = [f"seed{s}__cat_native_8_keys" for s in PHASE9_SEEDS]
    phase9_vectors = [p9[col].to_numpy(dtype=np.float64) for col in p9_columns]
    for seed, prediction in zip(PHASE9_SEEDS, phase9_vectors):
        model_id = f"phase9_native8_seed{seed}"
        MODEL_SPECS[model_id] = MODEL_SPECS["phase9_native8_seed_average"]
        add_model(
            models,
            model_rows,
            model_id,
            prediction,
            y,
            fold_assignment(y, seed),
            str(PHASE9),
            f"CatBoost random/CV seed {seed}",
        )
    p9_avg = np.mean(np.column_stack(phase9_vectors), axis=1)
    recorded_p9_avg = p9_avg
    add_model(
        models,
        model_rows,
        "phase9_native8_seed_average",
        p9_avg,
        y,
        None,
        str(PHASE9),
        "Mean of seed 42, 7, and 2026 OOF probability vectors",
    )
    seed_auc_mean_p9 = float(np.mean([auc(y, pred) for pred in phase9_vectors]))
    model_rows[-1]["seed_auc_mean"] = seed_auc_mean_p9
    model_rows[-1]["prediction_average_auc"] = auc(y, recorded_p9_avg)
    model_rows[-1]["reproducibility"] = (
        "Recomputed from three saved seed-specific OOF columns; aggregate score "
        "is pooled AUC of mean predictions, distinct from mean seed AUC"
    )
    p9_run_dir = PHASE9.parent
    p9_folds_path = p9_run_dir / "CATBOOST_FOLD_RESULTS.csv"
    p9_folds = pd.read_csv(p9_folds_path)
    for seed in PHASE9_SEEDS:
        seed_col = f"seed{seed}__cat_native_8_keys"
        vector = p9[seed_col].to_numpy(dtype=np.float64)
        assigned = fold_assignment(y, seed)
        for fold in range(1, 6):
            mask = assigned == fold
            recomputed_fold_auc = auc(y[mask], vector[mask])
            recorded_fold = p9_folds.loc[
                (p9_folds["seed"] == seed)
                & (p9_folds["fold"] == fold)
                & (p9_folds["variant"] == "cat_native_8_keys")
            ]
            ensure(
                len(recorded_fold) == 1
                and abs(
                    recomputed_fold_auc
                    - float(recorded_fold.iloc[0]["fold_auc"])
                )
                < 1e-10,
                f"Phase 9 native fold verification failed for seed {seed}, fold {fold}",
            )
            fold_rows.append(
                {
                    "model_id": f"phase9_native8_seed{seed}",
                    "fold_definition": f"native_seed_{seed}",
                    "fold": fold,
                    "n_rows": int(mask.sum()),
                    "fold_auc": recomputed_fold_auc,
                    "fold_comparable_to_phase15_primary": False,
                    "interpretation": "Native seed-specific validation fold",
                }
            )
    for fold in range(1, 6):
        mask = primary_folds == fold
        fold_rows.append(
            {
                "model_id": "phase9_native8_seed_average",
                "fold_definition": "common_reference_seed_20261113",
                "fold": fold,
                "n_rows": int(mask.sum()),
                "fold_auc": auc(y[mask], p9_avg[mask]),
                "fold_comparable_to_phase15_primary": True,
                "interpretation": "Descriptive common row partition; averaged components use different CV seeds",
            }
        )

    p11 = pd.read_csv(
        PHASE11, usecols=["row_index", "fold", "target", "product_keys_bagged"]
    )
    f11 = fold_assignment(y, 2026)
    actual11 = check_row_alignment(
        p11,
        train,
        "Phase 11",
        "row_index",
        "target",
        row_index_column="row_index",
        fold_column="fold",
        expected_fold=f11,
    )
    add_model(
        models,
        model_rows,
        "phase11_catboost_keys",
        p11["product_keys_bagged"].to_numpy(dtype=np.float64),
        y,
        actual11,
        str(PHASE11),
        "Two replicas per fold; CV split seed 2026",
    )

    p13 = pd.read_csv(PHASE13)
    f13 = primary_folds
    actual13 = check_row_alignment(
        p13,
        train,
        "Phase 13",
        "SK_ID_CURR",
        "TARGET",
        row_index_column="row_index",
        fold_column="primary_fold",
        expected_fold=f13,
    )
    add_model(
        models,
        model_rows,
        "phase13_C_raw_rowwise",
        p13["C_complete_raw_rowwise"].to_numpy(dtype=np.float64),
        y,
        actual13,
        str(PHASE13),
        "Primary five-fold seed 20261113",
    )
    add_model(
        models,
        model_rows,
        "phase13_E_frequency_ohe",
        p13["E_C_plus_keys_frequency_ohe"].to_numpy(dtype=np.float64),
        y,
        actual13,
        str(PHASE13),
        "Primary five-fold seed 20261113",
    )

    p15m = pd.read_csv(PHASE15_MULTI)
    check_row_alignment(
        p15m,
        train,
        "Phase 15 multiseed",
        "SK_ID_CURR",
        "TARGET",
        row_index_column="row_index",
    )
    p15_vectors = [
        p15m[f"phase15_secondary_seed_{seed}"].to_numpy(dtype=np.float64)
        if seed != 20261113
        else p15m["phase15_primary_seed_20261113"].to_numpy(dtype=np.float64)
        for seed in PHASE15_SEEDS
    ]
    recorded_p15_avg = p15m["phase15_three_seed_average"].to_numpy(dtype=np.float64)
    expected_p15_avg = np.mean(np.column_stack(p15_vectors), axis=1)
    ensure(
        np.allclose(recorded_p15_avg, expected_p15_avg, rtol=0.0, atol=1e-7),
        "Phase 15 saved ensemble is not the mean of saved seed predictions",
    )
    p15_seed_aucs = [auc(y, prediction) for prediction in p15_vectors]
    for seed, prediction in zip(PHASE15_SEEDS, p15_vectors):
        model_id = f"phase15_seed{seed}"
        MODEL_SPECS[model_id] = MODEL_SPECS["phase15_three_seed_average"]
        add_model(
            models,
            model_rows,
            model_id,
            prediction,
            y,
            fold_assignment(y, seed),
            str(PHASE15_MULTI),
            f"CatBoost CV seed {seed}",
        )
        model_rows[-1]["prediction_average_auc"] = ""
        model_rows[-1]["fold_scheme"] = f"StratifiedKFold(seed={seed})"
    add_model(
        models,
        model_rows,
        "phase15_three_seed_average",
        recorded_p15_avg,
        y,
        None,
        str(PHASE15_MULTI),
        "Mean of seed 20261113, 20261114, and 20261115 OOF vectors",
    )
    model_rows[-1]["seed_auc_mean"] = float(np.mean(p15_seed_aucs))
    model_rows[-1]["prediction_average_auc"] = auc(y, recorded_p15_avg)
    model_rows[-1]["reproducibility"] = (
        "Verified saved vector equals mean of three aligned seed OOF vectors; "
        "aggregate has no single native fold assignment"
    )
    for seed, prediction in zip(PHASE15_SEEDS, p15_vectors):
        assigned = fold_assignment(y, seed)
        for fold in range(1, 6):
            mask = assigned == fold
            fold_rows.append(
                {
                    "model_id": f"phase15_seed{seed}",
                    "fold_definition": f"native_seed_{seed}",
                    "fold": fold,
                    "n_rows": int(mask.sum()),
                    "fold_auc": auc(y[mask], prediction[mask]),
                    "fold_comparable_to_phase15_primary": seed == PRIMARY_SEED,
                    "interpretation": "Native validation fold for this seed",
                }
            )
    for fold in range(1, 6):
        mask = primary_folds == fold
        fold_rows.append(
            {
                "model_id": "phase15_three_seed_average",
                "fold_definition": "common_reference_seed_20261113",
                "fold": fold,
                "n_rows": int(mask.sum()),
                "fold_auc": auc(y[mask], recorded_p15_avg[mask]),
                "fold_comparable_to_phase15_primary": True,
                "interpretation": "Descriptive common partition; ensemble combines three CV seeds",
            }
        )

    p16 = pd.read_csv(PHASE16)
    check_row_alignment(
        p16,
        train,
        "Phase 16",
        "SK_ID_CURR",
        "TARGET",
        row_index_column="row_index",
        fold_column="fold",
        expected_fold=primary_folds,
    )
    add_model(
        models,
        model_rows,
        "phase16_nested_tuned_catboost",
        p16["phase16_nested_tuned_catboost"].to_numpy(dtype=np.float64),
        y,
        primary_folds,
        str(PHASE16),
        "Outer seed 20261113; nested three-fold model selection",
    )
    p17 = pd.read_csv(PHASE17)
    check_row_alignment(
        p17,
        train,
        "Phase 17",
        "SK_ID_CURR",
        "TARGET",
        row_index_column="row_index",
        fold_column="fold",
        expected_fold=primary_folds,
    )
    add_model(
        models,
        model_rows,
        "phase17_nested_te_lightgbm",
        p17["nested_te_constant_free"].to_numpy(dtype=np.float64),
        y,
        primary_folds,
        str(PHASE17),
        "Outer seed 20261113; nested target encoding",
    )
    p18 = pd.read_csv(PHASE18)
    check_row_alignment(
        p18,
        train,
        "Phase 18",
        "SK_ID_CURR",
        "TARGET",
        row_index_column="row_index",
        fold_column="fold",
        expected_fold=primary_folds,
    )
    add_model(
        models,
        model_rows,
        "phase18_nested_blend",
        p18["phase18_nested_blend"].to_numpy(dtype=np.float64),
        y,
        primary_folds,
        str(PHASE18),
        "Outer seed 20261113; fold-local nested blend selection",
    )

    for model_id in (
        "phase13_C_raw_rowwise",
        "phase13_E_frequency_ohe",
        "phase16_nested_tuned_catboost",
        "phase17_nested_te_lightgbm",
        "phase18_nested_blend",
    ):
        vector = models[model_id]
        for fold in range(1, 6):
            mask = primary_folds == fold
            fold_rows.append(
                {
                    "model_id": model_id,
                    "fold_definition": "native_seed_20261113",
                    "fold": fold,
                    "n_rows": int(mask.sum()),
                    "fold_auc": auc(y[mask], vector[mask]),
                    "fold_comparable_to_phase15_primary": True,
                    "interpretation": "Native validation fold; same split as Phase 15 primary",
                }
            )

    if mlp_run is not None:
        mlp_path = mlp_run / "OOF_PREDICTIONS.csv"
        ensure(mlp_path.is_file(), f"Phase 19 MLP OOF missing: {mlp_path}")
        mlp = pd.read_csv(mlp_path)
        check_row_alignment(
            mlp,
            train,
            "Phase 19 MLP",
            "SK_ID_CURR",
            "TARGET",
            row_index_column="row_index",
            fold_column="fold",
            expected_fold=primary_folds,
        )
        add_model(
            models,
            model_rows,
            "phase19_mlp_constant_free",
            mlp["phase19_mlp_constant_free"].to_numpy(dtype=np.float64),
            y,
            primary_folds,
            str(mlp_path),
            "Outer seed 20261113; per-fold early stopping",
        )
        for fold in range(1, 6):
            mask = primary_folds == fold
            fold_rows.append(
                {
                    "model_id": "phase19_mlp_constant_free",
                    "fold_definition": "native_seed_20261113",
                    "fold": fold,
                    "n_rows": int(mask.sum()),
                    "fold_auc": auc(y[mask], models["phase19_mlp_constant_free"][mask]),
                    "fold_comparable_to_phase15_primary": True,
                    "interpretation": "Native validation fold; same split as Phase 15 primary",
                }
            )

    frame = pd.DataFrame(model_rows)
    frame.to_csv(output_dir / "PHASE19_MODEL_COMPARISON.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "PHASE19_FOLD_METRICS.csv", index=False)

    ordered_models = list(models)
    corr_rows = []
    for i, left in enumerate(ordered_models):
        for right in ordered_models[i:]:
            corr_rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "pearson_correlation": float(pearsonr(models[left], models[right]).statistic),
                    "spearman_correlation": float(spearmanr(models[left], models[right]).statistic),
                    "rows": int(len(y)),
                    "aligned_ids_and_targets": True,
                }
            )
    pd.DataFrame(corr_rows).to_csv(
        output_dir / "PHASE19_PREDICTION_CORRELATIONS.csv", index=False
    )

    ablation_rows = []
    ablations = {
        "A_raw_numeric": "Raw numeric only",
        "B_raw_numeric_categories": "Raw numeric + raw categories",
        "C_complete_raw_rowwise": "Complete raw + approved rowwise features",
        "D_C_plus_keys_ohe": "C + eight product keys, ordinary one-hot",
        "E_C_plus_keys_frequency_ohe": "C + eight product keys, frequency-supported one-hot",
        "F_C_plus_supported_numeric_decompositions": "C + numerical product decompositions",
        "ablation_all_eight": "C + all eight product keys",
        "ablation_single_Q3": "C + Q3 only",
        "ablation_single_Q4": "C + Q4 only",
        "ablation_single_Q5": "C + Q5 only",
        "ablation_single_KQ3": "C + KQ3 only",
        "ablation_single_KQ4": "C + KQ4 only",
        "ablation_single_P4": "C + P4 only",
        "ablation_single_P5": "C + P5 only",
        "ablation_single_AG3": "C + AG3 only",
        "ablation_leave_out_Q3": "C + all keys except Q3",
        "ablation_leave_out_Q4": "C + all keys except Q4",
        "ablation_leave_out_Q5": "C + all keys except Q5",
        "ablation_leave_out_KQ3": "C + all keys except KQ3",
        "ablation_leave_out_KQ4": "C + all keys except KQ4",
        "ablation_leave_out_P4": "C + all keys except P4",
        "ablation_leave_out_P5": "C + all keys except P5",
        "ablation_leave_out_AG3": "C + all keys except AG3",
    }
    phase13_base = p13["C_complete_raw_rowwise"].to_numpy(dtype=np.float64)
    for column, description in ablations.items():
        if column not in p13:
            continue
        vector = p13[column].to_numpy(dtype=np.float64)
        ablation_rows.append(
            {
                "source_phase": "Phase 13",
                "variant": column,
                "feature_description": description,
                "pooled_oof_auc": auc(y, vector),
                "delta_vs_C_complete_raw_rowwise": auc(y, vector) - auc(y, phase13_base),
                "positive_folds_vs_C": int(
                    sum(
                        auc(y[primary_folds == f], vector[primary_folds == f])
                        > auc(y[primary_folds == f], phase13_base[primary_folds == f])
                        for f in range(1, 6)
                    )
                ),
                "fold_seed": PRIMARY_SEED,
                "eligibility": (
                    "All-eight variants include unresolved constants; numerical "
                    "decompositions are deterministic but require rule review"
                ),
                "status": "Existing matched OOF ablation; not a new independent test",
            }
        )
    for seed in PHASE9_SEEDS:
        for variant in (
            "cat_product_baseline",
            "cat_native_q4",
            "cat_native_8_keys",
            "cat_native_8_keys_plus_crosses",
        ):
            col = f"seed{seed}__{variant}"
            if col in p9:
                vector = p9[col].to_numpy(dtype=np.float64)
                base = p9[f"seed{seed}__cat_product_baseline"].to_numpy(dtype=np.float64)
                ablation_rows.append(
                    {
                        "source_phase": "Phase 9",
                        "variant": f"seed{seed}__{variant}",
                        "feature_description": variant,
                        "pooled_oof_auc": auc(y, vector),
                        "delta_vs_C_complete_raw_rowwise": auc(y, vector) - auc(y, base),
                        "positive_folds_vs_C": "",
                        "fold_seed": seed,
                        "eligibility": "Track B: unresolved formula constants and CTR status",
                        "status": "Existing within-seed family comparison; not matched to Phase 13 C",
                    }
                )
    pd.DataFrame(ablation_rows).to_csv(
        output_dir / "PHASE19_FEATURE_ABLATION.csv", index=False
    )

    error_rows = []
    if "phase19_mlp_constant_free" in models:
        from phase13_features import build_raw_feature_blocks

        full_train = pd.read_csv(TRAIN, low_memory=False)
        raw = full_train.drop(columns=["TARGET"])
        _, _, rowwise = build_raw_feature_blocks(raw)
        score_a = models["phase15_three_seed_average"]
        score_b = models["phase19_mlp_constant_free"]
        ages = rowwise["AGE_YEARS_FROM_DAYS_BIRTH"].to_numpy(dtype=np.float64)
        age_edges = np.quantile(ages[np.isfinite(ages)], [0.25, 0.5, 0.75])
        ext_count = full_train[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ].notna().sum(axis=1).to_numpy()
        groups: dict[str, np.ndarray] = {
            "cash_loans": full_train["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy(),
            "revolving_loans": full_train["NAME_CONTRACT_TYPE"].eq("Revolving loans").to_numpy(),
            "pensioner": full_train["NAME_INCOME_TYPE"].eq("Pensioner").to_numpy(),
            "ext_scores_0": ext_count == 0,
            "ext_scores_1": ext_count == 1,
            "ext_scores_2": ext_count == 2,
            "ext_scores_3": ext_count == 3,
            "age_q1": ages <= age_edges[0],
            "age_q2": (ages > age_edges[0]) & (ages <= age_edges[1]),
            "age_q3": (ages > age_edges[1]) & (ages <= age_edges[2]),
            "age_q4": ages > age_edges[2],
        }
        for name, mask in groups.items():
            if mask.sum() < 500 or np.unique(y[mask]).size != 2:
                continue
            auc15 = auc(y[mask], score_a[mask])
            auc19 = auc(y[mask], score_b[mask])
            error_rows.append(
                {
                    "subgroup": name,
                    "rows": int(mask.sum()),
                    "positive_rate": float(y[mask].mean()),
                    "phase15_three_seed_auc": auc15,
                    "phase19_mlp_auc": auc19,
                    "delta_phase19_minus_phase15": auc19 - auc15,
                    "interpretation": "Post-hoc descriptive subgroup; labels previously examined",
                }
            )
    pd.DataFrame(error_rows).to_csv(
        output_dir / "PHASE19_ERROR_SUBGROUPS.csv", index=False
    )

    matrix = frame.sort_values("pooled_oof_auc", ascending=False)
    best = matrix.iloc[0]
    phase15_row = matrix[matrix.model_id == "phase15_three_seed_average"].iloc[0]
    phase9_row = matrix[matrix.model_id == "phase9_native8_seed_average"].iloc[0]
    track_a = matrix[
        matrix["eligibility"].str.startswith(("Track A", "Conservative Track A"))
    ]
    best_track_a = track_a.iloc[0] if not track_a.empty else None
    phase15_key = "phase15_three_seed_average"
    correlations = pd.DataFrame(corr_rows)
    related = correlations[
        correlations["model_a"].eq(phase15_key)
        | correlations["model_b"].eq(phase15_key)
    ].copy()
    related["other_model"] = np.where(
        related["model_a"].eq(phase15_key),
        related["model_b"],
        related["model_a"],
    )
    related = related[related["other_model"] != phase15_key]
    strongest_diversity = (
        related.sort_values("spearman_correlation").iloc[0]
        if not related.empty
        else None
    )
    md = [
        "# Phase 19 — unified OOF recovery and error analysis",
        "",
        "All listed predictions were aligned to the same 171,202 training IDs and targets before scoring. "
        "Scores reuse historically exposed labels and are research diagnostics, not independent confirmation.",
        "",
        "## Main findings",
        "",
        f"- Phase 9 reported mean seed AUC: `{float(phase9_row['seed_auc_mean']):.9f}`; "
        f"recomputed rowwise mean-prediction AUC: `{float(phase9_row['pooled_oof_auc']):.9f}`.",
        f"- Phase 15 mean seed AUC: `{float(phase15_row['seed_auc_mean']):.9f}`; "
        f"recomputed/saved three-seed mean-prediction AUC: `{float(phase15_row['pooled_oof_auc']):.9f}`.",
        "- These seed-averaged vectors have no single native fold assignment. Their fold rows are explicitly marked as descriptive common partitions, not native validation folds.",
        "- Phase 9, Phase 11, and Phase 13–18 OOF sources use different CV fold definitions; only identical row IDs/targets justify pooled score and prediction-correlation comparisons. Fold deltas are interpreted only on matched splits.",
        f"- Highest score in this unified inventory: `{best['model_id']}` at `{float(best['pooled_oof_auc']):.9f}`. This is a retrospective same-label comparison, not a promotion decision.",
        (
            f"- Highest provisional Track A candidate: `{best_track_a['model_id']}` at "
            f"`{float(best_track_a['pooled_oof_auc']):.9f}`; this is not organizer certification."
            if best_track_a is not None
            else "- No provisional Track A model was available."
        ),
        (
            f"- Lowest prediction Spearman correlation with the Phase 15 three-seed vector: "
            f"`{strongest_diversity['other_model']}` at "
            f"`{float(strongest_diversity['spearman_correlation']):.6f}`. Low correlation alone "
            "does not establish useful ensemble lift."
            if strongest_diversity is not None
            else ""
        ),
        "",
        "## Recovery interpretation",
        "",
        "Phase 9 and Phase 15 are not controlled replications: their feature builders differ, and their multi-seed ensembles average predictions generated under different CV partitions. Phase 15's time-feature ablation was only about +0.000280 in its matched single-seed comparison, with a paired interval containing zero; Phase 16's nested parameter selection added only +0.000142 with an interval containing zero. Available evidence therefore does not isolate a causal score loss to one feature or parameter. The apparent Phase 9/15 ensemble difference is compatible with feature-builder and seed/fold variation plus averaging nonlinearity.",
        "",
        "The main structural ceiling is likely limited input information: local inventory contains only the supplied application-level table, not bureau, previous-application, installment, POS/cash, or credit-card histories. Those data are not locally available and local competition guidance prohibits external data. More model tuning on these 32 predictors cannot be promised to add 0.01 AUC.",
        "",
        "## Fold and subgroup diagnostics",
        "",
        "See `PHASE19_FOLD_METRICS.csv` for native and reference-fold definitions. `PHASE19_ERROR_SUBGROUPS.csv` contains post-hoc subgroup AUCs for Phase 15 and the Phase 19 MLP if its run is present. These subgroup estimates are descriptive and not fresh confirmation.",
        "",
        "## Decision",
        "",
        "No tiny OOF lift alone promotes a champion. The safest raw-only complete baseline is Phase 13 C; the Phase 19 MLP is the one new label-free neural-family candidate, evaluated with fold-local preprocessing. The best research OOF score remains a retrospective ensemble result, not a competition-ready score. The current evidence does not establish a path to 0.780–0.785.",
        "",
    ]
    (output_dir / "PHASE19_ERROR_ANALYSIS.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    eligibility = """# Phase 19 eligibility audit

## Track A — label-free supplied-data candidates

| Feature/model family | Label-derived? | Test fitting/coupling? | Status |
|---|---|---|---|
| Original supplied predictors with LightGBM / CatBoost / MLP | No feature target statistics; CatBoost native CTR is an exception and remains unresolved | No test data used in these OOF runs | Raw-only feature content is the conservative benchmark; no organizer certification is asserted |
| Deterministic raw/rowwise features (age, ratios, missing counts, employment-sentinel treatment) | No | Computed per row from supplied columns | Candidate Track A, subject to the local "applicable to other data" rule and competition interpretation |
| Q3/Q4/Q5/AG3 integerized ratios | No; no 130.5 or 0.0066 constants | No test data used; deterministic rowwise construction | Track A candidate, but derived-feature permission is not independently certified |
| Phase 19 fold-fitted medians/scales/category vocabularies | No labels used in fitting the transformations | Fit on each outer training fold only | Fold-safe; unknown categories map to an explicit unknown index |

## Track B — exploratory / unresolved

| Method | Reason |
|---|---|
| Phase 9/11/13/15/16/18 all-eight keys | P4/P5 and KQ3/KQ4 include unresolved 130.5/0.0066 hypotheses; CatBoost native target statistics/CTR rule status is unresolved |
| Phase 17 target rates and counts | Explicitly label-derived; organizer eligibility unresolved |
| Phase 18 blend | Inherits Phase 15 and Phase 17 unresolved methods |
| Any post-hoc OOF blend or selected candidate | Repeated exposure/adaptive selection on the same labels; not independent validation |

## Governance

No test rows, sample-submission values, external data, submissions, or champion files were used or changed in Phase 19. EXP85 was not imported. The Phase 19 neural model uses only train labels for supervised model fitting; preprocessing vocabularies, imputations, and scaling are fit inside each outer training fold. OOF labels have nevertheless been examined throughout prior project phases, so this is not clean independent confirmation. Local competition materials and absent related history tables constrain further progress.
"""
    (output_dir / "PHASE19_ELIGIBILITY_AUDIT.md").write_text(
        eligibility, encoding="utf-8"
    )
    execution_report = [
        "# Phase 19 — unified model recovery, validation, and discovery",
        "",
        "**Status:** Historical OOF recovery and matched source audit complete. "
        + (
            "A new five-fold CPU neural candidate was also completed."
            if "phase19_mlp_constant_free" in models
            else "The Phase 19 neural candidate is not yet included."
        )
        + " Research-only; no champion or submission was changed.",
        "",
        "## Unified score position",
        "",
        "| Model | Pooled OOF ROC-AUC | Seed mean AUC | Prediction-average AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in matrix.itertuples(index=False):
        seed_auc_value = optional_float(row.seed_auc_mean)
        average_auc_value = optional_float(row.prediction_average_auc)
        seed_auc = f"{seed_auc_value:.9f}" if seed_auc_value is not None else "—"
        average_auc = (
            f"{average_auc_value:.9f}"
            if average_auc_value is not None
            else "—"
        )
        execution_report.append(
            f"| {row.model_id} | {float(row.pooled_oof_auc):.9f} | {seed_auc} | {average_auc} |"
        )
    execution_report.extend(
        [
            "",
            f"Phase 9's reported mean seed AUC is `{float(phase9_row['seed_auc_mean']):.9f}`, "
            f"while its recomputed rowwise mean-prediction AUC is `{float(phase9_row['pooled_oof_auc']):.9f}`.",
            f"Phase 15's mean seed AUC is `{float(phase15_row['seed_auc_mean']):.9f}`, "
            f"while the saved/recomputed three-seed prediction average is `{float(phase15_row['pooled_oof_auc']):.9f}`.",
            "The saved Phase 19 MLP run is a historical CPU-only experiment "
            "(its summary records `gpu_available=false`, runtime `493.03` seconds, "
            "and pooled AUC `0.742694381`); it was not rerun or represented as a GPU result.",
            "The seed-averaged OOF vectors do not have a single native fold assignment. "
            "Fold metrics label the common Phase 13/15 primary partition as descriptive for those aggregates; "
            "native seed-fold results are separately retained.",
            "",
            "## Recovered model differences and decision",
            "",
            "Phase 9 and Phase 15 are not an exact matched replication. Their feature builders differ; "
            "Phase 15 adds numeric/time features, while Phase 9 used its earlier raw/rowwise builder. "
            "Both use GPU CatBoost native categorical statistics and eight keys. Phase 9 averages "
            "three OOF vectors from split seeds 42/7/2026; Phase 15 averages seeds 20261113/14/15. "
            "The Phase 15 time-feature matched lift was only +0.000280 (paired interval included zero), "
            "and Phase 16 nested tuning added +0.000142 (interval included zero). The records do not "
            "isolate one cause for the difference between aggregate scores.",
            "",
            "The highest-scoring conservative no-product-key reference in this inventory is the "
            "Phase 13 C benchmark at "
            f"`{float(matrix[matrix.model_id == 'phase13_C_raw_rowwise'].iloc[0]['pooled_oof_auc']):.9f}`. "
            "The Phase 19 MLP scored lower and does not change this position. "
            "Eligibility remains provisional and subject to organizer interpretation.",
            "The Phase 15 three-seed average remains the best complete result from Phase 15; "
            "Phase 9's 0.769839 retrospective rowwise average is higher but is not a controlled "
            "reproduction or independently validated champion. No basis was found to claim 0.780–0.785.",
            "",
            "## Validation, limitations, and next step",
            "",
            "All requested source OOF files were checked for row count, applicant IDs, target alignment, "
            "and native fold IDs where saved or reconstructable. Correlations use only exactly aligned "
            "OOF vectors. Full fold definitions, model settings, features, eligibility, and reproducibility "
            "status are in `PHASE19_MODEL_COMPARISON.csv`; fold detail is in `PHASE19_FOLD_METRICS.csv`.",
            "No new adaptive blend was selected. All historical labels have been repeatedly examined, "
            "so nested selection and paired row bootstrap do not provide independent confirmation. "
            "The local data contains no bureau/history tables; external data is not allowed. "
            "Stop tuning on the same OOF labels unless a genuinely untouched labeled evaluation becomes available.",
            "",
            "See `PHASE19_ERROR_ANALYSIS.md`, `PHASE19_FEATURE_ABLATION.csv`, "
            "`PHASE19_PREDICTION_CORRELATIONS.csv`, and `PHASE19_ELIGIBILITY_AUDIT.md`.",
            "",
        ]
    )
    (output_dir / "PHASE19_EXECUTION_REPORT.md").write_text(
        "\n".join(execution_report), encoding="utf-8"
    )

    sources = [
        TRAIN,
        PHASE9,
        p9_folds_path,
        PHASE11,
        PHASE13,
        PHASE15_MULTI,
        PHASE16,
        PHASE17,
        PHASE18,
    ]
    if mlp_run is not None:
        sources.extend(
            [
                mlp_run / "OOF_PREDICTIONS.csv",
                mlp_run / "FOLD_METRICS.csv",
                mlp_run / "SUMMARY.json",
                mlp_run / "SHA256_MANIFEST.csv",
            ]
        )
    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "PHASE19_FINAL_SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "kind": "output",
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in sources:
        manifest_rows.append(
            {
                "kind": "input",
                "file": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (output_dir / "PHASE19_FINAL_SHA256_MANIFEST.csv").write_text(
        pd.DataFrame(manifest_rows).to_csv(index=False), encoding="utf-8"
    )


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--mlp-run", type=Path)
    args = parser.parse_args()
    main(args.output_dir.resolve(), args.mlp_run.resolve() if args.mlp_run else None)


if __name__ == "__main__":
    cli()
