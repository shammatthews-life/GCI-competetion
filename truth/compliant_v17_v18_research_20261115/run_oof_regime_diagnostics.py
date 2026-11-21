from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
EXPERIMENTS = REPO / "experiments"
OUTPUT = ROOT / "runs" / "historical_oof_diagnostics_clean_v1"
SEED = 20261113
FOLDS = 5


def load_aligned(path: Path, train: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if (
        len(frame) != len(train)
        or not frame[id_col].equals(train[id_col])
        or not frame[target_col].equals(train[target_col])
    ):
        raise ValueError(f"OOF data do not align with official training rows: {path}")
    return frame


def rank_within_folds(values: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        if mask.sum() < 2:
            raise ValueError(f"Fold {fold} is missing or too small")
        result[mask] = (rankdata(values[mask], method="average") - 1) / (
            mask.sum() - 1
        )
    if not np.isfinite(result).all():
        raise ValueError("Fold rank transform produced invalid values")
    return result


def fold_auc_records(
    name: str,
    prediction: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    group_name: str,
    groups: np.ndarray,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for value in sorted(np.unique(groups)):
        group_mask = groups == value
        for fold in range(1, FOLDS + 1):
            mask = group_mask & (fold_ids == fold)
            positives = int(y[mask].sum())
            negatives = int(mask.sum() - positives)
            if positives == 0 or negatives == 0:
                continue
            records.append(
                {
                    "model": name,
                    "grouping": group_name,
                    "group": str(value),
                    "fold": fold,
                    "rows": int(mask.sum()),
                    "positives": positives,
                    "auc": float(roc_auc_score(y[mask], prediction[mask])),
                }
            )
    return records


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostics: {OUTPUT}")
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE",
                                       "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                                       "DAYS_EMPLOYED"])
    y = train["TARGET"].to_numpy(dtype=np.int8)
    base_dir = (
        EXPERIMENTS
        / "phase37_multiseed_xgboost_rules_first"
        / "runs"
        / "phase37_multiseed_20261107T173127Z"
    )
    phase37 = load_aligned(
        base_dir / "PHASE37_OOF_PREDICTIONS.csv", train, "SK_ID_CURR", "TARGET"
    )
    fold_ids = phase37["fold_seed_20261113"].to_numpy(dtype=np.int8)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(
        splitter.split(np.zeros(len(y)), y), start=1
    ):
        expected_fold_ids[valid] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Phase 37 fold IDs do not match the declared split")

    sources = {
        "phase52": (
            EXPERIMENTS
            / "phase52_gpu_archive_port"
            / "runs"
            / "phase52_gpu_20261108"
            / "phase52_oof.csv"
        ),
        "phase44": (
            EXPERIMENTS
            / "phase44_fold_safe_experts"
            / "runs"
            / "phase44_experts_20261108T095439Z"
            / "PHASE44_OOF_PREDICTIONS.csv"
        ),
        "phase45": (
            EXPERIMENTS
            / "phase45_nested_expert_calibration"
            / "runs"
            / "phase45_calibrated_20261108T095905Z"
            / "PHASE45_OOF_PREDICTIONS.csv"
        ),
        "phase17": (
            EXPERIMENTS
            / "phase17_nested_constant_free_target_encoding"
            / "runs"
            / "phase17_nested_te_20261105T140620Z"
            / "OOF_PREDICTIONS.csv"
        ),
        "phase18": (
            EXPERIMENTS
            / "phase18_nested_oof_ensemble"
            / "runs"
            / "phase18_nested_blend_20261105T142048Z"
            / "OOF_PREDICTIONS.csv"
        ),
    }
    loaded = {
        name: load_aligned(path, train, "SK_ID_CURR", "TARGET")
        for name, path in sources.items()
    }
    clean_catboost_path = ROOT / "runs" / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
    clean_catboost = load_aligned(clean_catboost_path, train, "SK_ID_CURR", "TARGET")
    if not np.array_equal(clean_catboost["fold"].to_numpy(dtype=np.int8), fold_ids):
        raise ValueError("Clean CatBoost OOF fold assignment does not align")
    for name in ("phase44", "phase45", "phase17", "phase18"):
        frame = loaded[name]
        if "fold" not in frame or not np.array_equal(
            frame["fold"].to_numpy(dtype=np.int8), fold_ids
        ):
            raise ValueError(f"{name} OOF fold assignment does not align")

    predictions: dict[str, np.ndarray] = {
        "phase37_reference": phase37[
            "oof_equal_blend_lgbm_xgb_three_seeds"
        ].to_numpy(dtype=np.float64),
        "phase52_label_free_feature_model": loaded["phase52"][
            "v7_label_free_features"
        ].to_numpy(dtype=np.float64),
        "phase44_contract_expert_mix": loaded["phase44"][
            "oof_contract_type_mixture_50_50"
        ].to_numpy(dtype=np.float64),
        "phase44_external_completeness_mix": loaded["phase44"][
            "oof_external_completeness_mixture_50_50"
        ].to_numpy(dtype=np.float64),
        "phase45_contract_calibrated_mix": loaded["phase45"][
            "oof_contract_type_platt_mixture_50_50"
        ].to_numpy(dtype=np.float64),
        "phase17_nested_constant_free_te": loaded["phase17"][
            "nested_te_constant_free"
        ].to_numpy(dtype=np.float64),
        "phase18_nested_mixed_expert": loaded["phase18"][
            "phase18_nested_blend"
        ].to_numpy(dtype=np.float64),
        "clean_constant_free_catboost": clean_catboost[
            "oof_clean_catboost"
        ].to_numpy(dtype=np.float64),
    }
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("At least one OOF prediction vector is non-finite")

    baseline = predictions["phase37_reference"]
    blend_specs = {
        "phase52_label_free_feature_model": (0.25, 0.50),
        "phase44_contract_expert_mix": (0.25, 0.50),
        "phase44_external_completeness_mix": (0.25, 0.50),
        "phase45_contract_calibrated_mix": (0.25, 0.50),
        "phase17_nested_constant_free_te": (0.25, 0.50),
        "phase18_nested_mixed_expert": (0.25, 0.50),
        "clean_constant_free_catboost": (0.25, 0.50),
    }
    for name, weights in blend_specs.items():
        candidate_rank = rank_within_folds(predictions[name], fold_ids)
        baseline_rank = rank_within_folds(baseline, fold_ids)
        for weight in weights:
            blend_name = f"phase37_plus_{name}_rank_{weight:.2f}"
            predictions[blend_name] = (
                (1.0 - weight) * baseline_rank + weight * candidate_rank
            )

    scores: dict[str, dict[str, object]] = {}
    for name, values in predictions.items():
        per_fold = [
            float(roc_auc_score(y[fold_ids == fold], values[fold_ids == fold]))
            for fold in range(1, FOLDS + 1)
        ]
        scores[name] = {
            "pooled_auc": float(roc_auc_score(y, values)),
            "mean_fold_auc": float(np.mean(per_fold)),
            "fold_auc": per_fold,
        }

    raw = pd.read_csv(
        TRAIN,
        usecols=[
            "NAME_CONTRACT_TYPE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "DAYS_EMPLOYED",
        ],
    )
    groups = {
        "contract_type": raw["NAME_CONTRACT_TYPE"]
        .astype("string")
        .fillna("MISSING")
        .to_numpy(dtype=str),
        "external_source_count": raw[
            ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
        ]
        .notna()
        .sum(axis=1)
        .astype(str)
        .to_numpy(),
        "employment_sentinel": (raw["DAYS_EMPLOYED"] == 365243)
        .astype(str)
        .to_numpy(),
    }
    regime_records: list[dict[str, object]] = []
    for grouping, values in groups.items():
        for model in (
            "phase37_reference",
            "phase52_label_free_feature_model",
            "phase44_contract_expert_mix",
            "phase17_nested_constant_free_te",
            "phase18_nested_mixed_expert",
            "clean_constant_free_catboost",
            "phase37_plus_clean_constant_free_catboost_rank_0.25",
            "phase37_plus_clean_constant_free_catboost_rank_0.50",
            "phase37_plus_phase17_nested_constant_free_te_rank_0.50",
            "phase37_plus_phase18_nested_mixed_expert_rank_0.50",
        ):
            regime_records.extend(
                fold_auc_records(model, predictions[model], y, fold_ids, grouping, values)
            )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(regime_records).to_csv(OUTPUT / "REGIME_FOLD_AUC.csv", index=False)
    (OUTPUT / "OOF_SUMMARY.json").write_text(
        json.dumps(
            {
                "status": "exploratory diagnostics on repeatedly examined labels; not an independent estimate",
                "validation": "fixed five-fold OOF; seed 20261113; every included artifact checked for ID/target alignment and fold identity",
                "interpretation": "AUC by feature-defined subgroup and fixed within-fold rank blend; confusion matrices are omitted because no probability threshold is specified by ROC-AUC",
                "rule_notes": {
                    "phase18": "not a compliant candidate; includes Phase 15 features/constants with unresolved provenance",
                    "phase17": "nested label-derived target statistics on constant-free product keys; follows the user's stated fold-local TE recommendation, but official wording is not explicit",
                    "clean_catboost": "new fold-local GPU CatBoost model; constant-free row-wise features; no test rows or peer-label features",
                    "phase44_phase45": "feature-only fold-safe experts",
                    "phase52": "label-free rowwise feature model",
                },
                "models": scores,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), "models": scores}, indent=2), flush=True)


if __name__ == "__main__":
    main()
