from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE33_OOF = Path(
    r"D:\gci\truth\experiments\phase33_rules_first_features"
    r"\runs\phase33_rules_first_20261107T170931Z\PHASE33_OOF_PREDICTIONS.csv"
)
PHASE34_OOF = Path(
    r"D:\gci\truth\experiments\phase34_xgboost_rules_first"
    r"\runs\phase34_xgb_20261107T171823Z\PHASE34_OOF_PREDICTIONS.csv"
)
CLEAN_CatBOOST_OOF = ROOT / "runs" / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "fold_aligned_catboost_blend_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_aligned(path: Path, train: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if (
        len(frame) != len(train)
        or not frame[ID].equals(train[ID])
        or not frame[TARGET].equals(train[TARGET])
    ):
        raise ValueError(f"OOF rows, IDs, or labels do not align: {path}")
    return frame


def rank_within_fold(values: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    ranked = np.full(len(values), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        if mask.sum() < 2:
            raise ValueError(f"Fold {fold} is missing or too small")
        ranked[mask] = (rankdata(values[mask], method="average") - 1) / (
            mask.sum() - 1
        )
    if not np.isfinite(ranked).all():
        raise ValueError("Fold rank transform produced non-finite values")
    return ranked


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT}")
    for path in (TRAIN, PHASE33_OOF, PHASE34_OOF, CLEAN_CatBOOST_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis input is missing: {path}")

    train = pd.read_csv(
        TRAIN,
        usecols=[
            ID,
            TARGET,
            "NAME_CONTRACT_TYPE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "DAYS_EMPLOYED",
        ],
    )
    phase33 = read_aligned(PHASE33_OOF, train)
    phase34 = read_aligned(PHASE34_OOF, train)
    catboost = read_aligned(CLEAN_CatBOOST_OOF, train)

    fold_ids = phase33["fold"].to_numpy(dtype=np.int8)
    if not np.array_equal(phase34["fold"].to_numpy(dtype=np.int8), fold_ids):
        raise ValueError("Phase 33 and Phase 34 fold assignments differ")
    if not np.array_equal(catboost["fold"].to_numpy(dtype=np.int8), fold_ids):
        raise ValueError("Clean CatBoost and reference fold assignments differ")
    if set(np.unique(fold_ids)) != set(range(1, FOLDS + 1)):
        raise ValueError("Unexpected fold identifiers")

    phase33_prediction = phase33["oof_keys_frequency"].to_numpy(dtype=np.float64)
    phase34_prediction = phase34["oof_xgboost"].to_numpy(dtype=np.float64)
    same_fold_baseline = phase34[
        "oof_fixed_equal_blend_with_phase33_lgbm"
    ].to_numpy(dtype=np.float64)
    expected_baseline = (phase33_prediction + phase34_prediction) / 2.0
    if not np.allclose(
        same_fold_baseline, expected_baseline, rtol=0.0, atol=1e-7
    ):
        raise ValueError("Saved same-fold baseline is not the fixed Phase 33/34 blend")

    cat_prediction = catboost["oof_clean_catboost"].to_numpy(dtype=np.float64)
    baseline_rank = rank_within_fold(same_fold_baseline, fold_ids)
    cat_rank = rank_within_fold(cat_prediction, fold_ids)
    rank_blend = 0.5 * baseline_rank + 0.5 * cat_rank
    y = train[TARGET].to_numpy(dtype=np.int8)

    vectors = {
        "same_fold_phase33_phase34_baseline": same_fold_baseline,
        "clean_constant_free_catboost": cat_prediction,
        "fixed_50_50_within_fold_rank_blend": rank_blend,
    }
    if any(not np.isfinite(values).all() for values in vectors.values()):
        raise ValueError("A prediction vector contains non-finite values")

    fold_records: list[dict[str, object]] = []
    summary_models: dict[str, dict[str, object]] = {}
    for name, prediction in vectors.items():
        fold_aucs: list[float] = []
        for fold in range(1, FOLDS + 1):
            mask = fold_ids == fold
            auc = float(roc_auc_score(y[mask], prediction[mask]))
            fold_aucs.append(auc)
            fold_records.append(
                {
                    "model": name,
                    "fold": fold,
                    "rows": int(mask.sum()),
                    "positives": int(y[mask].sum()),
                    "auc": auc,
                }
            )
        summary_models[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "mean_fold_auc": float(np.mean(fold_aucs)),
            "fold_auc": fold_aucs,
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
        for value in sorted(np.unique(values)):
            group_mask = values == value
            for fold in range(1, FOLDS + 1):
                mask = group_mask & (fold_ids == fold)
                positives = int(y[mask].sum())
                negatives = int(mask.sum() - positives)
                if positives == 0 or negatives == 0:
                    continue
                for name, prediction in vectors.items():
                    regime_records.append(
                        {
                            "grouping": grouping,
                            "group": str(value),
                            "model": name,
                            "fold": fold,
                            "rows": int(mask.sum()),
                            "positives": positives,
                            "auc": float(roc_auc_score(y[mask], prediction[mask])),
                        }
                    )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    predictions_path = OUTPUT / "OOF_PREDICTIONS.csv"
    folds_path = OUTPUT / "FOLD_METRICS.csv"
    regimes_path = OUTPUT / "REGIME_FOLD_AUC.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{name: values.astype(np.float32) for name, values in vectors.items()},
        }
    ).to_csv(predictions_path, index=False)
    pd.DataFrame(fold_records).to_csv(folds_path, index=False)
    pd.DataFrame(regime_records).to_csv(regimes_path, index=False)

    baseline_auc = summary_models["same_fold_phase33_phase34_baseline"][
        "mean_fold_auc"
    ]
    blend_auc = summary_models["fixed_50_50_within_fold_rank_blend"][
        "mean_fold_auc"
    ]
    summary = {
        "status": "exploratory fold-aligned OOF analysis on repeatedly examined labels; not independent confirmation",
        "validation": "Phase 33, Phase 34, and clean CatBoost use identical five-fold partitions; all labels in an evaluated fold are excluded from all component models used for that fold",
        "blend": "fixed 50/50 average of within-fold percentile ranks; no weight search",
        "metrics": summary_models,
        "mean_fold_auc_delta_blend_vs_aligned_baseline": float(blend_auc - baseline_auc),
        "positive_fold_delta_count": int(
            sum(
                summary_models["fixed_50_50_within_fold_rank_blend"]["fold_auc"][i]
                > summary_models["same_fold_phase33_phase34_baseline"]["fold_auc"][i]
                for i in range(FOLDS)
            )
        ),
        "rule_notes": {
            "inputs": "official training labels for model development and saved OOF component predictions",
            "test_rows": "not read",
            "label_transfer": "none",
            "transductive_statistics": "none",
            "fixed_fee_or_step_constants": "none in the new CatBoost feature builder",
            "fold_alignment": "checked row-by-row, label-by-label, and fold-by-fold",
        },
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                TRAIN,
                PHASE33_OOF,
                PHASE34_OOF,
                CLEAN_CatBOOST_OOF,
                Path(__file__).resolve(),
                predictions_path,
                folds_path,
                regimes_path,
            )
        },
    }
    summary_path = OUTPUT / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
