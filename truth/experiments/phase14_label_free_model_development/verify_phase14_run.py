from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from run_phase14 import SEED, sha256_file


ROOT = Path(__file__).resolve().parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE13_OOF = (
    ROOT.parent
    / "phase13_matched_baseline"
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)
RUN = ROOT / "runs" / "phase14_20261105T114015Z"
MODEL_VARIANTS = (
    "phase13_C",
    "phase13_D",
    "phase13_E",
    "lgbm_all8_augmented",
    "catboost_all8_augmented",
    "lgbm_constant_free_keys",
    "blend_lgbm_catboost_augmented_equal",
)


def verify_and_summarize() -> None:
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    phase13 = pd.read_csv(PHASE13_OOF, usecols=["row_index", "SK_ID_CURR", "TARGET"])
    oof = pd.read_csv(RUN / "OOF_PREDICTIONS.csv")
    if len(train) != len(oof) or len(phase13) != len(oof):
        raise ValueError("Train, Phase 13 reference, and Phase 14 OOF row counts differ")
    expected_rows = np.arange(len(train))
    for frame_name, frame in (("Phase 13", phase13), ("Phase 14", oof)):
        if not np.array_equal(frame["row_index"].to_numpy(), expected_rows):
            raise ValueError(f"{frame_name} row_index is not the original input order")
        if not np.array_equal(
            frame["SK_ID_CURR"].to_numpy(), train["SK_ID_CURR"].to_numpy()
        ):
            raise ValueError(f"{frame_name} IDs do not align with the input")
        if not np.array_equal(
            frame["TARGET"].to_numpy(dtype=np.int8),
            train["TARGET"].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"{frame_name} TARGET does not align with the input")

    y = train["TARGET"].to_numpy(dtype=np.int8)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    expected_folds = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, apply) in enumerate(
        split.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        expected_folds[apply] = fold
    if not np.array_equal(oof["primary_fold"].to_numpy(dtype=np.int8), expected_folds):
        raise ValueError("Phase 14 fold IDs do not match the preregistered split")

    predictions = oof.drop(
        columns=["row_index", "SK_ID_CURR", "TARGET", "primary_fold"]
    )
    if predictions.columns.duplicated().any():
        raise ValueError("Duplicate prediction columns found")
    if not np.isfinite(predictions.to_numpy(dtype=np.float64)).all():
        raise ValueError("Missing or non-finite OOF predictions found")
    model_results = pd.read_csv(RUN / "MODEL_RESULTS.csv").set_index("variant")
    fold_results = pd.read_csv(RUN / "FOLD_LEVEL_METRICS.csv")
    reference = pd.read_csv(
        PHASE13_OOF,
        usecols=[
            "row_index",
            "C_complete_raw_rowwise",
            "D_C_plus_keys_ohe",
            "E_C_plus_keys_frequency_ohe",
        ],
    )
    prediction_map = {
        "phase13_C": reference["C_complete_raw_rowwise"].to_numpy(dtype=float),
        "phase13_D": reference["D_C_plus_keys_ohe"].to_numpy(dtype=float),
        "phase13_E": reference["E_C_plus_keys_frequency_ohe"].to_numpy(dtype=float),
        **{
            name: oof[name].to_numpy(dtype=float)
            for name in predictions.columns
            if name not in {"phase13_C", "phase13_D", "phase13_E"}
        },
    }
    for name in MODEL_VARIANTS:
        pooled = roc_auc_score(y, prediction_map[name])
        if not np.isclose(pooled, model_results.loc[name, "pooled_oof_auc"], atol=1e-12):
            raise ValueError(f"Pooled AUC mismatch for {name}")
        for fold in range(1, 6):
            mask = expected_folds == fold
            saved = fold_results.loc[
                fold_results.variant.eq(name) & fold_results.outer_fold.eq(fold),
                "auc",
            ]
            if len(saved) != 1 or not np.isclose(
                roc_auc_score(y[mask], prediction_map[name][mask]),
                saved.iloc[0],
                atol=1e-12,
            ):
                raise ValueError(f"Fold {fold} AUC mismatch for {name}")

    selection = pd.read_csv(RUN / "INNER_MODEL_SELECTION.csv")
    catboost_categories = selection.loc[
        selection.model.eq("catboost"), "max_fit_category_count"
    ]
    if catboost_categories.isna().any() or (
        catboost_categories >= 10000
    ).any():
        raise ValueError("CatBoost fit category count reached CTR-permitting threshold")

    subgroup = pd.read_csv(RUN / "SUBGROUP_STABILITY.csv")
    subgroup["group_value"] = subgroup["group_value"].astype("string")
    reference_groups = subgroup.loc[
        subgroup.variant.eq("phase13_E"),
        ["outer_fold", "group", "group_value", "auc"],
    ].rename(columns={"auc": "reference_auc"})
    paired = subgroup.merge(
        reference_groups,
        on=["outer_fold", "group", "group_value"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Subgroup metric cannot be paired to Phase 13 E")
    paired = paired.drop(columns="_merge")
    paired["delta_vs_phase13_E"] = paired["auc"] - paired["reference_auc"]
    paired.to_csv(RUN / "SUBGROUP_STABILITY_PAIRED.csv", index=False)
    tracked = {
        "lgbm_all8_augmented",
        "catboost_all8_augmented",
        "lgbm_constant_free_keys",
        "blend_lgbm_catboost_augmented_equal",
    }
    summary = []
    for (variant, group, value), group_rows in paired.loc[
        paired.variant.isin(tracked)
    ].groupby(["variant", "group", "group_value"], sort=True):
        group_rows = group_rows.dropna(subset=["delta_vs_phase13_E"])
        if group_rows.empty:
            continue
        deltas = group_rows["delta_vs_phase13_E"].to_numpy(dtype=float)
        summary.append(
            {
                "variant": variant,
                "group": group,
                "group_value": value,
                "folds_present": len(group_rows),
                "mean_fold_auc": group_rows["auc"].mean(),
                "mean_delta_vs_phase13_E": deltas.mean(),
                "positive_folds_vs_phase13_E": int((deltas > 0).sum()),
                "min_fold_delta_vs_phase13_E": deltas.min(),
                "max_fold_delta_vs_phase13_E": deltas.max(),
                "rows_mean": group_rows["rows"].mean(),
                "positives_mean": group_rows["positives"].mean(),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(RUN / "SUBGROUP_STABILITY_SUMMARY.csv", index=False)

    train_hash = sha256_file(TRAIN)
    phase13_hash = sha256_file(PHASE13_OOF)
    oof_hash = sha256_file(RUN / "OOF_PREDICTIONS.csv")
    report = f"""# Phase 14 OOF integrity and independent artifact verification

- Status: **PASS**
- Training rows / unique IDs / positives: {len(train):,} / {train.SK_ID_CURR.nunique():,} / {int(y.sum()):,}
- Phase 13 input row/ID/target alignment: PASS
- Phase 14 OOF row/ID/target alignment: PASS
- Phase 14 stratified folds reproduced with seed {SEED}: PASS; fold sizes {json.dumps(pd.Series(expected_folds).value_counts().sort_index().to_dict(), sort_keys=True)}
- OOF prediction columns: {len(predictions.columns)}; missing/non-finite values: 0
- Pooled AUCs recomputed from saved OOF predictions: {len(MODEL_VARIANTS)}/{len(MODEL_VARIANTS)} match `MODEL_RESULTS.csv` within `1e-12`
- Fold AUCs recomputed from saved predictions: {len(MODEL_VARIANTS) * 5}/{len(MODEL_VARIANTS) * 5} match `FOLD_LEVEL_METRICS.csv` within `1e-12`
- CatBoost fit-partition maximum category cardinality: {int(catboost_categories.max()):,}; `one_hot_max_size`: 10,000; target-derived CTR threshold was not reached
- Paired subgroup outputs: {len(paired):,} fold/group/model rows; {len(summary_frame):,} model/group summaries. Subgroups are descriptive, not independent validation.
- Training SHA-256: `{train_hash}`
- Phase 13 OOF SHA-256: `{phase13_hash}`
- Phase 14 OOF SHA-256: `{oof_hash}`

No test data, external data, submissions, or champion files were used in this verification.
"""
    (RUN / "OOF_INTEGRITY_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    for name in ("SUBGROUP_STABILITY_PAIRED.csv", "SUBGROUP_STABILITY_SUMMARY.csv"):
        print(
            f"{name}: {sha256_file(RUN / name)} "
            f"({(RUN / name).stat().st_size} bytes)"
        )


if __name__ == "__main__":
    verify_and_summarize()
