from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci\truth")
TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEEDS = (42, 7, 2026)
N_SPLITS = 5
LGBM_VARIANTS = (
    "product_baseline", "te_q3", "te_q4", "te_q5", "te_q4_cash",
    "te_amount_ca", "te_amount_gc", "te_exact_gca", "te_k_only",
    "te_q4_plus_k", "ordinal_q4",
)
CAT_VARIANTS = (
    "cat_product_baseline", "cat_native_q4",
    "cat_native_8_keys", "cat_native_8_keys_plus_crosses",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    oof = pd.read_csv(run_dir / "OOF_PREDICTIONS.csv")
    require(len(train) == 171202, "Unexpected training row count")
    require(oof[ID].equals(train[ID]), "OOF IDs/order differ from training input")
    require(oof[TARGET].equals(train[TARGET]), "OOF TARGET/order differs from training input")
    y = train[TARGET].to_numpy(dtype=np.int8)

    expected_variants = {
        "LightGBM": LGBM_VARIANTS,
        "CatBoost_GPU": CAT_VARIANTS,
    }
    expected_prediction_columns = {
        f"seed{seed}__{variant}"
        for seed in SEEDS
        for variants in expected_variants.values()
        for variant in variants
    }
    prediction_columns = set(oof.columns) - {ID, TARGET}
    require(prediction_columns == expected_prediction_columns, "OOF prediction schema mismatch")
    require(
        np.isfinite(oof[list(expected_prediction_columns)].to_numpy(dtype=np.float64)).all(),
        "OOF predictions contain nonfinite values",
    )

    fold_ids_by_seed: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        fold_ids = np.empty(len(y), dtype=np.int8)
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (_, valid_idx) in enumerate(
            splitter.split(np.zeros(len(y)), y), start=1
        ):
            fold_ids[valid_idx] = fold
        fold_ids_by_seed[seed] = fold_ids

    metrics = pd.read_csv(run_dir / "SEED_METRICS.csv")
    fold_results = {
        "LightGBM": pd.read_csv(run_dir / "LIGHTGBM_FOLD_RESULTS.csv"),
        "CatBoost_GPU": pd.read_csv(run_dir / "CATBOOST_FOLD_RESULTS.csv"),
    }
    require(len(metrics) == 3 * (len(LGBM_VARIANTS) + len(CAT_VARIANTS)), "Seed metric row count mismatch")
    require(
        len(fold_results["LightGBM"]) == 3 * N_SPLITS * len(LGBM_VARIANTS),
        "LightGBM fold result row count mismatch",
    )
    require(
        len(fold_results["CatBoost_GPU"]) == 3 * N_SPLITS * len(CAT_VARIANTS),
        "CatBoost fold result row count mismatch",
    )

    recomputed_rows = []
    for family, variants in expected_variants.items():
        fold_frame = fold_results[family]
        for seed in SEEDS:
            baseline_name = "product_baseline" if family == "LightGBM" else "cat_product_baseline"
            baseline_pred = oof[f"seed{seed}__{baseline_name}"].to_numpy(dtype=np.float64)
            baseline_auc = roc_auc_score(y, baseline_pred)
            for variant in variants:
                pred = oof[f"seed{seed}__{variant}"].to_numpy(dtype=np.float64)
                pooled_auc = float(roc_auc_score(y, pred))
                recorded = metrics.loc[
                    (metrics.model_family == family)
                    & (metrics.seed == seed)
                    & (metrics.variant == variant)
                ]
                require(len(recorded) == 1, f"Missing/duplicate seed metric: {family}/{seed}/{variant}")
                recorded_row = recorded.iloc[0]
                require(
                    abs(pooled_auc - float(recorded_row.pooled_oof_auc)) < 1e-10,
                    f"Pooled OOF AUC mismatch: {family}/{seed}/{variant}",
                )
                fold_auc_values = []
                for fold in range(1, N_SPLITS + 1):
                    mask = fold_ids_by_seed[seed] == fold
                    auc = float(roc_auc_score(y[mask], pred[mask]))
                    fold_auc_values.append(auc)
                    fold_recorded = fold_frame.loc[
                        (fold_frame.seed == seed)
                        & (fold_frame.fold == fold)
                        & (fold_frame.variant == variant)
                    ]
                    require(len(fold_recorded) == 1, f"Missing/duplicate fold row: {family}/{seed}/{fold}/{variant}")
                    require(
                        abs(auc - float(fold_recorded.iloc[0].fold_auc)) < 1e-10,
                        f"Fold AUC mismatch: {family}/{seed}/{fold}/{variant}",
                    )
                delta = pooled_auc - baseline_auc
                require(
                    abs(delta - float(recorded_row.delta_vs_family_product_baseline)) < 1e-10,
                    f"Baseline delta mismatch: {family}/{seed}/{variant}",
                )
                recomputed_rows.append(
                    {
                        "model_family": family,
                        "seed": seed,
                        "variant": variant,
                        "independent_pooled_oof_auc": pooled_auc,
                        "recorded_pooled_oof_auc": float(recorded_row.pooled_oof_auc),
                        "delta_vs_family_baseline": delta,
                        "fold_auc_json": json.dumps(fold_auc_values),
                        "verification": "PASS",
                    }
                )

    integrity = pd.read_csv(run_dir / "TARGET_MAP_INTEGRITY.csv")
    require(len(integrity) == 3 * N_SPLITS * 9, "Target-map integrity row count mismatch")
    require(integrity.outer_fit_validation_disjoint.astype(bool).all(), "Outer fit/validation overlap flagged")
    require(integrity.inner_map_apply_disjoint.astype(bool).all(), "Inner map/apply overlap flagged")
    require((~integrity.outer_validation_labels_used_for_map.astype(bool)).all(), "Outer validation labels used in map")
    require((integrity.inner_folds == 5).all(), "Unexpected inner-fold count")

    aggregate = pd.read_csv(run_dir / "AGGREGATE_COMPARISON.csv")
    require(len(aggregate) == len(LGBM_VARIANTS) + len(CAT_VARIANTS), "Aggregate comparison row count mismatch")

    phase8_oof = pd.read_csv(
        ROOT
        / "experiments"
        / "phase8_nested_product_target_encoding"
        / "runs"
        / "phase8_20261104T080000Z"
        / "OOF_PREDICTIONS.csv"
    )
    subgroup_input = train[
        [
            ID, TARGET, "NAME_CONTRACT_TYPE", "DAYS_BIRTH", "NAME_INCOME_TYPE",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        ]
    ].merge(
        phase8_oof,
        on=[ID, TARGET],
        how="left",
        validate="one_to_one",
    )
    require(
        subgroup_input["seed42__product_plus_te8"].notna().all(),
        "Phase 8 subgroup OOF alignment failed",
    )
    age = -subgroup_input["DAYS_BIRTH"].to_numpy(dtype=np.float64) / 365.25
    age_cuts = np.quantile(age[np.isfinite(age)], [0.25, 0.5, 0.75])
    subgroup_masks = {
        "all": np.ones(len(subgroup_input), dtype=bool),
        "cash_loans": subgroup_input["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy(),
        "revolving_loans": subgroup_input["NAME_CONTRACT_TYPE"].eq("Revolving loans").to_numpy(),
        "pensioner": subgroup_input["NAME_INCOME_TYPE"].eq("Pensioner").to_numpy(),
        "age_q1_youngest": age <= age_cuts[0],
        "age_q2": (age > age_cuts[0]) & (age <= age_cuts[1]),
        "age_q3": (age > age_cuts[1]) & (age <= age_cuts[2]),
        "age_q4_oldest": age > age_cuts[2],
    }
    ext_count = np.isfinite(
        subgroup_input[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        .to_numpy(dtype=np.float64)
    ).sum(axis=1)
    subgroup_masks.update(
        {
            "ext_available_0": ext_count == 0,
            "ext_available_1": ext_count == 1,
            "ext_available_2plus": ext_count >= 2,
        }
    )
    saved_subgroups = pd.read_csv(run_dir / "SUBGROUP_OOF_DIAGNOSTICS.csv")
    for seed in SEEDS:
        ref = subgroup_input[f"seed{seed}__raw_plus_phase3_product"].to_numpy(dtype=np.float64)
        candidate = subgroup_input[f"seed{seed}__product_plus_te8"].to_numpy(dtype=np.float64)
        for group, mask in subgroup_masks.items():
            if mask.sum() < 200 or np.unique(y[mask]).size < 2:
                continue
            row = saved_subgroups.loc[
                (saved_subgroups.seed == seed) & (saved_subgroups.group == group)
            ]
            require(len(row) == 1, f"Missing/duplicate subgroup diagnostic: {seed}/{group}")
            expected_auc = float(roc_auc_score(y[mask], candidate[mask]))
            expected_delta = expected_auc - float(roc_auc_score(y[mask], ref[mask]))
            require(abs(expected_auc - float(row.iloc[0].nested_te8_auc)) < 1e-10, f"Subgroup AUC mismatch: {seed}/{group}")
            require(abs(expected_delta - float(row.iloc[0].delta_auc)) < 1e-10, f"Subgroup delta mismatch: {seed}/{group}")

    cash = train["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    amount = train.loc[cash, ["AMT_GOODS_PRICE", "AMT_CREDIT", "AMT_ANNUITY"]].dropna()
    adjacent = []
    for _, values in amount.groupby(["AMT_GOODS_PRICE", "AMT_CREDIT"])["AMT_ANNUITY"]:
        unique_values = np.sort(values.unique().astype(np.float64))
        if len(unique_values) > 1:
            adjacent.extend(np.diff(unique_values).tolist())
    adjacent = np.asarray(adjacent, dtype=np.float64)
    fee_audit = pd.read_csv(run_dir / "FEE_GAP_PLACEBO_AUDIT.csv")
    require(len(fee_audit) == 7, "Fee placebo row count mismatch")
    for row in fee_audit.itertuples(index=False):
        gap_mask = np.isclose(adjacent, row.adjacent_annuity_gap, rtol=0.0, atol=1e-6)
        require(int(gap_mask.sum()) == int(row.adjacent_gap_count), f"Fee-gap count mismatch: {row.adjacent_annuity_gap}")
        require(len(adjacent) == int(row.n_adjacent_gaps), "Fee-gap denominator mismatch")

    q4 = np.full(len(train), "MISSING", dtype=object)
    credit = train["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = train["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    valid_q4 = np.isfinite(credit) & np.isfinite(annuity) & (credit != 0)
    q4[valid_q4] = np.rint(annuity[valid_q4] / credit[valid_q4] * 10000).astype(np.int64).astype(str)
    residual_summary = pd.read_csv(run_dir / "PRODUCT_RESIDUAL_HETEROGENEITY.csv")
    for seed in SEEDS:
        pred = oof[f"seed{seed}__product_baseline"].to_numpy(dtype=np.float64)
        expected = np.clip(pred, 1e-8, 1 - 1e-8)
        stats = pd.DataFrame(
            {
                "key": q4,
                "residual": y - expected,
                "target": y.astype(np.float64),
                "variance": expected * (1 - expected),
                "age": age,
                "pensioner": train["NAME_INCOME_TYPE"].eq("Pensioner").to_numpy(dtype=np.float64),
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
        grouped["pearson_residual_z"] = grouped.residual_sum / np.sqrt(grouped.variance_sum)
        residual_path = run_dir / f"Q4_PRODUCT_RESIDUALS_SEED_{seed}.csv"
        saved = pd.read_csv(residual_path).set_index("key")
        require(set(saved.index.astype(str)) == set(grouped.index.astype(str)), f"Residual key set mismatch for seed {seed}")
        grouped.index = grouped.index.astype(str)
        saved.index = saved.index.astype(str)
        for column in ("n", "observed_rate", "pearson_residual_z"):
            require(
                np.allclose(
                    grouped[column].sort_index().to_numpy(dtype=np.float64),
                    saved[column].sort_index().to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                ),
                f"Residual table mismatch: seed {seed}, {column}",
            )
        summary_row = residual_summary.loc[residual_summary.seed == seed]
        require(len(summary_row) == 1, f"Residual summary missing for seed {seed}")
        expected_share = float((grouped["pearson_residual_z"].abs() > 3).mean())
        require(
            abs(expected_share - float(summary_row.iloc[0].share_abs_pearson_z_gt_3)) < 1e-10,
            f"Residual summary mismatch for seed {seed}",
        )

    gpu_info = (run_dir / "GPU_DEVICE_INFO.txt").read_text(encoding="utf-8").strip()
    require("NVIDIA" in gpu_info, "GPU metadata does not identify NVIDIA hardware")
    require("GPU:0" in (run_dir / "EXECUTION_LOG.txt").read_text(encoding="utf-8"), "Execution log does not confirm GPU model configuration")

    recomputed = pd.DataFrame(recomputed_rows)
    recomputed.to_csv(run_dir / "INDEPENDENT_AUC_RECOMPUTATION.csv", index=False)
    runner_source = ROOT / "experiments" / "phase9_product_mechanism_and_catboost" / "run_phase9_product_tests.py"
    report = f"""# Independent Phase 9 result verification

- Status: **PASS**
- Training rows: {len(train):,}; train input SHA-256: `{sha256_file(TRAIN_PATH)}`
- OOF rows and ID/TARGET order: exact match to the training input.
- Prediction columns: {len(expected_prediction_columns)}; all finite.
- Independently recalculated metrics: {len(recomputed):,} pooled seed/model AUCs and {sum(len(frame) for frame in fold_results.values()):,} fold AUCs; all match saved results within `1e-10`.
- Nested target-map integrity: {len(integrity):,} rows; all outer/inner disjointness and no-outer-validation-label checks pass.
- Post-hoc subgroup AUCs/deltas, exact adjacent-annuity placebo counts, and Q4 residual tables were independently recalculated and match.
- CatBoost device metadata: `{gpu_info}`; run configuration is GPU device 0.
- Runner SHA-256: `{sha256_file(runner_source)}`.

These are repeated-seed OOF checks on historically examined training labels, not independent confirmation or a competition-eligibility ruling. LightGBM's installed build lacks GPU tree support; LightGBM therefore remained CPU while the CatBoost comparisons ran on GPU.
"""
    (run_dir / "INDEPENDENT_VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")

    provenance = pd.DataFrame(
        [
            {"path": str(TRAIN_PATH), "sha256": sha256_file(TRAIN_PATH)},
            {"path": str(runner_source), "sha256": sha256_file(runner_source)},
            {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
            {"path": str(run_dir / "OOF_PREDICTIONS.csv"), "sha256": sha256_file(run_dir / "OOF_PREDICTIONS.csv")},
        ]
    )
    provenance.to_csv(run_dir / "INDEPENDENT_AUDIT_PROVENANCE.csv", index=False)

    manifest_rows = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    pd.DataFrame(manifest_rows).to_csv(run_dir / "SHA256_MANIFEST.csv", index=False)
    print(f"verification=PASS; verified_metrics={len(recomputed)}; output={run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    main(parser.parse_args().run_dir)
