from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
RUN = (
    ROOT
    / "experiments"
    / "phase8_nested_product_target_encoding"
    / "runs"
    / "phase8_20261104T080000Z"
)
TRAIN = Path(r"D:\gci\inpuy\train.csv")
AUDIT_SOURCE = Path(__file__).resolve()
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20261104


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc_delta_bootstrap(
    labels: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sample = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(
            labels[sample], candidate[sample]
        ) - roc_auc_score(labels[sample], reference[sample])
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "delta_mean": float(deltas.mean()),
        "delta_ci_2_5": float(np.quantile(deltas, 0.025)),
        "delta_ci_97_5": float(np.quantile(deltas, 0.975)),
        "bootstrap_fraction_delta_positive": float(np.mean(deltas > 0)),
    }


def load_phase8():
    source = (
        ROOT
        / "experiments"
        / "phase8_nested_product_target_encoding"
        / "run_phase8_nested_product_te.py"
    )
    spec = importlib.util.spec_from_file_location("phase8_audit_import", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Phase 8 source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not RUN.is_dir():
        raise FileNotFoundError(RUN)
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    saved = pd.read_csv(RUN / "OOF_PREDICTIONS.csv")
    metrics = pd.read_csv(RUN / "SEED_METRICS.csv")
    aggregate = pd.read_csv(RUN / "AGGREGATE_COMPARISON.csv")
    integrity = pd.read_csv(RUN / "TARGET_ENCODING_INTEGRITY.csv")

    if saved["SK_ID_CURR"].duplicated().any() or len(saved) != len(train):
        raise AssertionError("OOF ID uniqueness/count mismatch")
    aligned = train.set_index("SK_ID_CURR").reindex(saved["SK_ID_CURR"])
    if aligned["TARGET"].isna().any() or not np.array_equal(
        aligned["TARGET"].to_numpy(dtype=np.int8),
        saved["TARGET"].to_numpy(dtype=np.int8),
    ):
        raise AssertionError("OOF TARGET is not aligned to train IDs")
    if integrity["outer_fit_validation_disjoint"].ne(True).any():
        raise AssertionError("Outer target-map disjointness metadata failed")
    if integrity["inner_map_apply_disjoint"].ne(True).any():
        raise AssertionError("Inner target-map disjointness metadata failed")
    if integrity["outer_validation_labels_used_for_encoding"].ne(False).any():
        raise AssertionError("Outer validation labels were reported in target maps")

    seeds = (42, 7, 2026)
    variants = [
        "raw_baseline",
        "raw_plus_phase3_product",
        "product_plus_te_q4",
        "product_plus_te8",
        "product_plus_te8_and_counts",
        "product_plus_te10",
    ]
    labels = saved["TARGET"].to_numpy(dtype=np.int8)
    seed_avg_rows = []
    recomputed_rows = []
    averaged_predictions = {}
    for variant in variants:
        per_seed = []
        for seed in seeds:
            column = f"seed{seed}__{variant}"
            prediction = saved[column].to_numpy(dtype=np.float64)
            if not np.isfinite(prediction).all():
                raise AssertionError(f"Nonfinite OOF values in {column}")
            score = float(roc_auc_score(labels, prediction))
            stored = metrics.loc[
                (metrics.seed == seed) & (metrics.variant == variant),
                "pooled_oof_auc",
            ]
            if len(stored) != 1 or abs(score - float(stored.iloc[0])) > 1e-12:
                raise AssertionError(f"Stored AUC mismatch for {column}")
            per_seed.append(prediction)
            recomputed_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "recomputed_pooled_oof_auc": score,
                    "stored_pooled_oof_auc": float(stored.iloc[0]),
                    "absolute_auc_difference": abs(score - float(stored.iloc[0])),
                }
            )
        averaged_predictions[variant] = np.mean(np.stack(per_seed), axis=0)
        seed_avg_rows.append(
            {
                "variant": variant,
                "seed_average_oof_auc": float(
                    roc_auc_score(labels, averaged_predictions[variant])
                ),
                "mean_individual_seed_auc": float(
                    metrics.loc[metrics.variant == variant, "pooled_oof_auc"].mean()
                ),
                "seed_average_delta_vs_phase3_product": np.nan,
            }
        )

    phase3_auc = float(
        roc_auc_score(labels, averaged_predictions["raw_plus_phase3_product"])
    )
    for row in seed_avg_rows:
        row["seed_average_delta_vs_phase3_product"] = (
            row["seed_average_oof_auc"] - phase3_auc
        )
    seed_average = pd.DataFrame(seed_avg_rows)
    seed_average.to_csv(RUN / "SEED_AVERAGED_OOF_DIAGNOSTIC.csv", index=False)
    pd.DataFrame(recomputed_rows).to_csv(
        RUN / "INDEPENDENT_AUC_RECOMPUTATION.csv", index=False
    )

    reference = averaged_predictions["raw_plus_phase3_product"]
    candidate = averaged_predictions["product_plus_te8"]
    observed_delta = float(
        roc_auc_score(labels, candidate) - roc_auc_score(labels, reference)
    )
    boot = auc_delta_bootstrap(labels, reference, candidate)
    pd.DataFrame([{**boot, "observed_seed_averaged_delta": observed_delta}]).to_csv(
        RUN / "SEED_AVERAGED_PAIRED_BOOTSTRAP.csv", index=False
    )

    phase8 = load_phase8()
    toy_keys = np.array(["a", "a", "b", "b", "a", "c"], dtype=str)
    toy_y = np.array([0, 1, 1, 0, 1, 0], dtype=np.int8)
    fit_idx = np.array([0, 1, 2, 3], dtype=np.int64)
    apply_idx = np.array([4, 5], dtype=np.int64)
    original, original_counts = phase8.encode_keys(
        toy_keys, toy_y, fit_idx, apply_idx
    )
    changed_y = toy_y.copy()
    changed_y[apply_idx] = 1 - changed_y[apply_idx]
    changed, changed_counts = phase8.encode_keys(
        toy_keys, changed_y, fit_idx, apply_idx
    )
    if not np.array_equal(original, changed) or not np.array_equal(
        original_counts, changed_counts
    ):
        raise AssertionError("Application labels affected their own target encodings")
    try:
        phase8.encode_keys(toy_keys, toy_y, np.array([0, 1]), np.array([1, 2]))
    except ValueError:
        overlap_rejected = True
    else:
        raise AssertionError("Target encoder accepted overlapping fit/apply indices")

    source_paths = [
        AUDIT_SOURCE,
        RUN / "OOF_PREDICTIONS.csv",
        RUN / "SEED_METRICS.csv",
        RUN / "AGGREGATE_COMPARISON.csv",
        RUN / "TARGET_ENCODING_INTEGRITY.csv",
        TRAIN,
    ]
    source_table = pd.DataFrame(
        [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ]
    )
    source_table.to_csv(RUN / "INDEPENDENT_AUDIT_PROVENANCE.csv", index=False)

    primary = seed_average.loc[
        seed_average.variant == "product_plus_te8"
    ].iloc[0]
    count_variant = seed_average.loc[
        seed_average.variant == "product_plus_te8_and_counts"
    ].iloc[0]
    ten_variant = seed_average.loc[
        seed_average.variant == "product_plus_te10"
    ].iloc[0]
    report = f"""# Independent verification and seed-averaged OOF diagnostic

## Verification

- Train rows were loaded by ID and TARGET only; test data were not opened.
- All `{len(saved):,}` OOF IDs are unique and TARGET-aligned to the train file.
- Recomputed all `{len(recomputed_rows)}` seed/variant pooled OOF AUCs from saved predictions; each matches `SEED_METRICS.csv` within 1e-12.
- All outer fit/validation maps and inner map/application folds are recorded disjoint; no outer validation TARGET is reported as used for encoding.
- Synthetic perturbation test: changing only application-row labels leaves their encoding and count unchanged: `True`.
- Overlapping target-map fit/application indices are rejected: `{overlap_rejected}`.
- Source/data hashes are recorded in `INDEPENDENT_AUDIT_PROVENANCE.csv`.

## Fixed seed-average diagnostic

Each applicant's three predictions are averaged across the three fold seeds. For each of those three models, the applicant is held out, so this is a valid cross-fit average prediction diagnostic; it is **not** a full-data deployed-model score.

- Raw + Phase 3 product seed-averaged OOF AUC: `{phase3_auc:.9f}`.
- 8-key nested target encoding seed-averaged OOF AUC: `{primary.seed_average_oof_auc:.9f}` (delta `{primary.seed_average_delta_vs_phase3_product:+.9f}`).
- 8-key plus counts: `{count_variant.seed_average_oof_auc:.9f}` (delta `{count_variant.seed_average_delta_vs_phase3_product:+.9f}`).
- 10-key: `{ten_variant.seed_average_oof_auc:.9f}` (delta `{ten_variant.seed_average_delta_vs_phase3_product:+.9f}`).
- Descriptive paired stratified row-bootstrap 95% interval for the 8-key seed-averaged delta: `[{boot['delta_ci_2_5']:+.9f}, {boot['delta_ci_97_5']:+.9f}]`, over {boot['replicates']} resamples. Labels are historically exposed; this interval is not independent confirmation and does not adjust for feature/key search.

The seed-average improves over the mean of per-seed pooled AUC when its rank aggregation helps, but seed repeats use the same applicants and are correlated. It should not be represented as three independent datasets.
"""
    (RUN / "INDEPENDENT_VERIFICATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )

    manifest_rows = []
    for path in sorted(RUN.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(RUN / "SHA256_MANIFEST.csv", index=False)
    print(json.dumps(
        {
            "status": "VERIFIED",
            "oof_rows": len(saved),
            "auc_recomputations": len(recomputed_rows),
            "seed_averaged_auc": seed_average.to_dict(orient="records"),
            "observed_seed_average_delta": observed_delta,
            "paired_bootstrap": boot,
            "run_manifest_files": len(manifest_rows),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
