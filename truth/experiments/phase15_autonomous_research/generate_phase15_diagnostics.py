from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE9 = (
    REPO
    / "experiments"
    / "phase9_product_mechanism_and_catboost"
    / "runs"
    / "phase9_20261104T181500LOCAL"
    / "OOF_PREDICTIONS.csv"
)
PHASE13 = (
    REPO
    / "experiments"
    / "phase13_matched_baseline"
    / "runs"
    / "phase13_20261105T075946Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE15_PRIMARY = (
    ROOT / "runs" / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE15_MULTISEED = (
    ROOT / "runs" / "phase15_native_catboost_multiseed_20261105T125826Z"
    / "MULTISEED_OOF_PREDICTIONS.csv"
)
PHASE2_ARRAY = (
    REPO
    / "experiments"
    / "exp85_phase2_inner_early_stopping"
    / "runs"
    / "phase2_inner_stop_seed42"
    / "oof_phase2_blend.npy"
)
BOOTSTRAP_REPETITIONS = 300
BOOTSTRAP_SEED = 20263114


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    ids = train.SK_ID_CURR.to_numpy()
    y = train.TARGET.to_numpy(dtype=np.int8)
    phase9 = pd.read_csv(PHASE9)
    phase13 = pd.read_csv(PHASE13)
    primary = pd.read_csv(PHASE15_PRIMARY)
    multiseed = pd.read_csv(PHASE15_MULTISEED)
    if not (
        np.array_equal(phase9.SK_ID_CURR.to_numpy(), ids)
        and np.array_equal(phase13.SK_ID_CURR.to_numpy(), ids)
        and np.array_equal(primary.SK_ID_CURR.to_numpy(), ids)
        and np.array_equal(multiseed.SK_ID_CURR.to_numpy(), ids)
        and np.array_equal(phase9.TARGET.to_numpy(), y)
        and np.array_equal(phase13.TARGET.to_numpy(), y)
        and np.array_equal(primary.TARGET.to_numpy(), y)
        and np.array_equal(multiseed.TARGET.to_numpy(), y)
    ):
        raise ValueError("Historical OOF vectors do not align with train IDs/targets")
    p9 = phase9[
        [
            "seed42__cat_native_8_keys",
            "seed7__cat_native_8_keys",
            "seed2026__cat_native_8_keys",
        ]
    ].mean(axis=1).to_numpy(dtype=np.float64)
    p2 = np.load(PHASE2_ARRAY, allow_pickle=False).astype(np.float64)
    p13 = phase13["E_C_plus_keys_frequency_ohe"].to_numpy(dtype=np.float64)
    p15_primary = primary[
        "native_catboost_phase14_plus_time"
    ].to_numpy(dtype=np.float64)
    p15 = multiseed["phase15_three_seed_average"].to_numpy(dtype=np.float64)
    model_predictions = {
        "phase2_nested_research": p2,
        "phase9_three_seed_oof_mean": p9,
        "phase13_E_frequency_ohe": p13,
        "phase15_one_seed_time": p15_primary,
        "phase15_three_seed_average": p15,
    }
    if any(prediction.shape != y.shape for prediction in model_predictions.values()):
        raise ValueError("A historical OOF vector does not match the training row count")
    if any(not np.isfinite(prediction).all() for prediction in model_predictions.values()):
        raise ValueError("A historical OOF vector contains non-finite predictions")

    blend_rows = []
    direct_auc = {
        name: float(roc_auc_score(y, pred))
        for name, pred in model_predictions.items()
    }
    for name, auc in direct_auc.items():
        blend_rows.append(
            {
                "combination": name,
                "left_weight": 1.0,
                "right_weight": 0.0,
                "pooled_auc": auc,
                "status": "saved OOF vector; prior label exposure",
            }
        )
    for left, right in (
        ("phase15_three_seed_average", "phase13_E_frequency_ohe"),
        ("phase15_three_seed_average", "phase9_three_seed_oof_mean"),
        ("phase15_three_seed_average", "phase2_nested_research"),
        ("phase2_nested_research", "phase9_three_seed_oof_mean"),
    ):
        a = model_predictions[left]
        b = model_predictions[right]
        for right_weight in (0.25, 0.5, 0.75):
            prediction = (1.0 - right_weight) * a + right_weight * b
            blend_rows.append(
                {
                    "combination": f"{left} + {right}",
                    "left_weight": 1.0 - right_weight,
                    "right_weight": right_weight,
                    "pooled_auc": float(roc_auc_score(y, prediction)),
                    "status": (
                        "post-hoc OOF diagnostic; blend weight not nested or "
                        "independent; Phase 2/9 include unresolved Track B methods"
                    ),
                }
            )
    blend_rows.append(
        {
            "combination": "equal_phase15_phase9_phase2",
            "left_weight": 1 / 3,
            "right_weight": 1 / 3,
            "pooled_auc": float(roc_auc_score(y, (p15 + p9 + p2) / 3.0)),
            "status": "post-hoc three-model OOF diagnostic; not validation",
        }
    )
    pd.DataFrame(blend_rows).to_csv(
        ROOT / "PHASE15_FIXED_BLEND_DIAGNOSTICS.csv", index=False
    )

    correlation_rows = []
    names = list(model_predictions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            correlation_rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(
                        pd.Series(model_predictions[left]).corr(
                            pd.Series(model_predictions[right]), method="spearman"
                        )
                    ),
                }
            )
    pd.DataFrame(correlation_rows).to_csv(
        ROOT / "PHASE15_OOF_CORRELATIONS.csv", index=False
    )

    numeric = primary[
        "native_catboost_phase14_numeric"
    ].to_numpy(dtype=np.float64)
    time = primary[
        "native_catboost_phase14_plus_time"
    ].to_numpy(dtype=np.float64)
    fold_ids = primary["fold"].to_numpy(dtype=np.int8)
    fold_rows = []
    for fold in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold
        fold_rows.append(
            {
                "fold": int(fold),
                "numeric_auc": float(roc_auc_score(y[mask], numeric[mask])),
                "time_auc": float(roc_auc_score(y[mask], time[mask])),
                "time_delta": float(
                    roc_auc_score(y[mask], time[mask])
                    - roc_auc_score(y[mask], numeric[mask])
                ),
            }
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        sampled = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        deltas[index] = roc_auc_score(y[sampled], time[sampled]) - roc_auc_score(
            y[sampled], numeric[sampled]
        )
    time_report = {
        "comparison": "Phase 14 numeric enhancement + time vs numeric enhancement alone",
        "pooled_auc_delta": direct_auc["phase15_one_seed_time"]
        - float(
            roc_auc_score(
                y, primary["native_catboost_phase14_numeric"].to_numpy()
            )
        ),
        "positive_folds": int(sum(row["time_delta"] > 0 for row in fold_rows)),
        "fold_deltas": fold_rows,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_delta_ci95_low": float(np.quantile(deltas, 0.025)),
        "paired_delta_ci95_high": float(np.quantile(deltas, 0.975)),
        "scope": "Conditional on fitted OOF predictions; labels previously exposed.",
    }
    (ROOT / "PHASE15_TIME_FEATURE_PAIRED_BOOTSTRAP.json").write_text(
        json.dumps(time_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    folds = phase13["primary_fold"].to_numpy(dtype=np.int8)
    ids_auc = float(roc_auc_score(y, ids))
    id_fold_auc = [
        {
            "fold": int(fold),
            "rows": int(np.sum(folds == fold)),
            "id_auc": float(roc_auc_score(y[folds == fold], ids[folds == fold])),
        }
        for fold in sorted(np.unique(folds))
    ]
    pd.DataFrame(id_fold_auc).to_csv(
        ROOT / "PHASE15_ID_ORDER_DIAGNOSTICS.csv", index=False
    )

    input_hashes = {
        str(path): sha256_file(path)
        for path in (TRAIN, PHASE9, PHASE13, PHASE15_PRIMARY, PHASE15_MULTISEED, PHASE2_ARRAY)
    }
    report = {
        "training_rows": len(y),
        "ids_equal_source_row_index": bool(
            np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype))
        ),
        "id_only_pooled_auc": ids_auc,
        "id_fold_auc": id_fold_auc,
        "fixed_blend_rows": len(blend_rows),
        "prediction_spearman_correlations": correlation_rows,
        "time_feature_paired_bootstrap": time_report,
        "direct_oof_auc": direct_auc,
        "inputs_sha256": input_hashes,
        "method_warning": (
            "Post-hoc blends and all same-label intervals are descriptive, "
            "not independent validation."
        ),
    }
    (ROOT / "PHASE15_POSTHOC_DIAGNOSTICS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
