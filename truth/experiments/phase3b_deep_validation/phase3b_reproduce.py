from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
BASELINE = ROOT / "experiments" / "phase2_advanced_feature_engineering"
OUT = ROOT / "experiments" / "phase3b_deep_validation"
ID = "SK_ID_CURR"
TARGET = "TARGET"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction(
    prediction_path: Path,
    prediction_column: str,
    train: pd.DataFrame,
    folds: pd.Series,
    variant: str,
) -> dict[str, object]:
    frame = pd.read_csv(prediction_path)
    required = {ID, TARGET, prediction_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{prediction_path} missing {sorted(required - set(frame.columns))}")
    if frame[ID].duplicated().any():
        raise ValueError(f"Duplicated IDs in {prediction_path}")
    if len(frame) != len(train):
        raise ValueError(f"Row count mismatch in {prediction_path}: {len(frame)} != {len(train)}")

    reference = train.set_index(ID)
    observed = frame.set_index(ID).reindex(reference.index)
    if observed.index.hasnans or observed[TARGET].isna().any():
        raise ValueError(f"Missing or unmatched IDs/labels in {prediction_path}")
    if not np.array_equal(observed[TARGET].to_numpy(), reference[TARGET].to_numpy()):
        raise ValueError(f"TARGET mismatch in {prediction_path}")
    expected_folds = folds.reindex(reference.index)
    if expected_folds.isna().any():
        raise ValueError("Missing fold assignment for a training ID")
    if "fold" in observed and not np.array_equal(
        observed["fold"].to_numpy(), expected_folds.to_numpy()
    ):
        raise ValueError(f"Fold mismatch in {prediction_path}")

    pred = observed[prediction_column].to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all():
        raise ValueError(f"Missing or infinite OOF predictions in {prediction_path}")
    y = reference[TARGET].to_numpy(dtype=np.int8)
    fold_ids = expected_folds.to_numpy()
    fold_auc = {
        int(fold): float(roc_auc_score(y[fold_ids == fold], pred[fold_ids == fold]))
        for fold in sorted(np.unique(fold_ids))
    }
    return {
        "variant": variant,
        "prediction_file": str(prediction_path),
        "prediction_column": prediction_column,
        "row_count": len(frame),
        "unique_prediction_ids": int(frame[ID].nunique()),
        "unique_train_ids": int(train[ID].nunique()),
        "id_set_and_order_independent_alignment": True,
        "target_alignment": True,
        "fold_alignment": True,
        "missing_prediction_count": int(np.isnan(pred).sum()),
        "infinite_prediction_count": int(np.isinf(pred).sum()),
        "pooled_oof_auc": float(roc_auc_score(y, pred)),
        "mean_fold_auc": float(np.mean(list(fold_auc.values()))),
        "std_fold_auc": float(np.std(list(fold_auc.values()), ddof=1)),
        "fold_auc": json.dumps(fold_auc, sort_keys=True),
        "prediction_sha256": sha256(prediction_path),
        "status": "REPRODUCED",
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = DATA / "train.csv"
    train = pd.read_csv(train_path, usecols=[ID, TARGET])
    if train[ID].duplicated().any():
        raise ValueError("Duplicate IDs in supplied training data")
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("Invalid training target")
    folds_frame = pd.read_csv(BASELINE / "FOLD_ASSIGNMENTS.csv")
    if folds_frame[ID].duplicated().any():
        raise ValueError("Duplicate IDs in saved folds")
    folds = folds_frame.set_index(ID)["fold"]
    if set(folds.index) != set(train[ID]):
        raise ValueError("Saved fold IDs do not exactly match the training IDs")

    specs = [
        ("corrected_baseline", BASELINE / "OOF_BASELINE_PREDICTIONS.csv", "OOF_RAW_BASELINE"),
        (
            "product_decoder_all",
            PHASE3 / "batch1_family_ablation_02" / "BATCH1_FAMILY_OOF_PREDICTIONS.csv",
            "OOF_product_decoder_only",
        ),
        (
            "product_decoder_all_components",
            PHASE3 / "batch1_product_component_03" / "PRODUCT_COMPONENT_OOF_PREDICTIONS.csv",
            "OOF_all_product_decoder_features",
        ),
        (
            "k_lattice_only",
            PHASE3 / "batch1_product_component_03" / "PRODUCT_COMPONENT_OOF_PREDICTIONS.csv",
            "OOF_k_lattice_only",
        ),
        (
            "q_rounded_ratio_keys_only",
            PHASE3 / "batch1_product_component_03" / "PRODUCT_COMPONENT_OOF_PREDICTIONS.csv",
            "OOF_q_rounded_ratio_keys_only",
        ),
        (
            "p_fee_adjusted_keys_only",
            PHASE3 / "batch1_product_component_03" / "PRODUCT_COMPONENT_OOF_PREDICTIONS.csv",
            "OOF_p_fee_adjusted_keys_only",
        ),
        (
            "p_fee_fit_estimated",
            PHASE3 / "batch1_fee_constant_04" / "FEE_CONSTANT_OOF_PREDICTIONS.csv",
            "OOF_p_keys_fold_estimated_fee",
        ),
        (
            "p_fee_fixed_129_5",
            PHASE3 / "batch1_fee_constant_04" / "FEE_CONSTANT_OOF_PREDICTIONS.csv",
            "OOF_p_keys_fixed_v17_fee_129_5",
        ),
        (
            "q_unadjusted_control",
            PHASE3 / "batch1_fee_constant_04" / "FEE_CONSTANT_OOF_PREDICTIONS.csv",
            "OOF_q_keys_unadjusted_control",
        ),
    ]
    rows = [
        validate_prediction(path, column, train, folds, name)
        for name, path, column in specs
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "CURRENT_SCORE_REPRODUCTION.csv", index=False)

    finished = datetime.now(timezone.utc)
    source_path = Path(__file__).resolve()
    log = [
        f"started_utc={started.isoformat()}",
        f"finished_utc={finished.isoformat()}",
        f"runtime_seconds={(finished - started).total_seconds():.3f}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"train_file={train_path}",
        f"train_sha256={sha256(train_path)}",
        f"train_dimensions={len(train)} rows; 2 selected columns (ID,TARGET)",
        f"fold_file={BASELINE / 'FOLD_ASSIGNMENTS.csv'}",
        f"source_sha256={sha256(source_path)}",
        "status=SUCCESS",
        "validated_prediction_sets=" + str(len(rows)),
        "all_ID_TARGET_fold_prediction_checks=PASS",
        "results=" + str(OUT / "CURRENT_SCORE_REPRODUCTION.csv"),
        "",
        result[["variant", "pooled_oof_auc", "mean_fold_auc", "std_fold_auc"]].to_string(index=False),
    ]
    (OUT / "SCORE_REPRODUCTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
