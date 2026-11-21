from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
IDEAS_V1 = ROOT / "ideas_v1"
TRAIN = Path(r"D:\gci\inpuy\train.csv")
RUNS_V2 = IDEAS_DIR / "runs"
OUTPUT = RUNS_V2 / "five_expert_blend_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
BASELINE_COLUMN = "candidate_te20_blend"
PAIRWISE_PATH = IDEAS_V1 / "runs" / "xgboost_rank_pairwise_v1" / "OOF_PREDICTIONS.csv"
PAIRWISE_COLUMN = "oof_xgboost_rank_pairwise"
EXPERTS = {
    "group_conditional_rank_expert": RUNS_V2
    / "group_conditional_rank_v1"
    / "OOF_PREDICTIONS.csv",
    "covariate_residual_expert": RUNS_V2
    / "covariate_residual_expert_v1"
    / "OOF_PREDICTIONS.csv",
    "crossfit_model_view_expert": RUNS_V2
    / "crossfit_model_views_v1"
    / "OOF_PREDICTIONS.csv",
    "fold_local_category_te_expert": RUNS_V2
    / "fold_local_category_te_v1"
    / "OOF_PREDICTIONS.csv",
    "xgboost_rank_pairwise_expert": PAIRWISE_PATH,
}
BASELINE_WEIGHT = 0.90
EXPERT_WEIGHT = 0.02
PROMOTION_DELTA = 0.005


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    started = time.perf_counter()
    input_paths = [TRAIN]
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Official training file is missing: {TRAIN}")
    train = pd.read_csv(TRAIN, usecols=[ID, TARGET])
    frames: dict[str, pd.DataFrame] = {}
    for expert, path in EXPERTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required expert OOF file is missing: {path}")
        frame = pd.read_csv(path)
        input_paths.append(path)
        score_column = PAIRWISE_COLUMN if expert == "xgboost_rank_pairwise_expert" else expert
        if (
            len(frame) != len(train)
            or not train[ID].equals(frame[ID])
            or not train[TARGET].equals(frame[TARGET])
            or not train[ID].is_unique
            or "fold" not in frame
            or BASELINE_COLUMN not in frame
            or score_column not in frame
        ):
            raise ValueError(f"Expert OOF alignment/schema failed: {path}")
        frames[expert] = frame

    reference = next(iter(frames.values()))
    for expert, frame in frames.items():
        if not frame["fold"].equals(reference["fold"]):
            raise ValueError(f"Expert fold assignment differs: {expert}")
        if not np.array_equal(
            frame[BASELINE_COLUMN].to_numpy(),
            reference[BASELINE_COLUMN].to_numpy(),
        ):
            raise ValueError(f"Benchmark prediction differs: {expert}")

    y = train[TARGET].to_numpy(dtype=np.int8)
    folds = reference["fold"].to_numpy(dtype=np.int8)
    baseline = reference[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("Benchmark predictions contain non-finite values")

    combined_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        rows = np.flatnonzero(folds == fold)
        y_fold = y[rows]
        combined = BASELINE_WEIGHT * fold_rank(baseline[rows])
        for expert, frame in frames.items():
            column = (
                PAIRWISE_COLUMN
                if expert == "xgboost_rank_pairwise_expert"
                else expert
            )
            scores = frame[column].to_numpy(dtype=np.float64)[rows]
            combined += EXPERT_WEIGHT * fold_rank(scores)
        if not np.isfinite(combined).all():
            raise ValueError(f"Fold {fold}: combined ranks are non-finite")
        combined_oof[rows] = combined.astype(np.float32)
        baseline_auc = float(roc_auc_score(y_fold, baseline[rows]))
        blend_auc = float(roc_auc_score(y_fold, combined))
        record: dict[str, Any] = {
            "fold": int(fold),
            "baseline_auc": baseline_auc,
            "fixed_five_expert_blend_auc": blend_auc,
            "blend_delta": blend_auc - baseline_auc,
            "valid_rows": int(len(rows)),
        }
        for expert, frame in frames.items():
            column = (
                PAIRWISE_COLUMN
                if expert == "xgboost_rank_pairwise_expert"
                else expert
            )
            record[f"{expert}_auc"] = float(
                roc_auc_score(y_fold, frame[column].to_numpy(dtype=np.float64)[rows])
            )
        records.append(record)
        print(
            f"fold={fold} baseline={baseline_auc:.6f} "
            f"five_expert_blend={blend_auc:.6f} "
            f"gain={blend_auc - baseline_auc:+.6f}",
            flush=True,
        )

    if not np.isfinite(combined_oof).all():
        raise ValueError("Combined OOF vector is incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": folds,
            BASELINE_COLUMN: baseline,
            "fixed_five_expert_rank_blend": combined_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_five_expert_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fixed post-hoc ensemble of five fold-isolated OOF experts; exploratory",
        "formula": (
            "0.90*rank(TE20) + 0.02*sum(rank(group-conditional, "
            "covariate-residual, model-view, category-TE, XGBoost-pairwise)); "
            "all ranks within outer fold"
        ),
        "fixed_weights": {
            "baseline": BASELINE_WEIGHT,
            "each_expert": EXPERT_WEIGHT,
            "expert_count": len(EXPERTS),
        },
        "outer_folds": len(records),
        "baseline_mean_fold_auc": baseline_mean,
        "five_expert_blend_mean_fold_auc": blend_mean,
        "five_expert_blend_mean_fold_gain": gain,
        "positive_folds": int(sum(row["blend_delta"] > 0 for row in records)),
        "minimum_promotion_gain": PROMOTION_DELTA,
        "promotion_gate_passed": bool(
            gain >= PROMOTION_DELTA
            and all(row["blend_delta"] > 0 for row in records)
        ),
        "fold_metrics": records,
        "input_oof_files": {expert: str(path) for expert, path in EXPERTS.items()},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "official_train": (
                "read only to verify ID/label alignment and calculate OOF metrics"
            ),
            "expert_outputs": (
                "pre-existing outer-fold OOF predictions; no test prediction or "
                "training-row label transfer"
            ),
            "weights": "fixed equal 2% rank share per expert; not optimized",
            "external_data": "none",
            "manual_row_edits": "none",
            "test_predictions_or_submission": "none",
            "validation_caveat": (
                "post-hoc combination on repeatedly examined OOF folds; "
                "not an independent generalization estimate"
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in input_paths + [Path(__file__).resolve(), oof_path, fold_path]
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
