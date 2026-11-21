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
TRAIN = Path(r"D:\gci\inpuy\train.csv")
RUNS = IDEAS_DIR / "runs"
OUTPUT = RUNS / "complementary_expert_blend_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHTS = {
    "baseline": 0.90,
    "group_conditional_rank_expert": 0.025,
    "covariate_residual_expert": 0.025,
    "crossfit_model_view_expert": 0.025,
    "fold_local_category_te_expert": 0.025,
}
EXPERIMENTS = {
    "group_conditional_rank_expert": "group_conditional_rank_v1",
    "covariate_residual_expert": "covariate_residual_expert_v1",
    "crossfit_model_view_expert": "crossfit_model_views_v1",
    "fold_local_category_te_expert": "fold_local_category_te_v1",
}
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
    if not TRAIN.is_file():
        raise FileNotFoundError(f"Official training file is missing: {TRAIN}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, usecols=[ID, TARGET])
    model_frames: dict[str, pd.DataFrame] = {}
    input_paths = [TRAIN]
    for expert, run_name in EXPERIMENTS.items():
        path = RUNS / run_name / "OOF_PREDICTIONS.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required expert OOF file is missing: {path}")
        frame = pd.read_csv(path)
        input_paths.append(path)
        if (
            len(frame) != len(train)
            or not train[ID].equals(frame[ID])
            or not train[TARGET].equals(frame[TARGET])
            or not train[ID].is_unique
            or "fold" not in frame
            or BASELINE_COLUMN not in frame
            or expert not in frame
        ):
            raise ValueError(f"Expert OOF alignment/schema failed: {path}")
        model_frames[expert] = frame

    reference = next(iter(model_frames.values()))
    for expert, frame in model_frames.items():
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
    decile_records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        rows = np.flatnonzero(folds == fold)
        y_fold = y[rows]
        baseline_rank = fold_rank(baseline[rows])
        combined = BLEND_WEIGHTS["baseline"] * baseline_rank
        expert_ranks: dict[str, np.ndarray] = {}
        for expert, frame in model_frames.items():
            predictions = frame[expert].to_numpy(dtype=np.float64)[rows]
            expert_ranks[expert] = fold_rank(predictions)
            combined += BLEND_WEIGHTS[expert] * expert_ranks[expert]
        if not np.isfinite(combined).all():
            raise ValueError(f"Fold {fold}: combined ranks are non-finite")
        combined_oof[rows] = combined.astype(np.float32)
        baseline_auc = float(roc_auc_score(y_fold, baseline[rows]))
        combined_auc = float(roc_auc_score(y_fold, combined))
        record: dict[str, Any] = {
            "fold": int(fold),
            "baseline_auc": baseline_auc,
            "fixed_four_expert_blend_auc": combined_auc,
            "blend_delta": combined_auc - baseline_auc,
            "valid_rows": int(len(rows)),
        }
        for expert, frame in model_frames.items():
            record[f"{expert}_auc"] = float(
                roc_auc_score(y_fold, frame[expert].to_numpy(dtype=np.float64)[rows])
            )
        records.append(record)

        negative_rows = np.flatnonzero(y_fold == 0)
        negative_order = negative_rows[
            np.argsort(baseline[rows][negative_rows], kind="stable")
        ]
        for decile, negative_bin in enumerate(
            np.array_split(negative_order, 10), start=1
        ):
            pair_rows = np.concatenate((np.flatnonzero(y_fold == 1), negative_bin))
            base_auc_decile = roc_auc_score(y_fold[pair_rows], baseline[rows][pair_rows])
            blend_auc_decile = roc_auc_score(
                y_fold[pair_rows], combined[pair_rows]
            )
            decile_records.append(
                {
                    "fold": int(fold),
                    "baseline_negative_decile": decile,
                    "negative_count": int(len(negative_bin)),
                    "delta_auc": float(blend_auc_decile - base_auc_decile),
                    "pair_weighted_delta": float(
                        (blend_auc_decile - base_auc_decile)
                        * len(negative_bin)
                        / len(negative_rows)
                    ),
                }
            )
        print(
            f"fold={fold} baseline={baseline_auc:.6f} "
            f"fixed_four_expert_blend={combined_auc:.6f} "
            f"gain={combined_auc - baseline_auc:+.6f}",
            flush=True,
        )

    if not np.isfinite(combined_oof).all():
        raise ValueError("Combined OOF vector is incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    decile_path = OUTPUT / "NEGATIVE_DECILE_DECOMPOSITION.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": folds,
            BASELINE_COLUMN: baseline,
            "fixed_four_expert_rank_blend": combined_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)
    pd.DataFrame(decile_records).to_csv(decile_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_four_expert_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    weighted_deciles = (
        pd.DataFrame(decile_records)
        .groupby("baseline_negative_decile")["pair_weighted_delta"]
        .mean()
        .to_dict()
    )
    summary = {
        "status": "fixed post-hoc ensemble of fold-isolated OOF experts; exploratory",
        "formula": (
            "0.90*rank(TE20) + 0.025*rank(group-conditional) + "
            "0.025*rank(covariate-residual) + "
            "0.025*rank(cross-fitted-model-view) + "
            "0.025*rank(fold-local-category-TE), all ranks within outer fold"
        ),
        "fixed_weights": BLEND_WEIGHTS,
        "outer_folds": int(len(records)),
        "baseline_mean_fold_auc": baseline_mean,
        "four_expert_blend_mean_fold_auc": blend_mean,
        "four_expert_blend_mean_fold_gain": gain,
        "positive_folds": int(sum(row["blend_delta"] > 0 for row in records)),
        "minimum_promotion_gain": PROMOTION_DELTA,
        "promotion_gate_passed": bool(
            gain >= PROMOTION_DELTA
            and all(row["blend_delta"] > 0 for row in records)
        ),
        "fold_metrics": records,
        "mean_pair_weighted_contribution_by_negative_decile": {
            str(k): float(v) for k, v in weighted_deciles.items()
        },
        "input_oof_files": {
            expert: str(RUNS / run_name / "OOF_PREDICTIONS.csv")
            for expert, run_name in EXPERIMENTS.items()
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "training_data": (
                "official train.csv used only to verify row IDs/labels and calculate OOF metrics"
            ),
            "expert_outputs": (
                "pre-existing outer-fold OOF predictions; no test prediction or "
                "training-row label transfer"
            ),
            "weights": "fixed equal 2.5% rank share per expert; not optimized",
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
            for path in input_paths + [Path(__file__).resolve(), oof_path, fold_path, decile_path]
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
