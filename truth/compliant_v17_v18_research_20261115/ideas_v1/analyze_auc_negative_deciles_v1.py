from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


IDEAS_DIR = Path(__file__).resolve().parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASE_OOF = (
    IDEAS_DIR
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
CATBOOST_OOF = (
    IDEAS_DIR / "runs" / "catboost_pairlogit_auc_v1" / "OOF_PREDICTIONS.csv"
)
XGBOOST_OOF = (
    IDEAS_DIR / "runs" / "xgboost_rank_pairwise_v1" / "OOF_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "auc_negative_decile_decomposition_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLDS = 5
DECILES = 10
BASELINE = "candidate_te20_blend"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_aligned(
    path: Path,
    reference: pd.DataFrame,
    prediction_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required OOF artifact is missing: {path}")
    frame = pd.read_csv(path)
    if (
        len(frame) != len(reference)
        or not frame[ID].equals(reference[ID])
        or not frame[TARGET].equals(reference[TARGET])
        or not frame["fold"].equals(reference["fold"])
    ):
        raise ValueError(f"OOF rows, labels, or folds do not align: {path}")
    missing = set(prediction_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions missing {sorted(missing)} in {path}")
    if not np.isfinite(frame[list(prediction_columns)].to_numpy()).all():
        raise ValueError(f"OOF predictions contain non-finite values: {path}")
    return frame


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT}")
    for path in (TRAIN, BASE_OOF, CATBOOST_OOF, XGBOOST_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    train = pd.read_csv(TRAIN, usecols=[ID, TARGET])
    baseline = pd.read_csv(BASE_OOF)
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Training labels and baseline OOF rows do not align")
    if set(baseline["fold"].unique()) != set(range(1, FOLDS + 1)):
        raise ValueError("Unexpected fold identifiers in the baseline OOF file")

    cb = read_aligned(
        CATBOOST_OOF, baseline, ("candidate_te20_blend", "fixed_90_10_rank_blend")
    )
    xgb = read_aligned(
        XGBOOST_OOF, baseline, ("candidate_te20_blend", "fixed_90_10_rank_blend")
    )
    if not np.array_equal(
        baseline[BASELINE].to_numpy(dtype=np.float64),
        cb[BASELINE].to_numpy(dtype=np.float64),
    ) or not np.array_equal(
        baseline[BASELINE].to_numpy(dtype=np.float64),
        xgb[BASELINE].to_numpy(dtype=np.float64),
    ):
        raise ValueError("Pairwise artifacts do not use the same fixed baseline")

    y = train[TARGET].to_numpy(dtype=np.int8)
    folds = baseline["fold"].to_numpy(dtype=np.int8)
    scores = {
        "catboost_pairlogit_10pct": cb["fixed_90_10_rank_blend"].to_numpy(
            dtype=np.float64
        ),
        "xgboost_rank_pairwise_10pct": xgb["fixed_90_10_rank_blend"].to_numpy(
            dtype=np.float64
        ),
    }
    base_scores = baseline[BASELINE].to_numpy(dtype=np.float64)
    rows: list[dict[str, int | float | str]] = []
    summary: dict[str, dict[str, float | int]] = {}

    for model_name, candidate_scores in scores.items():
        fold_deltas = []
        contribution_by_decile = np.zeros(DECILES, dtype=np.float64)
        for fold in range(1, FOLDS + 1):
            valid = np.flatnonzero(folds == fold)
            valid_y = y[valid]
            positive_rows = valid[valid_y == 1]
            negative_rows = valid[valid_y == 0]
            if not len(positive_rows) or not len(negative_rows):
                raise ValueError(f"Fold {fold} lacks one target class")

            negative_order = np.argsort(
                base_scores[negative_rows], kind="mergesort"
            )
            negative_deciles = np.array_split(negative_rows[negative_order], DECILES)
            baseline_auc = float(
                roc_auc_score(valid_y, base_scores[valid])
            )
            candidate_auc = float(
                roc_auc_score(valid_y, candidate_scores[valid])
            )
            fold_delta = candidate_auc - baseline_auc
            fold_deltas.append(fold_delta)
            contribution_sum = 0.0
            for decile, decile_negatives in enumerate(negative_deciles, start=1):
                subset = np.concatenate((positive_rows, decile_negatives))
                subset_y = y[subset]
                base_decile_auc = float(
                    roc_auc_score(subset_y, base_scores[subset])
                )
                candidate_decile_auc = float(
                    roc_auc_score(subset_y, candidate_scores[subset])
                )
                weight = len(decile_negatives) / len(negative_rows)
                contribution = weight * (
                    candidate_decile_auc - base_decile_auc
                )
                contribution_by_decile[decile - 1] += contribution / FOLDS
                contribution_sum += contribution
                rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "negative_baseline_score_decile": decile,
                        "negative_rows": int(len(decile_negatives)),
                        "positive_rows_compared": int(len(positive_rows)),
                        "pair_count": int(len(decile_negatives) * len(positive_rows)),
                        "baseline_decile_auc": base_decile_auc,
                        "candidate_decile_auc": candidate_decile_auc,
                        "weighted_auc_delta_contribution": contribution,
                    }
                )
            if not np.isclose(contribution_sum, fold_delta, atol=1e-12, rtol=0):
                raise RuntimeError(
                    f"Fold {fold} decile contributions do not sum to its AUC delta"
                )

        mean_delta = float(np.mean(fold_deltas))
        if not np.isclose(
            contribution_by_decile.sum(), mean_delta, atol=1e-12, rtol=0
        ):
            raise RuntimeError(f"{model_name} mean contributions do not sum")
        summary[model_name] = {
            "mean_fold_auc_delta": mean_delta,
            "positive_fold_deltas": int(sum(delta > 0 for delta in fold_deltas)),
            "sum_mean_decile_contribution": float(contribution_by_decile.sum()),
        }
        for decile, contribution in enumerate(contribution_by_decile, start=1):
            rows.append(
                {
                    "model": model_name,
                    "fold": 0,
                    "negative_baseline_score_decile": decile,
                    "negative_rows": 0,
                    "positive_rows_compared": 0,
                    "pair_count": 0,
                    "baseline_decile_auc": np.nan,
                    "candidate_decile_auc": np.nan,
                    "weighted_auc_delta_contribution": float(contribution),
                }
            )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    contributions_path = OUTPUT / "DECILE_CONTRIBUTIONS.csv"
    pd.DataFrame(rows).to_csv(contributions_path, index=False)
    manifest = {
        "status": "descriptive OOF decomposition; no fitting or weight selection",
        "decomposition": "AUC delta = sum over baseline-score negative deciles of (negative-decile pair share * conditional AUC delta)",
        "decile_definition": "Within each validation fold, split negatives into ten near-equal groups by ascending baseline candidate score; positives are compared against every negative in each group.",
        "folds": FOLDS,
        "models": summary,
        "rule_controls": {
            "training_data": "official train.csv labels and previously generated OOF predictions only",
            "test_rows": "not read",
            "model_fitting": "none",
            "prediction_changes": "none",
        },
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                TRAIN,
                BASE_OOF,
                CATBOOST_OOF,
                XGBOOST_OOF,
                Path(__file__).resolve(),
                contributions_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
