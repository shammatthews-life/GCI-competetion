from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BASELINE = RUNS / "fold_aligned_catboost_blend_v1" / "OOF_PREDICTIONS.csv"
VIEWS = RUNS / "compliant_multiview_catboost_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = RUNS / "multiview_blend_screen_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_by_fold(values: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        result[mask] = (rankdata(values[mask], method="average") - 1) / (
            mask.sum() - 1
        )
    if not np.isfinite(result).all():
        raise ValueError("Ranked OOF vector contains non-finite values")
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT}")
    if not BASELINE.is_file() or not VIEWS.is_file():
        raise FileNotFoundError("Required aligned OOF files are missing")
    base = pd.read_csv(BASELINE)
    views = pd.read_csv(VIEWS)
    if (
        len(base) != len(views)
        or not base[ID].equals(views[ID])
        or not base[TARGET].equals(views[TARGET])
        or not base["fold"].equals(views["fold"])
    ):
        raise ValueError("Multi-view OOF rows, IDs, labels, or folds do not align")
    y = base[TARGET].to_numpy(dtype=np.int8)
    folds = base["fold"].to_numpy(dtype=np.int8)
    base_rank = rank_by_fold(
        base["fixed_50_50_within_fold_rank_blend"].to_numpy(dtype=np.float64),
        folds,
    )
    component_ranks = {
        name: rank_by_fold(views[f"oof_{name}"].to_numpy(dtype=np.float64), folds)
        for name in ("external", "loan", "person")
    }
    view_average = sum(component_ranks.values()) / 3.0
    candidates = {
        "previous_aligned_blend": base_rank,
        "prior_plus_three_views_10pct_total": 0.90 * base_rank + 0.10 * view_average,
        "prior_plus_three_views_20pct_total": 0.80 * base_rank + 0.20 * view_average,
        "prior_plus_three_views_30pct_total": 0.70 * base_rank + 0.30 * view_average,
    }
    fold_rows = []
    results = []
    baseline_aucs = {
        fold: float(roc_auc_score(y[folds == fold], base_rank[folds == fold]))
        for fold in range(1, FOLDS + 1)
    }
    for name, prediction in candidates.items():
        fold_aucs = []
        for fold in range(1, FOLDS + 1):
            mask = folds == fold
            auc = float(roc_auc_score(y[mask], prediction[mask]))
            fold_aucs.append(auc)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "auc": auc,
                    "delta_vs_previous": auc - baseline_aucs[fold],
                }
            )
        results.append(
            {
                "candidate": name,
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "delta_vs_previous": float(
                    np.mean(fold_aucs) - np.mean(list(baseline_aucs.values()))
                ),
                "positive_folds": int(
                    sum(
                        auc > baseline_aucs[fold]
                        for fold, auc in enumerate(fold_aucs, start=1)
                    )
                ),
                "fold_auc": fold_aucs,
            }
        )
    correlations = {
        name: float(spearmanr(base_rank, prediction).statistic)
        for name, prediction in component_ranks.items()
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    results_path = OUTPUT / "CANDIDATE_METRICS.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(results).drop(columns=["fold_auc"]).to_csv(
        results_path, index=False
    )
    summary = {
        "status": "exploratory multi-view blend screen on repeatedly examined labels",
        "primary_metric": "mean-fold ROC-AUC; pooled cross-fold rank AUC is not used",
        "blend_rule": "fixed equal average of three global view ranks; no row routing or optimized weights",
        "candidates": results,
        "spearman_correlation_to_prior": correlations,
        "selection_note": "All choices are exploratory on previously examined labels and require external leaderboard validation.",
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                BASELINE,
                VIEWS,
                Path(__file__).resolve(),
                fold_path,
                results_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
