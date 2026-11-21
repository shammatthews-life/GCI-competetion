from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
IDEAS_DIR = Path(__file__).resolve().parent
TE_OOF = IDEAS_DIR / "submissions" / "product_te_20pct_v1" / "OOF_CANDIDATE_PREDICTIONS.csv"
PAIRWISE_OOF = ROOT / "runs" / "pairwise_auc_linear_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "pairwise_te_blend_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT}")
    te = pd.read_csv(TE_OOF)
    pairwise = pd.read_csv(PAIRWISE_OOF)
    for column in (ID, TARGET, FOLD):
        if not te[column].equals(pairwise[column]):
            raise ValueError(f"OOF artifacts differ in {column}")

    y = te[TARGET].to_numpy(dtype=np.int8)
    folds = te[FOLD].to_numpy(dtype=np.int8)
    baseline = te["previous_aligned_blend"].to_numpy(dtype=np.float64)
    te_score = te["nested_te_lgbm"].to_numpy(dtype=np.float64)
    pairwise_score = pairwise["pairwise_auc_linear"].to_numpy(dtype=np.float64)
    metrics: list[dict[str, int | float]] = []
    for fold_number in sorted(np.unique(folds)):
        rows = np.flatnonzero(folds == fold_number)
        base_rank = rank01(baseline[rows])
        te_rank = rank01(te_score[rows])
        pairwise_rank = rank01(pairwise_score[rows])
        candidate = 0.80 * base_rank + 0.10 * te_rank + 0.10 * pairwise_rank
        baseline_auc = float(roc_auc_score(y[rows], baseline[rows]))
        te20_auc = float(
            roc_auc_score(
                y[rows],
                0.80 * base_rank + 0.20 * te_rank,
            )
        )
        candidate_auc = float(roc_auc_score(y[rows], candidate))
        metrics.append(
            {
                "fold": int(fold_number),
                "baseline_auc": baseline_auc,
                "prior_te20_auc": te20_auc,
                "prior_te_pairwise_80_10_10_auc": candidate_auc,
                "delta_vs_baseline": candidate_auc - baseline_auc,
                "delta_vs_prior_te20": candidate_auc - te20_auc,
            }
        )

    baseline_mean = float(np.mean([item["baseline_auc"] for item in metrics]))
    te20_mean = float(np.mean([item["prior_te20_auc"] for item in metrics]))
    candidate_mean = float(
        np.mean([item["prior_te_pairwise_80_10_10_auc"] for item in metrics])
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metrics_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    summary = {
        "status": "exploratory fixed-formula OOF screen on repeatedly examined labels",
        "formula": "80% prior rank + 10% nested target-encoding rank + 10% pairwise-linear rank, within each fold",
        "weight_search": "none in this analysis",
        "mean_fold_auc": {
            "baseline": baseline_mean,
            "prior_te20": te20_mean,
            "prior_te_pairwise_80_10_10": candidate_mean,
            "delta_vs_baseline": candidate_mean - baseline_mean,
            "delta_vs_prior_te20": candidate_mean - te20_mean,
        },
        "positive_folds_vs_baseline": int(
            sum(item["delta_vs_baseline"] > 0 for item in metrics)
        ),
        "positive_folds_vs_prior_te20": int(
            sum(item["delta_vs_prior_te20"] > 0 for item in metrics)
        ),
        "fold_metrics": metrics,
        "input_sha256": {
            str(TE_OOF): sha256_file(TE_OOF),
            str(PAIRWISE_OOF): sha256_file(PAIRWISE_OOF),
            str(metrics_path): sha256_file(metrics_path),
        },
        "caveat": (
            "The pairwise model and formula were evaluated on labels already used in "
            "earlier model development; results are not an independent estimate or "
            "leaderboard score."
        ),
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
