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
CATBOOST_BASE = RUNS / "clean_catboost_constant_free_v1" / "OOF_PREDICTIONS.csv"
SECOND_SEED = RUNS / "clean_catboost_second_seed_v1" / "OOF_PREDICTIONS.csv"
EXTERNAL_GEOMETRY = RUNS / "clean_external_geometry_catboost_v1" / "OOF_PREDICTIONS.csv"
NESTED_TE = RUNS / "nested_product_te_lgbm_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = RUNS / "ideas_v1_blend_screen_v2"
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_aligned(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF input: {path}")
    frame = pd.read_csv(path)
    if (
        len(frame) != len(reference)
        or not frame[ID].equals(reference[ID])
        or not frame[TARGET].equals(reference[TARGET])
        or not frame["fold"].equals(reference["fold"])
    ):
        raise ValueError(f"OOF rows, IDs, labels, or folds do not align: {path}")
    return frame


def fold_rank(values: np.ndarray, folds: np.ndarray) -> np.ndarray:
    ranked = np.full(len(values), np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = folds == fold
        if mask.sum() < 2:
            raise ValueError(f"Fold {fold} is missing or too small")
        ranked[mask] = (rankdata(values[mask], method="average") - 1) / (
            mask.sum() - 1
        )
    if not np.isfinite(ranked).all():
        raise ValueError("Rank transform contains non-finite values")
    return ranked


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {OUTPUT}")
    base = pd.read_csv(BASELINE)
    if (
        not base[ID].is_unique
        or set(base["fold"].unique()) != set(range(1, FOLDS + 1))
    ):
        raise ValueError("Baseline IDs or fold identifiers are invalid")
    cat_base = read_aligned(CATBOOST_BASE, base)
    cat_seed2 = read_aligned(SECOND_SEED, base)
    geometry = read_aligned(EXTERNAL_GEOMETRY, base)
    te = read_aligned(NESTED_TE, base)
    y = base[TARGET].to_numpy(dtype=np.int8)
    folds = base["fold"].to_numpy(dtype=np.int8)

    raw_scores = {
        "same_fold_general_baseline": base[
            "same_fold_phase33_phase34_baseline"
        ].to_numpy(dtype=np.float64),
        "clean_catboost_seed1": cat_base[
            "oof_clean_catboost"
        ].to_numpy(dtype=np.float64),
        "clean_catboost_seed2": cat_seed2[
            "oof_catboost_second_seed"
        ].to_numpy(dtype=np.float64),
        "external_geometry_catboost": geometry[
            "oof_external_geometry_catboost"
        ].to_numpy(dtype=np.float64),
        "nested_product_te_lgbm": te[
            "oof_nested_product_te_lgbm"
        ].to_numpy(dtype=np.float64),
    }
    if any(not np.isfinite(values).all() for values in raw_scores.values()):
        raise ValueError("An input OOF vector contains non-finite values")

    ranks = {
        name: fold_rank(values, folds) for name, values in raw_scores.items()
    }
    previous_best = base[
        "fixed_50_50_within_fold_rank_blend"
    ].to_numpy(dtype=np.float64)
    ranks["previous_aligned_blend"] = fold_rank(previous_best, folds)

    candidates: dict[str, np.ndarray] = {
        "previous_aligned_blend": ranks["previous_aligned_blend"],
        "two_seed_catboost_50_50": 0.5 * ranks["clean_catboost_seed1"]
        + 0.5 * ranks["clean_catboost_seed2"],
    }
    components = (
        "clean_catboost_seed2",
        "external_geometry_catboost",
        "nested_product_te_lgbm",
    )
    for component in components:
        for weight in (0.10, 0.20, 0.50):
            candidates[f"previous_plus_{component}_{weight:.2f}"] = (
                (1.0 - weight) * ranks["previous_aligned_blend"]
                + weight * ranks[component]
            )
    candidates["previous_plus_three_new_views_10pct_total"] = (
        0.90 * ranks["previous_aligned_blend"]
        + (0.10 / 3.0) * ranks["clean_catboost_seed2"]
        + (0.10 / 3.0) * ranks["external_geometry_catboost"]
        + (0.10 / 3.0) * ranks["nested_product_te_lgbm"]
    )

    baseline_auc = float(
        np.mean(
            [
                roc_auc_score(y[folds == fold], previous_best[folds == fold])
                for fold in range(1, FOLDS + 1)
            ]
        )
    )
    model_metrics: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []
    for name, prediction in candidates.items():
        aucs = []
        for fold in range(1, FOLDS + 1):
            mask = folds == fold
            auc = float(roc_auc_score(y[mask], prediction[mask]))
            aucs.append(auc)
            fold_metrics.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "rows": int(mask.sum()),
                    "positives": int(y[mask].sum()),
                    "auc": auc,
                    "delta_vs_previous_aligned_blend": float(
                        auc
                        - roc_auc_score(y[mask], previous_best[mask])
                    ),
                }
            )
        model_metrics.append(
            {
                "candidate": name,
                "pooled_auc_after_within_fold_rank_normalization_descriptive_only": float(
                    roc_auc_score(y, prediction)
                ),
                "mean_fold_auc": float(np.mean(aucs)),
                "delta_vs_previous_aligned_blend": float(
                    np.mean(aucs) - baseline_auc
                ),
                "positive_fold_delta_count": int(
                    sum(
                        auc
                        > roc_auc_score(y[folds == fold], previous_best[folds == fold])
                        for fold, auc in enumerate(aucs, start=1)
                    )
                ),
                "fold_auc": aucs,
            }
        )

    complementarity = {
        name: float(spearmanr(ranks["previous_aligned_blend"], vector).statistic)
        for name, vector in ranks.items()
        if name != "previous_aligned_blend"
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metrics_path = OUTPUT / "CANDIDATE_METRICS.csv"
    folds_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(model_metrics).drop(columns=["fold_auc"]).to_csv(
        metrics_path, index=False
    )
    pd.DataFrame(fold_metrics).to_csv(folds_path, index=False)
    summary = {
        "status": "exploratory fixed-weight blend screen on repeatedly examined labels; not independent confirmation",
        "folds": "all candidate OOF rows, IDs, labels, and outer-fold IDs aligned",
        "blend_rule": "within-fold average ranks; weights fixed as listed, no row-specific routing",
        "metric_note": "Mean-fold AUC is the primary comparison. Pooled AUC after fold-wise rank normalization is descriptive only because each fold has an independently normalized score scale.",
        "selection_note": "The finite weights and candidates were compared on the same repeatedly examined folds; the best observed blend is subject to model-selection optimism and requires an external leaderboard check.",
        "reference_mean_fold_auc": baseline_auc,
        "candidate_results": model_metrics,
        "spearman_rank_correlation_to_previous_blend": complementarity,
        "rule_controls": {
            "test_rows": "not read",
            "training_labels": "used only for fold AUC reporting; no model fit or test prediction in this script",
            "manual_row_edits": "none",
            "weight_search": "no continuous or per-row weight optimization; documented fixed-weight screen only",
        },
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                BASELINE,
                CATBOOST_BASE,
                SECOND_SEED,
                EXTERNAL_GEOMETRY,
                NESTED_TE,
                Path(__file__).resolve(),
                metrics_path,
                folds_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
