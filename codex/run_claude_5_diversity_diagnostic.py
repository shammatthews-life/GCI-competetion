import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci\codex")
SEEDS = (20261051, 20261052)
N_FOLDS = 3
N_BOOTSTRAPS = 1000


def _stratified_auc_delta_interval(y, candidate, reference, rng):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    deltas = np.empty(N_BOOTSTRAPS, dtype=np.float64)
    for i in range(N_BOOTSTRAPS):
        idx = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], reference[idx]
        )
    return np.quantile(deltas, [0.025, 0.975])


def main():
    archive = np.load(ROOT / "claude_5_feature_hypothesis_oof.npz")
    y = archive["y"]
    rows = []
    for seed in SEEDS:
        folds = StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=seed
        ).split(np.zeros(len(y)), y)
        for fold, (_, valid_idx) in enumerate(folds, start=1):
            fold_y = y[valid_idx]
            raw = {
                name: archive[f"seed{seed}_{name}"][valid_idx]
                for name in ("baseline", "household", "bureau_profile")
            }
            ranks = {
                name: rankdata(values, method="average") / len(valid_idx)
                for name, values in raw.items()
            }
            candidate = (
                ranks["baseline"] + ranks["household"] + ranks["bureau_profile"]
            ) / 3.0
            baseline_auc = roc_auc_score(fold_y, raw["baseline"])
            candidate_auc = roc_auc_score(fold_y, candidate)
            ci = _stratified_auc_delta_interval(
                fold_y,
                candidate,
                raw["baseline"],
                np.random.default_rng(seed * 10 + fold),
            )
            rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "baseline_auc": float(baseline_auc),
                    "household_auc": float(
                        roc_auc_score(fold_y, raw["household"])
                    ),
                    "bureau_profile_auc": float(
                        roc_auc_score(fold_y, raw["bureau_profile"])
                    ),
                    "equal_rank_blend_auc": float(candidate_auc),
                    "delta_vs_baseline": float(candidate_auc - baseline_auc),
                    "paired_bootstrap_ci_low": float(ci[0]),
                    "paired_bootstrap_ci_high": float(ci[1]),
                    "household_bureau_spearman": float(
                        spearmanr(raw["household"], raw["bureau_profile"]).statistic
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "claude_5_diversity_fold_diagnostics.csv", index=False)
    metrics = {
        "hypothesis": (
            "The complementary OLD/NEW member gain can be partly reproduced by "
            "equal rank blending a compact reference, household features, and "
            "bureau inquiry profile features."
        ),
        "blend_weights": {
            "baseline": 1 / 3,
            "household": 1 / 3,
            "bureau_profile": 1 / 3,
        },
        "folds": len(frame),
        "positive_fold_deltas": int((frame["delta_vs_baseline"] > 0).sum()),
        "mean_fold_delta": float(frame["delta_vs_baseline"].mean()),
        "median_fold_delta": float(frame["delta_vs_baseline"].median()),
        "min_fold_delta": float(frame["delta_vs_baseline"].min()),
        "max_fold_delta": float(frame["delta_vs_baseline"].max()),
        "mean_household_bureau_spearman": float(
            frame["household_bureau_spearman"].mean()
        ),
        "mean_of_fold_specific_paired_bootstrap_bounds": [
            float(frame["paired_bootstrap_ci_low"].mean()),
            float(frame["paired_bootstrap_ci_high"].mean()),
        ],
        "validation_scope": (
            "Exploratory fold-local rank blend on the same two seed partitions "
            "used to screen the feature blocks. It is not a fresh v15 comparison "
            "and not an unbiased confirmation after screening."
        ),
        "submission_created": False,
    }
    (ROOT / "claude_5_diversity_diagnostic.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
