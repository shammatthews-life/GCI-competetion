import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import run_claude_5_feature_hypothesis_screen as screen


ROOT = Path(r"D:\gci\codex")
DATA = Path(r"D:\gci\inpuy")
SEED = 20261053
N_FOLDS = 3
N_BOOTSTRAPS = 1000


def paired_bootstrap_ci(y, candidate, baseline, seed):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(N_BOOTSTRAPS, dtype=np.float64)
    for i in range(N_BOOTSTRAPS):
        idx = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], baseline[idx]
        )
    return np.quantile(deltas, [0.025, 0.975])


def main():
    started = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    base, blocks, product_keys, y, n_train = screen._build_frames(train, test)
    selected_blocks = {name: blocks[name] for name in ("household", "bureau_profile")}
    predictions, fold_metrics = screen._run_seed(
        SEED, base, selected_blocks, product_keys, y, n_train
    )

    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(n_train), y
        )
    )
    fold_rows = []
    for fold_number, (_, valid_idx) in enumerate(folds, start=1):
        fold_y = y[valid_idx]
        fold_predictions = {
            name: predictions[name][valid_idx]
            for name in ("baseline", "household", "bureau_profile")
        }
        fold_ranks = {
            name: rankdata(values, method="average") / len(valid_idx)
            for name, values in fold_predictions.items()
        }
        blend = (
            fold_ranks["baseline"]
            + fold_ranks["household"]
            + fold_ranks["bureau_profile"]
        ) / 3.0
        baseline_auc = roc_auc_score(fold_y, fold_predictions["baseline"])
        blend_auc = roc_auc_score(fold_y, blend)
        ci = paired_bootstrap_ci(
            fold_y, blend, fold_predictions["baseline"], SEED + fold_number
        )
        fold_rows.append(
            {
                "seed": SEED,
                "fold": fold_number,
                "baseline_auc": float(baseline_auc),
                "household_auc": float(
                    roc_auc_score(fold_y, fold_predictions["household"])
                ),
                "bureau_profile_auc": float(
                    roc_auc_score(fold_y, fold_predictions["bureau_profile"])
                ),
                "equal_rank_blend_auc": float(blend_auc),
                "delta_vs_baseline": float(blend_auc - baseline_auc),
                "paired_bootstrap_ci_low": float(ci[0]),
                "paired_bootstrap_ci_high": float(ci[1]),
                "household_bureau_spearman": float(
                    spearmanr(
                        fold_predictions["household"],
                        fold_predictions["bureau_profile"],
                    ).statistic
                ),
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(ROOT / "claude_6_fresh_diversity_confirmation.csv", index=False)
    np.savez_compressed(
        ROOT / "claude_6_fresh_diversity_confirmation_oof.npz",
        y=y,
        **{f"oof_{name}": pred for name, pred in predictions.items()},
    )
    summary = {
        "seed": SEED,
        "folds": N_FOLDS,
        "fixed_blend_weights": {
            "baseline": 1 / 3,
            "household": 1 / 3,
            "bureau_profile": 1 / 3,
        },
        "positive_fold_deltas": int((fold_frame["delta_vs_baseline"] > 0).sum()),
        "mean_fold_delta": float(fold_frame["delta_vs_baseline"].mean()),
        "min_fold_delta": float(fold_frame["delta_vs_baseline"].min()),
        "max_fold_delta": float(fold_frame["delta_vs_baseline"].max()),
        "mean_household_bureau_spearman": float(
            fold_frame["household_bureau_spearman"].mean()
        ),
        "runtime_seconds": round(time.time() - started, 2),
        "validation_scope": (
            "Fresh third split for a pre-fixed OLD/NEW-inspired equal-rank blend. "
            "This is still a compact-model result, not a matched v15 comparison."
        ),
        "submission_created": False,
    }
    (ROOT / "claude_6_fresh_diversity_confirmation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(fold_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
