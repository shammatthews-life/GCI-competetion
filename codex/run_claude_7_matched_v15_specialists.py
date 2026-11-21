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
CACHE = ROOT / "v15_cache" / "gci_cache_final"
SEED = 42
N_FOLDS = 5
BOOTSTRAPS = 1500
BLEND_WEIGHT = 0.01
MEMBER_WEIGHTS = {
    "qf_shallow": 0.24,
    "qf_reg": 0.12,
    "v1_allfam_oh": 0.32,
    "v2_allfam_oh": 0.16,
    "v2_mono": 0.16,
}


def rank(values):
    return rankdata(values, method="average") / len(values)


def paired_bootstrap_ci(y, candidate, reference, seed):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        idx = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas[i] = roc_auc_score(y[idx], candidate[idx]) - roc_auc_score(
            y[idx], reference[idx]
        )
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def load_v15_oof(y):
    branch = {}
    for tag in ("old", "e1", "e1et", "e1Aet"):
        blend = np.zeros(len(y), dtype=np.float64)
        for name, weight in MEMBER_WEIGHTS.items():
            path = CACHE / f"{tag}_{name}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing v15 member cache: {path}")
            with np.load(path) as archive:
                pred = archive["oof"]
            if pred.shape != y.shape or not np.isfinite(pred).all():
                raise ValueError(f"Invalid OOF cache array: {path}")
            blend += weight * rank(pred)
        branch[tag] = blend

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y)), y))
    cb_oof = np.zeros(len(y), dtype=np.float64)
    for fold_number, (_, valid_idx) in enumerate(folds, start=1):
        path = CACHE / f"catboost_product_fold_{fold_number}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing v15 CatBoost cache: {path}")
        with np.load(path) as archive:
            pred = archive["oof"]
        if pred.shape != (len(valid_idx),) or not np.isfinite(pred).all():
            raise ValueError(f"Invalid CatBoost fold cache array: {path}")
        cb_oof[valid_idx] = pred

    base = (branch["old"] + branch["e1"] + branch["e1et"]) / 3.0
    final = 0.90 * rank(0.75 * rank(base) + 0.25 * branch["e1Aet"])
    final += 0.10 * rank(cb_oof)
    return final, folds


def main():
    started = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    champion, folds = load_v15_oof(y)
    screen.OUTER_FOLDS = N_FOLDS
    base, blocks, product_keys, screen_y, n_train = screen._build_frames(train, test)
    if not np.array_equal(screen_y, y):
        raise ValueError("Feature-screen labels are misaligned with v15 labels")

    selected = {
        name: blocks[name] for name in ("household", "bureau_profile")
    }
    predictions, _ = screen._run_seed(
        SEED, base, selected, product_keys, y, n_train
    )
    if not np.array_equal(
        [valid_idx.size for _, valid_idx in folds],
        [np.count_nonzero(np.isin(np.arange(n_train), valid_idx)) for _, valid_idx in folds],
    ):
        raise AssertionError("Unexpected fold sizes")

    champion_rank = rank(champion)
    candidate_ranks = {
        name: rank(predictions[name])
        for name in ("household", "bureau_profile")
    }
    candidate_ranks["joint_equal_rank"] = (
        candidate_ranks["household"] + candidate_ranks["bureau_profile"]
    ) / 2.0

    results = []
    fold_rows = []
    for name, candidate_rank in candidate_ranks.items():
        blended = (1.0 - BLEND_WEIGHT) * champion_rank + (
            BLEND_WEIGHT * candidate_rank
        )
        delta = roc_auc_score(y, blended) - roc_auc_score(y, champion)
        ci = paired_bootstrap_ci(y, blended, champion, SEED + len(name))
        fold_deltas = []
        for fold_number, (_, valid_idx) in enumerate(folds, start=1):
            before = roc_auc_score(y[valid_idx], champion[valid_idx])
            after = roc_auc_score(y[valid_idx], blended[valid_idx])
            fold_delta = float(after - before)
            fold_deltas.append(fold_delta)
            fold_rows.append(
                {
                    "variant": name,
                    "fold": fold_number,
                    "champion_auc": float(before),
                    "candidate_blend_auc": float(after),
                    "delta_auc": fold_delta,
                }
            )
        results.append(
            {
                "variant": name,
                "fixed_challenger_weight": BLEND_WEIGHT,
                "champion_oof_auc": float(roc_auc_score(y, champion)),
                "specialist_oof_auc": float(
                    roc_auc_score(y, predictions[name])
                    if name != "joint_equal_rank"
                    else roc_auc_score(y, candidate_rank)
                ),
                "blended_oof_auc": float(roc_auc_score(y, blended)),
                "delta_auc_vs_v15": float(delta),
                "fold_deltas": fold_deltas,
                "positive_folds": int(np.sum(np.asarray(fold_deltas) > 0)),
                "paired_bootstrap_95_ci": ci,
                "spearman_with_v15": float(
                    spearmanr(champion_rank, candidate_rank).statistic
                ),
                "passes_project_gate": bool(
                    delta >= 0.00015
                    and np.sum(np.asarray(fold_deltas) > 0) >= 4
                    and ci[0] > 0
                ),
            }
        )

    pd.DataFrame(results).to_csv(
        ROOT / "claude_7_matched_v15_specialist_results.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        ROOT / "claude_7_matched_v15_specialist_folds.csv", index=False
    )
    np.savez_compressed(
        ROOT / "claude_7_matched_v15_specialist_oof.npz",
        y=y,
        v15_oof=champion,
        household_oof=predictions["household"],
        bureau_profile_oof=predictions["bureau_profile"],
    )
    metadata = {
        "v15_cache_dir": str(CACHE),
        "outer_folds": N_FOLDS,
        "fold_seed": SEED,
        "specialist_model": "compact formula/product-aware LightGBM screen",
        "locked_rank_blend_weight": BLEND_WEIGHT,
        "all_predictions_are_fold_aligned": True,
        "v15_rebuilt_from_all_member_oof_caches": True,
        "submission_created": False,
        "runtime_seconds": round(time.time() - started, 2),
        "interpretation": (
            "Matched v15 OOF comparison using the notebook's cached member predictions. "
            "Candidate branches are compact specialist models, not retrained full v15 "
            "members. Treat this as a component screen; do not extrapolate to leaderboard."
        ),
    }
    (ROOT / "claude_7_matched_v15_specialist_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("v15 reconstructed OOF AUC:", roc_auc_score(y, champion), flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
