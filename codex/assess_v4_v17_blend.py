import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(r"D:\gci")
DATA = ROOT / "inpuy"
OUT = ROOT / "codex"
WEIGHT_LGB = 0.10
BOOTSTRAPS = 500
MINIMUM_GAIN = 0.00015
MINIMUM_POSITIVE_FOLDS = 4


def rank(values):
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def paired_bootstrap_ci(y, candidate, reference, seed=20261043):
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    deltas = np.empty(BOOTSTRAPS, dtype=np.float64)
    for index in range(BOOTSTRAPS):
        rows = np.concatenate(
            (
                rng.choice(positives, len(positives), replace=True),
                rng.choice(negatives, len(negatives), replace=True),
            )
        )
        deltas[index] = roc_auc_score(
            y[rows], candidate[rows]
        ) - roc_auc_score(y[rows], reference[rows])
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def main():
    train = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    labels = train["TARGET"].to_numpy(dtype=np.int8)
    catboost = np.load(OUT / "v_3_standalone_v17_oof.npz")
    lightgbm = np.load(OUT / "v_4_lgb_seed42_oof.npz")
    if not np.array_equal(catboost["y"], labels) or not np.array_equal(
        lightgbm["y"], labels
    ):
        raise ValueError("OOF target rows are not aligned to train.csv")
    cat_oof = rank(catboost["oof"])
    lgb_oof = rank(lightgbm["oof_seed_42"])
    candidate_oof = (1 - WEIGHT_LGB) * cat_oof + WEIGHT_LGB * lgb_oof
    reference_auc = float(roc_auc_score(labels, cat_oof))
    candidate_auc = float(roc_auc_score(labels, candidate_oof))
    folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_gains = [
        float(
            roc_auc_score(labels[valid], candidate_oof[valid])
            - roc_auc_score(labels[valid], cat_oof[valid])
        )
        for _, valid in folds
    ]
    ci = paired_bootstrap_ci(labels, candidate_oof, cat_oof)
    passed = (
        candidate_auc - reference_auc >= MINIMUM_GAIN
        and sum(gain > 0 for gain in fold_gains) >= MINIMUM_POSITIVE_FOLDS
        and ci[0] > 0
    )

    metrics = {
        "version": "v_4",
        "experiment": "fixed 10% LightGBM rank blend into standalone v17 CatBoost-E1",
        "weight_lgb": WEIGHT_LGB,
        "reference_catboost_oof_auc": reference_auc,
        "candidate_oof_auc": candidate_auc,
        "delta_auc": candidate_auc - reference_auc,
        "fold_gains": fold_gains,
        "positive_folds": sum(gain > 0 for gain in fold_gains),
        "paired_bootstrap_95pct_ci": ci,
        "acceptance_rule": {
            "minimum_gain": MINIMUM_GAIN,
            "minimum_positive_folds": MINIMUM_POSITIVE_FOLDS,
            "bootstrap_lower_bound_above_zero": True,
        },
        "passed": passed,
        "important_limit": (
            "This compares two standalone branches, not the established v15 "
            "full ensemble. A pass would only justify a diagnostic probe, not "
            "claim an improvement over public 0.77891."
        ),
    }

    if passed:
        cat_test = rank(catboost["test_probability"])
        lgb_test = rank(lightgbm["test_seed_42"])
        predictions = (1 - WEIGHT_LGB) * cat_test + WEIGHT_LGB * lgb_test
        submission = sample.copy()
        submission["TARGET"] = predictions
        path = OUT / "v_4_v17_lgb_rank_blend.csv"
        submission.to_csv(path, index=False)
        checked = pd.read_csv(path)
        if (
            checked.shape != (len(sample), 2)
            or checked.columns.tolist() != ["SK_ID_CURR", "TARGET"]
            or not checked.SK_ID_CURR.is_unique
            or not np.array_equal(checked.SK_ID_CURR, sample.SK_ID_CURR)
            or not np.isfinite(checked.TARGET).all()
        ):
            raise ValueError("v_4 submission failed structural validation")
        metrics["submission"] = str(path)
        metrics["submission_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        metrics["submission"] = None
        print("Acceptance gate failed; no v_4 candidate CSV will be written.")

    (OUT / "v_4_blend_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
