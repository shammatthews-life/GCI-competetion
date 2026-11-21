"""Test fixed-v15 blend with a fold-local tariff-frequency CatBoost gate."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v15_product_frequency_gate"
OUT.mkdir(exist_ok=True)
WEIGHTS = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32,
           "v2_allfam_oh": .16, "v2_mono": .16}


def rank(x):
    return rankdata(x, method="average") / len(x)


def main():
    import run_claude_7_matched_v15_specialists as champion

    started = time.time()
    train = pd.read_csv(ROOT / "inpuy" / "train.csv")
    y = train.TARGET.to_numpy(dtype=np.int8)
    n = len(y)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    cache = HERE / "v15_cache" / "gci_cache_final"
    branches = {}
    for tag in ("old", "e1", "e1et", "e1Aet"):
        pred = np.zeros(n)
        for name, weight in WEIGHTS.items():
            z = np.load(cache / f"{tag}_{name}.npz")
            pred += weight * rank(z["oof"])
        branches[tag] = pred
    cb = np.zeros(n)
    for fi, (_, valid) in enumerate(folds, start=1):
        z = np.load(cache / f"catboost_product_fold_{fi}.npz")
        cb[valid] = z["oof"]
    b = (branches["old"] + branches["e1"] + branches["e1et"]) / 3
    inner = rank(.75 * rank(b) + .25 * branches["e1Aet"])
    main_rank = rank(inner)
    cb_rank = rank(cb)
    base = .90 * main_rank + .10 * cb_rank
    assert np.allclose(base, champion.load_v15_oof(y)[0], atol=1e-10)

    # Decode the dominant tariff Q4 family label-free. Fold-local counts mimic
    # the information available when predicting each held-out fold.
    cash = train.NAME_CONTRACT_TYPE.eq("Cash loans").to_numpy()
    credit = train.AMT_CREDIT.to_numpy(dtype=np.float64)
    annuity = train.AMT_ANNUITY.to_numpy(dtype=np.float64)
    q = annuity / credit
    q_fee = (annuity - 129.5) / credit
    use_fee = cash & np.isfinite(q_fee)
    p = np.where(use_fee, q_fee, q)
    keys = pd.Series(np.round(p, 4)).astype("string").fillna("__MISSING__").to_numpy()
    logcount = np.zeros(n)
    folds_for_row = np.empty(n, dtype=np.int8)
    for fi, (fit, valid) in enumerate(folds):
        counts = pd.Series(keys[fit]).value_counts()
        logcount[valid] = np.log1p(pd.Series(keys[valid]).map(counts).fillna(0).to_numpy())
        folds_for_row[valid] = fi

    # Descriptive grouped AUC: which branch tends to rank within each frequency
    # band? This is diagnostic only; the fixed symmetric gates are also scored.
    decile = pd.qcut(pd.Series(logcount).rank(method="first"), 10, labels=False)
    desc = []
    for d in range(10):
        idx = np.flatnonzero(decile.to_numpy() == d)
        desc.append({"frequency_decile": d + 1, "rows": len(idx),
                     "mean_training_count": float(np.expm1(logcount[idx]).mean()),
                     "v15_auc_within_band": float(roc_auc_score(y[idx], base[idx])),
                     "main_branch_auc_within_band": float(roc_auc_score(y[idx], main_rank[idx])),
                     "catboost_auc_within_band": float(roc_auc_score(y[idx], cb_rank[idx]))})

    # Candidate rule is frozen: shift the CatBoost weight linearly by +/-2.5pp
    # over the centered within-fold rank of log product frequency. Keep mean 10%.
    freq_rank = np.zeros(n)
    for fi, (_, valid) in enumerate(folds):
        local = pd.Series(logcount[valid]).rank(method="average", pct=True).to_numpy()
        freq_rank[valid] = local - .5
    candidate_more_cb_common = (.90 - .05 * freq_rank) * main_rank + (.10 + .05 * freq_rank) * cb_rank
    candidate_more_cb_rare = (.90 + .05 * freq_rank) * main_rank + (.10 - .05 * freq_rank) * cb_rank
    auc = lambda pred: float(roc_auc_score(y, pred))
    fold_rows = []
    for fi, (_, valid) in enumerate(folds):
        fold_rows.append({"fold": fi + 1, "v15_auc": float(roc_auc_score(y[valid], base[valid])),
                          "common_catboost_auc": float(roc_auc_score(y[valid], candidate_more_cb_common[valid])),
                          "rare_catboost_auc": float(roc_auc_score(y[valid], candidate_more_cb_rare[valid]))})
    result = {
        "hypothesis": "The native product CatBoost branch is more useful for repeated tariff keys; gate its global 10% weight by fold-local Q4 key frequency.",
        "rows": n, "v15_oof_auc": auc(base),
        "common_products_more_catboost_auc": auc(candidate_more_cb_common),
        "common_products_more_catboost_delta": auc(candidate_more_cb_common)-auc(base),
        "rare_products_more_catboost_auc": auc(candidate_more_cb_rare),
        "rare_products_more_catboost_delta": auc(candidate_more_cb_rare)-auc(base),
        "gate": "CatBoost weight = 0.10 +/- 0.05 * (within-fold percentile(log1p(train Q4 count)) - 0.5); mean held near 10%.",
        "fold_metrics": fold_rows,
        "elapsed_seconds": round(time.time()-started, 1),
        "note": "Same seed-42 OOF folds; count gates use no labels and are fit using outer-fold training rows only; no submission created. A positive selected-fold result requires a fresh confirmation.",
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(desc).to_csv(OUT / "frequency_deciles.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, main_rank=main_rank,
                        catboost_rank=cb_rank, log_product_frequency=logcount,
                        common_gate_oof=candidate_more_cb_common,
                        rare_gate_oof=candidate_more_cb_rare)
    print(json.dumps(result, indent=2), flush=True)
    print(pd.DataFrame(desc).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
