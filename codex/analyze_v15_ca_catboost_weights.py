"""Exploratory blend-weight sensitivity for the matched CA CatBoost candidate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
CACHE = HERE / "v15_cache" / "gci_cache_final"
OOF = np.load(HERE / "v15_ca_ext_catboost" / "oof.npz")
y = OOF["y"]
candidate_cb = OOF["candidate_catboost_oof"]
rk = lambda x: rankdata(x, method="average") / len(x)
w = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32,
     "v2_allfam_oh": .16, "v2_mono": .16}
branch = {}
for tag in ("old", "e1", "e1et", "e1Aet"):
    pred = np.zeros(len(y))
    for name, weight in w.items():
        pred += weight * rk(np.load(CACHE / f"{tag}_{name}.npz")["oof"])
    branch[tag] = pred
b = (branch["old"] + branch["e1"] + branch["e1et"]) / 3
main = rk(.75 * rk(b) + .25 * branch["e1Aet"])
old_cb = np.zeros(len(y))
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
for i, (_, valid) in enumerate(folds, 1):
    old_cb[valid] = np.load(CACHE / f"catboost_product_fold_{i}.npz")["oof"]
old_r, cand_r = rk(old_cb), rk(candidate_cb)
baseline = .90 * main + .10 * old_r
results = []
for alpha in (0, .05, .10, .15, .20, .25, .30, .40, .50):
    pred = (1-alpha) * main + alpha * cand_r
    fold_deltas = []
    for _, valid in folds:
        fold_deltas.append(float(roc_auc_score(y[valid], pred[valid]) - roc_auc_score(y[valid], baseline[valid])))
    results.append({"candidate_catboost_weight": alpha,
                    "auc": float(roc_auc_score(y, pred)),
                    "delta_vs_v15": float(roc_auc_score(y, pred)-roc_auc_score(y, baseline)),
                    "positive_folds_vs_v15": sum(x > 0 for x in fold_deltas),
                    "fold_deltas": fold_deltas})

rng = np.random.default_rng(12731)
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
base = baseline
cand = .90 * main + .10 * cand_r
deltas = np.empty(1000)
for i in range(len(deltas)):
    idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
    deltas[i] = roc_auc_score(y[idx], cand[idx]) - roc_auc_score(y[idx], base[idx])
summary = {
    "baseline_auc": float(roc_auc_score(y, baseline)),
    "candidate_10pct_auc": float(roc_auc_score(y, cand)),
    "candidate_10pct_delta": float(roc_auc_score(y, cand)-roc_auc_score(y, baseline)),
    "candidate_10pct_paired_bootstrap_95_ci": np.quantile(deltas, [.025, .5, .975]).tolist(),
    "weights": results,
    "warning": "Weight sweep uses the same five reused OOF folds and is exploratory; any selected alternative weight needs fresh confirmation."
}
out = HERE / "v15_ca_ext_catboost"
(out / "weight_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame(results).to_csv(out / "weight_sensitivity.csv", index=False)
print(json.dumps(summary, indent=2))
