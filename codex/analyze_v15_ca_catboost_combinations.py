"""Explore combinations of the two matched CA CatBoost OOF candidates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

H = Path(__file__).resolve().parent
C = H / "v15_cache" / "gci_cache_final"
z1 = np.load(H / "v15_ca_ext_catboost" / "oof.npz")
z2 = np.load(H / "v15_ca_catboost_only" / "oof.npz")
y, cb_ext, cb_only = z1["y"], z1["candidate_catboost_oof"], z2["candidate_catboost_oof"]
rk = lambda x: rankdata(x, method="average") / len(x)
w = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32, "v2_allfam_oh": .16, "v2_mono": .16}
br = {}
for tag in ("old", "e1", "e1et", "e1Aet"):
    p = np.zeros(len(y))
    for m, a in w.items():
        p += a * rk(np.load(C / f"{tag}_{m}.npz")["oof"])
    br[tag] = p
b = (br["old"] + br["e1"] + br["e1et"]) / 3
main = rk(.75 * rk(b) + .25 * br["e1Aet"])
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
old = np.zeros(len(y))
for i, (_, valid) in enumerate(folds, 1):
    old[valid] = np.load(C / f"catboost_product_fold_{i}.npz")["oof"]
baseline = .9 * main + .1 * rk(old)
results = []
for w_ext in (0, .25, .5, .75, 1):
    cb_mix = (1-w_ext) * rk(cb_only) + w_ext * rk(cb_ext)
    for alpha in (.10, .15, .20, .25):
        pred = (1-alpha) * main + alpha * cb_mix
        fd = [float(roc_auc_score(y[v], pred[v])-roc_auc_score(y[v], baseline[v])) for _, v in folds]
        results.append({"ca_ext_share_in_candidate": w_ext, "candidate_branch_weight": alpha,
                        "auc": float(roc_auc_score(y, pred)),
                        "delta_vs_v15": float(roc_auc_score(y, pred)-roc_auc_score(y, baseline)),
                        "positive_folds": sum(x > 0 for x in fd), "fold_deltas": fd})
summary = {"baseline_auc": float(roc_auc_score(y, baseline)),
           "candidate_catboost_spearman_ca_only_vs_ca_ext": float(spearmanr(cb_only, cb_ext).statistic),
           "results": results,
           "note": "Grid uses reused OOF folds and is exploratory; selected combinations require fresh confirmation."}
out = H / "v15_ca_ext_catboost"
(out / "combination_sensitivity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
pd.DataFrame(results).to_csv(out / "combination_sensitivity.csv", index=False)
print("Spearman:", summary["candidate_catboost_spearman_ca_only_vs_ca_ext"])
print(pd.DataFrame(results).sort_values("delta_vs_v15", ascending=False).head(12).to_string(index=False))
