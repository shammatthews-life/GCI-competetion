"""Matched v15 test of peer-relative feature ranks and robust deviations."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
CODEX = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
CACHE = CODEX / "v15_cache" / "gci_cache_final"
RESID = CODEX / "v_3_cache" / "resid16.pkl"
OUT = CODEX / "v15_peer_rank_experiment"
OUT.mkdir(exist_ok=True)
MEMBER_WEIGHTS = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32,
                  "v2_allfam_oh": .16, "v2_mono": .16}


def main():
    t0 = time.time()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    read = lambda i: "".join(nb["cells"][i].get("source", []))
    ns = {"__name__": "__v15_peer_rank_experiment__"}
    setup = read(1)
    old = ('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
           'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
           'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    new = f'base_dir = {str(CODEX)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)'
    if old not in setup:
        raise RuntimeError("Could not redirect v15 notebook cache paths safely")
    setup = setup.replace(old, new)
    exec(compile(setup, str(NOTEBOOK) + "#cell1", "exec"), ns)
    ns["CACHE_DIR"], ns["OUT_DIR"], ns["THREADS"] = str(CACHE), str(OUT), 3
    exec(compile(read(2), str(NOTEBOOK) + "#cell2", "exec"), ns)
    ns["get_resid"] = lambda seeds, name: pd.read_pickle(RESID)
    exec(compile(read(3), str(NOTEBOOK) + "#cell3", "exec"), ns)
    exec(compile(read(4), str(NOTEBOOK) + "#cell4", "exec"), ns)

    al = ns["al"].copy()
    n, ntr = int(ns["n"]), int(ns["ntr"])
    groups = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE",
              "NAME_EDUCATION_TYPE", "REGION_RATING_CLIENT"]
    raw_values = {
        "LOG_INCOME": np.log1p(al.AMT_INCOME_TOTAL.clip(lower=0)),
        "LOG_CREDIT": np.log1p(al.AMT_CREDIT.clip(lower=0)),
        "LOG_ANNUITY": np.log1p(al.AMT_ANNUITY.clip(lower=0)),
        "EXT1": al.EXT_SOURCE_1, "EXT2": al.EXT_SOURCE_2, "EXT3": al.EXT_SOURCE_3,
        "AGE": -al.DAYS_BIRTH / 365.25,
    }
    peer = {}
    for group in groups:
        key = al[group].astype("string").fillna("__MISSING__")
        for value_name, value in raw_values.items():
            grouped = value.groupby(key, dropna=False)
            count = grouped.transform("count")
            rank = value.groupby(key, dropna=False).rank(method="average", pct=True)
            med = grouped.transform("median")
            mad = value.groupby(key, dropna=False).transform(
                lambda s: (s - s.median()).abs().median()
            )
            robust = (value - med) / (1.4826 * mad.replace(0, np.nan))
            valid = count >= 30
            peer[f"PEER_{group}_{value_name}_RANK"] = rank.where(valid)
            peer[f"PEER_{group}_{value_name}_ROBUST_Z"] = robust.clip(-6, 6).where(valid)
    block = pd.DataFrame(peer, index=al.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    assert block.shape[1] == len(groups) * len(raw_values) * 2

    spec = {}
    for name, (X, Xt, params, rounds, full_rounds, seeds) in ns["SPEC1"].items():
        spec[name] = (X, Xt, {**params, "extra_trees": True}, rounds, full_rounds, seeds)
    X0, Xt0, p0, r0, rf0, seeds0 = spec["v1_allfam_oh"]
    Xp = pd.concat([X0.reset_index(drop=True), block.iloc[:n].reset_index(drop=True)], axis=1)
    Xpt = pd.concat([Xt0.reset_index(drop=True), block.iloc[ntr:].reset_index(drop=True)], axis=1)
    peer_spec = dict(spec)
    peer_spec["v1_allfam_oh"] = (Xp, Xpt, p0, r0, rf0, seeds0)

    print(f"Training E1Aet v1_allfam_oh with {block.shape[1]} peer-relative features", flush=True)
    pred_test, pred_oof = ns["run_member"]("v1_allfam_oh", peer_spec,
                                            "peer_rank_candidate", ns["VIEW_A"])
    y = np.asarray(ns["y"], dtype=np.int8)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    rk = lambda x: (rankdata(x, method="average") - 1) / (len(x) - 1)

    def cache_array(tag, member):
        return np.load(CACHE / f"{tag}_{member}.npz")["oof"].copy()

    def formula(replacement):
        branches = {}
        for tag in ("old", "e1", "e1et", "e1Aet"):
            pred = {}
            for member, weight in MEMBER_WEIGHTS.items():
                arr = replacement if tag == "e1Aet" and member == "v1_allfam_oh" else cache_array(tag, member)
                pred[member] = rk(arr)
            branches[tag] = sum(MEMBER_WEIGHTS[k] * pred[k] for k in MEMBER_WEIGHTS)
        b = (branches["old"] + branches["e1"] + branches["e1et"]) / 3
        cb = np.zeros(n)
        for i, (_, valid) in enumerate(folds, 1):
            z = np.load(CACHE / f"catboost_product_fold_{i}.npz")["oof"]
            cb[valid] = z
        return .90 * rk(.75 * rk(b) + .25 * branches["e1Aet"]) + .10 * rk(cb)

    candidate = formula(pred_oof)
    cached = np.load(CODEX / "claude_7_matched_v15_specialist_oof.npz")
    y0, base = cached["y"].copy(), cached["v15_oof"].copy()
    if not np.array_equal(y, y0):
        raise RuntimeError("V15 OOF label alignment changed")
    folds_out = []
    for fi, (_, valid) in enumerate(folds, 1):
        old_auc = float(roc_auc_score(y[valid], base[valid]))
        new_auc = float(roc_auc_score(y[valid], candidate[valid]))
        folds_out.append({"fold": fi, "v15_auc": old_auc, "candidate_auc": new_auc,
                          "delta_auc": new_auc - old_auc})
    result = {
        "hypothesis": "Within-group percentile ranks and robust deviations add peer context absent from v15 group-mean difference features.",
        "feature_groups": groups, "raw_variables": list(raw_values),
        "features_added_to_one_member": int(block.shape[1]),
        "feature_formula": "within-group percentile rank and median/MAD robust z-score; suppress estimates where group has fewer than 30 nonmissing values; label-free train+test covariate transform",
        "v15_oof_auc": float(roc_auc_score(y, base)),
        "candidate_oof_auc": float(roc_auc_score(y, candidate)),
        "delta_auc": float(roc_auc_score(y, candidate) - roc_auc_score(y, base)),
        "positive_folds": int(sum(row["delta_auc"] > 0 for row in folds_out)),
        "folds": folds_out, "candidate_member_spearman_to_v15_member": float(spearmanr(pred_oof, cache_array("e1Aet", "v1_allfam_oh")).statistic),
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": "Same 5 seed-42 folds and exact fixed v15 formula; transforms use covariates only; no submission created.",
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(folds_out).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, candidate_oof=candidate,
                        v15_member_oof=cache_array("e1Aet", "v1_allfam_oh"), candidate_member_oof=pred_oof,
                        candidate_test=pred_test)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
