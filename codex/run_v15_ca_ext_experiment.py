"""Matched full-v15 test of credit-annuity x external-score TE features."""
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
HERE = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
CACHE = HERE / "v15_cache" / "gci_cache_final"
RESID = HERE / "v_3_cache" / "resid16.pkl"
OUT = HERE / "v15_ca_ext_alpha10_experiment"
OUT.mkdir(exist_ok=True)
MEMBER_WEIGHTS = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32,
                  "v2_allfam_oh": .16, "v2_mono": .16}
ALPHA = 10.0


def main():
    t0 = time.time()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    read = lambda i: "".join(nb["cells"][i].get("source", []))
    ns = {"__name__": "__v15_ca_ext_experiment__"}
    setup = read(1)
    old = ('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
           'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
           'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    new = f'base_dir = {str(HERE)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)'
    if old not in setup:
        raise RuntimeError("Could not redirect notebook cache paths")
    exec(compile(setup.replace(old, new), str(NOTEBOOK) + "#cell1", "exec"), ns)
    ns["CACHE_DIR"], ns["OUT_DIR"], ns["THREADS"] = str(CACHE), str(OUT), 3
    exec(compile(read(2), str(NOTEBOOK) + "#cell2", "exec"), ns)
    ns["get_resid"] = lambda seeds, name: pd.read_pickle(RESID)
    exec(compile(read(3), str(NOTEBOOK) + "#cell3", "exec"), ns)
    exec(compile(read(4), str(NOTEBOOK) + "#cell4", "exec"), ns)

    al, y = ns["al"], np.asarray(ns["y"], dtype=np.int8)
    n, nt = int(ns["n"]), int(ns["nt"])
    credit = al.AMT_CREDIT.fillna(-1).round(0).astype("int64").astype(str)
    annuity = al.AMT_ANNUITY.fillna(-1).round(0).astype("int64").astype(str)
    ca = credit + "|" + annuity

    # Estimate decile cut points from outer-training covariates only for each
    # fold below. The archive's exploratory deciles were unsupervised, but this
    # stricter construction prevents held-out covariates shaping train features.
    ext_cols = ("EXT_SOURCE_1", "EXT_SOURCE_3")
    keys = {"CA": ca}
    fold_indices = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    fold_bins = {}
    for fi, (fit, valid) in enumerate(fold_indices):
        ext_bins = {}
        for col in ext_cols:
            train_values = al[col].iloc[fit].dropna().to_numpy(dtype=float)
            edges = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            all_values = al[col].to_numpy(dtype=float)
            bins = pd.cut(all_values, bins=edges, labels=False, include_lowest=True)
            ext_bins[col] = pd.Series(bins).fillna(-1).astype("int8").astype(str)
        fold_bins[fi] = ext_bins

    full_ext_bins = {}
    for col in ext_cols:
        vals = al[col].iloc[:n].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(vals, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        full_ext_bins[col] = pd.Series(
            pd.cut(al[col].to_numpy(dtype=float), bins=edges, labels=False, include_lowest=True)
        ).fillna(-1).astype("int8").astype(str)

    # CA-only encoding is identical across folds. Conditional CA x EXT keys
    # have fold-local decile edges; the held-out fold uses training edges.
    def summarize(key_values, fit_idx):
        frame = pd.DataFrame({"key": np.asarray(key_values, dtype=object)[fit_idx], "y": y[fit_idx]})
        stats = frame.groupby("key", sort=False).y.agg(["sum", "count"])
        return stats["sum"], stats["count"], float(y[fit_idx].mean())

    def encode(key_values, fit_idx, query_idx):
        sums, counts, prior = summarize(key_values, fit_idx)
        q = pd.Series(np.asarray(key_values, dtype=object)[query_idx])
        s = q.map(sums).fillna(0).to_numpy(dtype=float)
        c = q.map(counts).fillna(0).to_numpy(dtype=float)
        te = (s + ALPHA * prior) / (c + ALPHA)
        return te.astype(np.float32), np.log1p(c).astype(np.float32)

    def keys_for_fold(fi):
        ext_bins = fold_bins[fi]
        return {"CA": ca.to_numpy(dtype=object),
                "CAxEXT_SOURCE_1": (ca.astype(str) + "|" + ext_bins["EXT_SOURCE_1"].astype(str)).to_numpy(dtype=object),
                "CAxEXT_SOURCE_3": (ca.astype(str) + "|" + ext_bins["EXT_SOURCE_3"].astype(str)).to_numpy(dtype=object)}

    def keys_for_full():
        return {"CA": ca.to_numpy(dtype=object),
                "CAxEXT_SOURCE_1": (ca.astype(str) + "|" + full_ext_bins["EXT_SOURCE_1"].astype(str)).to_numpy(dtype=object),
                "CAxEXT_SOURCE_3": (ca.astype(str) + "|" + full_ext_bins["EXT_SOURCE_3"].astype(str)).to_numpy(dtype=object)}

    original_enc = ns["enc"]
    original_train_enc = ns["train_enc"]

    def enc_ca(fit_idx, query_idx, keys=None):
        # For validation rows, use the matching outer fold's decile boundaries.
        fi = None
        # Fold membership is determined by the query indices; training subsets
        # used inside nested encoding are always contained in one outer fold.
        for j, (outer_fit, outer_valid) in enumerate(fold_indices):
            if (len(query_idx) and np.isin(query_idx, outer_valid).all()) or (
                    len(fit_idx) == len(outer_fit) and np.array_equal(np.sort(fit_idx), np.sort(outer_fit))):
                fi = j
                break
        kd = keys_for_fold(fi) if fi is not None else keys_for_full()
        extras = {}
        for name, key_values in kd.items():
            te, cnt = encode(key_values, fit_idx, query_idx)
            extras[f"TE_{name}"] = te
            extras[f"LOGN_{name}"] = cnt
        return pd.concat([original_enc(fit_idx, query_idx, keys), pd.DataFrame(extras)], axis=1)

    def train_enc_ca(fit_idx):
        base_cols = original_enc(fit_idx[:10], fit_idx[:5]).columns
        base_values = np.zeros((len(fit_idx), len(base_cols)), dtype=np.float32)
        extra_names = [f"{typ}_{key}" for key in ("CA", "CAxEXT_SOURCE_1", "CAxEXT_SOURCE_3")
                       for typ in ("TE", "LOGN")]
        extra_values = np.zeros((len(fit_idx), len(extra_names)), dtype=np.float32)
        # Each candidate member outer fit_idx equals one of the v15 outer fit
        # partitions; fit deciles inside that outer fold from its train rows.
        outer_id = next((j for j, (outer_fit, _) in enumerate(fold_indices)
                         if len(fit_idx) == len(outer_fit) and np.array_equal(np.sort(fit_idx), np.sort(outer_fit))), None)
        if outer_id is None:
            # Full-data refit: fit unsupervised bins on all labeled training rows.
            kd = keys_for_full()
        else:
            kd = keys_for_fold(outer_id)
        inner = StratifiedKFold(5, shuffle=True, random_state=7)
        for ia, ib in inner.split(np.zeros(len(fit_idx)), y[fit_idx]):
            fit, query = fit_idx[ia], fit_idx[ib]
            base_values[ib] = original_enc(fit, query).to_numpy(dtype=np.float32)
            out = []
            for name, key_values in kd.items():
                te, cnt = encode(key_values, fit, query)
                out.extend((te, cnt))
            extra_values[ib] = np.column_stack(out)
        return pd.concat([pd.DataFrame(base_values, columns=base_cols),
                          pd.DataFrame(extra_values, columns=extra_names)], axis=1)

    spec = {}
    for name, (X, Xt, params, rounds, full_rounds, seeds) in ns["SPEC1"].items():
        spec[name] = (X, Xt, {**params, "extra_trees": True}, rounds, full_rounds, seeds)
    ns["enc"], ns["train_enc"] = enc_ca, train_enc_ca
    print("Training matched E1Aet v1_allfam_oh with CA, CA×EXT1-decile, CA×EXT3-decile encodings", flush=True)
    cand_test, cand_oof = ns["run_member"]("v1_allfam_oh", spec, "ca_ext_alpha10_candidate", ns["VIEW_A"])

    rk = lambda x: (rankdata(x, method="average") - 1) / (len(x) - 1)
    def cached(tag, member):
        return np.load(CACHE / f"{tag}_{member}.npz")["oof"].copy()
    def formula(replacement):
        branch = {}
        for tag in ("old", "e1", "e1et", "e1Aet"):
            vals = {}
            for name, weight in MEMBER_WEIGHTS.items():
                arr = replacement if tag == "e1Aet" and name == "v1_allfam_oh" else cached(tag, name)
                vals[name] = rk(arr)
            branch[tag] = sum(MEMBER_WEIGHTS[k] * vals[k] for k in MEMBER_WEIGHTS)
        b = (branch["old"] + branch["e1"] + branch["e1et"]) / 3
        cb = np.zeros(n)
        for fi, (_, valid) in enumerate(fold_indices, 1):
            cb[valid] = np.load(CACHE / f"catboost_product_fold_{fi}.npz")["oof"]
        return .90 * rk(.75 * rk(b) + .25 * branch["e1Aet"]) + .10 * rk(cb)

    candidate = formula(cand_oof)
    audit = np.load(HERE / "claude_7_matched_v15_specialist_oof.npz")
    base, y_audit = audit["v15_oof"].copy(), audit["y"].copy()
    if not np.array_equal(y, y_audit):
        raise RuntimeError("Labels do not align with cached full v15 OOF")
    folds = []
    for fi, (_, valid) in enumerate(fold_indices, 1):
        auc0 = float(roc_auc_score(y[valid], base[valid]))
        auc1 = float(roc_auc_score(y[valid], candidate[valid]))
        folds.append({"fold": fi, "v15_auc": auc0, "candidate_auc": auc1,
                      "delta_auc": auc1-auc0})
    auc0, auc1 = float(roc_auc_score(y, base)), float(roc_auc_score(y, candidate))
    result = {
        "source_screen": "pre info/GCI_CA_EXT_promising_research.zip compact 3-fold screen reported +0.004862; not previously matched to v15",
        "hypothesis": "Rounded exact credit-annuity pairs plus conditional EXT_SOURCE_1/3 decile TE expose applicant risk within repeated financing bundles.",
        "encoding": f"Nested five-fold target encoding inside each outer train split; alpha={ALPHA:g}; validation and test use outer-train statistics; EXT decile boundaries fitted from outer-train covariates only.",
        "features_added": ["TE_CA", "LOGN_CA", "TE_CAxEXT_SOURCE_1", "LOGN_CAxEXT_SOURCE_1", "TE_CAxEXT_SOURCE_3", "LOGN_CAxEXT_SOURCE_3"],
        "alpha": ALPHA,
        "v15_oof_auc": auc0, "candidate_oof_auc": auc1, "delta_auc": auc1-auc0,
        "positive_folds": int(sum(x["delta_auc"] > 0 for x in folds)), "folds": folds,
        "candidate_member_spearman_to_v15_member": float(spearmanr(cand_oof, cached("e1Aet", "v1_allfam_oh")).statistic),
        "elapsed_seconds": round(time.time()-t0, 1),
        "submission_written": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(folds).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, candidate_oof=candidate,
                        candidate_member_oof=cand_oof, candidate_test=cand_test)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
