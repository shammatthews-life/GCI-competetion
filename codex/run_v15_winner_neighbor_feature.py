"""Matched v15 test of the Home Aloan 500-neighbor target-rate feature.

The winner-derived neighborhood uses only EXT_SOURCE_1/2/3 and
AMT_CREDIT / AMT_ANNUITY.  Its target means are nested-cross-fitted inside
each outer training fold so held-out labels never enter model features.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree

ROOT = Path(__file__).resolve().parents[1]
CODEX = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
CACHE = CODEX / "v15_cache" / "gci_cache_final"
RESID = CODEX / "v_3_cache" / "resid16.pkl"
OUT = CODEX / "v15_winner_neighbor_feature"
OUT.mkdir(exist_ok=True)
K_NEIGHBORS = 500
INNER_SEED = 20261022


def rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def main() -> None:
    t0 = time.time()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    read = lambda i: "".join(nb["cells"][i].get("source", []))
    ns = {"__name__": "__v15_winner_neighbor_feature__"}
    setup = read(1)
    old = ('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
           'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
           'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    if old not in setup:
        raise RuntimeError("Could not safely redirect notebook cache paths")
    setup = setup.replace(old, f'base_dir = {str(CODEX)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    exec(compile(setup, str(NOTEBOOK) + "#cell1", "exec"), ns)
    ns["CACHE_DIR"], ns["OUT_DIR"], ns["THREADS"] = str(CACHE), str(OUT), 3
    exec(compile(read(2), str(NOTEBOOK) + "#cell2", "exec"), ns)
    ns["get_resid"] = lambda seeds, name: pd.read_pickle(RESID)
    exec(compile(read(3), str(NOTEBOOK) + "#cell3", "exec"), ns)
    exec(compile(read(4), str(NOTEBOOK) + "#cell4", "exec"), ns)

    train = pd.read_csv(ROOT / "inpuy" / "train.csv")
    y = train.TARGET.to_numpy(np.int8)
    n = len(y)
    # Exact four-input definition from the published winner-derived feature.
    raw = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
    raw["CREDIT_ANNUITY_RATIO"] = train.AMT_CREDIT / train.AMT_ANNUITY
    neigh_x = (raw.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64))
    if not np.isfinite(neigh_x).all():
        raise RuntimeError("Neighbor inputs contain non-finite values after zero fill")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))

    def neighbor_rate(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
        tree = KDTree(neigh_x[reference], leaf_size=40, metric="euclidean")
        ids = tree.query(neigh_x[query], k=K_NEIGHBORS, return_distance=False)
        return y[reference][ids].mean(axis=1).astype(np.float32)

    # The fixed 5-fold OOF feature used for full-data fitting, with each row's
    # own label excluded from its neighborhood.
    full_train_neighbor = np.empty(n, dtype=np.float32)
    for fi, (ref, query) in enumerate(folds, 1):
        full_train_neighbor[query] = neighbor_rate(ref, query)
        print(f"full-data neighbor cross-fit {fi}/5; elapsed {time.time()-t0:.0f}s", flush=True)

    # Recover the exact v15 E1Aet member frames, residual features, target
    # encodings, and consistent-view predictions from its source notebook.
    spec = {}
    for name, (X, Xt, params, rounds, full_rounds, seeds) in ns["SPEC1"].items():
        spec[name] = (X, Xt, {**params, "extra_trees": True}, rounds, full_rounds, seeds)
    X, Xt, params, rounds, full_rounds, seeds = spec["v1_allfam_oh"]
    ntr = int(ns["ntr"])
    view_a = ns["VIEW_A"]
    oof = np.zeros(n, dtype=np.float32)
    valid_features_auc = []
    test = pd.read_csv(ROOT / "inpuy" / "test.csv")
    test_raw = test[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
    test_raw["CREDIT_ANNUITY_RATIO"] = test.AMT_CREDIT / test.AMT_ANNUITY
    test_x = test_raw.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)

    lgb = ns["lgb"]
    te_test = np.zeros(len(test), dtype=np.float64)
    fold_rows = []
    for outer_i, (fit, valid) in enumerate(folds, 1):
        inner_oof = np.empty(len(fit), dtype=np.float32)
        inner = StratifiedKFold(5, shuffle=True, random_state=INNER_SEED + outer_i)
        for inner_i, (subfit_local, subvalid_local) in enumerate(inner.split(np.zeros(len(fit)), y[fit]), 1):
            subfit, subvalid = fit[subfit_local], fit[subvalid_local]
            inner_oof[subvalid_local] = neighbor_rate(subfit, subvalid)
            print(f"outer {outer_i}/5 inner neighbor {inner_i}/5; elapsed {time.time()-t0:.0f}s", flush=True)
        valid_neighbor = neighbor_rate(fit, valid)
        test_neighbor = neighbor_rate(fit, np.arange(ntr, ntr + len(test))) if False else None

        # Build the member inputs in the same order as notebook.run_member.
        Xa = pd.concat([
            X.iloc[fit].reset_index(drop=True),
            ns["train_enc"](fit),
            view_a["tr"][outer_i - 1].reset_index(drop=True),
            pd.Series(inner_oof, name="TARGET_NEIGHBORS_500_MEAN"),
        ], axis=1)
        Xb = pd.concat([
            X.iloc[valid].reset_index(drop=True),
            ns["enc"](fit, valid),
            view_a["va"][outer_i - 1].reset_index(drop=True),
            pd.Series(valid_neighbor, name="TARGET_NEIGHBORS_500_MEAN"),
        ], axis=1)
        model = lgb.train({**ns["BASE"], **params}, lgb.Dataset(Xa, y[fit]), rounds)
        oof[valid] = model.predict(Xb).astype(np.float32)
        v15_member = np.load(CACHE / "e1Aet_v1_allfam_oh.npz")["oof"]
        fold_rows.append({
            "fold": outer_i,
            "rows": len(valid),
            "v15_member_auc": float(roc_auc_score(y[valid], v15_member[valid])),
            "candidate_member_auc": float(roc_auc_score(y[valid], oof[valid])),
            "neighbor_feature_auc": float(roc_auc_score(y[valid], valid_neighbor)),
            "elapsed_seconds": round(time.time() - t0, 1),
        })
        print(f"outer {outer_i}/5 member AUC={fold_rows[-1]['candidate_member_auc']:.6f}; elapsed {time.time()-t0:.0f}s", flush=True)

    # Use the production formula with only the E1Aet v1_allfam_oh member
    # replaced. All other cached members, their fixed weights, and the CatBoost
    # leg remain identical to the v15 control.
    member_base = np.load(CACHE / "e1Aet_v1_allfam_oh.npz")["oof"]
    all_oof = {}
    for tag in ("old", "e1", "e1et", "e1Aet"):
        blend = np.zeros(n, dtype=np.float64)
        for member, weight in ns["W"].items():
            arr = oof if tag == "e1Aet" and member == "v1_allfam_oh" else np.load(CACHE / f"{tag}_{member}.npz")["oof"]
            blend += weight * rank(arr)
        all_oof[tag] = blend
    b = (all_oof["old"] + all_oof["e1"] + all_oof["e1et"]) / 3
    base = np.load(CODEX / "claude_7_matched_v15_specialist_oof.npz")["v15_oof"]
    cb = np.zeros(n, dtype=np.float64)
    for fi, (_, valid) in enumerate(folds, 1):
        cb[valid] = np.load(CACHE / f"catboost_product_fold_{fi}.npz")["oof"]
    candidate = .90 * rank(.75 * rank(b) + .25 * all_oof["e1Aet"]) + .10 * rank(cb)

    for fi, (_, valid) in enumerate(folds):
        old_auc = float(roc_auc_score(y[valid], base[valid]))
        new_auc = float(roc_auc_score(y[valid], candidate[valid]))
        fold_rows[fi]["v15_auc"] = old_auc
        fold_rows[fi]["candidate_full_v15_auc"] = new_auc
        fold_rows[fi]["delta_auc"] = new_auc - old_auc

    result = {
        "hypothesis": "A properly cross-fitted replica of the Home Aloan winner's 500-nearest-applicant target mean adds complementary signal to the v15 E1Aet LightGBM member.",
        "source_definition": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT / AMT_ANNUITY"],
        "distance": "raw Euclidean after NaN/inf -> 0, matching the published reference implementation",
        "neighbors": K_NEIGHBORS,
        "crossfit": "outer seed 42; within each outer training fold, 5-fold seed 20261022+outer_fold; validation labels are never searched",
        "v15_oof_auc": float(roc_auc_score(y, base)),
        "candidate_oof_auc": float(roc_auc_score(y, candidate)),
        "delta_auc": float(roc_auc_score(y, candidate) - roc_auc_score(y, base)),
        "positive_folds": int(sum(r["delta_auc"] > 0 for r in fold_rows)),
        "folds": fold_rows,
        "candidate_member_spearman_to_v15_member": float(spearmanr(oof, member_base).statistic),
        "neighbor_feature_oof_auc": float(roc_auc_score(y, full_train_neighbor)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "submission_written": False,
        "external_data_used": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, candidate_oof=candidate,
                        v15_member_oof=member_base, candidate_member_oof=oof,
                        winner_neighbor_mean_full_oof=full_train_neighbor)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
