"""Combine three distinct, leakage-safe winner-inspired application signals.

The block contains (1) a 500-neighbor target mean on the winner's four inputs,
(2) a smoothed exact-region target rate, and (3) application inputs divided by
EXT_SOURCE_3. All target-derived features are nested inside each outer fold.
The combined block replaces one E1Aet member; the rest of v15 is fixed.
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
OUT = CODEX / "v15_winner_feature_combo"
OUT.mkdir(exist_ok=True)
K = 500
REGION_ALPHA = 100.0


def rank(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def main() -> None:
    t0 = time.time()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    read = lambda i: "".join(nb["cells"][i].get("source", []))
    ns = {"__name__": "__v15_winner_feature_combo__"}
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
    test = pd.read_csv(ROOT / "inpuy" / "test.csv")
    y = np.asarray(ns["y"], dtype=np.int8)
    n, ntr = len(y), int(ns["ntr"])
    region_keys = train.REGION_POPULATION_RELATIVE.to_numpy()
    neighbor_raw = train[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
    neighbor_raw["CREDIT_ANNUITY_RATIO"] = train.AMT_CREDIT / train.AMT_ANNUITY
    neighbor_x = neighbor_raw.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float64)

    # Reuse the fixed EXT3 quotient block definition from its isolated run.
    al = pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True)
    den = al.EXT_SOURCE_3.where(al.EXT_SOURCE_3 > 0)
    sources = {
        "AMT_INCOME_TOTAL": al.AMT_INCOME_TOTAL, "AMT_CREDIT": al.AMT_CREDIT,
        "AMT_ANNUITY": al.AMT_ANNUITY, "AMT_GOODS_PRICE": al.AMT_GOODS_PRICE,
        "DAYS_BIRTH": al.DAYS_BIRTH,
        "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
        "DAYS_REGISTRATION": al.DAYS_REGISTRATION, "DAYS_ID_PUBLISH": al.DAYS_ID_PUBLISH,
        "DAYS_LAST_PHONE_CHANGE": al.DAYS_LAST_PHONE_CHANGE, "OWN_CAR_AGE": al.OWN_CAR_AGE,
        "CNT_CHILDREN": al.CNT_CHILDREN, "CNT_FAM_MEMBERS": al.CNT_FAM_MEMBERS,
        "REGION_POPULATION_RELATIVE": al.REGION_POPULATION_RELATIVE,
        "REGION_RATING_CLIENT": al.REGION_RATING_CLIENT,
        "AMT_REQ_CREDIT_BUREAU_HOUR": al.AMT_REQ_CREDIT_BUREAU_HOUR,
        "AMT_REQ_CREDIT_BUREAU_MON": al.AMT_REQ_CREDIT_BUREAU_MON,
        "AMT_REQ_CREDIT_BUREAU_QRT": al.AMT_REQ_CREDIT_BUREAU_QRT,
        "AMT_REQ_CREDIT_BUREAU_YEAR": al.AMT_REQ_CREDIT_BUREAU_YEAR,
        "EXT_SOURCE_1": al.EXT_SOURCE_1, "EXT_SOURCE_2": al.EXT_SOURCE_2,
    }
    quotient = pd.DataFrame({f"EXT3_DIV_{name}": (values / den).replace([np.inf, -np.inf], np.nan).astype(np.float32)
                             for name, values in sources.items()})
    quotient_train = quotient.iloc[:ntr].reset_index(drop=True)

    def neighbor_mean(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
        tree = KDTree(neighbor_x[ref], leaf_size=40, metric="euclidean")
        indices = tree.query(neighbor_x[query], k=K, return_distance=False)
        return y[ref][indices].mean(axis=1).astype(np.float32)

    def region_rate(ref: np.ndarray, query: np.ndarray) -> np.ndarray:
        prior = float(y[ref].mean())
        stats = pd.DataFrame({"key": region_keys[ref], "target": y[ref]}).groupby("key").target.agg(["sum", "count"])
        q = pd.Series(region_keys[query])
        count = q.map(stats["count"]).fillna(0).to_numpy(np.float64)
        total = q.map(stats["sum"]).fillna(0).to_numpy(np.float64)
        return ((total + REGION_ALPHA * prior) / (count + REGION_ALPHA)).astype(np.float32)

    spec = {}
    for name, (X, Xt, params, rounds, full_rounds, seeds) in ns["SPEC1"].items():
        spec[name] = (X, Xt, {**params, "extra_trees": True}, rounds, full_rounds, seeds)
    X, Xt, params, rounds, full_rounds, seeds = spec["v1_allfam_oh"]
    X = pd.concat([X.reset_index(drop=True), quotient_train], axis=1)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    view_a = ns["VIEW_A"]
    oof = np.zeros(n, dtype=np.float32)
    fold_rows = []
    lgb = ns["lgb"]
    member_base = np.load(CACHE / "e1Aet_v1_allfam_oh.npz")["oof"]

    for outer_i, (fit, valid) in enumerate(folds, 1):
        inner_oof = np.empty(len(fit), dtype=np.float32)
        inner_region = np.empty(len(fit), dtype=np.float32)
        inner = StratifiedKFold(5, shuffle=True, random_state=20261022 + outer_i)
        for inner_i, (subfit_local, subvalid_local) in enumerate(inner.split(np.zeros(len(fit)), y[fit]), 1):
            subfit, subvalid = fit[subfit_local], fit[subvalid_local]
            inner_oof[subvalid_local] = neighbor_mean(subfit, subvalid)
            inner_region[subvalid_local] = region_rate(subfit, subvalid)
            print(f"outer {outer_i}/5 inner cross-fit {inner_i}/5; elapsed {time.time()-t0:.0f}s", flush=True)
        valid_neighbor = neighbor_mean(fit, valid)
        valid_region = region_rate(fit, valid)
        Xa = pd.concat([
            X.iloc[fit].reset_index(drop=True), ns["train_enc"](fit),
            view_a["tr"][outer_i - 1].reset_index(drop=True),
            pd.Series(inner_oof, name="TARGET_NEIGHBORS_500_MEAN"),
            pd.Series(inner_region, name="TE_REGION_POPULATION_RELATIVE"),
        ], axis=1)
        Xb = pd.concat([
            X.iloc[valid].reset_index(drop=True), ns["enc"](fit, valid),
            view_a["va"][outer_i - 1].reset_index(drop=True),
            pd.Series(valid_neighbor, name="TARGET_NEIGHBORS_500_MEAN"),
            pd.Series(valid_region, name="TE_REGION_POPULATION_RELATIVE"),
        ], axis=1)
        model = lgb.train({**ns["BASE"], **params}, lgb.Dataset(Xa, y[fit]), rounds)
        oof[valid] = model.predict(Xb).astype(np.float32)
        fold_rows.append({"fold": outer_i, "rows": len(valid),
                          "v15_member_auc": float(roc_auc_score(y[valid], member_base[valid])),
                          "candidate_member_auc": float(roc_auc_score(y[valid], oof[valid])),
                          "neighbor_feature_auc": float(roc_auc_score(y[valid], valid_neighbor)),
                          "region_feature_auc": float(roc_auc_score(y[valid], valid_region)),
                          "elapsed_seconds": round(time.time() - t0, 1)})
        print(f"outer {outer_i}/5 member AUC={fold_rows[-1]['candidate_member_auc']:.6f}; elapsed {time.time()-t0:.0f}s", flush=True)

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
        fold_rows[fi]["v15_auc"] = float(roc_auc_score(y[valid], base[valid]))
        fold_rows[fi]["candidate_full_v15_auc"] = float(roc_auc_score(y[valid], candidate[valid]))
        fold_rows[fi]["delta_auc"] = fold_rows[fi]["candidate_full_v15_auc"] - fold_rows[fi]["v15_auc"]

    result = {
        "hypothesis": "The winner's nearest-peer target rate, exact-region risk prior, and EXT_SOURCE_3-normalized application values add complementary signal when used together in the same matched v15 member.",
        "features": {"neighbor_inputs": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT / AMT_ANNUITY"],
                     "neighbors": K, "region_key": "REGION_POPULATION_RELATIVE", "region_smoothing": REGION_ALPHA,
                     "ext3_quotient_count": int(quotient_train.shape[1])},
        "v15_oof_auc": float(roc_auc_score(y, base)),
        "candidate_oof_auc": float(roc_auc_score(y, candidate)),
        "delta_auc": float(roc_auc_score(y, candidate) - roc_auc_score(y, base)),
        "positive_folds": int(sum(r["delta_auc"] > 0 for r in fold_rows)),
        "folds": fold_rows,
        "candidate_member_spearman_to_v15_member": float(spearmanr(oof, member_base).statistic),
        "elapsed_seconds": round(time.time() - t0, 1),
        "submission_written": False,
        "external_data_used": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, candidate_oof=candidate,
                        v15_member_oof=member_base, candidate_member_oof=oof)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
