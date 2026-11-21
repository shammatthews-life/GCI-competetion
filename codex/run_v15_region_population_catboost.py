"""Test REGION_POPULATION_RELATIVE as a CatBoost categorical signal.

This is based on the Home Aloan winner's reported REGION_ID_POPULATION
categorical feature. It starts from the strongest recent matched CA/EXT
CatBoost candidate and adds only the supplied regional population field as a
native category. All scoring uses the original v15 seed-42 held-out folds.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
CACHE = HERE / "v15_cache" / "gci_cache_final"
OUT = HERE / "v15_region_population_catboost"
OUT.mkdir(exist_ok=True)


def rank(x: np.ndarray) -> np.ndarray:
    return rankdata(x, method="average") / len(x)


def main() -> None:
    t0 = time.time()
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    read = lambda i: "".join(nb["cells"][i].get("source", []))
    ns = {"__name__": "__v15_region_population_catboost__"}
    setup = read(1)
    old = ('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
           'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
           'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    if old not in setup:
        raise RuntimeError("Could not safely redirect notebook cache paths")
    new = f'base_dir = {str(HERE)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)'
    exec(compile(setup.replace(old, new), str(NOTEBOOK) + "#cell1", "exec"), ns)
    ns["CACHE_DIR"], ns["OUT_DIR"], ns["THREADS"] = str(CACHE), str(OUT), 6
    exec(compile(read(2), str(NOTEBOOK) + "#cell2", "exec"), ns)

    al = ns["al"]
    y = np.asarray(ns["y"], dtype=np.int8)
    n, ntr = len(y), int(ns["ntr"])
    cats = list(ns["CATS"])
    numeric = pd.concat([ns["BASEF"]["Xa"], ns["BASEF"]["Xat"]], ignore_index=True)
    cb = numeric.drop(columns=cats).reset_index(drop=True).copy()
    source_cats = al[cats].copy().reset_index(drop=True)
    for col in cats:
        source_cats[col] = source_cats[col].astype("string").fillna("__MISSING__").astype(str)
    cb = pd.concat([cb, source_cats], axis=1)
    native_cats = cats.copy()
    for key in ("Q3", "Q4", "Q5", "P3", "P4", "P5", "PF4"):
        name = "PROD_" + key
        cb[name] = ns["KS"][key].reset_index(drop=True).astype("string").fillna("__MISSING__").astype(str)
        native_cats.append(name)
    kd = pd.concat([ns["BASEF"]["Xak"]["K_DEC"], ns["BASEF"]["Xakt"]["K_DEC"]], ignore_index=True)
    cb["PROD_K_DEC"] = kd.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str)
    native_cats.append("PROD_K_DEC")
    cb["PROD_FEE"] = pd.Series(ns["FEE"]).reset_index(drop=True).astype(str)
    native_cats.append("PROD_FEE")

    raw_ca = (al.AMT_CREDIT.fillna(-1).round(0).astype("int64").astype(str) + "|" +
              al.AMT_ANNUITY.fillna(-1).round(0).astype("int64").astype(str))
    cb["PROD_CA"] = raw_ca.astype(str)
    native_cats.append("PROD_CA")
    for ext in ("EXT_SOURCE_1", "EXT_SOURCE_3"):
        train_values = al[ext].iloc[:n].dropna().to_numpy(dtype=float)
        edges = np.unique(np.quantile(train_values, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        bins = pd.Series(pd.cut(al[ext].to_numpy(dtype=float), bins=edges,
                                labels=False, include_lowest=True)).fillna(-1).astype("int8").astype(str)
        col = "PROD_CA_X_" + ext + "_DECILE"
        cb[col] = (raw_ca.astype(str) + "|" + bins.astype(str)).to_numpy()
        native_cats.append(col)

    # Winner-reported regional identifier; preserve its 81 observed values as
    # labels so CatBoost applies ordered target statistics within each fold.
    region = al.REGION_POPULATION_RELATIVE
    cb["REGION_POPULATION_RELATIVE_CAT"] = region.map(
        lambda x: "__MISSING__" if pd.isna(x) else f"{float(x):.6f}"
    ).astype(str)
    native_cats.append("REGION_POPULATION_RELATIVE_CAT")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    candidate_oof = np.zeros(n, dtype=np.float64)
    fold_rows = []
    params = dict(iterations=900, depth=7, learning_rate=.05, loss_function="Logloss",
                  eval_metric="AUC", l2_leaf_reg=8, random_seed=42, thread_count=10,
                  verbose=False, allow_writing_files=False, random_strength=.5,
                  bootstrap_type="Bernoulli", subsample=.8)
    tr = cb.iloc[:ntr].reset_index(drop=True)
    for fi, (fit, valid) in enumerate(folds, 1):
        fold_cache = OUT / f"candidate_catboost_fold_{fi}.npz"
        if fold_cache.exists():
            with np.load(fold_cache) as z:
                candidate_oof[valid] = z["oof"]
        else:
            model = CatBoostClassifier(**params)
            model.fit(tr.iloc[fit], y[fit], cat_features=native_cats)
            candidate_oof[valid] = model.predict_proba(tr.iloc[valid])[:, 1]
            np.savez_compressed(fold_cache, oof=candidate_oof[valid])
        fold_rows.append({"fold": fi, "rows": len(valid),
                          "candidate_catboost_auc": float(roc_auc_score(y[valid], candidate_oof[valid])),
                          "elapsed_seconds": round(time.time() - t0, 1)})
        print(f"regional CatBoost fold {fi}/5 AUC={fold_rows[-1]['candidate_catboost_auc']:.6f}; elapsed={time.time()-t0:.0f}s", flush=True)

    weights = {"qf_shallow": .24, "qf_reg": .12, "v1_allfam_oh": .32,
               "v2_allfam_oh": .16, "v2_mono": .16}
    branches = {}
    for tag in ("old", "e1", "e1et", "e1Aet"):
        pred = np.zeros(n)
        for name, weight in weights.items():
            pred += weight * rank(np.load(CACHE / f"{tag}_{name}.npz")["oof"])
        branches[tag] = pred
    b = (branches["old"] + branches["e1"] + branches["e1et"]) / 3
    main = .75 * rank(b) + .25 * branches["e1Aet"]
    base_cb = np.zeros(n)
    for fi, (_, valid) in enumerate(folds, 1):
        base_cb[valid] = np.load(CACHE / f"catboost_product_fold_{fi}.npz")["oof"]
    base = .90 * rank(main) + .10 * rank(base_cb)
    candidate = .90 * rank(main) + .10 * rank(candidate_oof)
    cached = np.load(HERE / "claude_7_matched_v15_specialist_oof.npz")
    if not np.array_equal(cached["y"], y) or not np.allclose(base, cached["v15_oof"], atol=1e-10):
        raise RuntimeError("Reconstructed v15 formula does not match the cached reference")

    ca_ext = np.load(HERE / "v15_ca_ext_catboost" / "oof.npz")["candidate_catboost_oof"]
    ca_ext_candidate = .90 * rank(main) + .10 * rank(ca_ext)
    for fi, (_, valid) in enumerate(folds):
        fold_rows[fi].update({
            "v15_full_auc": float(roc_auc_score(y[valid], base[valid])),
            "candidate_full_auc": float(roc_auc_score(y[valid], candidate[valid])),
            "delta_vs_v15": float(roc_auc_score(y[valid], candidate[valid]) - roc_auc_score(y[valid], base[valid])),
            "delta_vs_ca_ext": float(roc_auc_score(y[valid], candidate[valid]) - roc_auc_score(y[valid], ca_ext_candidate[valid])),
        })
    auc0, auc1 = float(roc_auc_score(y, base)), float(roc_auc_score(y, candidate))
    result = {
        "hypothesis": "The exact REGION_POPULATION_RELATIVE values behave as regional identifiers; adding this native category to the matched CA/EXT CatBoost candidate improves its complementary ranking signal.",
        "added_native_category": "REGION_POPULATION_RELATIVE_CAT",
        "distinct_region_values": int(region.nunique()),
        "catboost_settings": params,
        "v15_oof_auc": auc0,
        "ca_ext_oof_auc": float(roc_auc_score(y, ca_ext_candidate)),
        "candidate_oof_auc": auc1,
        "delta_vs_v15": auc1 - auc0,
        "delta_vs_ca_ext": float(roc_auc_score(y, candidate) - roc_auc_score(y, ca_ext_candidate)),
        "positive_folds_vs_v15": int(sum(row["delta_vs_v15"] > 0 for row in fold_rows)),
        "positive_folds_vs_ca_ext": int(sum(row["delta_vs_ca_ext"] > 0 for row in fold_rows)),
        "folds": fold_rows,
        "rank_correlation_to_original_catboost": float(spearmanr(candidate_oof, base_cb).statistic),
        "rank_correlation_to_ca_ext_catboost": float(spearmanr(candidate_oof, ca_ext).statistic),
        "elapsed_seconds": round(time.time() - t0, 1),
        "submission_written": False,
        "external_data_used": False,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, ca_ext_oof=ca_ext_candidate,
                        candidate_oof=candidate, candidate_catboost_oof=candidate_oof,
                        original_catboost_oof=base_cb)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
