"""Leakage-safe nearest-peer target-rate specialist, evaluated in the full v15 blend."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v15_peer_knn_experiment"
OUT.mkdir(exist_ok=True)

def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT / "inpuy" / "train.csv")
    y = tr.TARGET.to_numpy(np.int8)
    n = len(y)
    numcols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED",
               "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
               "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "OWN_CAR_AGE"]
    cats = ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE", "NAME_CONTRACT_TYPE", "OCCUPATION_TYPE"]
    Xn0 = tr[numcols].copy()
    Xn0["CREDIT_INCOME"] = tr.AMT_CREDIT / (tr.AMT_INCOME_TOTAL.abs() + 1)
    Xn0["ANNUITY_INCOME"] = tr.AMT_ANNUITY / (tr.AMT_INCOME_TOTAL.abs() + 1)
    Xn0["ANNUITY_CREDIT"] = tr.AMT_ANNUITY / (tr.AMT_CREDIT.abs() + 1)
    Xn0["EMPLOYMENT_AGE"] = tr.DAYS_EMPLOYED / (tr.DAYS_BIRTH.abs() + 1)
    Xn0["AGE_YEARS"] = -tr.DAYS_BIRTH / 365.25
    Xn0["EMPLOYMENT_YEARS"] = -tr.DAYS_EMPLOYED.where(tr.DAYS_EMPLOYED != 365243) / 365.25
    Xn0 = Xn0.replace([np.inf, -np.inf], np.nan)
    Xc0 = tr[cats].astype("string").fillna("__MISSING__")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    peer_oof = np.zeros(n, dtype=float)
    fold_rows = []
    for fi, (fit, valid) in enumerate(folds, 1):
        scaler = RobustScaler(quantile_range=(10, 90))
        xn_fit = scaler.fit_transform(Xn0.iloc[fit].fillna(Xn0.iloc[fit].median()))
        xn_val = scaler.transform(Xn0.iloc[valid].fillna(Xn0.iloc[fit].median()))
        # Distances are averaged across numeric dimensions; category mismatches
        # contribute a modest, fixed penalty via one-hot dimensions.
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        xc_fit = enc.fit_transform(Xc0.iloc[fit])
        xc_val = enc.transform(Xc0.iloc[valid])
        numeric_scale = np.sqrt(xn_fit.shape[1])
        cat_scale = 1.25
        # A fixed stratified reference sample makes the peer search tractable;
        # validation rows and labels remain excluded from this reference set.
        rng = np.random.RandomState(20261021 + fi)
        ref_local = np.concatenate([
            rng.choice(np.flatnonzero(y[fit] == cls),
                       size=min(int(25000 * np.mean(y[fit] == cls)), int(np.sum(y[fit] == cls))),
                       replace=False)
            for cls in (0, 1)
        ])
        ref_local.sort()
        fitmat = hstack([csr_matrix(xn_fit[ref_local] / numeric_scale), xc_fit[ref_local] * cat_scale], format="csr")
        valmat = hstack([csr_matrix(xn_val / numeric_scale), xc_val * cat_scale], format="csr")
        nn = NearestNeighbors(n_neighbors=101, algorithm="brute", metric="cosine", n_jobs=1)
        nn.fit(fitmat)
        dist, ind = nn.kneighbors(valmat, return_distance=True)
        # Smooth inverse-distance peer default rate; exclude no validation labels.
        kdist, kind = dist[:, 1:], ind[:, 1:]
        weights = np.exp(-kdist / 0.15)
        peer_oof[valid] = ((weights * y[fit][ref_local][kind]).sum(axis=1) + 2.0 * y[fit][ref_local].mean()) / (weights.sum(axis=1) + 2.0)
        fold_rows.append({"fold": fi, "rows": len(valid), "peer_auc": float(roc_auc_score(y[valid], peer_oof[valid])),
                          "elapsed_seconds": round(time.time()-t0, 1)})
        print(f"peer kNN fold {fi}/5 AUC={fold_rows[-1]['peer_auc']:.6f}", flush=True)

    cache = np.load(HERE / "claude_7_matched_v15_specialist_oof.npz")
    base, y0 = cache["v15_oof"], cache["y"]
    if not np.array_equal(y, y0):
        raise RuntimeError("v15 OOF rows do not align with train.csv")
    # Fixed 1% specialist rank blend, predeclared as a diagnostic weight.
    candidate = 0.99 * rankdata(base, method="average") / n + 0.01 * rankdata(peer_oof, method="average") / n
    for fi, (_, valid) in enumerate(folds, 1):
        fold_rows[fi-1]["v15_auc"] = float(roc_auc_score(y[valid], base[valid]))
        fold_rows[fi-1]["candidate_auc"] = float(roc_auc_score(y[valid], candidate[valid]))
        fold_rows[fi-1]["delta_auc"] = fold_rows[fi-1]["candidate_auc"] - fold_rows[fi-1]["v15_auc"]
    result = {"hypothesis": "Local peer default rates in a mixed numeric and categorical borrower space add risk signal beyond v15.",
              "features_numeric": numcols + ["CREDIT_INCOME", "ANNUITY_INCOME", "ANNUITY_CREDIT", "EMPLOYMENT_AGE", "AGE_YEARS", "EMPLOYMENT_YEARS"],
              "features_categorical": cats, "reference_rows_per_fold": 25000, "neighbors": 100, "kernel": "exp(-cosine_distance/0.15)",
              "smoothing": "2 prior pseudo-observations at training-fold base rate", "fixed_v15_blend_weight": 0.01,
              "v15_oof_auc": float(roc_auc_score(y, base)), "peer_oof_auc": float(roc_auc_score(y, peer_oof)),
              "candidate_oof_auc": float(roc_auc_score(y, candidate)),
              "delta_auc": float(roc_auc_score(y, candidate)-roc_auc_score(y, base)),
              "positive_folds": int(sum(r["delta_auc"] > 0 for r in fold_rows)), "folds": fold_rows,
              "peer_vs_v15_spearman": float(spearmanr(peer_oof, base).statistic),
              "elapsed_seconds": round(time.time()-t0, 1), "submission_written": False}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=base, peer_oof=peer_oof, candidate_oof=candidate)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
