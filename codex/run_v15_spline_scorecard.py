"""Matched five-fold spline scorecard diversity test against v15."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "v15_spline_scorecard"
OUT.mkdir(exist_ok=True)


def rank(x):
    return rankdata(x, method="average") / len(x)


def main():
    import run_claude_5_feature_hypothesis_screen as screen

    t0 = time.time()
    train = pd.read_csv(ROOT / "inpuy" / "train.csv")
    test = pd.read_csv(ROOT / "inpuy" / "test.csv")
    base, _, _, y, n = screen._build_frames(train, test)
    cache = HERE / "v15_cache" / "gci_cache_final"
    # Match the exact submitted champion OOF formula and fold assignment.
    from run_claude_7_matched_v15_specialists import load_v15_oof
    champion, folds = load_v15_oof(y)

    cat_cols = list(base.select_dtypes(include="category").columns)
    num_cols = [c for c in base.columns if c not in cat_cols]
    # Numeric splines add smooth nonlinearity; one-hot categories provide a
    # classic scorecard-like additive view without target-derived features.
    numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear"),
        StandardScaler(with_mean=False),
    )
    categorical = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
    )
    prep = ColumnTransformer(
        [("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        sparse_threshold=1.0,
    )
    model = LogisticRegression(C=0.03, solver="lbfgs", max_iter=1200, tol=5e-4)
    oof = np.zeros(n, dtype=np.float64)
    fold_rows = []
    for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
        xtr = prep.fit_transform(base.iloc[tr_idx])
        xva = prep.transform(base.iloc[va_idx])
        m = LogisticRegression(C=0.03, solver="lbfgs", max_iter=1200, tol=5e-4)
        m.fit(xtr, y[tr_idx])
        oof[va_idx] = m.predict_proba(xva)[:, 1]
        fold_auc = float(roc_auc_score(y[va_idx], oof[va_idx]))
        fold_rows.append({"fold": fi, "rows": len(va_idx), "scorecard_auc": fold_auc})
        print(f"fold {fi}/5 AUC={fold_auc:.6f} matrix={xtr.shape} elapsed={time.time()-t0:.0f}s", flush=True)

    # Blend weight is fixed at 1%; larger weights are sensitivity diagnostics only.
    blend = 0.99 * rank(champion) + 0.01 * rank(oof)
    result = {
        "candidate": "5-knot quantile spline + additive logistic scorecard",
        "rows": int(n), "features_raw": int(base.shape[1]),
        "spline_knots": 5, "logistic_C": 0.03,
        "standalone_oof_auc": float(roc_auc_score(y, oof)),
        "v15_oof_auc": float(roc_auc_score(y, champion)),
        "fixed_1pct_blend_oof_auc": float(roc_auc_score(y, blend)),
        "fixed_1pct_blend_delta": float(roc_auc_score(y, blend)-roc_auc_score(y, champion)),
        "folds": fold_rows,
        "elapsed_seconds": round(time.time()-t0, 1),
        "note": "Reused v15 seed-42 folds; development comparison only; no submission created.",
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    np.savez_compressed(OUT / "oof.npz", y=y, v15_oof=champion, scorecard_oof=oof, blend_1pct_oof=blend)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
