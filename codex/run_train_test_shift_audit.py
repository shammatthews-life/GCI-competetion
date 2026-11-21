"""Audit train/test covariate shift and target drift across sequential IDs."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "train_test_shift_audit"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(ROOT / "inpuy" / "train.csv")
    test = pd.read_csv(ROOT / "inpuy" / "test.csv")
    y = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    labels = np.r_[np.zeros(len(train), dtype=np.int8), np.ones(len(test), dtype=np.int8)]

    # Predictor-only domain classification, excluding ID and target so that
    # sequential IDs alone cannot trivially identify the test partition.
    raw = pd.concat([
        train.drop(columns=["TARGET", "SK_ID_CURR"]),
        test.drop(columns=["SK_ID_CURR"]),
    ], ignore_index=True)
    categorical = raw.select_dtypes(include="object").columns.tolist()
    X = raw.copy()
    for col in categorical:
        X[col] = X[col].astype("category")
    fold_auc = []
    importances = np.zeros(X.shape[1], dtype=np.float64)
    folds = StratifiedKFold(3, shuffle=True, random_state=741)
    for fi, (a, b) in enumerate(folds.split(X, labels), 1):
        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.035, num_leaves=31,
            min_child_samples=100, colsample_bytree=0.85,
            subsample=0.85, subsample_freq=1, reg_lambda=3,
            objective="binary", verbosity=-1, n_jobs=3,
            random_state=741 + fi,
        )
        model.fit(X.iloc[a], labels[a], categorical_feature=categorical or "auto")
        p = model.predict_proba(X.iloc[b])[:, 1]
        fold_auc.append(float(roc_auc_score(labels[b], p)))
        importances += model.feature_importances_ / 3

    shift_rows = []
    for col in raw.columns:
        a, b = raw.iloc[:len(train)][col], raw.iloc[len(train):][col]
        if col in categorical:
            av = a.fillna("__MISSING__").astype(str).value_counts(normalize=True)
            bv = b.fillna("__MISSING__").astype(str).value_counts(normalize=True)
            cats = av.index.union(bv.index)
            pa, pb = av.reindex(cats, fill_value=0.0), bv.reindex(cats, fill_value=0.0)
            # Jensen-Shannon distance as a bounded categorical shift summary.
            m = (pa + pb) / 2
            eps = 1e-12
            js = 0.5 * np.sum(pa * np.log2((pa + eps) / (m + eps))) + 0.5 * np.sum(pb * np.log2((pb + eps) / (m + eps)))
            shift_rows.append({"feature": col, "kind": "categorical", "shift_stat": float(js)})
        else:
            av, bv = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
            result = ks_2samp(av.dropna(), bv.dropna())
            shift_rows.append({"feature": col, "kind": "numeric", "shift_stat": float(result.statistic),
                               "ks_pvalue": float(result.pvalue),
                               "train_missing_rate": float(av.isna().mean()),
                               "test_missing_rate": float(bv.isna().mean())})
    importance_rows = pd.DataFrame({"feature": X.columns, "gain_count": importances})
    shift = pd.DataFrame(shift_rows)
    shift = shift.merge(importance_rows, on="feature", how="left").sort_values(
        ["gain_count", "shift_stat"], ascending=False
    )

    # Descriptive only: binned target rates along the ordered ID axis.
    order = np.argsort(ids, kind="stable")
    id_bins = np.array_split(order, 20)
    id_rows = []
    for i, idx in enumerate(id_bins, 1):
        id_rows.append({"id_bin": i, "id_min": int(ids[idx].min()), "id_max": int(ids[idx].max()),
                        "rows": int(len(idx)), "target_rate": float(y[idx].mean())})
    target_by_id = pd.DataFrame(id_rows)

    summary = {
        "train_rows": len(train), "test_rows": len(test),
        "id_ranges": {"train": [int(ids.min()), int(ids.max())],
                      "test": [int(test.SK_ID_CURR.min()), int(test.SK_ID_CURR.max())]},
        "target_rate": float(y.mean()),
        "domain_auc_without_id": {
            "folds": fold_auc, "mean": float(np.mean(fold_auc)),
            "note": "AUC near 0.5 implies weak predictor-only train/test separability; excludes ID and TARGET.",
        },
        "top_domain_discriminator_features": importance_rows.sort_values("gain_count", ascending=False).head(15).to_dict("records"),
        "top_univariate_shift_features": shift.sort_values("shift_stat", ascending=False).head(15).to_dict("records"),
        "target_rate_by_train_id_bin": id_rows,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    shift.to_csv(OUT / "feature_shift.csv", index=False)
    target_by_id.to_csv(OUT / "target_by_id_bin.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
