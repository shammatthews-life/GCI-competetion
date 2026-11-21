from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
OUT = ROOT / "experiments" / "phase3d_claude_hypothesis_probes"
PHASE3C = ROOT / "experiments" / "phase3c_model_intelligence"
ID, TARGET = "SK_ID_CURR", "TARGET"


def main() -> None:
    candidate_oof = pd.read_csv(OUT / "FOLLOWUP_OOF_PREDICTIONS.csv")
    prior = pd.read_csv(PHASE3C / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv").set_index(ID)
    prior = prior.loc[candidate_oof[ID]]
    y = candidate_oof[TARGET].to_numpy(dtype=np.int8)
    folds = candidate_oof["fold"].to_numpy()
    cat = prior["catboost__raw_plus_product"].to_numpy(dtype=np.float64)
    lgb = prior["lightgbm__raw_plus_product"].to_numpy(dtype=np.float64)
    selected = {
        "rounded_paid_ratio": candidate_oof[
            "term__total_paid_ratio_rounded_term_hypothesis"
        ].to_numpy(dtype=np.float64),
        "snapped_paid_ratio": candidate_oof[
            "term__total_paid_ratio_snapped_term_hypothesis"
        ].to_numpy(dtype=np.float64),
    }

    blends = {
        "prior_lgb_cat_probability_equal": 0.5 * lgb + 0.5 * cat,
    }
    for key, pred in selected.items():
        blends[f"{key}__cat_probability_equal"] = 0.5 * pred + 0.5 * cat
        blends[f"{key}__cat_rank_equal"] = 0.5 * rankdata(pred) + 0.5 * rankdata(cat)
        blends[f"{key}__prior_lgb_cat_probability_equal"] = (
            0.5 * pred + 0.25 * lgb + 0.25 * cat
        )

    rows = []
    fold_rows = []
    for name, pred in blends.items():
        fold_auc = []
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            auc = float(roc_auc_score(y[mask], pred[mask]))
            fold_auc.append(auc)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": int(fold),
                    "fold_auc": auc,
                    "n_valid": int(mask.sum()),
                }
            )
        base_auc = float(roc_auc_score(y, blends["prior_lgb_cat_probability_equal"]))
        rows.append(
            {
                "candidate": name,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "mean_fold_auc": float(np.mean(fold_auc)),
                "std_fold_auc": float(np.std(fold_auc, ddof=1)),
                "pooled_delta_vs_prior_lgb_cat_probability_equal": float(
                    roc_auc_score(y, pred) - base_auc
                ),
                "paired_fold_delta_vs_prior_lgb_cat_probability_equal": str(
                    [
                        round(
                            fold_auc[index]
                            - roc_auc_score(
                                y[folds == fold],
                                blends["prior_lgb_cat_probability_equal"][folds == fold],
                            ),
                            7,
                        )
                        for index, fold in enumerate(sorted(np.unique(folds)))
                    ]
                ),
                "folds_improved_vs_prior_blend": int(
                    sum(
                        fold_auc[index]
                        > roc_auc_score(
                            y[folds == fold],
                            blends["prior_lgb_cat_probability_equal"][folds == fold],
                        )
                        for index, fold in enumerate(sorted(np.unique(folds)))
                    )
                ),
                "spearman_vs_lgb_prediction": float(spearmanr(pred, lgb).statistic),
                "weight_selection": "fixed equal weights; exploratory after formula probe",
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "TERM_CANDIDATE_FIXED_BLEND_RESULTS.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "TERM_CANDIDATE_FIXED_BLEND_FOLDS.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
