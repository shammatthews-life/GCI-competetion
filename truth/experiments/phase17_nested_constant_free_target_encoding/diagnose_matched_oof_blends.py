from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
PHASE15 = (
    REPO
    / "experiments"
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE16 = (
    REPO
    / "experiments"
    / "phase16_nested_catboost_research"
    / "runs"
    / "phase16_nested_catboost_20261105T132953Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE17 = (
    ROOT
    / "runs"
    / "phase17_nested_te_20261105T140620Z"
    / "OOF_PREDICTIONS.csv"
)


def main() -> None:
    train = pd.read_csv(TRAIN, usecols=["SK_ID_CURR", "TARGET"])
    frames = {
        "phase15_native_catboost": pd.read_csv(PHASE15),
        "phase16_nested_tuned_catboost": pd.read_csv(PHASE16),
        "phase17_constant_free_nested_te": pd.read_csv(PHASE17),
    }
    y = train["TARGET"].to_numpy(dtype=np.int8)
    ids = train["SK_ID_CURR"].to_numpy()
    folds = frames["phase17_constant_free_nested_te"]["fold"].to_numpy()
    predictions: dict[str, np.ndarray] = {}
    source_columns = {
        "phase15_native_catboost": "native_catboost_phase14_plus_time",
        "phase16_nested_tuned_catboost": "phase16_nested_tuned_catboost",
        "phase17_constant_free_nested_te": "nested_te_constant_free",
    }
    for name, frame in frames.items():
        if not (
            len(frame) == len(train)
            and np.array_equal(frame["row_index"].to_numpy(), np.arange(len(y)))
            and np.array_equal(frame["SK_ID_CURR"].to_numpy(), ids)
            and np.array_equal(frame["TARGET"].to_numpy(), y)
            and np.array_equal(frame["fold"].to_numpy(), folds)
        ):
            raise ValueError(f"{name} OOF rows or fold labels do not match")
        prediction = frame[source_columns[name]].to_numpy(dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} predictions contain non-finite values")
        predictions[name] = prediction

    rows = []
    for name, prediction in predictions.items():
        rows.append(
            {
                "combination": name,
                "weights": "1.0",
                "pooled_auc": float(roc_auc_score(y, prediction)),
                "status": "same-fold OOF single model; labels previously exposed",
            }
        )
    names = list(predictions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            for right_weight in (0.25, 0.5, 0.75):
                prediction = (
                    (1.0 - right_weight) * predictions[left]
                    + right_weight * predictions[right]
                )
                rows.append(
                    {
                        "combination": f"{left} + {right}",
                        "weights": f"{1.0-right_weight:.2f}/{right_weight:.2f}",
                        "pooled_auc": float(roc_auc_score(y, prediction)),
                        "status": (
                            "post-hoc same-fold OOF blend; weight grid not nested; "
                            "not promotion or independent validation"
                        ),
                    }
                )
    rows.append(
        {
            "combination": "equal_three_model_blend",
            "weights": "1/3 each",
            "pooled_auc": float(
                roc_auc_score(y, sum(predictions.values()) / len(predictions))
            ),
            "status": "post-hoc same-fold OOF blend; not validation",
        }
    )
    pd.DataFrame(rows).to_csv(ROOT / "PHASE17_MATCHED_OOF_BLEND_DIAGNOSTICS.csv", index=False)

    correlations = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            correlations.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(
                        pd.Series(predictions[left]).corr(
                            pd.Series(predictions[right]), method="spearman"
                        )
                    ),
                }
            )
    pd.DataFrame(correlations).to_csv(
        ROOT / "PHASE17_MATCHED_OOF_CORRELATIONS.csv", index=False
    )


if __name__ == "__main__":
    main()
