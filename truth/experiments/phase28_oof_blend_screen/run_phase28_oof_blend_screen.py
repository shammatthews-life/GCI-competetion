from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V9_OOF = Path(r"D:\gci\oof_predictions_v9_ironclad_grade.csv")
PHASE15_OOF = (
    REPO
    / "experiments"
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_multiseed_20261105T125826Z"
    / "MULTISEED_OOF_PREDICTIONS.csv"
)
OUTPUT_DIR = ROOT / "runs" / "phase28_equal_blend_20261107"
ID = "SK_ID_CURR"
TARGET = "TARGET"
V9_WEIGHT = 0.5
PHASE15_WEIGHT = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite run: {OUTPUT_DIR}")
    if not V9_OOF.is_file() or not PHASE15_OOF.is_file():
        raise FileNotFoundError("One or more source OOF artifacts are missing")

    v9 = pd.read_csv(V9_OOF)
    phase15 = pd.read_csv(PHASE15_OOF)
    for frame, name in ((v9, "V9"), (phase15, "Phase 15")):
        if not {ID, TARGET}.issubset(frame.columns):
            raise ValueError(f"{name} OOF is missing ID or target columns")
    if not v9[[ID, TARGET]].equals(phase15[[ID, TARGET]]):
        raise ValueError("Source OOF row, ID, or target alignment differs")
    if not v9[ID].is_unique:
        raise ValueError("V9 OOF IDs are not unique")
    if not np.isfinite(v9["oof_ensemble"].to_numpy(dtype=np.float64)).all():
        raise ValueError("V9 OOF ensemble contains non-finite values")
    if not np.isfinite(
        phase15["phase15_three_seed_average"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Phase 15 OOF average contains non-finite values")

    y = v9[TARGET].to_numpy(dtype=np.int8)
    v9_prediction = v9["oof_ensemble"].to_numpy(dtype=np.float64)
    phase15_prediction = phase15[
        "phase15_three_seed_average"
    ].to_numpy(dtype=np.float64)
    blend = (
        V9_WEIGHT * v9_prediction
        + PHASE15_WEIGHT * phase15_prediction
    )
    metrics = {
        "v9_pooled_auc": float(roc_auc_score(y, v9_prediction)),
        "v9_log_loss": float(log_loss(y, v9_prediction)),
        "phase15_three_seed_pooled_auc": float(
            roc_auc_score(y, phase15_prediction)
        ),
        "phase15_three_seed_log_loss": float(log_loss(y, phase15_prediction)),
        "blend_pooled_auc": float(roc_auc_score(y, blend)),
        "blend_log_loss": float(log_loss(y, blend)),
        "blend_delta_vs_v9": float(
            roc_auc_score(y, blend) - roc_auc_score(y, v9_prediction)
        ),
        "blend_delta_vs_phase15": float(
            roc_auc_score(y, blend) - roc_auc_score(y, phase15_prediction)
        ),
        "component_prediction_correlation": float(
            np.corrcoef(v9_prediction, phase15_prediction)[0, 1]
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "PHASE28_OOF_BLEND.csv"
    output = pd.DataFrame(
        {
            ID: v9[ID],
            TARGET: y,
            "oof_v9_ensemble": v9_prediction,
            "oof_phase15_three_seed_average": phase15_prediction,
            "oof_equal_blend": blend,
        }
    )
    output.to_csv(output_path, index=False)
    summary_path = OUTPUT_DIR / "PHASE28_SUMMARY.json"
    summary = {
        "status": (
            "exploratory OOF diagnostic; post-hoc system comparison and "
            "repeated-label selection risk apply"
        ),
        "use": "OOF analysis only; not a test submission",
        "data_scope": "existing training-only OOF artifacts",
        "weights": {
            "v9": V9_WEIGHT,
            "phase15_three_seed_average": PHASE15_WEIGHT,
        },
        "rows": int(len(y)),
        "id_target_alignment": "exact",
        "phase15_oof_semantics": (
            "Each row's component predictions are out-of-fold; the three "
            "seed vectors use different fold assignments."
        ),
        "rule_status": (
            "Not a compliance determination. V9 target encodings and empirical "
            "feature hypotheses, and Phase 15 native categorical statistics/"
            "product-key hypotheses, have unresolved local rule interpretation."
        ),
        "limits": [
            "The blend was assessed on labels already used in prior research.",
            "This is not independent validation and may overstate generalization.",
            "No Phase 15 test predictions are available in these OOF artifacts.",
            "Do not create a competition submission directly from OOF predictions.",
        ],
        "metrics": metrics,
        "sha256": {
            str(path): sha256_file(path)
            for path in (V9_OOF, PHASE15_OOF, output_path)
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    run()
