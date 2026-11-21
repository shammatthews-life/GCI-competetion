from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
PHASE33_DIR = Path(r"D:\gci\truth\experiments\phase33_rules_first_features")
sys.path.insert(0, str(PHASE33_DIR))
import run_phase33_rules_first_features as phase33


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASE_OOF = (
    IDEAS_DIR
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "xgboost_rank_pairwise_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
N_ESTIMATORS = 1000
PAIRS_PER_SAMPLE = 8
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHT = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite prediction vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def validate_alignment(train: pd.DataFrame, baseline: pd.DataFrame) -> np.ndarray:
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Training rows, IDs, labels, and OOF rows do not align")
    if BASELINE_COLUMN not in baseline or "fold" not in baseline:
        raise ValueError("The aligned candidate OOF columns are incomplete")

    y = train[TARGET].to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold, (_, valid_rows) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Saved baseline OOF folds do not match the fixed split")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(y).all():
        raise ValueError("Training labels must be finite and binary")
    return fold_ids


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    for path in (TRAIN, BASE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    if xgb.__version__ != "3.4.1":
        raise RuntimeError(
            f"Expected the previously recorded XGBoost 3.4.1, found {xgb.__version__}"
        )

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASE_OOF)
    fold_ids = validate_alignment(train, baseline)
    y = train[TARGET].to_numpy(dtype=np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    blocks = phase33.build_safe_blocks(raw)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_scores).all():
        raise ValueError("Baseline OOF predictions contain non-finite values")

    ranker_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float64)
    records: list[dict[str, int | float]] = []

    for fold in range(1, FOLDS + 1):
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        if np.intersect1d(fit_rows, valid_rows).size:
            raise RuntimeError(f"Fold {fold} train/validation rows overlap")
        x_fit, x_valid = phase33.make_matrices(
            raw, blocks, fit_rows, valid_rows, "keys_frequency"
        )
        model = xgb.XGBRanker(
            n_estimators=N_ESTIMATORS,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=40,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_lambda=5.0,
            reg_alpha=0.0,
            objective="rank:pairwise",
            tree_method="hist",
            device="cuda",
            lambdarank_pair_method="mean",
            lambdarank_num_pair_per_sample=PAIRS_PER_SAMPLE,
            random_state=SEED + fold,
            n_jobs=8,
            verbosity=0,
        )
        model.fit(
            x_fit,
            y[fit_rows],
            qid=np.zeros(len(fit_rows), dtype=np.int32),
            verbose=False,
        )
        prediction = model.predict(x_valid)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite XGBoost rank prediction in fold {fold}")
        ranker_oof[valid_rows] = prediction.astype(np.float32)

        base_rank = fold_rank(baseline_scores[valid_rows])
        ranker_rank = fold_rank(prediction)
        blend = (1.0 - BLEND_WEIGHT) * base_rank + BLEND_WEIGHT * ranker_rank
        blend_oof[valid_rows] = blend
        base_auc = float(roc_auc_score(y[valid_rows], baseline_scores[valid_rows]))
        ranker_auc = float(roc_auc_score(y[valid_rows], prediction))
        blend_auc = float(roc_auc_score(y[valid_rows], blend))
        records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "baseline_auc": base_auc,
                "rank_pairwise_auc": ranker_auc,
                "blend_auc": blend_auc,
                "rank_pairwise_delta_vs_baseline": ranker_auc - base_auc,
                "blend_delta_vs_baseline": blend_auc - base_auc,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={base_auc:.6f} "
            f"rank_pairwise={ranker_auc:.6f} blend={blend_auc:.6f}",
            flush=True,
        )
        del model, x_fit, x_valid

    if not np.isfinite(ranker_oof).all() or not np.isfinite(blend_oof).all():
        raise ValueError("Pairwise rank OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    folds_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "candidate_te20_blend": baseline_scores,
            "oof_xgboost_rank_pairwise": ranker_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(folds_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    ranker_mean = float(np.mean([row["rank_pairwise_auc"] for row in records]))
    blend_mean = float(np.mean([row["blend_auc"] for row in records]))
    summary = {
        "status": "exploratory fold-isolated AUC-surrogate experiment; not an independent estimate",
        "objective": "XGBoost rank:pairwise (RankNet pairwise logistic) over one global group per training fold",
        "pair_construction": "XGBoost mean strategy; 8 sampled pairs per sample",
        "n_estimators": N_ESTIMATORS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "fixed_blend": f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 rank + {BLEND_WEIGHT:.2f} pairwise rank, normalized within each validation fold",
        "baseline_mean_fold_auc": baseline_mean,
        "rank_pairwise_mean_fold_auc": ranker_mean,
        "fixed_blend_mean_fold_auc": blend_mean,
        "fixed_blend_mean_fold_gain": blend_mean - baseline_mean,
        "positive_blend_fold_deltas": int(
            sum(row["blend_delta_vs_baseline"] > 0 for row in records)
        ),
        "fold_metrics": records,
        "versions": {
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "data": "official train.csv only; test.csv not read",
            "training_labels": "used only for model fitting inside each outer training fold",
            "validation_labels": "used only for AUC reporting; excluded from all training pairs",
            "feature_fitting": "Phase 33 key-frequency matrices fit on each outer training fold and applied to validation rows",
            "test_statistics": "none",
            "external_data": "none",
            "manual_prediction_changes": "none",
            "OOF_baseline": "read only to verify row/fold alignment and evaluate a pre-specified blend; not used as a model feature",
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (TRAIN, BASE_OOF, Path(__file__).resolve(), oof_path, folds_path)
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
