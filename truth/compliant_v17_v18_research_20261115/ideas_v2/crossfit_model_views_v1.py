from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
PHASE33_DIR = Path(r"D:\gci\truth\experiments\phase33_rules_first_features")
sys.path.insert(0, str(PHASE33_DIR))
import run_phase33_rules_first_features as phase33


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE_OOF = (
    ROOT
    / "ideas_v1"
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "crossfit_model_views_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_FOLDS = 3
VIEW_VARIANTS = ("base", "financial", "peer", "keys_frequency")
META_VARIANT = "all_safe"
BASELINE_COLUMN = "candidate_te20_blend"
BLEND_WEIGHT = 0.10
PROMOTION_DELTA = 0.005
MODEL_TREES = 600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alignment(
    train: pd.DataFrame, baseline: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with train.csv")
    if BASELINE_COLUMN not in baseline or "fold" not in baseline:
        raise ValueError("Baseline OOF file is missing required columns")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_rows) in enumerate(
        splitter.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Baseline OOF fold assignment differs from the declared split")
    return y, fold_ids


def make_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=MODEL_TREES,
        num_leaves=63,
        min_child_samples=100,
        learning_rate=0.03,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=127,
        device_type="cpu",
        n_jobs=8,
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )


def to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite score vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    if not lgb.__version__:
        raise RuntimeError("Unable to determine LightGBM version")
    for path in (TRAIN, BASELINE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    y, fold_ids = validate_alignment(train, baseline)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_scores).all():
        raise ValueError("Baseline OOF predictions contain non-finite scores")

    blocks = phase33.build_safe_blocks(raw)
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []

    for fold in range(1, FOLDS + 1):
        fold_started = time.perf_counter()
        fit_rows = np.flatnonzero(fold_ids != fold)
        valid_rows = np.flatnonzero(fold_ids == fold)
        inner_splitter = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=SEED + fold * 1000,
        )
        inner_splits = list(
            inner_splitter.split(np.zeros(len(fit_rows), dtype=np.int8), y[fit_rows])
        )
        fit_views = np.full(
            (len(fit_rows), len(VIEW_VARIANTS)), np.nan, dtype=np.float32
        )
        valid_view_logits = np.zeros(
            (len(valid_rows), len(VIEW_VARIANTS)), dtype=np.float64
        )

        for inner_number, (train_local, valid_local) in enumerate(
            inner_splits, start=1
        ):
            inner_fit = fit_rows[train_local]
            inner_valid = fit_rows[valid_local]
            combined_apply = np.concatenate((inner_valid, valid_rows))
            for view_index, variant in enumerate(VIEW_VARIANTS):
                x_inner_fit, x_combined = phase33.make_matrices(
                    raw,
                    blocks,
                    inner_fit,
                    combined_apply,
                    variant,
                )
                model = make_model(SEED + fold * 10000 + inner_number * 100 + view_index)
                model.fit(x_inner_fit, y[inner_fit])
                combined_logits = to_logit(
                    model.predict_proba(x_combined)[:, 1]
                )
                fit_views[valid_local, view_index] = combined_logits[: len(inner_valid)]
                valid_view_logits[:, view_index] += (
                    combined_logits[len(inner_valid) :] / INNER_FOLDS
                )
                del model, x_inner_fit, x_combined
            print(
                f"fold={fold} inner={inner_number}/{INNER_FOLDS} views complete",
                flush=True,
            )

        if not np.isfinite(fit_views).all() or not np.isfinite(
            valid_view_logits
        ).all():
            raise ValueError(f"Fold {fold}: cross-fitted view features are incomplete")

        x_meta_fit, x_meta_valid = phase33.make_matrices(
            raw,
            blocks,
            fit_rows,
            valid_rows,
            META_VARIANT,
        )
        meta_fit = hstack(
            (x_meta_fit, csr_matrix(fit_views, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        meta_valid = hstack(
            (x_meta_valid, csr_matrix(valid_view_logits, dtype=np.float32)),
            format="csr",
            dtype=np.float32,
        )
        meta_model = make_model(SEED + fold * 100)
        meta_model.fit(meta_fit, y[fit_rows])
        prediction = meta_model.predict_proba(meta_valid)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"Fold {fold}: model-view expert scores are non-finite")
        expert_oof[valid_rows] = prediction.astype(np.float32)

        baseline_rank = fold_rank(baseline_scores[valid_rows])
        expert_rank = fold_rank(prediction)
        blend_oof[valid_rows] = (
            (1.0 - BLEND_WEIGHT) * baseline_rank
            + BLEND_WEIGHT * expert_rank
        ).astype(np.float32)
        baseline_auc = float(roc_auc_score(y[valid_rows], baseline_scores[valid_rows]))
        expert_auc = float(roc_auc_score(y[valid_rows], prediction))
        blend_auc = float(roc_auc_score(y[valid_rows], blend_oof[valid_rows]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": baseline_auc,
                "model_view_expert_auc": expert_auc,
                "fixed_90_10_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - baseline_auc,
                "blend_delta": blend_auc - baseline_auc,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "view_variants": "|".join(VIEW_VARIANTS),
                "meta_variant": META_VARIANT,
                "feature_count": int(meta_fit.shape[1]),
                "runtime_seconds": time.perf_counter() - fold_started,
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={baseline_auc:.6f} "
            f"model_view_expert={expert_auc:.6f} blend={blend_auc:.6f} "
            f"seconds={records[-1]['runtime_seconds']:.1f}",
            flush=True,
        )
        del meta_model, meta_fit, meta_valid, x_meta_fit, x_meta_valid

    if not np.isfinite(expert_oof).all() or not np.isfinite(blend_oof).all():
        raise ValueError("Outer OOF predictions are incomplete")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            BASELINE_COLUMN: baseline_scores,
            "crossfit_model_view_expert": expert_oof,
            "fixed_90_10_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["model_view_expert_auc"] for row in records]))
    blend_mean = float(
        np.mean([row["fixed_90_10_rank_blend_auc"] for row in records])
    )
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated exploratory model-view stack; not an independent estimate",
        "hypothesis": (
            "Cross-fitted supervised scores from distinct rule-first feature "
            "views may provide complementary ranking information to a "
            "fold-safe all-feature meta learner."
        ),
        "view_variants": list(VIEW_VARIANTS),
        "meta_variant": META_VARIANT,
        "inner_crossfit_folds": INNER_FOLDS,
        "outer_folds": FOLDS,
        "seed": SEED,
        "model": {
            "family": "LightGBM",
            "trees": MODEL_TREES,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "max_bin": 127,
        },
        "fixed_blend": (
            f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{BLEND_WEIGHT:.2f} model-view expert within-fold rank"
        ),
        "minimum_promotion_gain": PROMOTION_DELTA,
        "baseline_mean_fold_auc": baseline_mean,
        "expert_mean_fold_auc": expert_mean,
        "blend_mean_fold_auc": blend_mean,
        "blend_mean_fold_gain": gain,
        "positive_blend_folds": int(sum(row["blend_delta"] > 0 for row in records)),
        "fold_metrics": records,
        "promotion_gate_passed": bool(
            gain >= PROMOTION_DELTA
            and all(row["blend_delta"] > 0 for row in records)
        ),
        "submission_generated": False,
        "versions": {
            "python": sys.version.split()[0],
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "training_labels": (
                "used only for supervised model fitting within the relevant "
                "inner/outer training partitions"
            ),
            "view_scores": (
                "training-row view scores are inner out-of-fold; outer-validation "
                "view scores use models trained only on outer-fit rows"
            ),
            "feature_views": (
                "existing phase33 base, standard financial ratios, peer features, "
                "rounded ratio keys/frequencies, and all_safe meta features"
            ),
            "external_data": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only for alignment, fold checks, and benchmark comparison"
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                TRAIN,
                BASELINE_OOF,
                Path(__file__).resolve(),
                oof_path,
                fold_path,
            )
        },
    }
    (OUTPUT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
