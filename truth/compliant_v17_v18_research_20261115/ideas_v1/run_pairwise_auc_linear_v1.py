from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE = ROOT / "runs" / "fold_aligned_catboost_blend_v1" / "OOF_PREDICTIONS.csv"
OUTPUT = ROOT / "runs" / "pairwise_auc_linear_v1"
EXPECTED_TRAIN_SHA256 = "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1"
SEED = 20261113
FOLDS = 5
PAIR_SAMPLES = 100_000
EXPERT_WEIGHT = 0.10
ID = "SK_ID_CURR"
TARGET = "TARGET"
BASELINE_PREDICTION = "fixed_50_50_within_fold_rank_blend"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = list(
        features.select_dtypes(include=["object", "category", "string"]).columns
    )
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(with_mean=False),
    )
    categorical_pipeline = make_pipeline(
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=2,
            dtype=np.float32,
        )
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def sampled_pair_differences(
    matrix: csr_matrix,
    labels: np.ndarray,
    pair_count: int,
    seed: int,
) -> tuple[csr_matrix, np.ndarray]:
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Pairwise training requires both target classes")
    rng = np.random.default_rng(seed)
    positive_rows = rng.choice(positive, size=pair_count, replace=True)
    negative_rows = rng.choice(negative, size=pair_count, replace=True)
    positive_minus_negative = (
        matrix[positive_rows] - matrix[negative_rows]
    ).tocsr()
    pair_matrix = vstack(
        (positive_minus_negative, -positive_minus_negative),
        format="csr",
        dtype=np.float32,
    )
    pair_labels = np.concatenate(
        (
            np.ones(pair_count, dtype=np.int8),
            np.zeros(pair_count, dtype=np.int8),
        )
    )
    return pair_matrix, pair_labels


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite run: {OUTPUT}")
    if not TRAIN.is_file() or not BASELINE.is_file():
        raise FileNotFoundError("Required official train or aligned OOF comparator is missing")
    train_hash = sha256_file(TRAIN)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError(f"Training file hash differs from the verified input: {train_hash}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE, usecols=[ID, TARGET, "fold", BASELINE_PREDICTION])
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with official train")
    y = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Training labels must be binary")

    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    features, _ = make_features(raw)
    categorical = list(
        features.select_dtypes(include=["object", "category", "string"]).columns
    )
    numeric = [column for column in features.columns if column not in categorical]
    features[numeric] = (
        features[numeric]
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float32)
    )

    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(split.split(np.zeros(len(y), dtype=np.int8), y))
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    for fold_number, (_, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
    if not np.array_equal(fold_ids, baseline["fold"].to_numpy(dtype=np.int8)):
        raise ValueError("Comparator fold assignments do not match this experiment")

    predictions = np.full(len(y), np.nan, dtype=np.float32)
    model_records: list[dict[str, int | float]] = []
    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        preprocessor = make_preprocessor(features)
        fit_matrix = preprocessor.fit_transform(features.iloc[fit_rows]).tocsr()
        valid_matrix = preprocessor.transform(features.iloc[valid_rows]).tocsr()
        fit_matrix = fit_matrix.astype(np.float32, copy=False)
        valid_matrix = valid_matrix.astype(np.float32, copy=False)
        pair_matrix, pair_labels = sampled_pair_differences(
            fit_matrix,
            y[fit_rows],
            PAIR_SAMPLES,
            SEED + fold_number,
        )
        ranker = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            fit_intercept=False,
            max_iter=30,
            tol=1e-3,
            average=True,
            random_state=SEED + 10_000 + fold_number,
        )
        ranker.fit(pair_matrix, pair_labels)
        prediction = ranker.decision_function(valid_matrix)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite pairwise score in fold {fold_number}")
        predictions[valid_rows] = prediction.astype(np.float32)
        model_records.append(
            {
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "positive_negative_pairs_sampled": PAIR_SAMPLES,
                "directed_pair_examples": int(len(pair_labels)),
                "features_after_fold_local_encoding": int(fit_matrix.shape[1]),
                "pairwise_auc_surrogate": "mean logistic loss on uniformly sampled positive-negative feature differences",
                "auc_pairwise_linear_model": float(
                    roc_auc_score(y[valid_rows], prediction)
                ),
                "training_pair_matrix_nnz": int(pair_matrix.nnz),
            }
        )
        print(
            f"fold={fold_number}/{FOLDS} "
            f"features={fit_matrix.shape[1]} "
            f"pair_examples={len(pair_labels)} "
            f"AUC={model_records[-1]['auc_pairwise_linear_model']:.6f}",
            flush=True,
        )
        del pair_matrix, pair_labels, ranker, preprocessor, fit_matrix, valid_matrix

    if not np.isfinite(predictions).all():
        raise ValueError("OOF pairwise scores are incomplete or non-finite")
    baseline_scores = baseline[BASELINE_PREDICTION].to_numpy(dtype=np.float64)
    blended_scores = np.full(len(y), np.nan, dtype=np.float64)
    blend_records: list[dict[str, int | float]] = []
    baseline_auc: list[float] = []
    blend_auc: list[float] = []
    for fold_number in range(1, FOLDS + 1):
        valid_rows = np.flatnonzero(fold_ids == fold_number)
        base_rank = fold_rank(baseline_scores[valid_rows])
        pairwise_rank = fold_rank(predictions[valid_rows])
        blended_scores[valid_rows] = (
            (1.0 - EXPERT_WEIGHT) * base_rank + EXPERT_WEIGHT * pairwise_rank
        )
        base_value = float(roc_auc_score(y[valid_rows], baseline_scores[valid_rows]))
        blend_value = float(roc_auc_score(y[valid_rows], blended_scores[valid_rows]))
        baseline_auc.append(base_value)
        blend_auc.append(blend_value)
        blend_records.append(
            {
                "fold": fold_number,
                "baseline_auc": base_value,
                "fixed_90_10_blend_auc": blend_value,
                "delta": blend_value - base_value,
            }
        )

    fold_deltas = np.asarray(blend_auc) - np.asarray(baseline_auc)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    oof_path = OUTPUT / "OOF_PREDICTIONS.csv"
    model_metrics_path = OUTPUT / "MODEL_FOLD_METRICS.csv"
    blend_metrics_path = OUTPUT / "BLEND_FOLD_METRICS.csv"
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            "pairwise_auc_linear": predictions,
            "aligned_baseline": baseline_scores,
            "fixed_90_10_rank_blend": blended_scores,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(model_records).to_csv(model_metrics_path, index=False)
    pd.DataFrame(blend_records).to_csv(blend_metrics_path, index=False)
    manifest = {
        "status": "training-only exploratory OOF experiment",
        "data_scope": "verified official train.csv only; no test or sample-submission rows read",
        "input_sha256": {
            str(TRAIN): train_hash,
            str(BASELINE): sha256_file(BASELINE),
        },
        "method": {
            "objective": "pairwise logistic AUC surrogate",
            "pair_sampling": "uniform positive and negative training-row sampling with replacement inside each outer fit fold",
            "directed_examples": "each sampled positive-minus-negative difference is paired with its negation",
            "preprocessing": "median imputation, missingness indicators, standardization, and one-hot encoding fit on each outer training fold only",
            "pair_samples_per_fold": PAIR_SAMPLES,
            "sgd_alpha": 1e-4,
            "sgd_max_iter": 30,
            "blend_rule": "90% within-fold rank of aligned reference + 10% within-fold rank of pairwise model",
            "blend_weight_selected_on_current_folds": False,
        },
        "validation": {
            "type": "five-fold stratified OOF, seed 20261113",
            "baseline_mean_fold_auc": float(np.mean(baseline_auc)),
            "candidate_mean_fold_auc": float(np.mean(blend_auc)),
            "mean_fold_auc_gain": float(np.mean(blend_auc) - np.mean(baseline_auc)),
            "fold_delta_sample_sd": float(np.std(fold_deltas, ddof=1)),
            "positive_fold_deltas": int(np.sum(fold_deltas > 0)),
            "model_fold_metrics": model_records,
            "blend_fold_metrics": blend_records,
            "selection_note": (
                "Exploratory test on repeatedly examined labels; pairwise loss is a surrogate "
                "for AUC, not the AUC metric itself."
            ),
        },
        "rule_controls": {
            "external_data": "none",
            "matched_label_transfer": "none; pair examples are generated only from training labels within the current fit fold",
            "manual_predictions": "none",
            "cross_dataset_statistics": "none",
            "validation_labels": "not used in preprocessing, pair generation, or fitting",
            "cached_predictions": "aligned OOF used only as a comparator, never as model input",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "output_sha256": {
            str(path): sha256_file(path)
            for path in (oof_path, model_metrics_path, blend_metrics_path)
        },
    }
    manifest_path = OUTPUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
