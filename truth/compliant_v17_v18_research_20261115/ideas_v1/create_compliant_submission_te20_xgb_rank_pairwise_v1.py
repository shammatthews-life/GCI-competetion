from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
PHASE33_DIR = Path(r"D:\gci\truth\experiments\phase33_rules_first_features")
PHASE34_DIR = Path(r"D:\gci\truth\experiments\phase34_xgboost_rules_first")
sys.path.insert(0, str(PHASE33_DIR))
sys.path.insert(0, str(PHASE34_DIR))
import run_phase33_rules_first_features as phase33
import run_phase34_xgboost_rules_first as phase34

sys.path.insert(0, str(IDEAS_DIR))
from run_nested_product_te_lgbm_v1 import (
    FOLDS,
    ID,
    INNER_TE_FOLDS,
    SEED,
    TARGET,
    TE_ALPHA,
    TE_KEYS,
    apply_training_te,
    make_sparse_pair,
    nested_training_te,
    new_model as new_te_model,
)

sys.path.insert(0, str(ROOT))
from run_clean_v17v18_catboost_probe import make_features


TRAIN = Path(r"D:\gci\inpuy\train.csv")
TEST = Path(r"D:\gci\inpuy\test.csv")
SAMPLE = Path(r"D:\gci\inpuy\sample_submission.csv")
OUTPUT = IDEAS_DIR / "submissions" / "te20_xgb_rank_pairwise_v1"
TE_WEIGHT = 0.20
PAIRWISE_WEIGHT = 0.10
PAIRWISE_ITERATIONS = 1000
PAIRWISE_PAIRS_PER_SAMPLE = 8
EXPECTED_INPUT_SHA256 = {
    TRAIN: "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    TEST: "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
    SAMPLE: "6e5ad587b7c0b3d9fa39c90ce03e161ff5ef292193e117a6a02b11075dffe7db",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> None:
    expected_submission_columns = [ID, TARGET]
    if list(sample.columns) != expected_submission_columns:
        raise ValueError("Sample submission columns/order are not the required schema")
    if list(test.columns) != [column for column in train.columns if column != TARGET]:
        raise ValueError("Train/test predictor columns or order differ")
    if not train[ID].is_unique or not test[ID].is_unique or not sample[ID].is_unique:
        raise ValueError("Training, test, and sample IDs must be unique")
    if not test[ID].equals(sample[ID]):
        raise ValueError("Test IDs/order do not match the sample submission")
    labels = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(labels).all()
        or not np.isin(labels, (0, 1)).all()
        or set(np.unique(labels)) != {0.0, 1.0}
    ):
        raise ValueError("Training target must contain finite binary labels")


def fold_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Cannot rank a short or non-finite prediction vector")
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def concat_blocks(
    train_blocks: tuple[pd.DataFrame, ...],
    test_blocks: tuple[pd.DataFrame, ...],
) -> tuple[pd.DataFrame, ...]:
    if len(train_blocks) != len(test_blocks):
        raise ValueError("Train/test feature block counts differ")
    combined = []
    for train_block, test_block in zip(train_blocks, test_blocks):
        if not train_block.columns.equals(test_block.columns):
            raise ValueError("Train/test feature block schemas differ")
        combined.append(
            pd.concat([train_block, test_block], ignore_index=True)
        )
    return tuple(combined)


def assert_sparse_equal(left: Any, right: Any, name: str) -> None:
    left = left.tocsr(copy=True)
    right = right.tocsr(copy=True)
    left.sort_indices()
    right.sort_indices()
    same_structure = (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    )
    same_values = same_structure and np.array_equal(
        left.data, right.data, equal_nan=True
    )
    if not same_values:
        raise RuntimeError(f"{name} changed when only test covariates were perturbed")


def test_set_perturbation_check(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    train_blocks: tuple[pd.DataFrame, ...],
) -> None:
    fit_stop = min(3072, max(2, len(raw_train) // 2))
    apply_stop = min(fit_stop + 1024, len(raw_train))
    fit_rows = np.arange(fit_stop, dtype=np.int64)
    apply_rows = np.arange(fit_stop, apply_stop, dtype=np.int64)
    original_blocks = concat_blocks(
        train_blocks, phase33.build_safe_blocks(raw_test)
    )
    x_fit, x_apply = phase33.make_matrices(
        raw_train, original_blocks, fit_rows, apply_rows, "keys_frequency"
    )

    perturbed_test = raw_test.iloc[::-1].reset_index(drop=True).copy()
    perturbed_test["AMT_CREDIT"] = (
        pd.to_numeric(perturbed_test["AMT_CREDIT"], errors="raise").fillna(0)
        + 12345.0
    )
    perturbed_test["CODE_GENDER"] = "__TEST_PERTURBATION__"
    perturbed_blocks = concat_blocks(
        train_blocks, phase33.build_safe_blocks(perturbed_test)
    )
    changed_fit, changed_apply = phase33.make_matrices(
        raw_train, perturbed_blocks, fit_rows, apply_rows, "keys_frequency"
    )
    assert_sparse_equal(x_fit, changed_fit, "Training matrix")
    assert_sparse_equal(x_apply, changed_apply, "Validation matrix")


def validation_label_perturbation_check(
    features: pd.DataFrame,
    y: np.ndarray,
    fit_rows: np.ndarray,
    valid_rows: np.ndarray,
) -> None:
    original = apply_training_te(features, y, fit_rows, valid_rows)
    changed_y = y.copy()
    changed_y[valid_rows] = 1 - changed_y[valid_rows]
    changed = apply_training_te(features, changed_y, fit_rows, valid_rows)
    if not np.array_equal(original, changed):
        raise RuntimeError("Validation labels changed fold-local target encodings")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite submission run: {OUTPUT}")
    for path in (TRAIN, TEST, SAMPLE):
        if not path.is_file():
            raise FileNotFoundError(f"Required official input is missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = EXPECTED_INPUT_SHA256[path]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Official input hash mismatch for {path}: "
                f"expected {expected_hash}, found {actual_hash}"
            )
    if int(get_gpu_device_count()) < 1:
        raise RuntimeError("CatBoost cannot see a CUDA GPU; refusing CPU fallback")
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA; refusing GPU candidate generation")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    test = pd.read_csv(TEST, low_memory=False)
    sample = pd.read_csv(SAMPLE)
    validate_inputs(train, test, sample)
    y = train[TARGET].to_numpy(dtype=np.int8)
    n_train = len(train)
    test_rows = np.arange(n_train, n_train + len(test), dtype=np.int64)
    raw_train = train.drop(columns=[TARGET]).reset_index(drop=True)
    raw_test = test.reset_index(drop=True)

    train_blocks = phase33.build_safe_blocks(raw_train)
    test_blocks = phase33.build_safe_blocks(raw_test)
    blocks = concat_blocks(train_blocks, test_blocks)
    test_set_perturbation_check(raw_train, raw_test, train_blocks)
    cat_train, cat_categories = make_features(raw_train)
    cat_test, test_categories = make_features(raw_test)
    if cat_categories != test_categories or not cat_train.columns.equals(cat_test.columns):
        raise ValueError("Train/test CatBoost feature schemas differ")
    if not np.array_equal(
        cat_train[cat_categories].notna().all(axis=1).to_numpy(),
        np.ones(n_train, dtype=bool),
    ):
        raise ValueError("CatBoost categorical preprocessing left missing strings")

    te_train, te_categories = make_features(raw_train)
    te_test, test_te_categories = make_features(raw_test)
    if te_categories != test_te_categories or not te_train.columns.equals(te_test.columns):
        raise ValueError("Train/test target-encoding feature schemas differ")
    te_features = pd.concat([te_train, te_test], ignore_index=True)
    if set(TE_KEYS).difference(te_features.columns):
        raise ValueError("A target-encoding product key is missing")

    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(split.split(np.zeros(n_train, dtype=np.int8), y))
    validation_label_perturbation_check(
        te_features,
        y,
        splits[0][0],
        splits[0][1],
    )
    oof = {
        name: np.full(n_train, np.nan, dtype=np.float32)
        for name in (
            "phase33_lgbm",
            "phase34_xgboost",
            "clean_catboost",
            "nested_te_lgbm",
            "xgb_rank_pairwise",
        )
    }
    test_predictions = {
        name: np.zeros(len(test), dtype=np.float64) for name in oof
    }
    fold_records: list[dict[str, Any]] = []
    cat_categories_index = [cat_train.columns.get_loc(column) for column in cat_categories]

    for fold, (fit_rows, valid_rows) in enumerate(splits, start=1):
        fold_seed = SEED + fold * 100
        test_apply = test_rows
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.12,
            random_state=fold_seed,
        )
        inner_fit_local, inner_valid_local = next(
            inner_splitter.split(np.zeros(len(fit_rows)), y[fit_rows])
        )
        inner_fit = fit_rows[inner_fit_local]
        inner_valid = fit_rows[inner_valid_local]

        x_inner_fit, x_inner_valid = phase33.make_matrices(
            raw_train, train_blocks, inner_fit, inner_valid, "keys_frequency"
        )
        selector33 = phase33._model(fold_seed, phase33.MAX_ITERATIONS)
        selector33.fit(
            x_inner_fit,
            y[inner_fit],
            eval_X=x_inner_valid,
            eval_y=y[inner_valid],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(phase33.STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        rounds33 = max(
            100, int(selector33.best_iteration_ or phase33.MAX_ITERATIONS)
        )
        del selector33, x_inner_fit, x_inner_valid

        x_fit, x_valid = phase33.make_matrices(
            raw_train, train_blocks, fit_rows, valid_rows, "keys_frequency"
        )
        _, x_test33 = phase33.make_matrices(
            raw_train, blocks, fit_rows, test_apply, "keys_frequency"
        )
        model33 = phase33._model(fold_seed, rounds33)
        model33.fit(x_fit, y[fit_rows])
        prediction33 = model33.predict_proba(x_valid)[:, 1]
        test33 = model33.predict_proba(x_test33)[:, 1]
        oof["phase33_lgbm"][valid_rows] = prediction33.astype(np.float32)
        test_predictions["phase33_lgbm"] += test33 / FOLDS
        del model33, x_fit, x_valid, x_test33

        x_inner_fit, x_inner_valid = phase33.make_matrices(
            raw_train, train_blocks, inner_fit, inner_valid, "keys_frequency"
        )
        selector34 = phase34._model(fold_seed, phase34.MAX_ITERATIONS)
        selector34.fit(
            x_inner_fit,
            y[inner_fit],
            eval_set=[(x_inner_valid, y[inner_valid])],
            verbose=False,
        )
        best_iteration34 = getattr(selector34, "best_iteration", None)
        rounds34 = (
            max(100, int(best_iteration34) + 1)
            if best_iteration34 is not None
            else phase34.MAX_ITERATIONS
        )
        del selector34, x_inner_fit, x_inner_valid

        x_fit, x_valid = phase33.make_matrices(
            raw_train, train_blocks, fit_rows, valid_rows, "keys_frequency"
        )
        _, x_test34 = phase33.make_matrices(
            raw_train, blocks, fit_rows, test_apply, "keys_frequency"
        )
        model34 = phase34._model(fold_seed, rounds34)
        model34.fit(x_fit, y[fit_rows], verbose=False)
        prediction34 = model34.predict_proba(x_valid)[:, 1]
        test34 = model34.predict_proba(x_test34)[:, 1]
        oof["phase34_xgboost"][valid_rows] = prediction34.astype(np.float32)
        test_predictions["phase34_xgboost"] += test34 / FOLDS
        del model34

        model_rank = xgb.XGBRanker(
            n_estimators=PAIRWISE_ITERATIONS,
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
            lambdarank_num_pair_per_sample=PAIRWISE_PAIRS_PER_SAMPLE,
            random_state=fold_seed,
            n_jobs=8,
            verbosity=0,
        )
        model_rank.fit(
            x_fit,
            y[fit_rows],
            qid=np.zeros(len(fit_rows), dtype=np.int32),
            verbose=False,
        )
        prediction_rank = model_rank.predict(x_valid)
        test_rank = model_rank.predict(x_test34)
        if not np.isfinite(prediction_rank).all() or not np.isfinite(test_rank).all():
            raise ValueError(f"Nonfinite pairwise XGBoost prediction in fold {fold}")
        oof["xgb_rank_pairwise"][valid_rows] = prediction_rank.astype(np.float32)
        test_predictions["xgb_rank_pairwise"] += fold_rank(test_rank) / FOLDS
        del model_rank, x_fit, x_valid, x_test34

        cat_model = CatBoostClassifier(
            iterations=700,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.72,
            border_count=128,
            one_hot_max_size=10,
            max_ctr_complexity=1,
            random_seed=SEED + fold,
            allow_writing_files=False,
            verbose=False,
        )
        cat_model.fit(
            cat_train.iloc[fit_rows],
            y[fit_rows],
            cat_features=cat_categories_index,
        )
        prediction_cat = cat_model.predict_proba(
            cat_train.iloc[valid_rows]
        )[:, 1]
        test_cat = cat_model.predict_proba(cat_test)[:, 1]
        oof["clean_catboost"][valid_rows] = prediction_cat.astype(np.float32)
        test_predictions["clean_catboost"] += test_cat / FOLDS
        del cat_model

        inner_te_fit_rows, inner_te_stop_rows = train_test_split(
            fit_rows,
            test_size=0.1,
            random_state=SEED + 700 + fold,
            stratify=y[fit_rows],
        )
        inner_te_fit = nested_training_te(
            te_features, y, inner_te_fit_rows, fold + 100
        )
        inner_te_valid = apply_training_te(
            te_features, y, inner_te_fit_rows, inner_te_stop_rows
        )
        x_inner_fit, x_inner_valid, _ = make_sparse_pair(
            te_features,
            inner_te_fit_rows,
            inner_te_stop_rows,
            inner_te_fit,
            inner_te_valid,
        )
        selector_te = new_te_model(2200, SEED + fold * 100)
        selector_te.fit(
            x_inner_fit,
            y[inner_te_fit_rows],
            eval_set=[(x_inner_valid, y[inner_te_stop_rows])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=120, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        rounds_te = max(100, int(selector_te.best_iteration_ or 2200))
        del selector_te, x_inner_fit, x_inner_valid

        fit_te = nested_training_te(te_features, y, fit_rows, fold)
        valid_te = apply_training_te(te_features, y, fit_rows, valid_rows)
        test_te = apply_training_te(te_features, y, fit_rows, test_apply)
        x_fit, x_valid, _ = make_sparse_pair(
            te_features, fit_rows, valid_rows, fit_te, valid_te
        )
        _, x_test_te, _ = make_sparse_pair(
            te_features, fit_rows, test_apply, fit_te, test_te
        )
        model_te = new_te_model(rounds_te, SEED + fold * 100)
        model_te.fit(x_fit, y[fit_rows])
        prediction_te = model_te.predict_proba(x_valid)[:, 1]
        test_te_prediction = model_te.predict_proba(x_test_te)[:, 1]
        oof["nested_te_lgbm"][valid_rows] = prediction_te.astype(np.float32)
        test_predictions["nested_te_lgbm"] += test_te_prediction / FOLDS
        del model_te, x_fit, x_valid, x_test_te

        fold_records.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_rows)),
                "valid_rows": int(len(valid_rows)),
                "phase33_iterations": rounds33,
                "phase34_iterations": rounds34,
                "pairwise_xgboost_iterations": PAIRWISE_ITERATIONS,
                "catboost_iterations": 700,
                "nested_te_iterations": rounds_te,
                "phase33_auc": float(roc_auc_score(y[valid_rows], prediction33)),
                "phase34_auc": float(roc_auc_score(y[valid_rows], prediction34)),
                "pairwise_xgboost_auc": float(
                    roc_auc_score(y[valid_rows], prediction_rank)
                ),
                "catboost_auc": float(roc_auc_score(y[valid_rows], prediction_cat)),
                "nested_te_auc": float(roc_auc_score(y[valid_rows], prediction_te)),
            }
        )
        print(
            f"fold={fold}/{FOLDS} p33={fold_records[-1]['phase33_auc']:.6f} "
            f"p34={fold_records[-1]['phase34_auc']:.6f} "
            f"cat={fold_records[-1]['catboost_auc']:.6f} "
            f"te={fold_records[-1]['nested_te_auc']:.6f}",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("An OOF component is incomplete or non-finite")
    if any(not np.isfinite(values).all() for values in test_predictions.values()):
        raise ValueError("A test component is incomplete or non-finite")

    fold_ids = np.full(n_train, -1, dtype=np.int8)
    for fold, (_, valid_rows) in enumerate(splits, start=1):
        fold_ids[valid_rows] = fold

    oof_previous = np.full(n_train, np.nan, dtype=np.float64)
    oof_te20 = np.full(n_train, np.nan, dtype=np.float64)
    oof_candidate = np.full(n_train, np.nan, dtype=np.float64)
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        general_probability = 0.5 * oof["phase33_lgbm"][mask] + 0.5 * oof[
            "phase34_xgboost"
        ][mask]
        prior_blend = 0.5 * fold_rank(general_probability) + 0.5 * fold_rank(
            oof["clean_catboost"][mask]
        )
        oof_previous[mask] = prior_blend
        oof_te20[mask] = (
            (1.0 - TE_WEIGHT) * fold_rank(prior_blend)
            + TE_WEIGHT * fold_rank(oof["nested_te_lgbm"][mask])
        )
        oof_candidate[mask] = (
            (1.0 - PAIRWISE_WEIGHT) * fold_rank(oof_te20[mask])
            + PAIRWISE_WEIGHT * fold_rank(oof["xgb_rank_pairwise"][mask])
        )

    fold_auc_records = []
    baseline_fold_auc = []
    candidate_fold_auc = []
    for fold in range(1, FOLDS + 1):
        mask = fold_ids == fold
        base_auc = float(roc_auc_score(y[mask], oof_te20[mask]))
        candidate_auc = float(roc_auc_score(y[mask], oof_candidate[mask]))
        baseline_fold_auc.append(base_auc)
        candidate_fold_auc.append(candidate_auc)
        fold_auc_records.append(
            {
                "fold": fold,
                "te20_baseline_auc": base_auc,
                "candidate_auc": candidate_auc,
                "delta": candidate_auc - base_auc,
            }
        )
    baseline_mean = float(np.mean(baseline_fold_auc))
    candidate_mean = float(np.mean(candidate_fold_auc))
    positive_folds = int(
        sum(candidate > baseline for candidate, baseline in zip(candidate_fold_auc, baseline_fold_auc))
    )
    if candidate_mean <= baseline_mean or positive_folds != FOLDS:
        raise RuntimeError(
            "Recomputed pairwise OOF candidate did not improve over TE20 in all folds; "
            "no submission was written."
        )

    general_test = 0.5 * test_predictions["phase33_lgbm"] + 0.5 * test_predictions[
        "phase34_xgboost"
    ]
    previous_test = (
        0.5 * fold_rank(general_test)
        + 0.5 * fold_rank(test_predictions["clean_catboost"])
    )
    te20_test = (
        (1.0 - TE_WEIGHT) * fold_rank(previous_test)
        + TE_WEIGHT * fold_rank(test_predictions["nested_te_lgbm"])
    )
    final_prediction = (
        (1.0 - PAIRWISE_WEIGHT) * fold_rank(te20_test)
        + PAIRWISE_WEIGHT * fold_rank(test_predictions["xgb_rank_pairwise"])
    )
    if (
        not np.isfinite(final_prediction).all()
        or np.any(final_prediction < 0)
        or np.any(final_prediction > 1)
    ):
        raise ValueError("Final submission predictions are invalid")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    submission_path = OUTPUT / "submission_compliant_te20_xgb_rank_pairwise_v1.csv"
    component_path = OUTPUT / "TEST_COMPONENT_PREDICTIONS.csv"
    oof_path = OUTPUT / "OOF_CANDIDATE_PREDICTIONS.csv"
    fold_path = OUTPUT / "FOLD_METRICS.csv"
    pd.DataFrame({ID: test[ID], TARGET: final_prediction}).to_csv(
        submission_path, index=False
    )
    pd.DataFrame(
        {
            ID: test[ID],
            **test_predictions,
            "previous_aligned_blend": previous_test,
            "candidate_te20_blend": te20_test,
            "candidate_te20_xgb_rank_pairwise_blend": final_prediction,
        }
    ).to_csv(component_path, index=False)
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **oof,
            "previous_aligned_blend": oof_previous,
            "candidate_te20_blend": oof_te20,
            "candidate_te20_xgb_rank_pairwise_blend": oof_candidate,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(fold_auc_records).to_csv(fold_path, index=False)
    if list(pd.read_csv(submission_path, nrows=0).columns) != [ID, TARGET]:
        raise ValueError("Generated submission columns do not match the required schema")
    result = pd.read_csv(submission_path)
    if (
        len(result) != len(test)
        or not result[ID].equals(sample[ID])
        or not result[TARGET].between(0, 1).all()
        or not np.isfinite(result[TARGET]).all()
    ):
        raise ValueError("Generated submission failed row, ID, or value validation")

    source_files = (
        Path(__file__).resolve(),
        ROOT / "run_clean_v17v18_catboost_probe.py",
        IDEAS_DIR / "run_nested_product_te_lgbm_v1.py",
        PHASE33_DIR / "run_phase33_rules_first_features.py",
        PHASE34_DIR / "run_phase34_xgboost_rules_first.py",
        Path(r"D:\gci\truth\experiments\phase13_matched_baseline\phase13_features.py"),
    )
    manifest = {
        "status": "generated; local exploratory validation only; not organizer-certified",
        "submission": str(submission_path),
        "data_scope": "official train.csv and test.csv; sample_submission.csv read only for schema and ID/order validation",
        "rows": int(len(result)),
        "prediction_min": float(result[TARGET].min()),
        "prediction_max": float(result[TARGET].max()),
        "blend": {
            "prior_model_views": "50% within-test rank of Phase 33/34 probability average + 50% within-test rank of clean CatBoost",
            "nested_product_target_encoding_lightgbm_weight": TE_WEIGHT,
            "pairwise_rank_weight": PAIRWISE_WEIGHT,
            "pairwise_ranker": {
                "objective": "XGBoost rank:pairwise",
                "pair_construction": "mean; 8 sampled pairs per sample in one global applicant group per training fold",
                "iterations": PAIRWISE_ITERATIONS,
                "feature_matrix": "Phase 33 key-frequency matrix fit on each training fold",
            },
            "final_rule": "90% rank of the fixed TE20 candidate + 10% rank of the pairwise ranker; the pairwise test rank is averaged across fold models",
            "weights": "fixed and documented; no row-specific routing or prediction edits",
        },
        "validation": {
            "type": "five-fold stratified OOF using seed 20261113; folds shared across components",
            "te20_baseline_mean_fold_auc": baseline_mean,
            "candidate_mean_fold_auc": candidate_mean,
            "mean_fold_auc_gain": candidate_mean - baseline_mean,
            "positive_fold_deltas": positive_folds,
            "fold_metrics": fold_auc_records,
            "weight_selection_caveat": "The TE20 baseline weight was selected on repeatedly examined labels. The 10% pairwise addition was pre-specified from prior pairwise screens, but this comparison is still exploratory and subject to model-selection optimism. The leaderboard score is unknown until submitted by the user.",
        },
        "rule_controls": {
            "external_data": "none",
            "label_transfer": "none; only smoothed category target statistics from the official TARGET column, cross-fitted within training partitions",
            "manual_test_predictions": "none",
            "train_test_statistical_pooling": "none; row-wise features are built separately, CatBoost train/test frames remain separate, and encoders/counts/target maps fit only on training partitions",
            "test_set_perturbation": "passed; changing/reordering test covariates did not change sampled training or validation matrices",
            "validation_label_perturbation": "passed; changing only outer-validation labels did not change the fold-local target encodings",
            "test_set_use": "model inference only; sample file used only to validate output schema and order",
            "reproducibility": "fixed seeds; source/data hashes and package versions recorded",
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "catboost": catboost.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (TRAIN, TEST, SAMPLE, *source_files, submission_path, oof_path, fold_path, component_path)
        },
    }
    manifest_path = OUTPUT / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
