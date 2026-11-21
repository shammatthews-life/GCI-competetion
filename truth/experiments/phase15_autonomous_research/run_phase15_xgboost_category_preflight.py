from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PHASE13_DIR = Path(__file__).resolve().parents[1] / "phase13_matched_baseline"
PHASE14_DIR = Path(__file__).resolve().parents[1] / "phase14_label_free_model_development"
sys.path.insert(0, str(PHASE14_DIR))
sys.path.insert(0, str(PHASE13_DIR))
import run_phase13 as phase13
import run_phase14 as phase14


DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
SEED = 20261127
SAMPLE_ROWS = 30000
VALIDATION_SIZE = 0.2
KEY_COLUMNS = (
    "PRODUCT_KEY_Q3",
    "PRODUCT_KEY_Q4",
    "PRODUCT_KEY_Q5",
    "PRODUCT_KEY_AG3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_categories(
    fit: pd.DataFrame,
    apply: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    fit = fit.copy()
    apply = apply.copy()
    support: dict[str, int] = {}
    for column in columns:
        categories = pd.Index(fit[column].astype("string").fillna("MISSING").unique())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        fit_values = fit[column].astype("string").fillna("MISSING")
        apply_values = apply[column].astype("string").fillna("MISSING")
        apply_values = apply_values.where(apply_values.isin(categories), np.nan)
        fit[column] = pd.Categorical(fit_values, dtype=dtype)
        apply[column] = pd.Categorical(apply_values, dtype=dtype)
        support[column] = len(categories)
    return fit, apply, support


def _xgb_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=700,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=8,
        max_cat_threshold=64,
        early_stopping_rounds=60,
        n_jobs=4,
        random_state=seed,
        verbosity=1,
    )


def run(train_path: Path, output_root: Path) -> Path:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime(
        "phase15_xgboost_category_preflight_%Y%m%dT%H%M%SZ"
    )
    output_dir = output_root / run_name
    output_dir.mkdir(parents=False, exist_ok=False)
    log_path = output_dir / "EXECUTION_LOG.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    started = time.perf_counter()
    logging.info("Starting native-categorical XGBoost representative preflight")
    logging.info("Only the supplied train file is read; test/submission excluded")

    raw_train = pd.read_csv(train_path, low_memory=False)
    y_all = raw_train[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    if len(raw_train) < SAMPLE_ROWS:
        raise ValueError(f"Need at least {SAMPLE_ROWS} rows for the benchmark")
    sample_idx, _ = train_test_split(
        np.arange(len(raw_train)),
        train_size=SAMPLE_ROWS,
        stratify=y_all,
        random_state=SEED,
    )
    sample = raw_train.iloc[sample_idx].reset_index(drop=True)
    y = sample[phase13.TARGET_COLUMN].to_numpy(dtype=np.int8)
    fit_idx, apply_idx = train_test_split(
        np.arange(len(sample)),
        test_size=VALIDATION_SIZE,
        stratify=y,
        random_state=SEED + 1,
    )
    raw = sample.drop(columns=[phase13.TARGET_COLUMN])
    numeric, categorical, rowwise = phase13.build_raw_feature_blocks(raw)
    keys = phase13.build_product_keys(raw)
    enhancements = phase14.build_numeric_enhancements(
        raw.drop(columns=[phase13.ID_COLUMN])
    )
    numeric_base = pd.concat(
        [
            numeric.reset_index(drop=True),
            rowwise.reset_index(drop=True),
            enhancements.reset_index(drop=True),
        ],
        axis=1,
    )
    raw_categories = categorical.reset_index(drop=True)
    key_categories = keys.loc[:, KEY_COLUMNS].reset_index(drop=True)
    variants = {
        "xgboost_native_categories_no_product_keys": (
            numeric_base,
            raw_categories,
        ),
        "xgboost_native_categories_four_constant_free_keys": (
            numeric_base,
            pd.concat([raw_categories, key_categories], axis=1),
        ),
    }

    result_rows = []
    prediction_frames = []
    for variant_index, (name, (numeric_features, categories)) in enumerate(
        variants.items(), start=1
    ):
        full = pd.concat(
            [numeric_features, categories],
            axis=1,
        )
        category_columns = tuple(categories.columns)
        fit_frame = full.iloc[fit_idx].reset_index(drop=True)
        apply_frame = full.iloc[apply_idx].reset_index(drop=True)
        fit_frame, apply_frame, support = align_categories(
            fit_frame, apply_frame, category_columns
        )
        model = _xgb_model(SEED + variant_index)
        fit_started = time.perf_counter()
        model.fit(
            fit_frame,
            y[fit_idx],
            eval_set=[(apply_frame, y[apply_idx])],
            verbose=100,
        )
        prediction = model.predict_proba(apply_frame)[:, 1]
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} produced non-finite predictions")
        auc = float(roc_auc_score(y[apply_idx], prediction))
        best_iteration = int(model.best_iteration)
        result_rows.append(
            {
                "variant": name,
                "seed": SEED + variant_index,
                "sample_rows": SAMPLE_ROWS,
                "fit_rows": len(fit_idx),
                "validation_rows": len(apply_idx),
                "auc": auc,
                "best_iteration": best_iteration,
                "fit_seconds": time.perf_counter() - fit_started,
                "numeric_feature_count": int(numeric_features.shape[1]),
                "categorical_feature_count": len(category_columns),
                "fit_category_support": json.dumps(support, sort_keys=True),
                "xgboost_version": xgboost.__version__,
                "tree_method": "hist",
                "device": "CPU",
                "competition_status": (
                    "Track A candidate without keys"
                    if "no_product_keys" in name
                    else "Research-only: key constants 130.5/0.0066 unresolved"
                ),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "sample_row_index": sample_idx[apply_idx],
                    phase13.ID_COLUMN: sample.iloc[apply_idx][
                        phase13.ID_COLUMN
                    ].to_numpy(),
                    phase13.TARGET_COLUMN: y[apply_idx],
                    name: prediction,
                }
            )
        )
        logging.info(
            "%s auc=%.6f best_iteration=%d fit_seconds=%.1f",
            name,
            auc,
            best_iteration,
            time.perf_counter() - fit_started,
        )
        del model

    pd.DataFrame(result_rows).to_csv(output_dir / "PREFLIGHT_RESULTS.csv", index=False)
    predictions = prediction_frames[0].merge(
        prediction_frames[1],
        on=["sample_row_index", phase13.ID_COLUMN, phase13.TARGET_COLUMN],
        validate="one_to_one",
    )
    predictions.to_csv(output_dir / "PREFLIGHT_PREDICTIONS.csv", index=False)
    report = {
        "status": "single representative sample screen; not an OOF model result",
        "train_sha256": sha256_file(train_path),
        "seed": SEED,
        "sample_rows": SAMPLE_ROWS,
        "sampled_row_indices_sha256": hashlib.sha256(
            np.asarray(sample_idx, dtype=np.int64).tobytes()
        ).hexdigest(),
        "fit_rows": len(fit_idx),
        "validation_rows": len(apply_idx),
        "results": result_rows,
        "runtime_seconds": time.perf_counter() - started,
        "categorical_schema_fit_on_fit_rows_only": True,
        "no_test_external_or_submission_data": True,
        "eligibility": (
            "The no-product-key model uses supplied columns and deterministic "
            "rowwise transforms only; this screen does not certify submission "
            "eligibility. The four-key model has unresolved constants."
        ),
    }
    (output_dir / "PREFLIGHT_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    logging.info("Completed in %.1f seconds; run=%s", time.perf_counter() - started, output_dir)
    manifest_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "kind": "output",
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest_rows.append(
        {
            "kind": "input",
            "file": str(train_path),
            "bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
        }
    )
    manifest_rows.append(
        {
            "kind": "source",
            "file": str(Path(__file__).resolve()),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    with (output_dir / "SHA256_MANIFEST.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("kind", "file", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(run(args.train.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
