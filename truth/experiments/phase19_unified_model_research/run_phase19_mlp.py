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
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
sys.path.insert(0, str(PHASE13))
import phase13_features as features


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
OUTER_SEED = 20261113
OUTER_FOLDS = 5
INNER_STOP_FRACTION = 0.10
MAX_EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 2048
SEED_KEYS = (
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


def configure_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("phase19_mlp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    for handler in (
        logging.FileHandler(output_dir / "EXECUTION_LOG.txt", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def feature_frames(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric, categorical, rowwise = features.build_raw_feature_blocks(raw)
    numeric = numeric.copy()
    numeric["DAYS_EMPLOYED"] = rowwise["DAYS_EMPLOYED_CLEAN"]
    decompositions = features.build_label_free_decompositions(raw)
    numeric = pd.concat(
        [
            numeric.reset_index(drop=True),
            rowwise.reset_index(drop=True),
            decompositions.reset_index(drop=True),
        ],
        axis=1,
    )
    keys = features.build_product_keys(raw).loc[:, SEED_KEYS].copy()
    categorical = pd.concat(
        [categorical.reset_index(drop=True), keys.reset_index(drop=True)], axis=1
    )
    return numeric, categorical


def encode_pair(
    numeric: pd.DataFrame,
    categorical: pd.DataFrame,
    fit: np.ndarray,
    apply: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    np.ndarray,
    list[int],
]:
    x_fit_num = numeric.iloc[fit].to_numpy(dtype=np.float64, copy=True)
    x_apply_num = numeric.iloc[apply].to_numpy(dtype=np.float64, copy=True)
    x_fit_num[~np.isfinite(x_fit_num)] = np.nan
    x_apply_num[~np.isfinite(x_apply_num)] = np.nan
    medians = np.nanmedian(x_fit_num, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    x_fit_num = np.where(np.isfinite(x_fit_num), x_fit_num, medians)
    x_apply_num = np.where(np.isfinite(x_apply_num), x_apply_num, medians)
    means = x_fit_num.mean(axis=0)
    scales = x_fit_num.std(axis=0)
    scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
    x_fit_num = ((x_fit_num - means) / scales).astype(np.float32)
    x_apply_num = ((x_apply_num - means) / scales).astype(np.float32)

    fit_categories: list[np.ndarray] = []
    apply_categories: list[np.ndarray] = []
    cardinalities: list[int] = []
    for column in categorical.columns:
        fit_values = categorical.iloc[fit][column].astype(str)
        apply_values = categorical.iloc[apply][column].astype(str)
        vocabulary = pd.Index(fit_values.unique())
        fit_codes = vocabulary.get_indexer(fit_values).astype(np.int64) + 1
        apply_codes = vocabulary.get_indexer(apply_values).astype(np.int64) + 1
        apply_codes[apply_codes < 0] = 0
        fit_categories.append(fit_codes)
        apply_categories.append(apply_codes)
        cardinalities.append(len(vocabulary) + 1)
    return (
        x_fit_num,
        x_apply_num,
        np.stack(fit_categories, axis=1),
        cardinalities,
    ), np.stack(apply_categories, axis=1), cardinalities


class TabularMLP(nn.Module):
    def __init__(self, numeric_count: int, cardinalities: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    cardinality,
                    min(32, max(4, int(round(cardinality**0.25)) * 4)),
                )
                for cardinality in cardinalities
            ]
        )
        embedding_count = sum(embedding.embedding_dim for embedding in self.embeddings)
        self.network = nn.Sequential(
            nn.Linear(numeric_count + embedding_count, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        return self.network(torch.cat([numeric, *embeddings], dim=1)).squeeze(1)


def fit_epochs(
    x_fit_num: np.ndarray,
    x_fit_cat: np.ndarray,
    y_fit: np.ndarray,
    cardinalities: list[int],
    seed: int,
    epochs: int,
    x_valid_num: np.ndarray | None = None,
    x_valid_cat: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
) -> tuple[TabularMLP, int, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    model = TabularMLP(x_fit_num.shape[1], cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.from_numpy(x_fit_num),
        torch.from_numpy(x_fit_cat),
        torch.from_numpy(y_fit.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=True,
    )
    best_auc = -np.inf
    best_epoch = 1
    stale_epochs = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_num, batch_cat, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_num, batch_cat), batch_y)
            loss.backward()
            optimizer.step()
        if x_valid_num is None:
            continue
        model.eval()
        with torch.no_grad():
            valid_logits = model(
                torch.from_numpy(x_valid_num), torch.from_numpy(x_valid_cat)
            )
            valid_predictions = torch.sigmoid(valid_logits).numpy()
        auc = float(roc_auc_score(y_valid, valid_predictions))
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break
    return model, best_epoch, time.perf_counter() - start


def predict(model: TabularMLP, numeric: np.ndarray, categorical: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.from_numpy(numeric), torch.from_numpy(categorical))
        ).numpy()


def run(train_path: Path = TRAIN_PATH, folds_to_run: int = OUTER_FOLDS) -> Path:
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file is missing: {train_path}")
    if folds_to_run != OUTER_FOLDS:
        raise ValueError("Phase 19 MLP runner requires all five outer folds")
    output_dir = ROOT / "runs" / datetime.now(timezone.utc).strftime(
        "phase19_mlp_%Y%m%dT%H%M%SZ"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = configure_logger(output_dir)
    started = time.perf_counter()
    train = pd.read_csv(train_path, low_memory=False)
    y = train[features.TARGET_COLUMN].to_numpy(dtype=np.int8)
    ids = train[features.ID_COLUMN].to_numpy()
    raw = train.drop(columns=[features.TARGET_COLUMN]).reset_index(drop=True)
    numeric, categorical = feature_frames(raw)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    fold_rows: list[dict[str, Any]] = []
    outer = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED)

    for fold, (outer_fit, outer_apply) in enumerate(
        outer.split(np.zeros(len(y), dtype=np.int8), y), start=1
    ):
        logger.info("Starting outer fold %d/%d", fold, OUTER_FOLDS)
        stop_fit, stop_apply = train_test_split(
            outer_fit,
            test_size=INNER_STOP_FRACTION,
            random_state=OUTER_SEED + fold * 100,
            stratify=y[outer_fit],
        )
        stop_features = encode_pair(numeric, categorical, stop_fit, stop_apply)
        x_fit_num, x_stop_num, x_fit_cat, cardinalities = stop_features[0]
        x_stop_cat = stop_features[1]
        stop_model, best_epoch, stop_seconds = fit_epochs(
            x_fit_num,
            x_fit_cat,
            y[stop_fit],
            cardinalities,
            seed=OUTER_SEED + fold * 1000 + 1,
            epochs=MAX_EPOCHS,
            x_valid_num=x_stop_num,
            x_valid_cat=x_stop_cat,
            y_valid=y[stop_apply],
        )
        del stop_model

        final_features = encode_pair(numeric, categorical, outer_fit, outer_apply)
        x_outer_fit_num, x_outer_apply_num, x_outer_fit_cat, final_cards = (
            final_features[0]
        )
        x_outer_apply_cat = final_features[1]
        model, _, fit_seconds = fit_epochs(
            x_outer_fit_num,
            x_outer_fit_cat,
            y[outer_fit],
            final_cards,
            seed=OUTER_SEED + fold * 1000 + 2,
            epochs=best_epoch,
        )
        prediction = predict(model, x_outer_apply_num, x_outer_apply_cat)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Nonfinite MLP predictions in outer fold {fold}")
        oof[outer_apply] = prediction
        fold_ids[outer_apply] = fold
        fold_auc = float(roc_auc_score(y[outer_apply], prediction))
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": int(len(outer_fit)),
                "validation_rows": int(len(outer_apply)),
                "inner_selected_epochs": int(best_epoch),
                "inner_stopping_seconds": float(stop_seconds),
                "outer_refit_seconds": float(fit_seconds),
                "fold_auc": fold_auc,
                "category_cardinalities": json.dumps(final_cards),
            }
        )
        logger.info(
            "outer=%d best_epochs=%d auc=%.6f refit_seconds=%.1f",
            fold,
            best_epoch,
            fold_auc,
            fit_seconds,
        )
        del model

    if not np.isfinite(oof).all() or np.any(fold_ids < 1):
        raise ValueError("OOF coverage is incomplete")
    pooled_auc = float(roc_auc_score(y, oof))
    pd.DataFrame(fold_rows).to_csv(output_dir / "FOLD_METRICS.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            features.ID_COLUMN: ids,
            features.TARGET_COLUMN: y,
            "fold": fold_ids,
            "phase19_mlp_constant_free": oof,
        }
    ).to_csv(output_dir / "OOF_PREDICTIONS.csv", index=False)
    summary = {
        "status": "completed, five-fold label-free neural-family research candidate",
        "run_id": output_dir.name,
        "rows": int(len(y)),
        "outer_folds": OUTER_FOLDS,
        "outer_seed": OUTER_SEED,
        "pooled_oof_auc": pooled_auc,
        "fold_aucs": [row["fold_auc"] for row in fold_rows],
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "cpu_threads": 4,
        "gpu_available": bool(torch.cuda.is_available()),
        "numeric_feature_count": int(numeric.shape[1]),
        "categorical_feature_count": int(categorical.shape[1]),
        "categorical_columns": list(categorical.columns),
        "constant_free_keys_only": list(SEED_KEYS),
        "eligibility": (
            "Track A candidate: supplied columns, fold-fitted numeric imputation/"
            "scaling and category vocabularies, no target encoding or test data. "
            "Derived constant-free keys still need organizer-rule interpretation."
        ),
        "training_sha256": sha256_file(train_path),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Completed MLP pooled OOF AUC=%.6f", pooled_auc)

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
    for path in (train_path, Path(__file__).resolve(), PHASE13 / "phase13_features.py"):
        manifest_rows.append(
            {
                "kind": "input" if path == train_path else "source",
                "file": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
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
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--folds", type=int, default=OUTER_FOLDS)
    args = parser.parse_args()
    print(run(args.train, args.folds))


if __name__ == "__main__":
    main()
