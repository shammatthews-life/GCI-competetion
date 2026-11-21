from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder


IDEAS_DIR = Path(__file__).resolve().parent
ROOT = IDEAS_DIR.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
BASELINE_OOF = (
    ROOT
    / "ideas_v1"
    / "submissions"
    / "product_te_20pct_v2"
    / "OOF_CANDIDATE_PREDICTIONS.csv"
)
OUTPUT = IDEAS_DIR / "runs" / "tabperiodic_swiglu_v1"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 4096
BLEND_WEIGHT = 0.20
PROMOTION_DELTA = 0.005
BASELINE_COLUMN = "candidate_te20_blend"
CATEGORICAL_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    np.divide(numerator, denominator, out=output, where=valid)
    return output


def make_numeric_frame(raw: pd.DataFrame) -> pd.DataFrame:
    numeric = raw.drop(
        columns=[ID, TARGET, *CATEGORICAL_COLUMNS], errors="ignore"
    ).select_dtypes(include=[np.number]).copy()
    credit = pd.to_numeric(raw["AMT_CREDIT"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    annuity = pd.to_numeric(raw["AMT_ANNUITY"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    goods = pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    income = pd.to_numeric(raw["AMT_INCOME_TOTAL"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    family = pd.to_numeric(raw["CNT_FAM_MEMBERS"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(
        dtype=np.float64
    )
    observed = np.isfinite(ext)
    observed_count = observed.sum(axis=1)
    ext_mean = np.divide(
        np.nansum(ext, axis=1),
        observed_count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=observed_count > 0,
    )
    ext_min = np.min(np.where(observed, ext, np.inf), axis=1)
    ext_max = np.max(np.where(observed, ext, -np.inf), axis=1)
    ext_min[observed_count == 0] = np.nan
    ext_max[observed_count == 0] = np.nan
    employment = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    employment[employment == 365243] = np.nan

    engineered = {
        "SAFE_CREDIT_TO_INCOME": safe_ratio(credit, income),
        "SAFE_ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "SAFE_ANNUITY_TO_CREDIT": safe_ratio(annuity, credit),
        "SAFE_GOODS_TO_CREDIT": safe_ratio(goods, credit),
        "SAFE_INCOME_PER_FAMILY": safe_ratio(income, family),
        "SAFE_CREDIT_PER_FAMILY": safe_ratio(credit, family),
        "SAFE_EMPLOYMENT_CLEAN": employment,
        "SAFE_EXT_MEAN": ext_mean,
        "SAFE_EXT_MIN": ext_min,
        "SAFE_EXT_MAX": ext_max,
        "SAFE_EXT_RANGE": ext_max - ext_min,
        "SAFE_EXT_OBSERVED_COUNT": observed_count.astype(np.float64),
    }
    for column, values in engineered.items():
        numeric[column] = values
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(np.float64)
    if numeric.columns.duplicated().any():
        raise ValueError("Numeric feature names are not unique")
    return numeric.reset_index(drop=True)


def fit_preprocessor(
    numeric_frame: pd.DataFrame, raw: pd.DataFrame, fit_rows: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    numeric_fit = numeric_frame.iloc[fit_rows].to_numpy(dtype=np.float64)
    missing_fit = ~np.isfinite(numeric_fit)
    medians = np.zeros(numeric_fit.shape[1], dtype=np.float64)
    for column in range(numeric_fit.shape[1]):
        valid = numeric_fit[np.isfinite(numeric_fit[:, column]), column]
        medians[column] = float(np.median(valid)) if len(valid) else 0.0
    filled_fit = np.where(missing_fit, medians, numeric_fit)
    lower = np.percentile(filled_fit, 0.5, axis=0)
    upper = np.percentile(filled_fit, 99.5, axis=0)
    clipped_fit = np.clip(filled_fit, lower, upper)
    means = clipped_fit.mean(axis=0)
    scales = clipped_fit.std(axis=0)
    scales[scales < 1e-6] = 1.0
    scaled_fit = ((clipped_fit - means) / scales).astype(np.float32)

    categorical_fit = (
        raw.iloc[fit_rows][list(CATEGORICAL_COLUMNS)]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        dtype=np.float32,
        sparse_output=False,
    )
    encoded_fit = encoder.fit_transform(categorical_fit).astype(np.float32)
    preprocessor = {
        "columns": list(numeric_frame.columns),
        "medians": medians,
        "lower": lower,
        "upper": upper,
        "means": means,
        "scales": scales,
        "encoder": encoder,
    }
    return preprocessor, np.column_stack(
        (scaled_fit, missing_fit.astype(np.float32))
    ), encoded_fit


def transform(
    preprocessor: dict[str, Any],
    numeric_frame: pd.DataFrame,
    raw: pd.DataFrame,
    apply_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if list(numeric_frame.columns) != preprocessor["columns"]:
        raise ValueError("Numeric feature schema changed between fit and apply")
    values = numeric_frame.iloc[apply_rows].to_numpy(dtype=np.float64)
    missing = ~np.isfinite(values)
    filled = np.where(missing, preprocessor["medians"], values)
    clipped = np.clip(filled, preprocessor["lower"], preprocessor["upper"])
    scaled = ((clipped - preprocessor["means"]) / preprocessor["scales"]).astype(
        np.float32
    )
    categorical = (
        raw.iloc[apply_rows][list(CATEGORICAL_COLUMNS)]
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )
    encoded = preprocessor["encoder"].transform(categorical).astype(np.float32)
    return np.column_stack((scaled, missing.astype(np.float32))), encoded


class PeriodicEmbedding(nn.Module):
    def __init__(self, numeric_count: int, frequencies: int = 8, sigma: float = 0.12):
        super().__init__()
        self.frequency = nn.Parameter(
            torch.randn(numeric_count, frequencies) * sigma
        )
        self.projection = nn.Linear(numeric_count * frequencies * 2, 128)

    def forward(self, numeric: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * np.pi * numeric.unsqueeze(-1) * self.frequency.unsqueeze(0)
        return self.projection(
            torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(1)
        )


class TabPeriodicSwiGLU(nn.Module):
    def __init__(self, numeric_count: int, mask_count: int, category_count: int):
        super().__init__()
        self.numeric_count = numeric_count
        self.periodic = PeriodicEmbedding(numeric_count)
        self.linear = nn.Linear(numeric_count + mask_count + category_count, 128)
        self.norm = nn.LayerNorm(256)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(256),
                        "gate": nn.Linear(256, 512),
                        "projection": nn.Linear(256, 256),
                        "dropout": nn.Dropout(0.25),
                    }
                )
                for _ in range(2)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        numeric_with_mask: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        numeric = numeric_with_mask[:, : self.numeric_count]
        full = torch.cat((numeric_with_mask, categorical), dim=1)
        hidden = self.norm(
            torch.cat((self.linear(full), self.periodic(numeric)), dim=1)
        )
        for block in self.blocks:
            normalized = block["norm"](hidden)
            gate, value = block["gate"](normalized).chunk(2, dim=1)
            hidden = hidden + 0.5 * block["dropout"](
                block["projection"](F.silu(gate) * value)
            )
        return self.head(hidden).squeeze(1)


def tensors(
    numeric: np.ndarray,
    categorical: np.ndarray,
    labels: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x_numeric = torch.as_tensor(numeric, dtype=torch.float32, device="cuda")
    x_categorical = torch.as_tensor(categorical, dtype=torch.float32, device="cuda")
    y = (
        torch.as_tensor(labels, dtype=torch.float32, device="cuda")
        if labels is not None
        else None
    )
    return x_numeric, x_categorical, y


def fit_epochs(
    x_numeric: torch.Tensor,
    x_categorical: torch.Tensor,
    y: torch.Tensor,
    numeric_count: int,
    mask_count: int,
    seed: int,
    epochs: int,
    x_valid_numeric: torch.Tensor | None = None,
    x_valid_categorical: torch.Tensor | None = None,
    y_valid: np.ndarray | None = None,
) -> tuple[TabPeriodicSwiGLU, int, float | None]:
    set_seeds(seed)
    model = TabPeriodicSwiGLU(
        numeric_count,
        mask_count,
        int(x_categorical.shape[1]),
    ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.025
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    best_epoch = 1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(y), device="cuda")
        for start in range(0, len(y), BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_numeric[batch], x_categorical[batch])
            loss = loss_fn(logits, y[batch])
            loss.backward()
            optimizer.step()
        scheduler.step()

        if x_valid_numeric is not None:
            model.eval()
            with torch.no_grad():
                predictions = []
                for start in range(0, len(x_valid_numeric), 8192):
                    stop = min(start + 8192, len(x_valid_numeric))
                    predictions.append(
                        torch.sigmoid(
                            model(
                                x_valid_numeric[start:stop],
                                x_valid_categorical[start:stop],
                            )
                        )
                        .cpu()
                        .numpy()
                    )
            auc = float(
                roc_auc_score(y_valid, np.concatenate(predictions))
            )
            if auc > best_auc + 1e-6:
                best_auc = auc
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= PATIENCE:
                    break

    return model, best_epoch, best_auc if best_auc >= 0 else None


def predict(
    model: TabPeriodicSwiGLU,
    numeric: np.ndarray,
    categorical: np.ndarray,
) -> np.ndarray:
    model.eval()
    x_numeric, x_categorical, _ = tensors(numeric, categorical)
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_numeric), 8192):
            stop = min(start + 8192, len(x_numeric))
            pieces.append(
                torch.sigmoid(
                    model(x_numeric[start:stop], x_categorical[start:stop])
                )
                .cpu()
                .numpy()
            )
    del x_numeric, x_categorical
    return np.concatenate(pieces)


def fold_rank(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing CPU fallback")
    for path in (TRAIN, BASELINE_OOF):
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")

    started = time.perf_counter()
    train = pd.read_csv(TRAIN, low_memory=False)
    baseline = pd.read_csv(BASELINE_OOF)
    if (
        len(train) != len(baseline)
        or not train[ID].equals(baseline[ID])
        or not train[TARGET].equals(baseline[TARGET])
        or not train[ID].is_unique
        or BASELINE_COLUMN not in baseline
        or "fold" not in baseline
    ):
        raise ValueError("Baseline IDs, labels, or rows do not align with train.csv")

    y = train[TARGET].to_numpy(dtype=np.int8)
    fold_ids = baseline["fold"].to_numpy(dtype=np.int8)
    expected_fold_ids = np.full(len(y), -1, dtype=np.int8)
    outer_splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    outer_splits = list(outer_splitter.split(np.zeros(len(y)), y))
    for fold, (_, valid_rows) in enumerate(outer_splits, start=1):
        expected_fold_ids[valid_rows] = fold
    if not np.array_equal(fold_ids, expected_fold_ids):
        raise ValueError("Baseline OOF fold assignment differs from the declared split")

    missing_categories = set(CATEGORICAL_COLUMNS).difference(train.columns)
    if missing_categories:
        raise ValueError(f"Missing categorical columns: {sorted(missing_categories)}")
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    numeric_frame = make_numeric_frame(raw)
    baseline_scores = baseline[BASELINE_COLUMN].to_numpy(dtype=np.float64)
    expert_oof = np.full(len(y), np.nan, dtype=np.float32)
    blend_oof = np.full(len(y), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    device_name = torch.cuda.get_device_name(0)

    for fold, (outer_fit, outer_valid) in enumerate(outer_splits, start=1):
        seed = SEED + fold * 100
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.12,
            random_state=seed,
        )
        inner_fit_local, inner_valid_local = next(
            inner_splitter.split(np.zeros(len(outer_fit)), y[outer_fit])
        )
        inner_fit = outer_fit[inner_fit_local]
        inner_valid = outer_fit[inner_valid_local]

        inner_preprocessor, inner_fit_numeric, inner_fit_categorical = fit_preprocessor(
            numeric_frame, raw, inner_fit
        )
        inner_valid_numeric, inner_valid_categorical = transform(
            inner_preprocessor, numeric_frame, raw, inner_valid
        )
        x_num, x_cat, y_fit = tensors(
            inner_fit_numeric, inner_fit_categorical, y[inner_fit]
        )
        x_valid_num, x_valid_cat, _ = tensors(
            inner_valid_numeric, inner_valid_categorical
        )
        inner_model, selected_epochs, stopping_auc = fit_epochs(
            x_num,
            x_cat,
            y_fit,
            numeric_count=inner_fit_numeric.shape[1] // 2,
            mask_count=inner_fit_numeric.shape[1] // 2,
            seed=seed,
            epochs=MAX_EPOCHS,
            x_valid_numeric=x_valid_num,
            x_valid_categorical=x_valid_cat,
            y_valid=y[inner_valid],
        )
        del inner_model, x_num, x_cat, y_fit, x_valid_num, x_valid_cat
        torch.cuda.empty_cache()

        outer_preprocessor, outer_fit_numeric, outer_fit_categorical = fit_preprocessor(
            numeric_frame, raw, outer_fit
        )
        outer_valid_numeric, outer_valid_categorical = transform(
            outer_preprocessor, numeric_frame, raw, outer_valid
        )
        x_num, x_cat, y_fit = tensors(
            outer_fit_numeric, outer_fit_categorical, y[outer_fit]
        )
        final_model, _, _ = fit_epochs(
            x_num,
            x_cat,
            y_fit,
            numeric_count=outer_fit_numeric.shape[1] // 2,
            mask_count=outer_fit_numeric.shape[1] // 2,
            seed=seed + 1,
            epochs=selected_epochs,
        )
        prediction = predict(
            final_model, outer_valid_numeric, outer_valid_categorical
        )
        expert_oof[outer_valid] = prediction.astype(np.float32)

        base_rank = fold_rank(baseline_scores[outer_valid])
        expert_rank = fold_rank(prediction)
        blend_oof[outer_valid] = (
            (1.0 - BLEND_WEIGHT) * base_rank + BLEND_WEIGHT * expert_rank
        ).astype(np.float32)
        base_auc = float(roc_auc_score(y[outer_valid], baseline_scores[outer_valid]))
        expert_auc = float(roc_auc_score(y[outer_valid], prediction))
        blend_auc = float(roc_auc_score(y[outer_valid], blend_oof[outer_valid]))
        records.append(
            {
                "fold": fold,
                "baseline_auc": base_auc,
                "tabperiodic_swiglu_auc": expert_auc,
                "fixed_80_20_rank_blend_auc": blend_auc,
                "expert_delta": expert_auc - base_auc,
                "blend_delta": blend_auc - base_auc,
                "selected_epochs": selected_epochs,
                "inner_stopping_auc": stopping_auc,
                "fit_rows": int(len(outer_fit)),
                "valid_rows": int(len(outer_valid)),
                "numeric_features": int(numeric_frame.shape[1]),
                "categorical_onehot_features": int(outer_fit_categorical.shape[1]),
            }
        )
        print(
            f"fold={fold}/{FOLDS} baseline={base_auc:.6f} "
            f"tabperiodic={expert_auc:.6f} blend={blend_auc:.6f} "
            f"epochs={selected_epochs}",
            flush=True,
        )
        del final_model, x_num, x_cat, y_fit
        torch.cuda.empty_cache()

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
            "candidate_te20_blend": baseline_scores,
            "tabperiodic_swiglu": expert_oof,
            "fixed_80_20_rank_blend": blend_oof,
        }
    ).to_csv(oof_path, index=False)
    pd.DataFrame(records).to_csv(fold_path, index=False)

    baseline_mean = float(np.mean([row["baseline_auc"] for row in records]))
    expert_mean = float(np.mean([row["tabperiodic_swiglu_auc"] for row in records]))
    blend_mean = float(np.mean([row["fixed_80_20_rank_blend_auc"] for row in records]))
    gain = blend_mean - baseline_mean
    summary = {
        "status": "fold-isolated exploratory neural experiment; not an independent estimate",
        "hypothesis": (
            "Periodic Fourier embeddings and gated residual layers may learn "
            "smooth threshold patterns and interactions missed by axis-aligned "
            "boosted trees, adding a diverse ranking stream."
        ),
        "outer_folds": FOLDS,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "fixed_blend": (
            f"{1.0 - BLEND_WEIGHT:.2f} candidate TE20 within-fold rank + "
            f"{BLEND_WEIGHT:.2f} TabPeriodic-SwiGLU within-fold rank"
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
        "hardware": device_name,
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "rule_controls": {
            "model_data": "official train.csv only; test.csv not read",
            "labels": "used only for fold-local training and inner stopping",
            "preprocessing": (
                "numeric medians, clipping bounds, scales, and categorical "
                "vocabulary fit inside each inner/outer training partition"
            ),
            "external_data": "none",
            "target_encoding": "none",
            "label_transfer": "none",
            "manual_test_adjustments": "none",
            "test_covariate_pooling": "none",
            "baseline_oof": (
                "read only as a fold-aligned benchmark; never used as a model feature"
            ),
            "gpu": "CUDA required; refuses CPU fallback",
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
