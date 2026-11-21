from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
PHASE44 = Path(__file__).resolve().parents[1] / "phase44_fold_safe_experts"
sys.path[:0] = [str(PHASE33), str(PHASE38), str(PHASE44)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38
import run_phase44_fold_safe_experts as phase44


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
INNER_WEIGHT_FOLDS = 3
WEIGHT_GRID = tuple(np.round(np.linspace(0.0, 1.0, 11), 1))
MIN_EXPERT_FIT_ROWS = 5000
MIN_EXPERT_VALID_ROWS = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_expert_weight(
    y: np.ndarray,
    global_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    candidates: tuple[float, ...] = WEIGHT_GRID,
) -> tuple[float, dict[str, float]]:
    if not (
        len(y) == len(global_prediction) == len(expert_prediction)
        and len(np.unique(y)) == 2
        and np.isfinite(global_prediction).all()
        and np.isfinite(expert_prediction).all()
    ):
        raise ValueError("Invalid nested OOF vectors for mixture-weight selection")
    results: list[tuple[float, float, float]] = []
    for weight in candidates:
        mixture = (1.0 - weight) * global_prediction + weight * expert_prediction
        auc = float(roc_auc_score(y, mixture))
        loss = float(log_loss(y, mixture))
        results.append((auc, loss, float(weight)))
    best_auc = max(item[0] for item in results)
    tied = [item for item in results if np.isclose(item[0], best_auc, atol=1e-12)]
    best = min(tied, key=lambda item: (item[1], abs(item[2] - 0.5), item[2]))
    return best[2], {
        "inner_oof_auc": best[0],
        "inner_oof_log_loss": best[1],
    }


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file does not exist: {train_path}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = pd.read_csv(train_path)
    if ID not in train or TARGET not in train or not train[ID].is_unique:
        raise ValueError("Training file must contain unique IDs and TARGET")
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(y_float).all()
        or not np.isin(y_float, (0, 1)).all()
        or set(np.unique(y_float)) != {0, 1}
    ):
        raise ValueError("TARGET must contain finite binary values")
    y = y_float.astype(np.int8)
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    membership = phase44.make_gate_memberships(raw)["contract_type"]
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    predictions = {
        name: np.full(len(y), np.nan, dtype=np.float32)
        for name in (
            "global",
            "contract_type_hard",
            "contract_type_mixture_50_50",
            "contract_type_mixture_nested_weight",
        )
    }
    fold_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        inner_global = np.full(len(fit_rows), np.nan, dtype=np.float64)
        inner_expert = np.full(len(fit_rows), np.nan, dtype=np.float64)
        inner_splitter = StratifiedKFold(
            n_splits=INNER_WEIGHT_FOLDS,
            shuffle=True,
            random_state=SEED + fold_number * 1000 + 131,
        )
        for inner_fold, (inner_fit_local, inner_valid_local) in enumerate(
            inner_splitter.split(np.zeros(len(fit_rows)), y[fit_rows]), start=1
        ):
            inner_fit = fit_rows[inner_fit_local]
            inner_valid = fit_rows[inner_valid_local]
            inner_seed = (
                SEED + fold_number * 100000 + inner_fold * 1000 + 50000
            )
            global_oof, _ = phase44.train_predict(
                blocks, inner_fit, inner_valid, y, inner_seed
            )
            expert_oof = np.full(len(inner_valid), np.nan, dtype=np.float64)
            for group in (0, 1):
                group_fit = inner_fit[membership[inner_fit] == group]
                group_valid_mask = membership[inner_valid] == group
                group_valid = inner_valid[group_valid_mask]
                if (
                    len(group_fit) < MIN_EXPERT_FIT_ROWS
                    or len(group_valid) < MIN_EXPERT_VALID_ROWS
                ):
                    raise ValueError(
                        f"Nested expert support insufficient: outer={fold_number}, "
                        f"inner={inner_fold}, group={group}, "
                        f"fit={len(group_fit)}, valid={len(group_valid)}"
                    )
                group_prediction, _ = phase44.train_predict(
                    blocks,
                    group_fit,
                    group_valid,
                    y,
                    inner_seed + group + 1,
                )
                expert_oof[group_valid_mask] = group_prediction
            if not np.isfinite(expert_oof).all():
                raise ValueError("Inner routed expert predictions are incomplete")
            inner_global[inner_valid_local] = global_oof
            inner_expert[inner_valid_local] = expert_oof
        if not (
            np.isfinite(inner_global).all() and np.isfinite(inner_expert).all()
        ):
            raise ValueError("Nested weight-selection OOF is incomplete")

        chosen_weight, inner_metrics = choose_expert_weight(
            y[fit_rows], inner_global, inner_expert
        )
        selection_records.append(
            {
                "outer_fold": fold_number,
                "weight": chosen_weight,
                **inner_metrics,
                "inner_rows": int(len(fit_rows)),
                "candidate_weights": list(WEIGHT_GRID),
                "weight_selection_scope": "outer-training rows only, using nested OOF predictions",
            }
        )
        fold_seed = SEED + fold_number * 100
        global_prediction, global_iterations = phase44.train_predict(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        hard_prediction = np.full(len(valid_rows), np.nan, dtype=np.float64)
        expert_iterations: dict[str, int] = {}
        for group in (0, 1):
            group_fit = fit_rows[membership[fit_rows] == group]
            group_valid_mask = membership[valid_rows] == group
            group_valid = valid_rows[group_valid_mask]
            if (
                len(group_fit) < MIN_EXPERT_FIT_ROWS
                or len(group_valid) < MIN_EXPERT_VALID_ROWS
            ):
                raise ValueError(
                    f"Outer expert support insufficient: fold={fold_number}, "
                    f"group={group}"
                )
            group_prediction, iterations = phase44.train_predict(
                blocks, group_fit, group_valid, y, fold_seed + group + 1
            )
            hard_prediction[group_valid_mask] = group_prediction
            expert_iterations[str(group)] = iterations
        if not np.isfinite(hard_prediction).all():
            raise ValueError("Outer routed expert predictions are incomplete")
        values = {
            "global": global_prediction,
            "contract_type_hard": hard_prediction,
            "contract_type_mixture_50_50": (
                0.5 * global_prediction + 0.5 * hard_prediction
            ),
            "contract_type_mixture_nested_weight": (
                (1.0 - chosen_weight) * global_prediction
                + chosen_weight * hard_prediction
            ),
        }
        for name, value in values.items():
            predictions[name][valid_rows] = value.astype(np.float32)
            fold_records.append(
                {
                    "model": name,
                    "fold": fold_number,
                    "auc": float(roc_auc_score(y[valid_rows], value)),
                    "log_loss": float(log_loss(y[valid_rows], value)),
                    "validation_rows": int(len(valid_rows)),
                    "expert_weight": (
                        chosen_weight
                        if name == "contract_type_mixture_nested_weight"
                        else (0.5 if name == "contract_type_mixture_50_50" else 0.0)
                    ),
                    "global_iterations": global_iterations,
                    "expert_iterations": json.dumps(expert_iterations, sort_keys=True),
                }
            )
        print(
            f"fold={fold_number}/{FOLDS} selected_expert_weight={chosen_weight:.1f} "
            f"global_auc={roc_auc_score(y[valid_rows], values['global']):.6f} "
            f"fixed_auc={roc_auc_score(y[valid_rows], values['contract_type_mixture_50_50']):.6f} "
            f"nested_auc={roc_auc_score(y[valid_rows], values['contract_type_mixture_nested_weight']):.6f}",
            flush=True,
        )

    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise ValueError("One or more OOF prediction vectors are incomplete")
    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE46_FOLD_RESULTS.csv"
    weight_path = output_dir / "PHASE46_WEIGHT_SELECTION.csv"
    oof_path = output_dir / "PHASE46_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE46_SUMMARY.json"
    pd.DataFrame(fold_records).to_csv(fold_path, index=False)
    pd.DataFrame(selection_records).to_csv(weight_path, index=False)
    pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{f"oof_{name}": value for name, value in predictions.items()},
        }
    ).to_csv(oof_path, index=False)
    metrics: dict[str, dict[str, float]] = {}
    for name, prediction in predictions.items():
        metrics[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "pooled_log_loss": float(log_loss(y, prediction)),
            "mean_fold_auc": float(
                np.mean(
                    [
                        roc_auc_score(y[valid], prediction[valid])
                        for _, valid in folds
                    ]
                )
            ),
        }
    summary: dict[str, Any] = {
        "phase": "Phase 46 nested OOF selection of contract-expert mixture weight",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "inner_weight_folds": INNER_WEIGHT_FOLDS,
        "candidate_expert_weights": list(WEIGHT_GRID),
        "gate": "NAME_CONTRACT_TYPE: cash=0, revolving=1",
        "model": "Phase 38 XGBoost configuration with fold-fitted one-hot and key-frequency features",
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "routing": "feature-only contract gate; no target-based routing",
            "inner_weight_selection": "selected separately per outer fold using only outer-training labels and nested OOF predictions",
            "outer_validation": "never used for weight selection, fitting, or early stopping",
            "target_encoding": "none",
            "native_categorical_target_statistics": "none",
            "manual_prediction_edits": "none",
            "test_statistics": "none",
        },
        "caveats": [
            "Nested selection avoids outer-fold label leakage, but the result remains subject to repeated project-wide model selection.",
            "Weight selection maximizes inner OOF AUC over a fixed grid and can itself be noisy; the held-out pooled OOF metric is the decision criterion.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
                PHASE44 / "run_phase44_fold_safe_experts.py",
                fold_path,
                weight_path,
                oof_path,
            )
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase46_nested_weight_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
