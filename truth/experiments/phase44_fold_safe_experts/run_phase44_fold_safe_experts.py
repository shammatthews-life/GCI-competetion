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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


PHASE33 = Path(__file__).resolve().parents[1] / "phase33_rules_first_features"
PHASE38 = Path(__file__).resolve().parents[1] / "phase38_key_frequency_thresholds"
sys.path[:0] = [str(PHASE33), str(PHASE38)]
import run_phase33_rules_first_features as phase33
import run_phase38_key_frequency_thresholds as phase38


TRAIN_DEFAULT = Path(r"D:\gci\inpuy\train.csv")
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 20261113
FOLDS = 5
KEY_MIN_FREQUENCY = 50
MAX_ITERATIONS = 3000
MIN_EXPERT_FIT_ROWS = 5000
MIN_EXPERT_VALID_ROWS = 1000
GATES = ("contract_type", "external_completeness")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_gate_memberships(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "NAME_CONTRACT_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Gate columns missing from input: {sorted(missing)}")
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING")
    contract_values = contract.to_numpy(dtype=str)
    if set(np.unique(contract_values)) != {"Cash loans", "Revolving loans"}:
        raise ValueError("Unexpected contract categories; refusing implicit routing")
    contract_gate = (contract_values == "Revolving loans").astype(np.int8)
    ext = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    complete_gate = np.isfinite(ext).all(axis=1).astype(np.int8)
    return {
        "contract_type": contract_gate,
        "external_completeness": complete_gate,
    }


def train_predict(
    blocks: tuple[pd.DataFrame, ...],
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    if len(np.unique(y[fit_rows])) != 2:
        raise ValueError("Model training partition must contain both target classes")
    inner = StratifiedShuffleSplit(
        n_splits=1, test_size=0.12, random_state=seed
    )
    inner_fit_idx, inner_valid_idx = next(
        inner.split(np.zeros(len(fit_rows)), y[fit_rows])
    )
    inner_fit = fit_rows[inner_fit_idx]
    inner_valid = fit_rows[inner_valid_idx]
    if len(np.unique(y[inner_valid])) != 2:
        raise ValueError("Inner stopping partition must contain both classes")

    x_inner_fit, x_inner_valid = phase38.make_matrices(
        blocks, inner_fit, inner_valid, KEY_MIN_FREQUENCY
    )
    selector = phase38._model(seed, MAX_ITERATIONS, stopping=True)
    selector.fit(
        x_inner_fit,
        y[inner_fit],
        eval_set=[(x_inner_valid, y[inner_valid])],
        verbose=False,
    )
    iterations = max(100, int(selector.best_iteration) + 1)
    del selector, x_inner_fit, x_inner_valid

    x_fit, x_apply = phase38.make_matrices(
        blocks, fit_rows, apply_rows, KEY_MIN_FREQUENCY
    )
    model = phase38._model(seed, iterations, stopping=False)
    model.fit(x_fit, y[fit_rows], verbose=False)
    prediction = model.predict_proba(x_apply)[:, 1]
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced nonfinite predictions")
    return prediction.astype(np.float32), iterations


def _metric_record(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "pooled_log_loss": float(log_loss(y, prediction)),
        "mean_fold_auc": float(
            np.mean(
                [
                    roc_auc_score(y[valid], prediction[valid])
                    for _, valid in StratifiedKFold(
                        n_splits=FOLDS, shuffle=True, random_state=SEED
                    ).split(np.zeros(len(y)), y)
                ]
            )
        ),
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
    gates = make_gate_memberships(raw)
    blocks = phase33.build_safe_blocks(raw)
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    predictions: dict[str, np.ndarray] = {
        "global": np.full(len(y), np.nan, dtype=np.float32)
    }
    for gate_name in GATES:
        predictions[f"{gate_name}_hard"] = np.full(
            len(y), np.nan, dtype=np.float32
        )
        predictions[f"{gate_name}_mixture_50_50"] = np.full(
            len(y), np.nan, dtype=np.float32
        )
    records: list[dict[str, Any]] = []
    segment_counts: dict[str, dict[str, int]] = {}

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        global_prediction, global_iterations = train_predict(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        predictions["global"][valid_rows] = global_prediction
        records.append(
            {
                "model": "global",
                "gate": "none",
                "group": "all",
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "iterations": global_iterations,
                "validation_auc": float(
                    roc_auc_score(y[valid_rows], global_prediction)
                ),
            }
        )

        for gate_name in GATES:
            membership = gates[gate_name]
            hard_prediction = np.full(len(valid_rows), np.nan, dtype=np.float32)
            gate_counts: dict[str, int] = {}
            for group in (0, 1):
                group_fit = fit_rows[membership[fit_rows] == group]
                group_valid_mask = membership[valid_rows] == group
                group_valid = valid_rows[group_valid_mask]
                gate_counts[f"train_group_{group}"] = int(len(group_fit))
                gate_counts[f"valid_group_{group}"] = int(len(group_valid))
                if (
                    len(group_fit) < MIN_EXPERT_FIT_ROWS
                    or len(group_valid) < MIN_EXPERT_VALID_ROWS
                ):
                    raise ValueError(
                        f"Gate {gate_name} group {group} below support floor "
                        f"in fold {fold_number}: fit={len(group_fit)}, "
                        f"valid={len(group_valid)}"
                    )
                expert_prediction, expert_iterations = train_predict(
                    blocks, group_fit, group_valid, y, fold_seed + group + 1
                )
                hard_prediction[group_valid_mask] = expert_prediction
                records.append(
                    {
                        "model": "expert",
                        "gate": gate_name,
                        "group": str(group),
                        "fold": fold_number,
                        "fit_rows": int(len(group_fit)),
                        "validation_rows": int(len(group_valid)),
                        "iterations": expert_iterations,
                        "validation_auc": float(
                            roc_auc_score(y[group_valid], expert_prediction)
                        ),
                    }
                )
            if not np.isfinite(hard_prediction).all():
                raise ValueError(
                    f"Hard route incomplete for {gate_name}, fold {fold_number}"
                )
            hard_key = f"{gate_name}_hard"
            mixture_key = f"{gate_name}_mixture_50_50"
            predictions[hard_key][valid_rows] = hard_prediction
            predictions[mixture_key][valid_rows] = (
                0.5 * global_prediction + 0.5 * hard_prediction
            )
            segment_counts.setdefault(gate_name, {})[f"fold_{fold_number}"] = (
                gate_counts
            )
            print(
                f"fold={fold_number}/{FOLDS} gate={gate_name} "
                f"routed_auc={roc_auc_score(y[valid_rows], hard_prediction):.6f} "
                f"mix_auc={roc_auc_score(y[valid_rows], predictions[mixture_key][valid_rows]):.6f}",
                flush=True,
            )

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("At least one OOF prediction vector is incomplete")
    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE44_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE44_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE44_SUMMARY.json"
    pd.DataFrame(records).to_csv(fold_path, index=False)
    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "fold": fold_ids,
            **{f"oof_{name}": values for name, values in predictions.items()},
        }
    )
    oof_frame.to_csv(oof_path, index=False)
    metrics = {
        name: _metric_record(y, values)
        for name, values in predictions.items()
    }
    summary: dict[str, Any] = {
        "phase": "Phase 44 fold-safe feature-routed experts",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "model": "Phase 38 XGBoost configuration; inner 12% stopping split; outer-fit refit",
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "gates": {
            "contract_type": "NAME_CONTRACT_TYPE: cash=0, revolving=1",
            "external_completeness": "all three supplied EXT_SOURCE fields finite=1; otherwise=0",
        },
        "fixed_mixture_weight": 0.5,
        "support_floors": {
            "expert_training_rows": MIN_EXPERT_FIT_ROWS,
            "expert_validation_rows": MIN_EXPERT_VALID_ROWS,
        },
        "segment_counts_by_fold": segment_counts,
        "metrics": metrics,
        "rule_controls": {
            "input": "supplied training CSV only; no test or external data read",
            "gate_membership": "deterministic supplied covariates only; no TARGET-based routing",
            "expert_training": "each expert fit only on its outer-training subgroup",
            "iteration_selection": "inner split within the expert's outer-training subgroup",
            "outer_validation": "never used for fitting, early stopping, or gate selection",
            "custom_target_encoding": "none",
            "native_categorical_target_statistics": "none; categorical inputs one-hot encoded",
            "test_statistics": "none",
            "unverified_empirical_constants": "none",
            "manual_prediction_edits": "none",
        },
        "caveats": [
            "Phase 20/21 previously found small and inconsistent segmentation effects; this is a matched XGBoost replication on the newer Phase 38 covariate representation.",
            "A hard gate is only eligible if its pooled all-row OOF metric improves; segment-only AUC is not a promotion criterion.",
            "Repeated reuse of the training labels creates selection bias; no candidate is independent confirmation or a leaderboard guarantee.",
        ],
        "runtime_seconds": time.perf_counter() - started,
        "sha256": {
            str(path): sha256_file(path)
            for path in (
                train_path,
                Path(__file__).resolve(),
                PHASE33 / "run_phase33_rules_first_features.py",
                PHASE38 / "run_phase38_key_frequency_thresholds.py",
                fold_path,
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
        / datetime.now(timezone.utc).strftime("phase44_experts_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)
