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
GATES = (
    "contract_type",
    "goods_credit_ratio",
    "pensioner_status",
    "contract_goods_bivariate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_gate_memberships(raw: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "NAME_CONTRACT_TYPE",
        "AMT_GOODS_PRICE",
        "AMT_CREDIT",
        "DAYS_EMPLOYED",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Gate columns missing from input: {sorted(missing)}")

    # 1. Contract Type Gate
    contract = raw["NAME_CONTRACT_TYPE"].astype("string").fillna("MISSING")
    contract_values = contract.to_numpy(dtype=str)
    if set(np.unique(contract_values)) != {"Cash loans", "Revolving loans"}:
        raise ValueError("Unexpected contract categories; refusing implicit routing")
    contract_gate = (contract_values == "Revolving loans").astype(np.int8)

    # 2. Goods-to-Credit Ratio Gate (F10 from Phase 20 study: ratio < 0.90 indicates cash fee/add-on margin)
    gp = pd.to_numeric(raw["AMT_GOODS_PRICE"], errors="coerce").to_numpy(dtype=np.float64)
    cr = pd.to_numeric(raw["AMT_CREDIT"], errors="coerce").to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cr > 0, gp / cr, np.nan)
    # Group 0: ratio >= 0.90 (standard merchandise retail loan)
    # Group 1: ratio < 0.90 or missing (pure cash loan / financed fees)
    goods_gate = ((~np.isfinite(ratio)) | (ratio < 0.90)).astype(np.int8)

    # 3. Pensioner / Employment Sentinel Gate (F02 from Phase 20 study)
    emp = pd.to_numeric(raw["DAYS_EMPLOYED"], errors="coerce").to_numpy(dtype=np.float64)
    # Group 0: Actively employed (DAYS_EMPLOYED != 365243)
    # Group 1: Pensioners / State retirement sentinel (DAYS_EMPLOYED == 365243)
    pensioner_gate = (emp == 365243.0).astype(np.int8)

    # 4. Bivariate Contract x Goods Triage Gate (3-way partition)
    # Group 0: Revolving loans
    # Group 1: Cash loans with goods/credit >= 0.90 (direct merchandise POS)
    # Group 2: Cash loans with goods/credit < 0.90 or missing (cash loans with added fee margins)
    bivariate_gate = np.zeros(len(raw), dtype=np.int8)
    bivariate_gate[contract_values == "Revolving loans"] = 0
    bivariate_gate[(contract_values == "Cash loans") & (ratio >= 0.90)] = 1
    bivariate_gate[(contract_values == "Cash loans") & ((~np.isfinite(ratio)) | (ratio < 0.90))] = 2

    return {
        "contract_type": contract_gate,
        "goods_credit_ratio": goods_gate,
        "pensioner_status": pensioner_gate,
        "contract_goods_bivariate": bivariate_gate,
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
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
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


def _metric_record(y: np.ndarray, prediction: np.ndarray, folds_indices: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    fold_aucs = [
        float(roc_auc_score(y[valid], prediction[valid]))
        for _, valid in folds_indices
    ]
    return {
        "pooled_auc": float(roc_auc_score(y, prediction)),
        "pooled_log_loss": float(log_loss(y, prediction)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
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
    y_float = pd.to_numeric(train[TARGET], errors="raise").to_numpy(dtype=np.float64)
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

    # Predeclare predictions dictionary
    predictions: dict[str, np.ndarray] = {
        "global": np.full(len(y), np.nan, dtype=np.float32)
    }
    for gate_name in GATES:
        predictions[f"{gate_name}_hard"] = np.full(len(y), np.nan, dtype=np.float32)
        predictions[f"{gate_name}_mixture_50_50"] = np.full(len(y), np.nan, dtype=np.float32)
    predictions["consensus_expert_mixture"] = np.full(len(y), np.nan, dtype=np.float32)

    records: list[dict[str, Any]] = []
    segment_counts: dict[str, dict[str, int]] = {}

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        fold_seed = SEED + fold_number * 100
        print(f"\n--- Fold {fold_number}/{FOLDS} (Fit rows: {len(fit_rows):,}, Valid rows: {len(valid_rows):,}) ---", flush=True)

        # 1. Global Baseline
        t0 = time.perf_counter()
        global_prediction, global_iterations = train_predict(
            blocks, fit_rows, valid_rows, y, fold_seed
        )
        predictions["global"][valid_rows] = global_prediction
        global_val_auc = roc_auc_score(y[valid_rows], global_prediction)
        print(f"  [Global Model] validation AUC: {global_val_auc:.6f} ({global_iterations} trees, {time.perf_counter()-t0:.1f}s)", flush=True)
        records.append(
            {
                "model": "global",
                "gate": "none",
                "group": "all",
                "fold": fold_number,
                "fit_rows": int(len(fit_rows)),
                "validation_rows": int(len(valid_rows)),
                "iterations": global_iterations,
                "validation_auc": float(global_val_auc),
            }
        )

        # 2. Gate Specialists
        for gate_name in GATES:
            membership = gates[gate_name]
            unique_groups = sorted(np.unique(membership))
            hard_prediction = np.full(len(valid_rows), np.nan, dtype=np.float32)
            gate_counts: dict[str, int] = {}

            for group in unique_groups:
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
                    blocks, group_fit, group_valid, y, fold_seed + int(group) + 1
                )
                hard_prediction[group_valid_mask] = expert_prediction
                group_auc = roc_auc_score(y[group_valid], expert_prediction)
                records.append(
                    {
                        "model": "expert",
                        "gate": gate_name,
                        "group": str(group),
                        "fold": fold_number,
                        "fit_rows": int(len(group_fit)),
                        "validation_rows": int(len(group_valid)),
                        "iterations": expert_iterations,
                        "validation_auc": float(group_auc),
                    }
                )

            if not np.isfinite(hard_prediction).all():
                raise ValueError(f"Hard route incomplete for {gate_name}, fold {fold_number}")

            hard_key = f"{gate_name}_hard"
            mix_key = f"{gate_name}_mixture_50_50"
            predictions[hard_key][valid_rows] = hard_prediction
            predictions[mix_key][valid_rows] = 0.5 * global_prediction + 0.5 * hard_prediction
            segment_counts.setdefault(gate_name, {})[f"fold_{fold_number}"] = gate_counts

            hard_auc = roc_auc_score(y[valid_rows], hard_prediction)
            mix_auc = roc_auc_score(y[valid_rows], predictions[mix_key][valid_rows])
            print(
                f"  [{gate_name:<24}] hard AUC: {hard_auc:.6f} | 50/50 mix AUC: {mix_auc:.6f} "
                f"(Δ vs global: {(mix_auc - global_val_auc)*10000:+.1f} bps)",
                flush=True,
            )

        # 3. Multi-Expert Consensus Mixture:
        # Predeclared combination: 0.5 * global + 0.25 * contract_specialist + 0.25 * goods_specialist
        c_hard = predictions["contract_type_hard"][valid_rows]
        g_hard = predictions["goods_credit_ratio_hard"][valid_rows]
        consensus_val = 0.50 * global_prediction + 0.25 * c_hard + 0.25 * g_hard
        predictions["consensus_expert_mixture"][valid_rows] = consensus_val
        cons_auc = roc_auc_score(y[valid_rows], consensus_val)
        print(
            f"  [consensus_expert_mixture] validation AUC: {cons_auc:.6f} "
            f"(Δ vs global: {(cons_auc - global_val_auc)*10000:+.1f} bps)",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("At least one OOF prediction vector is incomplete")

    output_dir.mkdir(parents=False, exist_ok=False)
    fold_path = output_dir / "PHASE47_FOLD_RESULTS.csv"
    oof_path = output_dir / "PHASE47_OOF_PREDICTIONS.csv"
    summary_path = output_dir / "PHASE47_SUMMARY.json"

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
        name: _metric_record(y, values, folds)
        for name, values in predictions.items()
    }

    summary: dict[str, Any] = {
        "phase": "Phase 47 fold-safe evidence-partition expert models",
        "status": "exploratory; repeated-label research; not organizer-certified",
        "rows": int(len(train)),
        "seed": SEED,
        "outer_folds": FOLDS,
        "model": "Phase 38 XGBoost configuration; inner 12% stopping split; outer-fit refit",
        "key_min_frequency": KEY_MIN_FREQUENCY,
        "gates": {
            "contract_type": "NAME_CONTRACT_TYPE: cash=0, revolving=1",
            "goods_credit_ratio": "AMT_GOODS_PRICE / AMT_CREDIT: >=0.90=0, <0.90 or missing=1",
            "pensioner_status": "DAYS_EMPLOYED == 365243: employed=0, pensioner=1",
            "contract_goods_bivariate": "revolving=0, cash goods>=0.90=1, cash goods<0.90 or missing=2",
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
    print("\n" + "=" * 80)
    print("  PHASE 47 COMPLETE: POOLED OOF METRICS")
    print("=" * 80)
    for name, m in metrics.items():
        delta_auc = (m["pooled_auc"] - metrics["global"]["pooled_auc"]) * 10000.0
        print(f"  {name:<32} Pooled AUC: {m['pooled_auc']:.6f} ({delta_auc:+.1f} bps) | LogLoss: {m['pooled_log_loss']:.6f} | Mean Fold: {m['mean_fold_auc']:.6f} (±{m['std_fold_auc']:.6f})")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(timezone.utc).strftime("phase47_experts_%Y%m%dT%H%M%SZ"),
    )
    args = parser.parse_args()
    run(args.train, args.output_dir)

