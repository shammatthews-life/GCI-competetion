from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PHASE13 = REPO / "experiments" / "phase13_matched_baseline"
PHASE21 = REPO / "experiments" / "phase21_focused_validation"
sys.path[:0] = [str(PHASE21), str(PHASE13)]

import run_phase21_focused_gpu as phase21
import phase13_features as features13


SEED = 20261113
FOLDS = 5
INNER_FOLDS = 5
MIN_FREE_MIB = 2500
MAX_GPU_UTILIZATION = 10
CONSTANT_FREE_KEYS = (
    "PRODUCT_KEY_Q3",
    "PRODUCT_KEY_Q4",
    "PRODUCT_KEY_Q5",
    "PRODUCT_KEY_AG3",
)
TARGET = features13.TARGET_COLUMN
ID = features13.ID_COLUMN


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(raw: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(raw[column], errors="raise").to_numpy(dtype=np.float64)


def _category(raw: pd.DataFrame, column: str) -> np.ndarray:
    return raw[column].astype("string").fillna("<MISSING>").astype(str).to_numpy()


def _integerized(values: np.ndarray, decimals: int) -> np.ndarray:
    scaled = np.rint(values * (10**decimals))
    result = np.full(len(values), "MISSING", dtype=object)
    finite = np.isfinite(scaled)
    limits = np.iinfo(np.int64)
    if np.any(finite & ((scaled < limits.min) | (scaled > limits.max))):
        raise ValueError("A constant-free product key exceeds the int64 range")
    result[finite] = scaled[finite].astype(np.int64).astype(str)
    return result.astype(str)


def build_compliant_feature_blocks(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    numeric, categorical, rowwise = features13.build_raw_feature_blocks(raw)
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    goods = _numeric(raw, "AMT_GOODS_PRICE")
    q_ratio = np.divide(
        annuity,
        credit,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(credit) & (credit != 0),
    )
    ag_ratio = np.divide(
        annuity,
        goods,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(goods) & (goods != 0),
    )
    keys = pd.DataFrame(
        {
            "PRODUCT_KEY_Q3": _integerized(q_ratio, 3),
            "PRODUCT_KEY_Q4": _integerized(q_ratio, 4),
            "PRODUCT_KEY_Q5": _integerized(q_ratio, 5),
            "PRODUCT_KEY_AG3": _integerized(ag_ratio, 3),
        }
    )
    decompositions = features13.build_label_free_decompositions(raw)
    without_id = raw.drop(columns=[ID])
    enhancements = phase21.features14.build_numeric_enhancements(without_id)
    time_features = phase21.build_time_lifecycle_features(without_id)
    if enhancements.columns.intersection(time_features.columns).size:
        raise ValueError("Phase 14 and Phase 15 feature names collide")
    enhancements = pd.concat(
        [enhancements.reset_index(drop=True), time_features.reset_index(drop=True)],
        axis=1,
    )
    document_flags = raw.filter(regex=r"^FLAG_DOCUMENT_")
    return (
        numeric,
        categorical,
        rowwise,
        keys,
        decompositions,
        enhancements,
        document_flags,
    )


def _peer_percentile(
    fit_values: np.ndarray,
    fit_groups: np.ndarray,
    apply_values: np.ndarray,
    apply_groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fit_result = np.full(len(fit_values), 0.5, dtype=np.float64)
    apply_result = np.full(len(apply_values), 0.5, dtype=np.float64)
    fit_frame = pd.DataFrame({"group": fit_groups, "value": fit_values})
    for group, positions in fit_frame.groupby("group", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        values = fit_values[positions]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        usable_positions = positions[finite]
        usable_values = values[finite]
        fit_result[usable_positions] = (
            pd.Series(usable_values).rank(method="average", pct=True).to_numpy()
        )
        reference = np.sort(usable_values)
        apply_positions = np.flatnonzero(apply_groups == group)
        candidate = apply_values[apply_positions]
        valid = np.isfinite(candidate)
        if valid.any():
            left = np.searchsorted(reference, candidate[valid], side="left")
            right = np.searchsorted(reference, candidate[valid], side="right")
            apply_result[apply_positions[valid]] = (
                left + right
            ) / (2.0 * len(reference))
    return fit_result, apply_result


def formula_peer_features(
    raw: pd.DataFrame,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    credit = _numeric(raw, "AMT_CREDIT")
    annuity = _numeric(raw, "AMT_ANNUITY")
    ext = raw.loc[:, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
        axis=1, skipna=True
    ).to_numpy(dtype=np.float64)

    payment_rate = np.divide(
        annuity,
        credit,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(credit) & (credit > 0),
    )
    occupation = _category(raw, "OCCUPATION_TYPE")

    fit_features = pd.DataFrame(
        {
            "P25_PAYMENT_RATE": payment_rate[fit_rows],
            "P25_PAYMENT_RATE_X_EXT_MEAN": payment_rate[fit_rows] * ext[fit_rows],
        }
    )
    apply_features = pd.DataFrame(
        {
            "P25_PAYMENT_RATE": payment_rate[apply_rows],
            "P25_PAYMENT_RATE_X_EXT_MEAN": payment_rate[apply_rows] * ext[apply_rows],
        }
    )
    fit_rank, apply_rank = _peer_percentile(
        ext[fit_rows],
        occupation[fit_rows],
        ext[apply_rows],
        occupation[apply_rows],
    )
    fit_features["P25_EXT_PCT_OCC"] = fit_rank
    apply_features["P25_EXT_PCT_OCC"] = apply_rank
    for frame, rows in (
        (fit_features, fit_rows),
        (apply_features, apply_rows),
    ):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return fit_features, apply_features


def _ratio_keys(raw: pd.DataFrame, rows: np.ndarray, decimals: int) -> np.ndarray:
    credit = _numeric(raw, "AMT_CREDIT")[rows]
    annuity = _numeric(raw, "AMT_ANNUITY")[rows]
    ratio = np.divide(
        annuity,
        credit,
        out=np.full(len(rows), np.nan, dtype=np.float64),
        where=np.isfinite(credit) & (credit > 0),
    )
    keys = np.full(len(rows), "<MISSING>", dtype=object)
    finite = np.isfinite(ratio)
    keys[finite] = np.round(ratio[finite], decimals).astype(str)
    return keys


def _smoothed_map(
    keys: np.ndarray, labels: np.ndarray, smoothing: float
) -> tuple[dict[str, float], float]:
    prior = float(labels.mean())
    table = pd.DataFrame({"key": keys, "target": labels}).groupby("key")[
        "target"
    ].agg(["sum", "count"])
    encoded = (table["sum"] + smoothing * prior) / (
        table["count"] + smoothing
    )
    return encoded.to_dict(), prior


def ratio_target_encoding_features(
    raw: pd.DataFrame,
    y_fit: np.ndarray,
    fit_rows: np.ndarray,
    apply_rows: np.ndarray,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_output = pd.DataFrame(index=pd.RangeIndex(len(fit_rows)))
    apply_output = pd.DataFrame(index=pd.RangeIndex(len(apply_rows)))
    for decimals, smoothing in ((3, 100.0), (4, 30.0)):
        fit_keys = _ratio_keys(raw, fit_rows, decimals)
        apply_keys = _ratio_keys(raw, apply_rows, decimals)
        oof = np.full(len(fit_rows), np.nan, dtype=np.float64)
        n_splits = min(INNER_FOLDS, len(y_fit))
        if n_splits < 2:
            raise ValueError("Inner target encoding requires at least two rows")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + decimals,
        )
        for inner_fit, inner_apply in splitter.split(
            np.zeros(len(y_fit)), y_fit
        ):
            mapping, prior = _smoothed_map(
                fit_keys[inner_fit], y_fit[inner_fit], smoothing
            )
            oof[inner_apply] = (
                pd.Series(fit_keys[inner_apply])
                .map(mapping)
                .fillna(prior)
                .to_numpy(dtype=np.float64)
            )
        if not np.isfinite(oof).all():
            raise ValueError(f"Target encoding Q{decimals} is incomplete")
        full_map, prior = _smoothed_map(fit_keys, y_fit, smoothing)
        fit_output[f"P25_TARGET_RATE_Q{decimals}"] = oof
        apply_output[f"P25_TARGET_RATE_Q{decimals}"] = (
            pd.Series(apply_keys)
            .map(full_map)
            .fillna(prior)
            .to_numpy(dtype=np.float64)
        )
    return fit_output, apply_output


def _gpu_status() -> tuple[float, float]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = [float(part.strip()) for part in result.stdout.split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout!r}")
    return fields[0], fields[1] - fields[2]


def _append_features(
    x_fit: pd.DataFrame,
    x_apply: pd.DataFrame,
    new_fit: pd.DataFrame,
    new_apply: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.concat(
        [x_fit.reset_index(drop=True), new_fit.reset_index(drop=True)], axis=1
    )
    apply = pd.concat(
        [x_apply.reset_index(drop=True), new_apply.reset_index(drop=True)], axis=1
    )
    if not fit.columns.equals(apply.columns):
        raise ValueError("Augmented feature schemas differ")
    return fit, apply


def run(train_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    utilization, free_mib = _gpu_status()
    if free_mib < MIN_FREE_MIB or utilization > MAX_GPU_UTILIZATION:
        raise RuntimeError(
            "Shared GPU is busy: "
            f"utilization={utilization:.0f}% (max {MAX_GPU_UTILIZATION}%), "
            f"free_memory={free_mib:.0f} MiB (need {MIN_FREE_MIB} MiB)"
        )

    started = time.perf_counter()
    data = pd.read_csv(train_path)
    if TARGET not in data or ID not in data:
        raise ValueError(f"Training data must contain {ID!r} and {TARGET!r}")
    y = pd.to_numeric(data[TARGET], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Target must contain only finite binary values 0 and 1")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1")
    y = y.astype(np.int32)
    raw = data.drop(columns=[TARGET])
    folds = list(
        StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    baseline, formula, target_encoded, combined = (
        np.full(len(y), np.nan, dtype=np.float64) for _ in range(4)
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    fold_records: list[dict[str, Any]] = []
    feature_blocks = build_compliant_feature_blocks(raw)

    for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
        fold_ids[valid_rows] = fold_number
        print(
            f"FOLD {fold_number}/{FOLDS} fit={len(fit_rows)} valid={len(valid_rows)}",
            flush=True,
        )
        x_fit, x_valid, cat_columns = phase21.phase15_matrix(
            feature_blocks,
            fit_rows,
            valid_rows,
            key_columns=CONSTANT_FREE_KEYS,
        )
        formula_fit, formula_valid = formula_peer_features(
            raw, fit_rows, valid_rows
        )
        te_fit, te_valid = ratio_target_encoding_features(
            raw, y[fit_rows], fit_rows, valid_rows, SEED + fold_number * 100
        )
        matrices = {
            "baseline": (x_fit, x_valid),
            "formula_peer": _append_features(
                x_fit, x_valid, formula_fit, formula_valid
            ),
            "ratio_target_encoding": _append_features(
                x_fit, x_valid, te_fit, te_valid
            ),
            "combined": _append_features(
                x_fit,
                x_valid,
                pd.concat([formula_fit, te_fit], axis=1),
                pd.concat([formula_valid, te_valid], axis=1),
            ),
        }
        for candidate, (candidate_fit, candidate_valid) in matrices.items():
            prediction, seconds = phase21.fit_predict(
                candidate_fit,
                y[fit_rows],
                candidate_valid,
                cat_columns,
                SEED + fold_number * 100,
            )
            {"baseline": baseline, "formula_peer": formula,
             "ratio_target_encoding": target_encoded, "combined": combined}[
                candidate
            ][valid_rows] = prediction
            fold_auc = roc_auc_score(y[valid_rows], prediction)
            fold_records.append(
                {
                    "candidate": candidate,
                    "fold": fold_number,
                    "auc": float(fold_auc),
                    "fit_seconds": float(seconds),
                    "fit_rows": int(len(fit_rows)),
                    "valid_rows": int(len(valid_rows)),
                }
            )
            print(
                f"  {candidate}: AUC={fold_auc:.6f} fit={seconds:.1f}s",
                flush=True,
            )
        del matrices, x_fit, x_valid
        gc.collect()

    predictions = {
        "baseline": baseline,
        "formula_peer": formula,
        "ratio_target_encoding": target_encoded,
        "combined": combined,
    }
    if np.any(fold_ids < 1):
        raise ValueError("Outer-fold assignment is incomplete")
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("One or more candidate OOF vectors are incomplete")

    baseline_auc = float(roc_auc_score(y, baseline))
    fold_frame = pd.DataFrame(fold_records)
    baseline_fold_aucs = fold_frame.loc[
        fold_frame["candidate"] == "baseline", "auc"
    ].to_numpy()
    scores: list[dict[str, Any]] = []
    for name, values in predictions.items():
        candidate_rows = fold_frame.loc[fold_frame["candidate"] == name]
        candidate_fold_aucs = candidate_rows["auc"].to_numpy()
        auc = float(roc_auc_score(y, values))
        scores.append(
            {
                "candidate": name,
                "pooled_auc": auc,
                "mean_fold_auc": float(candidate_fold_aucs.mean()),
                "delta_vs_baseline": auc - baseline_auc,
                "mean_fold_delta_vs_baseline": float(
                    (candidate_fold_aucs - baseline_fold_aucs).mean()
                ),
                "positive_folds": int(
                    (
                        candidate_fold_aucs > baseline_fold_aucs
                    ).sum()
                ),
            }
        )
    output_dir.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(output_dir / "PHASE25_AUC_RESULTS.csv", index=False)
    fold_frame.to_csv(output_dir / "PHASE25_FOLD_AUCS.csv", index=False)
    pd.DataFrame(
        {
            ID: raw[ID].to_numpy(),
            TARGET: y,
            "FOLD": fold_ids,
            **{f"OOF_{name}": values for name, values in predictions.items()},
        }
    ).to_csv(output_dir / "PHASE25_OOF_PREDICTIONS.csv", index=False)

    input_hash = sha256_file(train_path)
    report = [
        "# Phase 25 - EXP02 fold-safe, train-only feature ablation",
        "",
        f"- Input train CSV SHA-256: `{input_hash}`",
        f"- Rows: {len(raw)}; positive rate: {y.mean():.8f}.",
        f"- Protocol: {FOLDS}-fold stratified OOF, seed {SEED}; "
        "CatBoost GPU, 700 iterations, depth 6, learning rate 0.05.",
        "- Fixed parameters across folds; no validation-fold early stopping, "
        "feature selection, or blend fitting.",
        "- Read train.csv only; no test file, external data, cached predictions, "
        "or submission was used or written.",
        "- Excluded train-label-to-test linkage, UID/identity label features, "
        "test/test smoothing, pooled train/test statistics, and the unverified "
        "129.5/0.0132/130.5/0.0066 product constants. The matched base model "
        "uses only constant-free Q3/Q4/Q5/AG3 ratio keys.",
        "- Formula/peer features are deterministic or reference only each outer "
        "fit fold. Product-ratio target encodings use inner OOF values for "
        "outer-fit rows and mappings learned only from the outer-fit labels.",
        "- The local compliance matrix allows nested target encoding with "
        "leakage controls, but the official README does not explicitly address "
        "it; treat that candidate as research-only until organizer clarification.",
        "- Formula/peer features are annuity/credit, its interaction with the "
        "provided EXT_SOURCE mean, and an occupation peer percentile learned "
        "from each fit fold. No inferred loan terms, guessed rate anchors, or "
        "test-population statistics are used.",
        "- No blend weights were fitted. Prior label exposure and multiple "
        "candidate comparisons make these exploratory results, not independent "
        "confirmation or submission approval.",
        "",
        "| Candidate | Pooled AUC | Mean fold AUC | Pooled delta | Mean fold delta | Folds improved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scores:
        report.append(
            f"| {row['candidate']} | {row['pooled_auc']:.9f} | "
            f"{row['mean_fold_auc']:.9f} | {row['delta_vs_baseline']:+.9f} | "
            f"{row['mean_fold_delta_vs_baseline']:+.9f} | "
            f"{row['positive_folds']}/5 |"
        )
    report.extend(
        [
            "",
            f"- Runtime: {(time.perf_counter() - started):.1f} seconds.",
            f"- GPU utilization before run: {utilization:.0f}%.",
            f"- GPU free before run: {free_mib:.0f} MiB.",
        ]
    )
    (output_dir / "PHASE25_EXECUTION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "PHASE25_SHA256_MANIFEST.json"
    )
    manifest = {
        "train_csv": {"path": str(train_path), "sha256": input_hash},
        "code_sha256": {
            "phase25_runner": sha256_file(Path(__file__)),
            "phase21_model_helpers": sha256_file(
                PHASE21 / "run_phase21_focused_gpu.py"
            ),
            "phase13_features": sha256_file(PHASE13 / "phase13_features.py"),
            "phase14_features": sha256_file(
                REPO
                / "experiments"
                / "phase14_label_free_model_development"
                / "phase14_features.py"
            ),
            "phase15_features": sha256_file(
                REPO
                / "experiments"
                / "phase15_autonomous_research"
                / "phase15_features.py"
            ),
        },
        "protocol": {
            "metric": "ROC-AUC",
            "outer_seed": SEED,
            "outer_folds": FOLDS,
            "inner_target_encoding_folds": INNER_FOLDS,
            "catboost": {
                "task_type": "GPU",
                "iterations": phase21.ITERATIONS,
                "depth": 6,
                "learning_rate": 0.05,
            },
            "candidates": [
                "baseline",
                "formula_peer",
                "ratio_target_encoding",
                "combined",
            ],
        },
        "gpu_free_before_run_mib": free_mib,
        "artifacts": {
            path.name: sha256_file(path)
            for path in artifacts
        },
    }
    (output_dir / "PHASE25_SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.train, args.output_dir)


if __name__ == "__main__":
    main()
