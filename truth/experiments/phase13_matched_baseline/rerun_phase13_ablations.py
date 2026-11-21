from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase13_features import (
    ID_COLUMN,
    TARGET_COLUMN,
    build_label_free_decompositions,
    build_product_keys,
    build_raw_feature_blocks,
)
from run_phase13 import (
    CONFIGS,
    OUTER_FOLDS,
    configure_logger,
    outer_splits,
    run_ablations,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = Path(r"D:\gci\inpuy\train.csv")


def run(
    original_run: Path,
    train_path: Path,
    output_root: Path,
    seed: int,
) -> Path:
    if not original_run.is_dir():
        raise FileNotFoundError(f"Original Phase 13 run not found: {original_run}")
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    original_oof_path = original_run / "OOF_PREDICTIONS.csv"
    original_fits_path = original_run / "OUTER_MODEL_FITS.csv"
    for path in (original_oof_path, original_fits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required original run artifact not found: {path}")

    start = datetime.now(timezone.utc)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"phase13_ablation_seed_matched_{start.strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(exist_ok=False)
    logger = configure_logger(output)
    input_hash = sha256_file(train_path)
    original_oof_hash = sha256_file(original_oof_path)
    train = pd.read_csv(train_path, low_memory=False)
    original_oof = pd.read_csv(original_oof_path)
    if len(train) != len(original_oof):
        raise ValueError("Original OOF predictions do not match training row count")
    if not np.array_equal(original_oof["row_index"].to_numpy(), np.arange(len(train))):
        raise ValueError("Original OOF row indices are not aligned")
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if not np.array_equal(original_oof[TARGET_COLUMN].to_numpy(dtype=np.int8), y):
        raise ValueError("Original OOF targets differ from the training labels")
    if not np.array_equal(
        original_oof[ID_COLUMN].to_numpy(), train[ID_COLUMN].to_numpy()
    ):
        raise ValueError("Original OOF IDs differ from the training row order")
    fold_ids = np.zeros(len(y), dtype=np.int8)
    for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
        fold_ids[apply] = fold
    if not np.array_equal(original_oof["primary_fold"].to_numpy(dtype=np.int8), fold_ids):
        raise ValueError("The requested seed does not match the saved outer folds")

    fits = pd.read_csv(original_fits_path)
    key_fits = fits[fits["variant"] == "D_C_plus_keys_ohe"]
    if len(key_fits) != OUTER_FOLDS:
        raise ValueError("Expected one saved D configuration per outer fold")
    selected = {}
    for row in key_fits.itertuples(index=False):
        config = next((item for item in CONFIGS if item["name"] == row.config), None)
        if config is None:
            raise ValueError(f"Unknown saved model configuration: {row.config}")
        selected[int(row.fold)] = {
            "D_C_plus_keys_ohe": (config, int(row.selected_rounds))
        }
    if set(selected) != set(range(1, OUTER_FOLDS + 1)):
        raise ValueError("Saved D configurations do not cover the five outer folds")

    raw = train.drop(columns=[TARGET_COLUMN])
    numeric, categorical, rowwise = build_raw_feature_blocks(raw)
    keys = build_product_keys(raw)
    decompositions = build_label_free_decompositions(raw)
    blocks = (numeric, categorical, rowwise, keys, decompositions)
    support_rows: list[dict[str, object]] = []
    predictions, results = run_ablations(
        y,
        blocks,
        selected,
        seed,
        logger,
        support_rows,
    )
    oof = pd.DataFrame(
        {
            "row_index": np.arange(len(y)),
            ID_COLUMN: train[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y,
            "primary_fold": original_oof["primary_fold"].to_numpy(dtype=np.int8),
        }
    )
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise AssertionError(f"Missing corrected OOF values for {name}")
        oof[f"ablation_{name}"] = prediction
    oof.to_csv(output / "PRODUCT_KEY_ABLATION_SEED_MATCHED_OOF.csv", index=False)
    results.to_csv(output / "PRODUCT_KEY_ABLATION_SEED_MATCHED.csv", index=False)
    pd.DataFrame(support_rows).to_csv(
        output / "CATEGORY_SUPPORT_REPORT.csv", index=False
    )

    complete = original_oof["C_complete_raw_rowwise"].to_numpy(dtype=np.float64)
    fold_auc = {}
    for name, prediction in predictions.items():
        for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
            fold_auc[(name, fold)] = float(roc_auc_score(y[apply], prediction[apply]))
    results["paired_delta_vs_C"] = results.apply(
        lambda row: fold_auc[(row["variant"], int(row["fold"]))]
        - float(
            roc_auc_score(
                y[
                    original_oof["primary_fold"].to_numpy(dtype=np.int8)
                    == int(row["fold"])
                ],
                complete[
                    original_oof["primary_fold"].to_numpy(dtype=np.int8)
                    == int(row["fold"])
                ],
            )
        ),
        axis=1,
    )
    results.to_csv(output / "PRODUCT_KEY_ABLATION_SEED_MATCHED.csv", index=False)

    if sha256_file(train_path) != input_hash or sha256_file(original_oof_path) != original_oof_hash:
        raise AssertionError("Training or original OOF input changed during rerun")
    for fold, (_, apply) in enumerate(outer_splits(y, seed), start=1):
        if not np.all(oof.loc[apply, "primary_fold"].to_numpy() == fold):
            raise AssertionError(f"Corrected ablation fold {fold} is misaligned")
    pooled = results.groupby("variant")["pooled_oof_auc"].first()
    summary_rows = []
    for name, group in results.groupby("variant"):
        delta = group["paired_fold_delta_vs_all_eight"]
        summary_rows.append(
            {
                "variant": name,
                "pooled_oof_auc": float(pooled[name]),
                "mean_paired_fold_delta_vs_all_eight": float(delta.mean()),
                "folds_better_than_all_eight": int((delta > 0).sum()),
                "folds_worse_than_all_eight": int((delta < 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        "pooled_oof_auc", ascending=False
    )
    summary.to_csv(output / "ABLATION_SUMMARY.csv", index=False)
    report_lines = [
        "# Phase 13 matched-seed key ablation correction",
        "",
        f"- UTC start: {start.isoformat()}",
        f"- Original Phase 13 run: `{original_run}`",
        f"- Original OOF SHA-256: `{original_oof_hash}`; train SHA-256: `{input_hash}`.",
        f"- Rows: {len(train):,}; seed: {seed}; folds: {OUTER_FOLDS}.",
        "- Refit each key subset with the same fold-specific LightGBM seed, D-selected configuration, and D-selected iteration count for that fold. Only the key columns vary across ablation variants.",
        "- This rerun corrects the earlier ablation-only seed mismatch. It does not alter the primary A–F benchmark, selected configurations/rounds, controls, or secondary-seed run.",
        "- All transformations are fitted on each outer-fit partition; all rows are predicted exactly once out of fold.",
        "",
        "| Variant | Pooled OOF AUC | Mean fold delta vs all eight | Folds better than all eight |",
        "|---|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        report_lines.append(
            f"| {row['variant']} | {row['pooled_oof_auc']:.6f} | "
            f"{row['mean_paired_fold_delta_vs_all_eight']:+.6f} | "
            f"{row['folds_better_than_all_eight']}/5 |"
        )
    report_lines.extend(
        [
            "",
            "This is a methodological correction to the Phase 13 ablation sub-analysis. The original run artifacts are preserved. These reused OOF labels remain exploratory and are not independent confirmation.",
            "",
        ]
    )
    (output / "ABLATION_SEED_MATCHED_CORRECTION.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    for handler in logger.handlers:
        handler.flush()
    logger.handlers.clear()
    logging.shutdown()

    manifest = []
    for path in sorted(
        item for item in output.iterdir() if item.is_file() and item.name != "SHA256_MANIFEST.csv"
    ):
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for path in (
        Path(__file__),
        ROOT / "run_phase13.py",
        ROOT / "phase13_features.py",
    ):
        manifest.append(
            {
                "file": f"source:{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest).to_csv(output / "SHA256_MANIFEST.csv", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun Phase 13 product-key ablations with matched seeds."
    )
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    print(run(args.original_run, args.train, args.output_root, args.seed))


if __name__ == "__main__":
    main()
