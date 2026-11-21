from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase13_features import ID_COLUMN, TARGET_COLUMN
from run_phase13 import BOOTSTRAP_REPETITIONS, paired_bootstrap


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(run_dir: Path, output_root: Path) -> Path:
    source = run_dir / "OOF_PREDICTIONS.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Phase 13 OOF predictions not found: {source}")
    start = datetime.now(timezone.utc)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"phase13_comparison_supplement_{start.strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(exist_ok=False)
    source_hash = sha256_file(source)
    oof = pd.read_csv(
        source,
        usecols=[
            "row_index",
            ID_COLUMN,
            TARGET_COLUMN,
            "primary_fold",
            "D_C_plus_keys_ohe",
            "E_C_plus_keys_frequency_ohe",
        ],
    )
    y = oof[TARGET_COLUMN].to_numpy(dtype=np.int8)
    baseline = oof["D_C_plus_keys_ohe"].to_numpy(dtype=np.float64)
    candidate = oof["E_C_plus_keys_frequency_ohe"].to_numpy(dtype=np.float64)
    if len(oof) != 171202 or not np.array_equal(
        oof["row_index"].to_numpy(), np.arange(len(oof))
    ):
        raise ValueError("Phase 13 OOF rows do not match the fixed training row order")
    if oof[ID_COLUMN].isna().any() or oof[ID_COLUMN].duplicated().any():
        raise ValueError("Phase 13 OOF IDs must be present and unique")
    if set(oof["primary_fold"].unique()) != {1, 2, 3, 4, 5}:
        raise ValueError("Phase 13 OOF must contain all five primary folds")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("A or E OOF predictions are incomplete")
    baseline_auc = float(roc_auc_score(y, baseline))
    candidate_auc = float(roc_auc_score(y, candidate))
    low, high = paired_bootstrap(
        y,
        baseline,
        candidate,
        seed=20261115,
    )
    result = pd.DataFrame(
        [
            {
                "candidate": "E_C_plus_keys_frequency_ohe",
                "reference": "D_C_plus_keys_ohe",
                "reference_pooled_oof_auc": baseline_auc,
                "candidate_pooled_oof_auc": candidate_auc,
                "paired_delta": candidate_auc - baseline_auc,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "bootstrap_seed": 20261115,
                "interpretation": "Conditional on fitted OOF predictions; not independent confirmation.",
            }
        ]
    )
    result.to_csv(output / "FREQUENCY_VS_ORDINARY_PAIRED_BOOTSTRAP.csv", index=False)
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                "# Phase 13 direct frequency-support comparison supplement",
                "",
                f"- Source Phase 13 run: `{run_dir}`",
                f"- OOF SHA-256: `{source_hash}`",
                "- Comparison: E (fit-fold frequency-supported one-hot) versus D (ordinary one-hot), evaluated on identical primary outer-fold rows.",
                f"- D pooled OOF AUC: {baseline_auc:.9f}",
                f"- E pooled OOF AUC: {candidate_auc:.9f}",
                f"- E-D: {candidate_auc - baseline_auc:+.9f}",
                f"- Paired stratified bootstrap 95% interval: [{low:+.9f}, {high:+.9f}] from {BOOTSTRAP_REPETITIONS:,} resamples.",
                "- This interval is conditional on the fitted OOF predictions and the previously examined labels. It is not independent confirmation. The interval includes zero, so the nominal E-over-D gain is not conclusive evidence that frequency support is superior.",
                "- The secondary-seed run compared selected E against C, not E against D; it therefore does not independently replicate the E-D contrast.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "EXECUTION_LOG.txt").write_text(
        f"Started UTC: {start.isoformat()}\n"
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"OOF SHA-256: {source_hash}\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(
        item
        for item in output.iterdir()
        if item.is_file() and item.name != "SHA256_MANIFEST.csv"
    ):
        manifest.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    for path in (Path(__file__), ROOT / "run_phase13.py"):
        manifest.append(
            {
                "file": f"source:{path.name}",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(manifest).to_csv(output / "SHA256_MANIFEST.csv", index=False)
    if sha256_file(source) != source_hash:
        raise AssertionError("OOF source changed during the bootstrap")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap the direct E versus D OOF contrast."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    print(run(args.run_dir, args.output_root))


if __name__ == "__main__":
    main()
