from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phase13_features import PRODUCT_KEYS


ROOT = Path(__file__).resolve().parent
WEIGHTS = (0.05, 0.10, 0.20)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(hidden_run: Path, output_root: Path) -> Path:
    rate_path = hidden_run / "PRODUCT_KEY_CROSSFIT_RATE_OOF.csv"
    result_path = hidden_run / "HIDDEN_KEY_CANDIDATE_RESULTS.csv"
    for path in (rate_path, result_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required hidden-screen file not found: {path}")
    start = datetime.now(timezone.utc)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"hidden_key_weight_supplement_{start.strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(exist_ok=False)
    rate_hash = sha256_file(rate_path)
    result_hash = sha256_file(result_path)
    rates = pd.read_csv(rate_path)
    if len(rates) != 171202 or not np.array_equal(
        rates["row_index"].to_numpy(), np.arange(len(rates))
    ):
        raise ValueError("Saved cross-fit key rates are not row-aligned")
    y = rates["TARGET"].to_numpy(dtype=np.int8)
    baseline = rates["baseline_C_oof"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline).all():
        raise ValueError("Baseline OOF predictions are incomplete")
    baseline_auc = float(roc_auc_score(y, baseline))
    existing = pd.read_csv(result_path).set_index("candidate")
    rows = []
    for key in PRODUCT_KEYS:
        name = f"product_key:{key}"
        if name not in rates or name not in existing.index:
            raise ValueError(f"Missing positive-control key: {name}")
        encoded = rates[name].to_numpy(dtype=np.float64)
        if not np.isfinite(encoded).all():
            raise ValueError(f"Incomplete cross-fit target rate: {name}")
        encoded_auc = float(roc_auc_score(y, encoded))
        category_count = int(existing.loc[name, "unique_categories"])
        for weight in WEIGHTS:
            prediction = (1.0 - weight) * baseline + weight * encoded
            auc = float(roc_auc_score(y, prediction))
            rows.append(
                {
                    "candidate": name,
                    "key": key,
                    "unique_categories": category_count,
                    "target_rate_oof_auc": encoded_auc,
                    "blend_weight": weight,
                    "baseline_oof_auc": baseline_auc,
                    "blend_oof_auc": auc,
                    "delta_vs_C": auc - baseline_auc,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output / "PRODUCT_KEY_BLEND_WEIGHTS.csv", index=False)
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                "# Hidden-screen product-key blend-weight supplement",
                "",
                f"- Source screen: `{hidden_run}`",
                f"- Cross-fit rate file SHA-256: `{rate_hash}`",
                f"- Candidate-results file SHA-256: `{result_hash}`",
                f"- Rows: {len(rates):,}; baseline C pooled OOF AUC: {baseline_auc:.9f}.",
                "- Recomputed fixed blends for all eight positive-control keys from their saved, five-fold cross-fitted target-rate vectors. This fills the 0.05 and 0.20 weights not present for explicit keys in the initial screen output.",
                "- These label-dependent encodings are exploratory and rule eligibility remains unresolved. The blend weights are descriptive checks on exposed OOF labels, not nested model selection.",
                "",
                "| Key | Categories | Weight | Blend OOF AUC | Delta vs C |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {row.key} | {row.unique_categories:,} | {row.blend_weight:.2f} | "
                    f"{row.blend_oof_auc:.6f} | {row.delta_vs_C:+.6f} |"
                    for row in result.itertuples(index=False)
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "EXECUTION_LOG.txt").write_text(
        f"Started UTC: {start.isoformat()}\n"
        f"Completed UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Cross-fit key-rate SHA-256: {rate_hash}\n",
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
    for path in (
        Path(__file__),
        ROOT / "run_hidden_key_screen.py",
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
    if sha256_file(rate_path) != rate_hash or sha256_file(result_path) != result_hash:
        raise AssertionError("Hidden-screen source files changed during supplement")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute blend-weight sensitivity for all product keys."
    )
    parser.add_argument("--hidden-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    print(run(args.hidden_run, args.output_root))


if __name__ == "__main__":
    main()
