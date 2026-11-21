from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kendalltau, rankdata, spearmanr


ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT.parent / "v22_verified_reference.csv"
CANDIDATE_PATH = (
    ROOT / "runs" / "v1_gpu_final_20261114T2112" / "submission_clean_v1.csv"
)
OUTPUT_PATH = ROOT / "V1_COMPARISON.json"
RUN_SUMMARY_PATH = CANDIDATE_PATH.parent / "V1_SUMMARY.json"
EXPECTED_ROWS = 61_500
ID = "SK_ID_CURR"
TARGET = "TARGET"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    reference = pd.read_csv(REFERENCE_PATH)
    candidate = pd.read_csv(CANDIDATE_PATH)
    if reference.columns.tolist() != [ID, TARGET]:
        raise ValueError("Reference CSV schema is not SK_ID_CURR,TARGET")
    if candidate.columns.tolist() != [ID, TARGET]:
        raise ValueError("Candidate CSV schema is not SK_ID_CURR,TARGET")
    if len(reference) != EXPECTED_ROWS or len(candidate) != EXPECTED_ROWS:
        raise ValueError("CSV row count differs from the expected test size")
    if not reference[ID].is_unique or not candidate[ID].is_unique:
        raise ValueError("CSV IDs are not unique")
    if not np.array_equal(
        reference[ID].to_numpy(), candidate[ID].to_numpy()
    ):
        raise ValueError("Candidate IDs or row order differ from reference")

    reference_target = reference[TARGET].to_numpy(dtype=np.float64)
    candidate_target = candidate[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(reference_target).all() or not np.isfinite(
        candidate_target
    ).all():
        raise ValueError("Found non-finite target values")

    reference_bytes = REFERENCE_PATH.read_bytes()
    candidate_bytes = CANDIDATE_PATH.read_bytes()
    reference_lines = reference_bytes.splitlines()
    candidate_lines = candidate_bytes.splitlines()
    exact_numeric_matches = int(
        np.count_nonzero(reference_target == candidate_target)
    )
    exact_text_matches = sum(
        ref_line.split(b",", 1)[1] == candidate_line.split(b",", 1)[1]
        for ref_line, candidate_line in zip(
            reference_lines[1:], candidate_lines[1:]
        )
    )
    reference_ranks = rankdata(reference_target, method="average")
    candidate_ranks = rankdata(candidate_target, method="average")
    reference_rank_percentiles = (reference_ranks - 1.0) / (EXPECTED_ROWS - 1.0)
    candidate_rank_percentiles = (candidate_ranks - 1.0) / (EXPECTED_ROWS - 1.0)
    run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    result = {
        "reference": str(REFERENCE_PATH),
        "candidate": str(CANDIDATE_PATH),
        "rows": EXPECTED_ROWS,
        "id_order_matches": True,
        "exact_target_numeric_matches": exact_numeric_matches,
        "exact_target_text_matches": int(exact_text_matches),
        "target_match_percent": 100.0 * exact_text_matches / EXPECTED_ROWS,
        "whole_file_byte_equal": reference_bytes == candidate_bytes,
        "exact_normalized_rank_percentile_matches": int(
            np.count_nonzero(
                reference_rank_percentiles == candidate_rank_percentiles
            )
        ),
        "max_normalized_rank_percentile_difference": float(
            np.max(
                np.abs(
                    reference_rank_percentiles
                    - candidate_rank_percentiles
                )
            )
        ),
        "reference_unique_targets": int(np.unique(reference_target).size),
        "candidate_unique_targets": int(np.unique(candidate_target).size),
        "max_absolute_target_difference": float(
            np.max(np.abs(reference_target - candidate_target))
        ),
        "spearman_target_and_rank_correlation": float(
            spearmanr(reference_target, candidate_target).statistic
        ),
        "kendall_rank_correlation": float(
            kendalltau(reference_ranks, candidate_ranks).statistic
        ),
        "reference_sha256": sha256_file(REFERENCE_PATH),
        "candidate_sha256": sha256_file(CANDIDATE_PATH),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scope_note": (
            "V1 is a new clean raw-data-trained replacement and removes more "
            "than one historical risk at once. It is not a controlled "
            "single-feature ablation of the V17/V18 models; those model "
            "predictions must be regenerated to measure the isolated effect "
            "of removing partner-label features."
        ),
        "validation": {
            "pooled_oof_auc": run_summary["pooled_auc"],
            "mean_fold_auc": run_summary["mean_fold_auc"],
            "std_fold_auc": run_summary["std_fold_auc"],
            "log_loss": run_summary["log_loss"],
            "runtime_seconds": run_summary["runtime_seconds"],
            "gpu_requested": "device=\"cuda\""
            in (ROOT / "run_clean_v1.py").read_text(encoding="utf-8"),
            "rule_controls": run_summary["rule_controls"],
        },
        "interpretation": (
            "This is a distinct raw-data-trained model, not a reconstruction "
            "of the historical V17/V18 predictions. Rank correlation is a "
            "descriptive comparison, not a leaderboard score."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
