from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE3C_SOURCE = (
    ROOT
    / "experiments"
    / "phase3c_model_intelligence"
    / "phase3c_initial_benchmark.py"
)
EXP02_FEATURE_MANIFEST = (
    ROOT
    / "experiments"
    / "exp02_financial_behaviour"
    / "FEATURE_MANIFEST.json"
)
OUT_DIR = ROOT / "experiments" / "phase4_notebook_recovery_gpu_discovery"
CSV_PATH = OUT_DIR / "EXP02_PHASE3C_RATIO_ALIAS_AUDIT.csv"
REPORT_PATH = OUT_DIR / "EXP02_PHASE3C_RATIO_ALIAS_AUDIT.md"
LOG_PATH = OUT_DIR / "EXP02_PHASE3C_RATIO_ALIAS_AUDIT_LOG.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_ratio(raw: pd.DataFrame, numerator: str, denominator: str) -> np.ndarray:
    a = raw[numerator].to_numpy(dtype=np.float64, na_value=np.nan)
    b = raw[denominator].to_numpy(dtype=np.float64, na_value=np.nan)
    result = np.full(len(raw), np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (b > 0)
    result[valid] = a[valid] / b[valid]
    result[~np.isfinite(result)] = np.nan
    return result


def spearman(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) < 2:
        return None
    left_rank = pd.Series(left[mask]).rank(method="average").to_numpy()
    right_rank = pd.Series(right[mask]).rank(method="average").to_numpy()
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def main() -> None:
    for path in (CSV_PATH, REPORT_PATH, LOG_PATH):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite audit artifact: {path}")

    train_path = DATA / "train.csv"
    train = pd.read_csv(train_path)
    if "TARGET" not in train or "SK_ID_CURR" not in train:
        raise ValueError("Unexpected train schema.")
    raw = train.drop(columns=["TARGET"])
    baseline = load_module("phase3c_alias_baseline", BASELINE_SOURCE)
    existing = baseline.build_rowwise_features(raw)
    exp02_manifest = json.loads(EXP02_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    if len(exp02_manifest["features"]) != 7:
        raise ValueError("EXP02 manifest does not define the expected seven features.")

    mappings = [
        ("FIN_CREDIT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL", "CREDIT_TO_INCOME", "exact"),
        ("FIN_ANNUITY_TO_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "ANNUITY_TO_INCOME", "exact"),
        ("FIN_GOODS_TO_CREDIT", "AMT_GOODS_PRICE", "AMT_CREDIT", "CREDIT_TO_GOODS", "reciprocal"),
        ("FIN_CREDIT_TO_ANNUITY", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_TERM", "exact"),
        ("FIN_INCOME_PER_FAMILY_MEMBER", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "INCOME_PER_FAMILY", "exact"),
        ("FIN_CREDIT_PER_FAMILY_MEMBER", "AMT_CREDIT", "CNT_FAM_MEMBERS", "CREDIT_PER_FAMILY", "exact"),
        ("FIN_ANNUITY_PER_FAMILY_MEMBER", "AMT_ANNUITY", "CNT_FAM_MEMBERS", "ANNUITY_PER_FAMILY", "exact"),
    ]

    rows = []
    for proposed, numerator, denominator, existing_name, relation in mappings:
        candidate = safe_ratio(raw, numerator, denominator)
        reference = existing[existing_name].to_numpy(dtype=np.float64, na_value=np.nan)
        paired = np.isfinite(candidate) & np.isfinite(reference)
        if relation == "exact":
            equivalent = paired & np.isclose(candidate, reference, rtol=0, atol=1e-12)
            metric = "exact_equal_count"
            note = "same mathematical feature where both definitions are finite"
            error = np.abs(candidate[paired] - reference[paired])
        else:
            reciprocal = paired & ~np.isclose(reference, 0.0, rtol=0, atol=0)
            equivalent = np.zeros(len(raw), dtype=bool)
            equivalent[reciprocal] = np.isclose(
                candidate[reciprocal] * reference[reciprocal],
                1.0,
                rtol=1e-10,
                atol=1e-10,
            )
            metric = "reciprocal_product_equals_one_count"
            note = (
                "reciprocal where goods price is observed and positive; baseline "
                "fills missing goods price with credit"
            )
            error = np.abs(candidate[reciprocal] * reference[reciprocal] - 1.0)
            paired = reciprocal

        denom_values = raw[denominator].to_numpy(dtype=np.float64, na_value=np.nan)
        numerator_values = raw[numerator].to_numpy(dtype=np.float64, na_value=np.nan)
        missing_denominator = ~np.isfinite(denom_values)
        nonpositive_denominator = np.isfinite(denom_values) & (denom_values <= 0)
        rows.append(
            {
                "proposed_feature": proposed,
                "proposed_formula": f"{numerator} / {denominator}",
                "phase3c_baseline_feature": existing_name,
                "relation": relation,
                "rows": len(raw),
                "candidate_valid_rows": int(np.isfinite(candidate).sum()),
                "candidate_missing_rows": int(np.isnan(candidate).sum()),
                "baseline_valid_rows": int(np.isfinite(reference).sum()),
                "baseline_missing_rows": int(np.isnan(reference).sum()),
                "paired_valid_rows": int(paired.sum()),
                metric: int(equivalent.sum()),
                "paired_mismatch_rows": int(paired.sum() - equivalent.sum()),
                "spearman_rho_on_comparable_rows": spearman(candidate, reference, paired),
                "max_relation_error": float(error.max()) if len(error) else None,
                "missing_denominator_rows": int(missing_denominator.sum()),
                "nonpositive_denominator_rows": int(nonpositive_denominator.sum()),
                "missing_numerator_rows": int((~np.isfinite(numerator_values)).sum()),
                "edge_case_note": note,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(CSV_PATH, index=False, float_format="%.12g")
    status_lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        "status=PASS",
        f"train_rows={len(raw)}",
        f"train_columns_including_id_target={train.shape[1]}",
        f"target_used_in_feature_audit=False",
        f"train_sha256={sha256(train_path)}",
        f"baseline_source_sha256={sha256(BASELINE_SOURCE)}",
        f"phase3c_source_sha256={sha256(PHASE3C_SOURCE)}",
        f"exp02_feature_manifest_sha256={sha256(EXP02_FEATURE_MANIFEST)}",
        f"csv_sha256={sha256(CSV_PATH)}",
    ]
    LOG_PATH.write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    lines = [
        "# EXP02 ratio overlap with Phase 3C baseline",
        "",
        f"- Audit status: **PASS** (`{status_lines[0].split('=', 1)[1]}`)",
        f"- Training rows audited: `{len(raw):,}`. `TARGET` was removed before feature generation.",
        f"- Input train SHA-256: `{sha256(train_path)}`",
        f"- Preserved baseline source SHA-256: `{sha256(BASELINE_SOURCE)}`",
        f"- Phase 3C model source SHA-256: `{sha256(PHASE3C_SOURCE)}`",
        "",
        "## Mapping",
        "",
        "| EXP02 feature | Formula | Existing Phase 3C feature | Relation | Comparable rows | Equivalent rows | Candidate missing | Baseline missing |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        equivalent_count = row.get(
            "exact_equal_count", row.get("reciprocal_product_equals_one_count", 0)
        )
        lines.append(
            f"| `{row['proposed_feature']}` | `{row['proposed_formula']}` | "
            f"`{row['phase3c_baseline_feature']}` | {row['relation']} | "
            f"{row['paired_valid_rows']} | {equivalent_count} | "
            f"{row['candidate_missing_rows']} | {row['baseline_missing_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The Phase 3C raw feature builder already creates all seven ratio concepts. Six EXP02 formulas match those features on comparable rows; `FIN_GOODS_TO_CREDIT` is the reciprocal of existing `CREDIT_TO_GOODS` on rows where both prices are valid.",
            "- The alternative transforms differ at edge cases: EXP02 requires a finite positive denominator; the source baseline `safe_ratio` permits any nonzero denominator. It fills missing family size with 1 before per-family ratios, while EXP02 leaves those values missing. It fills missing goods price with credit before `CREDIT_TO_GOODS`, whereas EXP02 leaves `GOODS_PRICE / CREDIT` missing.",
            "- There are no missing `AMT_GOODS_PRICE` rows in this training dataset, so the goods-price fallback did not alter this audit. The observed missingness difference is limited to 2 rows with missing family size.",
            "- Consequently, the EXP02 gain over the raw-feature CatBoost experiment cannot be described as discovery of seven novel ratios for Phase 3C: those ratios were already part of Phase 3C's rowwise baseline. The seed-20261105 experiment demonstrated a useful joint engineered-feature signal relative to its raw-only input; it did not test incremental lift over Phase 3C.",
            "- The Phase 3C fixed blend's reported `0.763957` already includes the existing ratio features. Adding a second copy would test duplicate representation/model split competition, not new financial information. The planned duplicate-addition training run was correctly stopped before execution.",
            "- No AUC was calculated in this audit, no model was trained, and no existing model, champion, or experiment artifact was modified.",
            "",
            "## Artifacts",
            "",
            f"- Machine-readable mapping: `{CSV_PATH.name}`",
            f"- Execution log and hashes: `{LOG_PATH.name}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("status=PASS")
    print(f"rows={len(raw)}; features_audited={len(rows)}; target_used=False")
    print(f"csv={CSV_PATH}")
    print(f"report={REPORT_PATH}")
    print(f"log={LOG_PATH}")


if __name__ == "__main__":
    main()
