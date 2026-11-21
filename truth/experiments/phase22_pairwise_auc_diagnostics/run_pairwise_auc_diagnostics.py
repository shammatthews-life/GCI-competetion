from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
PHASE15_OOF = (
    REPO
    / "experiments"
    / "phase15_autonomous_research"
    / "runs"
    / "phase15_native_catboost_oof_20261105T124238Z"
    / "OOF_PREDICTIONS.csv"
)
PHASE18_OOF = (
    REPO
    / "experiments"
    / "phase18_nested_oof_ensemble"
    / "runs"
    / "phase18_nested_blend_20261105T142048Z"
    / "OOF_PREDICTIONS.csv"
)
DEFAULT_OUTPUT = ROOT / "runs" / "phase22_pairwise_20261106_final"
N_BANDS = 10
ID = "SK_ID_CURR"
TARGET = "TARGET"
FOLD = "fold"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alignment(
    train: pd.DataFrame, phase15: pd.DataFrame, phase18: pd.DataFrame
) -> None:
    required_train = {ID, TARGET}
    required_oof = {ID, TARGET, FOLD, "row_index"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"Training file is missing {required_train - set(train.columns)}")
    for name, frame in (("Phase 15", phase15), ("Phase 18", phase18)):
        if not required_oof.issubset(frame.columns):
            raise ValueError(f"{name} OOF is missing {required_oof - set(frame.columns)}")
        if len(frame) != len(train):
            raise ValueError(f"{name} OOF row count differs from train")
        if not np.array_equal(frame[ID].to_numpy(), train[ID].to_numpy()):
            raise ValueError(f"{name} OOF IDs do not align with train")
        if not np.array_equal(
            frame[TARGET].to_numpy(dtype=np.int8),
            train[TARGET].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"{name} OOF targets do not align with train")
        if not np.array_equal(frame["row_index"].to_numpy(), np.arange(len(train))):
            raise ValueError(f"{name} OOF row_index is not the train row order")
    if not np.array_equal(phase15[FOLD].to_numpy(), phase18[FOLD].to_numpy()):
        raise ValueError("Phase 15 and Phase 18 OOF folds differ")
    if not np.array_equal(
        phase15["native_catboost_phase14_plus_time"].to_numpy(dtype=np.float64),
        phase18["phase15_catboost"].to_numpy(dtype=np.float64),
    ):
        raise ValueError("Phase 18 CatBoost component differs from Phase 15 reference")


def score_deciles(scores: np.ndarray, n_bands: int = N_BANDS) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    if n_bands < 2:
        raise ValueError("At least two score bands are required")
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bands + 1))
    bands = np.searchsorted(edges[1:-1], values, side="right")
    return bands.astype(np.int8)


def pairwise_cell_counts(
    labels: np.ndarray,
    bands: np.ndarray,
    scores: np.ndarray,
    n_bands: int = N_BANDS,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int8)
    band_ids = np.asarray(bands, dtype=np.int8)
    prediction = np.asarray(scores, dtype=np.float64)
    if not (len(y) == len(band_ids) == len(prediction)):
        raise ValueError("Labels, bands, and scores must have equal length")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(prediction).all():
        raise ValueError("Labels must be binary and scores finite")
    counts = np.zeros((n_bands, n_bands), dtype=np.float64)
    for positive_band in range(n_bands):
        positive_scores = prediction[(y == 1) & (band_ids == positive_band)]
        if not len(positive_scores):
            continue
        for negative_band in range(n_bands):
            negative_scores = prediction[(y == 0) & (band_ids == negative_band)]
            if not len(negative_scores):
                continue
            ordered_negative = np.sort(negative_scores)
            lower = np.searchsorted(ordered_negative, positive_scores, side="left")
            upper = np.searchsorted(ordered_negative, positive_scores, side="right")
            counts[positive_band, negative_band] = np.sum(
                lower + 0.5 * (upper - lower)
            )
    return counts


def make_diagnostics(
    labels: np.ndarray,
    base_scores: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    y = np.asarray(labels, dtype=np.int8)
    base = np.asarray(base_scores, dtype=np.float64)
    bands = score_deciles(base)
    positive_counts = np.bincount(bands[y == 1], minlength=N_BANDS)
    negative_counts = np.bincount(bands[y == 0], minlength=N_BANDS)
    pair_total = int(positive_counts.sum() * negative_counts.sum())
    if pair_total == 0:
        raise ValueError("AUC diagnostics require at least one positive and one negative")
    base_counts = pairwise_cell_counts(y, bands, base)
    if not np.isclose(base_counts.sum() / pair_total, roc_auc_score(y, base)):
        raise RuntimeError("Pairwise band decomposition does not reproduce baseline AUC")

    cell_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    aucs: dict[str, float] = {"baseline": float(roc_auc_score(y, base))}
    candidate_comparison: dict[str, dict[str, float]] = {}
    for name, candidate in candidates.items():
        prediction = np.asarray(candidate, dtype=np.float64)
        if prediction.shape != base.shape or not np.isfinite(prediction).all():
            raise ValueError(f"{name} candidate scores are invalid")
        candidate_counts = pairwise_cell_counts(y, bands, prediction)
        auc = float(roc_auc_score(y, prediction))
        if not np.isclose(candidate_counts.sum() / pair_total, auc):
            raise RuntimeError(f"{name} pairwise decomposition does not reproduce AUC")
        aucs[name] = auc
        candidate_comparison[name] = {
            "auc": auc,
            "delta_auc_vs_baseline": auc - aucs["baseline"],
            "net_pair_change": float(candidate_counts.sum() - base_counts.sum()),
            "net_pair_change_bottom_half_positive": float(
                candidate_counts[: N_BANDS // 2].sum()
                - base_counts[: N_BANDS // 2].sum()
            ),
        }

        for positive_band in range(N_BANDS):
            baseline_wrong = 0.0
            baseline_pair_count = 0
            candidate_concordant = 0.0
            for negative_band in range(N_BANDS):
                pairs = int(positive_counts[positive_band] * negative_counts[negative_band])
                base_concordant = float(base_counts[positive_band, negative_band])
                cand_concordant = float(candidate_counts[positive_band, negative_band])
                baseline_wrong += pairs - base_concordant
                baseline_pair_count += pairs
                candidate_concordant += cand_concordant
                cell_rows.append(
                    {
                        "candidate": name,
                        "positive_baseline_risk_decile": positive_band + 1,
                        "negative_baseline_risk_decile": negative_band + 1,
                        "positive_rows": int(positive_counts[positive_band]),
                        "negative_rows": int(negative_counts[negative_band]),
                        "pair_count": pairs,
                        "baseline_concordant_pairs": base_concordant,
                        "candidate_concordant_pairs": cand_concordant,
                        "baseline_pair_auc": (
                            base_concordant / pairs if pairs else np.nan
                        ),
                        "candidate_pair_auc": (
                            cand_concordant / pairs if pairs else np.nan
                        ),
                        "net_pair_change": cand_concordant - base_concordant,
                        "delta_auc_contribution": (
                            (cand_concordant - base_concordant) / pair_total
                        ),
                    }
                )
            risk_rows.append(
                {
                    "candidate": name,
                    "positive_baseline_risk_decile": positive_band + 1,
                    "positive_rows": int(positive_counts[positive_band]),
                    "all_negative_pairs": baseline_pair_count,
                    "baseline_concordant_pairs": float(
                        base_counts[positive_band].sum()
                    ),
                    "candidate_concordant_pairs": candidate_concordant,
                    "baseline_wrong_pairs": baseline_wrong,
                    "baseline_wrong_pair_share": (
                        baseline_wrong / (pair_total - base_counts.sum())
                        if pair_total != base_counts.sum()
                        else 0.0
                    ),
                    "baseline_pair_auc": (
                        base_counts[positive_band].sum() / baseline_pair_count
                        if baseline_pair_count
                        else np.nan
                    ),
                    "candidate_pair_auc": (
                        candidate_concordant / baseline_pair_count
                        if baseline_pair_count
                        else np.nan
                    ),
                    "net_pair_change": candidate_concordant
                    - base_counts[positive_band].sum(),
                }
            )

    summary = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "positive_negative_pairs": pair_total,
        "baseline_auc": aucs["baseline"],
        "candidate_comparison": candidate_comparison,
        "candidate_aucs": candidates_auc(aucs),
        "baseline_error_pairs": float(pair_total - base_counts.sum()),
        "baseline_error_pair_share_by_positive_risk_half": {},
    }
    base_error_pairs = pair_total - base_counts.sum()
    for region, selected_bands in (
        ("bottom_half", range(0, N_BANDS // 2)),
        ("top_half", range(N_BANDS // 2, N_BANDS)),
    ):
        wrong = float(
            sum(
                int(positive_counts[band]) * int(negative_counts.sum())
                - base_counts[band].sum()
                for band in selected_bands
            )
        )
        summary["baseline_error_pair_share_by_positive_risk_half"][region] = {
            "baseline_wrong_pairs": wrong,
            "share_of_all_baseline_wrong_pairs": (
                wrong / base_error_pairs if base_error_pairs else 0.0
            ),
            "positive_rows": int(positive_counts[list(selected_bands)].sum()),
        }
    for name, stats in candidate_comparison.items():
        net = stats["net_pair_change"]
        if not np.isclose(net / pair_total, aucs[name] - aucs["baseline"]):
            raise RuntimeError(f"{name} pairwise net change does not match AUC delta")
    return pd.DataFrame(cell_rows), pd.DataFrame(risk_rows), summary


def candidates_auc(aucs: dict[str, float]) -> dict[str, float]:
    return {name: score for name, score in aucs.items() if name != "baseline"}


def write_report(output_dir: Path, summary: dict[str, Any], risk: pd.DataFrame) -> None:
    baseline = summary["baseline_auc"]
    lines = [
        "# Phase 22 — pairwise AUC error decomposition",
        "",
        "## Scope",
        "",
        "This is a post-hoc diagnostic of saved out-of-fold predictions. It trains no models and reads only `train.csv`, Phase 15 OOF predictions, and Phase 18 OOF predictions. It does not read test data. These labels have been examined in earlier phases; findings are exploratory, not independent confirmation.",
        "",
        "## Rank-pair accounting",
        "",
        f"- Rows: {summary['rows']:,}; positives: {summary['positives']:,} ({summary['positive_rate']:.3%}); negatives: {summary['negatives']:,}.",
        f"- Positive-negative pairs: {summary['positive_negative_pairs']:,}.",
        f"- Phase 15 baseline OOF AUC: {baseline:.9f}.",
        f"- Baseline incorrectly ranked pairs: {summary['baseline_error_pairs']:,.1f} ({1.0 - baseline:.3%}).",
        "",
        "| Candidate | OOF AUC | Delta AUC | Net pair change | Net gain from bottom-half positives |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in summary["candidate_comparison"].items():
        lines.append(
            f"| {name} | {stats['auc']:.9f} | "
            f"{stats['delta_auc_vs_baseline']:+.9f} | "
            f"{stats['net_pair_change']:+,.1f} | "
            f"{stats['net_pair_change_bottom_half_positive']:+,.1f} |"
        )
    lines.extend(
        [
            "",
            "The pair matrix groups both positives and negatives into deciles of the Phase 15 score. Each cell reports all positive-negative comparisons in that pair of score regions, so its contribution is weighted by the number of pairs, not by the number of rows.",
        "",
        "### Where baseline ranking errors occur",
        "",
        "| Positive rows' baseline score region | Positive rows | Baseline wrong pairs | Share of all baseline wrong pairs |",
        "|---|---:|---:|---:|",
    ]
    )
    for region, stats in summary[
        "baseline_error_pair_share_by_positive_risk_half"
    ].items():
        lines.append(
            f"| {region.replace('_', ' ')} | {stats['positive_rows']:,} | "
            f"{stats['baseline_wrong_pairs']:,.1f} | "
            f"{stats['share_of_all_baseline_wrong_pairs']:.2%} |"
        )
    lines.extend(
        [
            "",
            "### Candidate error corrections by positive-risk decile",
            "",
            "Positive-risk deciles are defined by the baseline score. `Net pair change` is candidate concordant pairs minus baseline concordant pairs over every negative; it is a net count (pairs corrected minus pairs newly broken), not the gross number corrected.",
            "",
            "| Candidate | Pos. risk decile | Positives | Baseline pair AUC | Candidate pair AUC | Net pair change |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in risk.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.positive_baseline_risk_decile} | "
            f"{row.positive_rows:,} | {row.baseline_pair_auc:.6f} | "
            f"{row.candidate_pair_auc:.6f} | {row.net_pair_change:,.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A large share of errors in the bottom-half positive-score region would support investigating low-scored positives, but would not prove that adding more features there improves total AUC. The full 10x10 matrix should guide any later, predeclared candidate design. No model selection or blend tuning is performed here.",
            "",
            "Per-model calibration does not change AUC if it is a single strictly increasing transform applied to every row. It can change a blend's cross-model ordering, which is why the Phase 18 mixture is compared directly by pairwise outcomes.",
        ]
    )
    (output_dir / "PHASE22_EXECUTION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_manifest(output_dir: Path, inputs: list[Path], source: Path) -> None:
    rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "PHASE22_SHA256_MANIFEST.csv":
            rows.append(
                {
                    "kind": "output",
                    "path": str(path.relative_to(REPO)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in [*inputs, source]:
        rows.append(
            {
                "kind": "source_or_input",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "PHASE22_SHA256_MANIFEST.csv",
        index=False,
        lineterminator="\n",
    )


def run(train_path: Path, phase15_path: Path, phase18_path: Path, output_dir: Path) -> None:
    for path in (train_path, phase15_path, phase18_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path, low_memory=False)
    phase15 = pd.read_csv(phase15_path)
    phase18 = pd.read_csv(phase18_path)
    validate_alignment(train, phase15, phase18)
    labels = train[TARGET].to_numpy(dtype=np.int8)
    base = phase15["native_catboost_phase14_plus_time"].to_numpy(dtype=np.float64)
    candidates = {
        "phase18_catboost_component": phase18["phase15_catboost"].to_numpy(
            dtype=np.float64
        ),
        "phase17_nested_te_lightgbm": phase18[
            "phase17_nested_te_lightgbm"
        ].to_numpy(dtype=np.float64),
        "phase18_nested_blend": phase18["phase18_nested_blend"].to_numpy(
            dtype=np.float64
        ),
    }
    cells, risk, summary = make_diagnostics(labels, base, candidates)
    summary["train_sha256"] = sha256_file(train_path)
    summary["phase15_oof_sha256"] = sha256_file(phase15_path)
    summary["phase18_oof_sha256"] = sha256_file(phase18_path)
    summary["score_deciles"] = N_BANDS
    summary["fold_ids_match"] = True
    summary["independent_confirmation"] = False
    summary["models_trained"] = 0
    summary["test_data_read"] = False
    cells.to_csv(output_dir / "PHASE22_PAIRWISE_CELLS.csv", index=False, lineterminator="\n")
    risk.to_csv(
        output_dir / "PHASE22_POSITIVE_RISK_BANDS.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "PHASE22_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(output_dir, summary, risk)
    write_manifest(
        output_dir,
        [train_path, phase15_path, phase18_path],
        Path(__file__).resolve(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--phase15-oof", type=Path, default=PHASE15_OOF)
    parser.add_argument("--phase18-oof", type=Path, default=PHASE18_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        args.train.resolve(),
        args.phase15_oof.resolve(),
        args.phase18_oof.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
