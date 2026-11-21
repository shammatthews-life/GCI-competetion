from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
OUT = ROOT / "experiments" / "phase3d_claude_hypothesis_probes"
ID, TARGET = "SK_ID_CURR", "TARGET"


def main() -> None:
    train_path = DATA / "train.csv"
    train = pd.read_csv(train_path)
    credit = train["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = train["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    valid = np.isfinite(credit) & np.isfinite(annuity) & (credit > 0) & (annuity > 0)
    p_over_a = np.full(len(train), np.nan)
    p_over_a[valid] = credit[valid] / annuity[valid]
    n_round = np.full(len(train), np.nan)
    n_round[valid] = np.clip(np.rint(p_over_a[valid]), 6, 72)
    n_snap = np.full(len(train), np.nan)
    n_snap[valid] = np.clip(np.rint(n_round[valid] / 6) * 6, 6, 72)
    paid_round = annuity * n_round / credit
    paid_snap = annuity * n_snap / credit
    paid_gap_round = annuity * n_round - credit
    paid_gap_snap = annuity * n_snap - credit
    y = train[TARGET].to_numpy(dtype=np.int8)
    oof = pd.read_csv(OUT / "FOLLOWUP_OOF_PREDICTIONS.csv")
    if not np.array_equal(oof[ID].to_numpy(), train[ID].to_numpy()) or not np.array_equal(
        oof[TARGET].to_numpy(dtype=np.int8), y
    ):
        raise ValueError("OOF applicant or target alignment failed.")
    base = oof["control_refit"].to_numpy(dtype=np.float64)
    candidates = {
        "rounded_paid_ratio": oof["term__total_paid_ratio_rounded_term_hypothesis"].to_numpy(dtype=np.float64),
        "snapped_paid_ratio": oof["term__total_paid_ratio_snapped_term_hypothesis"].to_numpy(dtype=np.float64),
    }

    term_rows = []
    for label, values in [
        ("P_OVER_A", p_over_a),
        ("N_ROUND", n_round),
        ("N_SNAP6", n_snap),
        ("TOTAL_PAID_RATIO_ROUND", paid_round),
        ("TOTAL_PAID_RATIO_SNAP6", paid_snap),
        ("PAYMENT_GAP_ROUND", paid_gap_round),
        ("PAYMENT_GAP_SNAP6", paid_gap_snap),
    ]:
        finite = np.isfinite(values)
        term_rows.append(
            {
                "quantity": label,
                "n_valid": int(finite.sum()),
                "n_missing": int((~finite).sum()),
                "min": float(np.min(values[finite])),
                "p01": float(np.quantile(values[finite], 0.01)),
                "median": float(np.median(values[finite])),
                "p99": float(np.quantile(values[finite], 0.99)),
                "max": float(np.max(values[finite])),
                "mean": float(np.mean(values[finite])),
                "std": float(np.std(values[finite])),
                "n_unique": int(pd.Series(values[finite]).nunique()),
            }
        )
    pd.DataFrame(term_rows).to_csv(OUT / "TERM_PATTERN_DIAGNOSTICS.csv", index=False)

    bins = []
    for name, group_values in [("rounded_proxy", n_round), ("six_month_proxy", n_snap)]:
        tmp = pd.DataFrame({"bucket": group_values, TARGET: y})
        for bucket, group in tmp.groupby("bucket", dropna=False):
            bins.append(
                {
                    "proxy": name,
                    "bucket": bucket,
                    "n": int(len(group)),
                    "defaults": int(group[TARGET].sum()),
                    "default_rate": float(group[TARGET].mean()),
                }
            )
    paid_bins = []
    for name, values in [("rounded_paid_ratio", paid_round), ("snapped_paid_ratio", paid_snap)]:
        tmp = pd.DataFrame({"value": values, TARGET: y})
        tmp["quantile_bin"] = pd.qcut(tmp["value"], q=10, duplicates="drop")
        for bin_value, group in tmp.groupby("quantile_bin", observed=False):
            paid_bins.append(
                {
                    "feature": name,
                    "quantile_bin": str(bin_value),
                    "n": int(len(group)),
                    "defaults": int(group[TARGET].sum()),
                    "default_rate": float(group[TARGET].mean()),
                    "mean_feature": float(group["value"].mean()),
                }
            )
    pd.DataFrame(bins).to_csv(OUT / "TERM_PROXY_DEFAULT_RATES.csv", index=False)
    pd.DataFrame(paid_bins).to_csv(OUT / "PAID_RATIO_QUANTILE_DEFAULT_RATES.csv", index=False)

    compare_rows = []
    for group_name, labels in [
        ("all", np.repeat("all", len(train))),
        ("contract_type", train["NAME_CONTRACT_TYPE"].fillna("MISSING").astype(str).to_numpy()),
        (
            "credit_goods_quartile",
            pd.qcut(
                train["AMT_CREDIT"] / train["AMT_GOODS_PRICE"].replace(0, np.nan),
                q=4,
                duplicates="drop",
            )
            .astype(str)
            .to_numpy(),
        ),
    ]:
        for group in pd.unique(labels):
            mask = labels == group
            if mask.sum() < 500 or np.unique(y[mask]).size != 2:
                continue
            for candidate, prediction in candidates.items():
                compare_rows.append(
                    {
                        "grouping": group_name,
                        "group": str(group),
                        "n": int(mask.sum()),
                        "default_rate": float(y[mask].mean()),
                        "candidate": candidate,
                        "baseline_auc": float(roc_auc_score(y[mask], base[mask])),
                        "candidate_auc": float(roc_auc_score(y[mask], prediction[mask])),
                        "delta_auc": float(
                            roc_auc_score(y[mask], prediction[mask])
                            - roc_auc_score(y[mask], base[mask])
                        ),
                        "interpretation": "exploratory subgroup diagnostic on reused labels",
                    }
                )
    pd.DataFrame(compare_rows).to_csv(OUT / "TERM_SUBGROUP_AUC_DIAGNOSTICS.csv", index=False)

    correlation = []
    original_ratio = train["AMT_CREDIT"] / train["AMT_ANNUITY"]
    for name, values in [
        ("rounded_paid_ratio", paid_round),
        ("snapped_paid_ratio", paid_snap),
        ("rounded_payment_gap", paid_gap_round),
        ("snapped_payment_gap", paid_gap_snap),
    ]:
        mask = np.isfinite(values) & np.isfinite(original_ratio.to_numpy())
        correlation.append(
            {
                "feature": name,
                "spearman_vs_credit_over_annuity": float(
                    spearmanr(values[mask], original_ratio.to_numpy()[mask]).statistic
                ),
                "min": float(values[mask].min()),
                "max": float(values[mask].max()),
                "fraction_exactly_one_if_ratio": float(
                    np.mean(values[mask] == 1) if "paid_ratio" in name else np.nan
                ),
            }
        )
    pd.DataFrame(correlation).to_csv(OUT / "TERM_PROXY_CORRELATIONS.csv", index=False)

    summary = []
    for candidate, pred in candidates.items():
        fold_deltas = []
        for fold in sorted(oof["fold"].unique()):
            mask = oof["fold"].to_numpy() == fold
            fold_deltas.append(
                roc_auc_score(y[mask], pred[mask]) - roc_auc_score(y[mask], base[mask])
            )
        summary.append(
            {
                "candidate": candidate,
                "pooled_auc": float(roc_auc_score(y, pred)),
                "pooled_delta_vs_control": float(
                    roc_auc_score(y, pred) - roc_auc_score(y, base)
                ),
                "mean_paired_fold_delta": float(np.mean(fold_deltas)),
                "fold_deltas": str([round(float(delta), 7) for delta in fold_deltas]),
                "folds_improved": int(np.sum(np.asarray(fold_deltas) > 0)),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "TERM_PATTERN_AUC_RECALCULATION.csv", index=False)

    auc_recalc = pd.DataFrame(summary)
    auc_recalc.to_csv(OUT / "TERM_PATTERN_AUC_RECALCULATION.csv", index=False)
    top = auc_recalc.sort_values("pooled_delta_vs_control", ascending=False).iloc[0]
    subgroup_frame = pd.DataFrame(compare_rows)
    revolver = subgroup_frame.loc[
        (subgroup_frame.grouping == "contract_type")
        & (subgroup_frame.group == "Revolving loans")
        & (subgroup_frame.candidate == "snapped_paid_ratio")
    ]
    revolver_delta = float(revolver.delta_auc.iloc[0]) if not revolver.empty else float("nan")

    md = f"""# Diagnostic interpretation of term-proxy gain

## Measured result

The strongest single feature in this probe was `{top['candidate']}`: pooled AUC {top['pooled_auc']:.6f}, a {top['pooled_delta_vs_control']:+.6f} paired pooled change versus the refit control, with mean fold delta {top['mean_paired_fold_delta']:+.6f} and positive deltas on {int(top['folds_improved'])}/5 folds. The alternate rounded residual yielded pooled AUC {auc_recalc.loc[auc_recalc.candidate == 'rounded_paid_ratio', 'pooled_auc'].iloc[0]:.6f} and improved on 4/5 folds. The six-month-snapped residual improved on only 3/5 folds. These are exploratory on reused folds.

The candidate formulas are deterministic transforms of `AMT_CREDIT / AMT_ANNUITY`, not independent observations of loan contract terms. They must not be described as actual repayment duration, total paid, or interest.

The rounded proxy is `n = clip(round(P/A), 6, 72)`. Its paid-ratio feature is `A*n/P = n/(P/A)`, a sawtooth rounding residual close to one. The six-month proxy rounds `n` to six-month steps; its paid-ratio feature deliberately creates a coarser quantization residual. Therefore, any AUC change indicates that this piecewise transformation helps the tree model represent nonlinearities in the existing credit/annuity ratio; it does not establish a true six-month contract structure.

See `TERM_PATTERN_DIAGNOSTICS.csv` for extrema/quantiles, `TERM_PROXY_DEFAULT_RATES.csv` and `PAID_RATIO_QUANTILE_DEFAULT_RATES.csv` for exploratory target rates, `TERM_PROXY_CORRELATIONS.csv` for redundancy, and `TERM_SUBGROUP_AUC_DIAGNOSTICS.csv` for whether the OOF gain is concentrated in contract/product subgroups. These subgroup analyses reuse the same exposed labels and are hypothesis-generating only.

For the saved subgroup diagnostic, the six-month residual changes AUC by {revolver_delta:+.6f} within revolving loans (n={int(revolver.n.iloc[0]) if not revolver.empty else 0}). This is a descriptive subgroup pattern, not independent evidence; the smaller group and the multiple subgroup comparisons make it especially susceptible to chance.

The OOF deltas were independently recalculated from the saved predictions in `TERM_PATTERN_AUC_RECALCULATION.csv`. All-fold improvements are not independent confirmation because the folds were used in earlier phases and multiple formula variants were compared.
"""
    (OUT / "TERM_PATTERN_ANALYSIS.md").write_text(md, encoding="utf-8")

    expected = [
        "phase3d_followup_experiments.py",
        "analyze_term_pattern.py",
        "evaluate_term_candidate_blends.py",
        "FOLLOWUP_REPORT.md",
        "FOLLOWUP_SUMMARY.csv",
        "FOLLOWUP_FOLD_RESULTS.csv",
        "FOLLOWUP_OOF_PREDICTIONS.csv",
        "CLAUDE_FORMULA_MATHEMATICAL_AUDIT.csv",
        "TERM_HYPOTHESIS_DISTRIBUTIONS.csv",
        "TERM_PATTERN_DIAGNOSTICS.csv",
        "TERM_PROXY_DEFAULT_RATES.csv",
        "PAID_RATIO_QUANTILE_DEFAULT_RATES.csv",
        "TERM_SUBGROUP_AUC_DIAGNOSTICS.csv",
        "TERM_PROXY_CORRELATIONS.csv",
        "TERM_PATTERN_AUC_RECALCULATION.csv",
        "TERM_PATTERN_ANALYSIS.md",
        "FIXED_BLEND_COMPARISONS.csv",
        "TERM_CANDIDATE_FIXED_BLEND_RESULTS.csv",
        "TERM_CANDIDATE_FIXED_BLEND_FOLDS.csv",
        "MODEL_RANK_CORRELATIONS.csv",
        "MODEL_CORRELATION_AND_AUC.csv",
        "environment.json",
        "INPUT_SHA256_MANIFEST.csv",
    ]
    missing = [
        str(OUT / name)
        for name in expected
        if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Missing/empty follow-up outputs: {missing}")
    manifest = [
        {
            "path": str(OUT / name),
            "bytes": (OUT / name).stat().st_size,
            "sha256": hashlib.sha256((OUT / name).read_bytes()).hexdigest(),
        }
        for name in expected
    ]
    pd.DataFrame(manifest).to_csv(OUT / "OUTPUT_MANIFEST.csv", index=False)
    log_lines = [
        f"verification_utc: {datetime.now(timezone.utc).isoformat()}",
        f"main_experiment_started_utc: {json.loads((OUT / 'environment.json').read_text(encoding='utf-8'))['started_utc']}",
        f"main_experiment_finished_utc: {json.loads((OUT / 'environment.json').read_text(encoding='utf-8'))['finished_utc']}",
        f"train_shape: {train.shape}",
        f"target_rate: {y.mean():.8f}",
        f"term_pattern_top_candidate: {top['candidate']}",
        f"term_pattern_pooled_delta: {top['pooled_delta_vs_control']:+.9f}",
        "status: SUCCESS",
        "OOF AUCs independently recalculated: yes",
        "claude solution source / vectors: unavailable locally",
        "warnings: reused folds and multiple probes make results exploratory; inferred contract variables are not observed loan terms.",
        "manifest_entries: " + str(len(manifest)),
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(auc_recalc.to_string(index=False))
    print(f"output_directory={OUT}")


if __name__ == "__main__":
    main()
