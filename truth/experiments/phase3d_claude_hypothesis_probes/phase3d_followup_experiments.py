from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE3B = ROOT / "experiments" / "phase3b_deep_validation"
PHASE3C = ROOT / "experiments" / "phase3c_model_intelligence"
PHASE3D = ROOT / "experiments" / "phase3d_competition_compliance_and_domain_research"
OUT = ROOT / "experiments" / "phase3d_claude_hypothesis_probes"
ID, TARGET = "SK_ID_CURR", "TARGET"
SEED = 20261104


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def term_hypotheses(raw: pd.DataFrame) -> pd.DataFrame:
    credit = raw["AMT_CREDIT"].to_numpy(dtype=np.float64)
    annuity = raw["AMT_ANNUITY"].to_numpy(dtype=np.float64)
    valid = np.isfinite(credit) & np.isfinite(annuity) & (credit > 0) & (annuity > 0)
    term = np.full(len(raw), np.nan)
    term[valid] = np.clip(np.rint(credit[valid] / annuity[valid]), 6, 72)
    snapped = np.full(len(raw), np.nan)
    snapped[valid] = np.clip(np.rint(term[valid] / 6) * 6, 6, 72)
    paid_ratio = np.full(len(raw), np.nan)
    payment_gap = np.full(len(raw), np.nan)
    paid_ratio_snap = np.full(len(raw), np.nan)
    payment_gap_snap = np.full(len(raw), np.nan)
    paid_ratio[valid] = annuity[valid] * term[valid] / credit[valid]
    payment_gap[valid] = annuity[valid] * term[valid] - credit[valid]
    paid_ratio_snap[valid] = annuity[valid] * snapped[valid] / credit[valid]
    payment_gap_snap[valid] = annuity[valid] * snapped[valid] - credit[valid]
    return pd.DataFrame(
        {
            "IMPLIED_TERM_ROUNDED_CLIPPED_HYPOTHESIS": term,
            "IMPLIED_TERM_SNAPPED_6_HYPOTHESIS": snapped,
            "TOTAL_PAID_RATIO_ROUNDED_TERM_HYPOTHESIS": paid_ratio,
            "CONTRACT_GAP_ROUNDED_TERM_HYPOTHESIS": payment_gap,
            "TOTAL_PAID_RATIO_SNAPPED_TERM_HYPOTHESIS": paid_ratio_snap,
            "CONTRACT_GAP_SNAPPED_TERM_HYPOTHESIS": payment_gap_snap,
        },
        index=raw.index,
    )


def external_fixed_weight(raw: pd.DataFrame) -> pd.DataFrame:
    values = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].to_numpy(dtype=np.float64)
    weights = np.array([0.20, 0.35, 0.45], dtype=np.float64)
    observed = np.isfinite(values)
    denominator = (observed * weights).sum(axis=1)
    numerator = np.where(observed, values, 0.0) @ weights
    score = np.divide(
        numerator,
        denominator,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return pd.DataFrame({"EXT_WEIGHTED_020_035_045": score}, index=raw.index)


def crossfit_external_logit(raw_fit, y_fit, raw_valid, fold):
    columns = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    x_fit = raw_fit[columns].to_numpy(dtype=np.float64)
    x_valid = raw_valid[columns].to_numpy(dtype=np.float64)
    train_score = np.full(len(raw_fit), np.nan, dtype=np.float64)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + int(fold))
    for sub_fit, sub_valid in inner.split(x_fit, y_fit):
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
        )
        model.fit(x_fit[sub_fit], y_fit[sub_fit])
        train_score[sub_valid] = model.predict_proba(x_fit[sub_valid])[:, 1]
    final_model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
    )
    final_model.fit(x_fit, y_fit)
    valid_score = final_model.predict_proba(x_valid)[:, 1]
    if not np.isfinite(train_score).all() or not np.isfinite(valid_score).all():
        raise ValueError("Cross-fitted external-score model produced nonfinite features.")
    return (
        pd.DataFrame({"EXT_LOGIT_CROSSFIT": train_score}),
        pd.DataFrame({"EXT_LOGIT_CROSSFIT": valid_score}),
    )


def add_constraint(feature_names, name):
    constraints = np.zeros(len(feature_names), dtype=np.int8)
    if name == "mono_days_birth_increases":
        constraints[feature_names.index("DAYS_BIRTH")] = 1
    elif name == "mono_days_birth_decreases":
        constraints[feature_names.index("DAYS_BIRTH")] = -1
    elif name == "mono_external_scores_decrease":
        for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
            constraints[feature_names.index(column)] = -1
    else:
        raise ValueError(f"Unknown constraint candidate: {name}")
    return constraints.tolist()


def main() -> None:
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_module(
        "phase3d_followup_baseline",
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
    )
    phase3b = load_module("phase3d_followup_phase3b", PHASE3B / "phase3b_ablation.py")

    train_path = DATA / "train.csv"
    train = pd.read_csv(train_path)
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    folds = pd.read_csv(PHASE2 / "FOLD_ASSIGNMENTS.csv").set_index(ID).loc[ids, "fold"].to_numpy(dtype=np.int8)
    if len(np.unique(folds)) != 5:
        raise ValueError("The preserved five folds were not found.")
    rounds_table = pd.read_csv(PHASE3C / "MODEL_RUNTIME_RESULTS.csv")
    rounds_by_fold = (
        rounds_table.loc[
            (rounds_table.model == "lightgbm") & (rounds_table.stage == "inner_round_selection"),
            ["fold", "selected_rounds_or_trees"],
        ]
        .set_index("fold")["selected_rounds_or_trees"]
        .to_dict()
    )
    benchmark_oof = pd.read_csv(PHASE3C / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv").set_index(ID).loc[ids]
    if not np.array_equal(benchmark_oof[TARGET].to_numpy(dtype=np.int8), y):
        raise ValueError("Phase 3C OOF target alignment failed.")
    if not np.array_equal(benchmark_oof["fold"].to_numpy(dtype=np.int8), folds):
        raise ValueError("Phase 3C OOF fold alignment failed.")
    baseline_ref = benchmark_oof["lightgbm__raw_plus_product"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_ref).all():
        raise ValueError("Preserved Phase 3C baseline OOF is incomplete.")

    term = term_hypotheses(raw)
    fixed_weight = external_fixed_weight(raw)
    candidate_features = {
        f"term__{column.lower()}": [column] for column in term.columns
    }
    candidate_features["external__fixed_020_035_045"] = ["EXT_WEIGHTED_020_035_045"]
    candidate_features["external__crossfit_logistic_score"] = ["EXT_LOGIT_CROSSFIT"]
    candidates = [
        "control_refit",
        *candidate_features,
        "mono_days_birth_increases",
        "mono_days_birth_decreases",
        "mono_external_scores_decrease",
    ]
    predictions = {name: np.full(len(train), np.nan, dtype=np.float64) for name in candidates}
    fold_rows = []
    formula_feature_rows = []

    for fold in sorted(np.unique(folds)):
        valid_idx = np.flatnonzero(folds == fold)
        fit_idx = np.flatnonzero(folds != fold)
        raw_fit, raw_valid = raw.iloc[fit_idx], raw.iloc[valid_idx]
        fit_base = baseline.build_rowwise_features(raw_fit)
        valid_base = baseline.build_rowwise_features(raw_valid)
        product_params = phase3b.shared.fit_fee_and_step(raw_fit)
        fit_product, valid_product, _ = phase3b.make_family(
            raw_fit, raw_valid, feature_params=product_params, family_name="all_product"
        )
        fit_base = pd.concat([fit_base, fit_product.reset_index(drop=True)], axis=1)
        valid_base = pd.concat([valid_base, valid_product.reset_index(drop=True)], axis=1)
        rounds = int(rounds_by_fold[int(fold)])
        ext_fit_cf, ext_valid_cf = crossfit_external_logit(raw_fit, y[fit_idx], raw_valid, fold)
        extras_fit = {
            **{name: term.iloc[fit_idx][[column]].reset_index(drop=True) for name, column in zip(candidate_features, list(term.columns) + ["EXT_WEIGHTED_020_035_045", "EXT_LOGIT_CROSSFIT"]) if name.startswith("term__")},
            "external__fixed_020_035_045": fixed_weight.iloc[fit_idx].reset_index(drop=True),
            "external__crossfit_logistic_score": ext_fit_cf.reset_index(drop=True),
        }
        extras_valid = {
            **{name: term.iloc[valid_idx][[column]].reset_index(drop=True) for name, column in zip(candidate_features, list(term.columns) + ["EXT_WEIGHTED_020_035_045", "EXT_LOGIT_CROSSFIT"]) if name.startswith("term__")},
            "external__fixed_020_035_045": fixed_weight.iloc[valid_idx].reset_index(drop=True),
            "external__crossfit_logistic_score": ext_valid_cf.reset_index(drop=True),
        }
        print(f"fold={fold} rounds={rounds} train={len(fit_idx)} valid={len(valid_idx)}", flush=True)

        for candidate in candidates:
            fit_frame = fit_base.copy()
            valid_frame = valid_base.copy()
            if candidate in candidate_features:
                fit_frame = pd.concat([fit_frame.reset_index(drop=True), extras_fit[candidate]], axis=1)
                valid_frame = pd.concat([valid_frame.reset_index(drop=True), extras_valid[candidate]], axis=1)
            fit_x, valid_x = baseline.make_lightgbm_frames(fit_frame, valid_frame)
            params = {
                key: value
                for key, value in baseline.MODEL_CONFIG["lightgbm"].items()
                if key not in ("device", "num_boost_round", "early_stopping_rounds")
            }
            if candidate.startswith("mono_"):
                params["monotone_constraints"] = add_constraint(list(fit_x.columns), candidate)
            model = lgb.train(
                {**params, "seed": baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_x, label=y[fit_idx]),
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            pred = model.predict(valid_x, num_iteration=rounds)
            if not np.isfinite(pred).all():
                raise ValueError(f"Nonfinite OOF predictions for {candidate}, fold {fold}.")
            predictions[candidate][valid_idx] = pred
            auc = float(roc_auc_score(y[valid_idx], pred))
            base_auc = float(roc_auc_score(y[valid_idx], baseline_ref[valid_idx]))
            fold_rows.append(
                {
                    "candidate": candidate,
                    "fold": int(fold),
                    "fold_auc": auc,
                    "baseline_reference_fold_auc": base_auc,
                    "paired_delta_vs_phase3c_oof": auc - base_auc,
                    "rounds": rounds,
                    "outer_validation_used_for_early_stopping": False,
                }
            )
            print(f"  {candidate}: AUC={auc:.6f} delta={auc-base_auc:+.6f}", flush=True)

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(OUT / "FOLLOWUP_FOLD_RESULTS.csv", index=False)
    oof = pd.DataFrame({ID: ids, TARGET: y, "fold": folds})
    for name, pred in predictions.items():
        if not np.isfinite(pred).all():
            raise ValueError(f"OOF output incomplete: {name}")
        oof[name] = pred
    oof.to_csv(OUT / "FOLLOWUP_OOF_PREDICTIONS.csv", index=False)

    summary_rows = []
    for name, pred in predictions.items():
        subset = folds_df[folds_df.candidate == name].sort_values("fold")
        aucs = subset.fold_auc.to_numpy(dtype=np.float64)
        deltas = subset.paired_delta_vs_phase3c_oof.to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "candidate": name,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "mean_fold_auc": float(aucs.mean()),
                "std_fold_auc": float(aucs.std(ddof=1)),
                "mean_fold_delta_vs_phase3c_reference": float(deltas.mean()),
                "fold_delta_std": float(deltas.std(ddof=1)),
                "folds_improved": int((deltas > 0).sum()),
                "interpretation": "exploratory; previously studied folds",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "FOLLOWUP_SUMMARY.csv", index=False)

    model_names = {
        "lgb_product": "lightgbm__raw_plus_product",
        "cat_product": "catboost__raw_plus_product",
        "xgb_product": "xgboost__raw_plus_product",
        "hgb_product": "hist_gradient_boosting__raw_plus_product",
        "rf_product": "random_forest__raw_plus_product",
    }
    base_prediction = {name: benchmark_oof[column].to_numpy(dtype=np.float64) for name, column in model_names.items()}
    ranks = {name: (rankdata(pred) - 1) / (len(pred) - 1) for name, pred in base_prediction.items()}
    blend_sets = {
        "rank_blend_lgb_cat_equal": ["lgb_product", "cat_product"],
        "rank_blend_lgb_cat_xgb_equal": ["lgb_product", "cat_product", "xgb_product"],
        "rank_blend_all_5_equal": list(model_names),
        "prob_blend_lgb_cat_equal": ["lgb_product", "cat_product"],
    }
    blend_rows = []
    blend_predictions = {}
    for name, members in blend_sets.items():
        if name == "prob_blend_lgb_cat_equal":
            pred = np.mean([base_prediction[m] for m in members], axis=0)
        else:
            pred = np.mean([ranks[m] for m in members], axis=0)
        blend_predictions[name] = pred
        fold_aucs = [roc_auc_score(y[folds == fold], pred[folds == fold]) for fold in sorted(np.unique(folds))]
        blend_rows.append(
            {
                "candidate": name,
                "members": json.dumps(members),
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "mean_fold_auc": float(np.mean(fold_aucs)),
                "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
                "weight_selection": "fixed equal weights; not tuned on these labels",
            }
        )
    pd.DataFrame(blend_rows).to_csv(OUT / "FIXED_BLEND_COMPARISONS.csv", index=False)
    rank_matrix = pd.DataFrame(ranks)
    rank_corr = rank_matrix.corr(method="pearson")
    rank_corr.to_csv(OUT / "MODEL_RANK_CORRELATIONS.csv")
    model_auc_rows = []
    for key, column in model_names.items():
        pred = base_prediction[key]
        model_auc_rows.append(
            {
                "model": key,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "spearman_rank_corr_with_lgb": float(rank_matrix[key].corr(rank_matrix["lgb_product"])),
            }
        )
    pd.DataFrame(model_auc_rows).to_csv(OUT / "MODEL_CORRELATION_AND_AUC.csv", index=False)

    formula_rows = [
        {
            "hypothesis": "n = round(credit / annuity), clipped 6..72",
            "mathematical_status": "not observed term; heuristic only",
            "reason": "Payment ratio generally understates duration for positive interest and does not identify both rate and term.",
        },
        {
            "hypothesis": "six-month snapping",
            "mathematical_status": "assumption",
            "reason": "No supplied dictionary field establishes term granularity.",
        },
        {
            "hypothesis": "TOTAL_PAID_RATIO = annuity * n / credit",
            "mathematical_status": "depends on heuristic n",
            "reason": "When n is round(credit/annuity), this is approximately one except clipping/rounding; not evidence of actual total paid.",
        },
        {
            "hypothesis": "ANN_X_TERM_MINUS_CRED = annuity * n - credit",
            "mathematical_status": "depends on heuristic n",
            "reason": "The residual is driven by assumed term, not observed contract amortization.",
        },
        {
            "hypothesis": "fixed-n implied rate",
            "mathematical_status": "not separately modeled",
            "reason": "For fixed n the annuity equation yields a monotone function of annuity/credit, already in baseline; it cannot add ranking information absent numerical/model approximation.",
        },
        {
            "hypothesis": "DISPOSABLE_INC = income/12 - annuity",
            "mathematical_status": "baseline duplicate under scaling",
            "reason": "Exactly (income - 12*annuity)/12, a scalar multiple of baseline DISPOSABLE_INCOME; period semantics remain unverified.",
        },
        {
            "hypothesis": "SOLVENCY_RATIO = 3*income/credit",
            "mathematical_status": "arbitrary scale of reciprocal baseline ratio",
            "reason": "A constant times income/credit; baseline has credit/income and tree models can represent reciprocal relation through splits only imperfectly but it is no independent measurement.",
        },
        {
            "hypothesis": "weighted external score 0.20/0.35/0.45",
            "mathematical_status": "tested hypothesis",
            "reason": "Weights are user-proposed, not dictionary-derived; missing scores are renormalized over observed weights.",
        },
        {
            "hypothesis": "cross-fitted logistic external score",
            "mathematical_status": "nested supervised representation",
            "reason": "Weights are estimated only in each outer training partition; each training row receives an internal out-of-fold estimate.",
        },
        {
            "hypothesis": "Days-birth monotonic sign",
            "mathematical_status": "tested both directions",
            "reason": "DAYS_BIRTH is negative elapsed days; +1 constraint asserts default risk rises as the value increases (younger age).",
        },
        {
            "hypothesis": "external score monotonic direction",
            "mathematical_status": "tested decreasing on the three raw columns",
            "reason": "Constraint is a predictive hypothesis only; does not constrain derived aggregate features and is not a causal statement.",
        },
    ]
    pd.DataFrame(formula_rows).to_csv(OUT / "CLAUDE_FORMULA_MATHEMATICAL_AUDIT.csv", index=False)

    dataset_summary = []
    for column in term.columns:
        values = term[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        dataset_summary.append(
            {
                "feature": column,
                "n_valid": int(valid.sum()),
                "n_missing": int((~valid).sum()),
                "minimum": float(values[valid].min()),
                "maximum": float(values[valid].max()),
                "mean": float(values[valid].mean()),
                "std": float(values[valid].std()),
                "n_unique": int(pd.Series(values[valid]).nunique()),
            }
        )
    pd.DataFrame(dataset_summary).to_csv(OUT / "TERM_HYPOTHESIS_DISTRIBUTIONS.csv", index=False)

    # Claude source discovery was performed before execution; do not reconstruct an unknown seven-model pipeline.
    environment = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lgb.__version__,
        "train_shape": list(train.shape),
        "target_rate": float(y.mean()),
        "folds": 5,
        "elapsed_seconds": time.perf_counter() - clock,
        "claude_solution_source": "not found locally; seven-model reconstruction blocked",
        "external_crossfit": "5 inner stratified folds within each outer training partition",
    }
    (OUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    input_paths = [
        train_path,
        PHASE2 / "FOLD_ASSIGNMENTS.csv",
        PHASE3C / "MODEL_RUNTIME_RESULTS.csv",
        PHASE3C / "MODEL_BENCHMARK_OOF_PREDICTIONS.csv",
        PHASE3B / "phase3b_ablation.py",
        ROOT / "research_archive" / "exp85_phase2_source_snapshot" / "project" / "source" / "run_independent_baseline.py",
    ]
    pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in input_paths]
    ).to_csv(OUT / "INPUT_SHA256_MANIFEST.csv", index=False)
    report = f"""# Phase 3D follow-up: contract, external-score, monotonicity, and fixed blends

## Result

All formula/model probes completed on the preserved five folds. Baseline reference is the Phase 3C LightGBM raw-plus-product OOF vector (AUC 0.762491). The fixed fold iteration counts and preprocessing are unchanged; all fold-fitted transformations use outer-training rows only. These are exploratory because the folds have previously been examined.

See `FOLLOWUP_SUMMARY.csv` for each candidate's pooled OOF AUC, paired fold delta, and number of fold wins. `FOLLOWUP_FOLD_RESULTS.csv` has per-fold values. `FIXED_BLEND_COMPARISONS.csv` and `MODEL_RANK_CORRELATIONS.csv` compare predeclared equal-weight rank/probability blends; no blend weights were tuned on these reported labels.

## Interpretation of the contract hypotheses

No inferred term or rate is an observed contract quantity. `round(credit/annuity)` is only a scale proxy; interest makes the amortization duration differ from credit/annuity. The 6-month snap and 6..72 clipping are assumptions. `annuity*n/credit` is near one by construction for that heuristic, except where rounding/clipping intervenes. `annuity*n-credit` is therefore a heuristic residual. A fixed-term inverse-rate is monotone in annuity/credit, already supplied to the model, so it was not presented as an independent financial signal.

The experiment does test the rounded/clipped and snapped proxy variants separately. Their outcomes do not establish true term or rate; look for robustness in the fold deltas rather than treating a small AUC change as contract discovery.

## External scores and monotonic constraints

The user-proposed 0.20/0.35/0.45 combination is tested as a fixed row-wise weighted mean (weights renormalized across available scores). A second feature uses logistic-regression weights derived only from each outer training fold; training-row scores are cross-fitted internally. Baseline already includes mean/min/max/std/count/products/differences, so an added score is an alternate summary, not new source information.

Monotonic experiments (positive/negative `DAYS_BIRTH` and negative `EXT_SOURCE_1..3`) are reported separately. A positive constraint on `DAYS_BIRTH` means risk must increase as its numeric value increases (becomes less negative/younger). A constraint on three raw external scores does not constrain their engineered combinations; interpret it as a limited model restriction, not a global causal claim.

## Blend caution

The fixed blends use equal weights and no OOF label fitting. They can be reproduced from the preserved Phase 3C predictions but are not independent validation. Nested optimization of weights was not performed: available OOF vectors were generated across the global fold partition, so using those vectors to fit weights inside one held-out fold is not a rigorously nested outer evaluation. A proper nested rank-weight experiment requires retraining every base estimator within each outer training fold. The Claude `solution.py`, notebook, model OOF files and its seven-model definition were not found locally; its 0.7647 score cannot be reconstructed or attributed to these models.

## Files and reproducibility

The execution script, OOF predictions, formula audit, term distributions, fold results, summary, fixed blend table, model correlations, environment, and input hashes are saved in this folder. No champion or earlier experiment was modified.
"""
    (OUT / "FOLLOWUP_REPORT.md").write_text(report, encoding="utf-8")

    expected = [
        "phase3d_followup_experiments.py",
        "FOLLOWUP_REPORT.md",
        "FOLLOWUP_SUMMARY.csv",
        "FOLLOWUP_FOLD_RESULTS.csv",
        "FOLLOWUP_OOF_PREDICTIONS.csv",
        "CLAUDE_FORMULA_MATHEMATICAL_AUDIT.csv",
        "TERM_HYPOTHESIS_DISTRIBUTIONS.csv",
        "FIXED_BLEND_COMPARISONS.csv",
        "MODEL_RANK_CORRELATIONS.csv",
        "MODEL_CORRELATION_AND_AUC.csv",
        "environment.json",
        "INPUT_SHA256_MANIFEST.csv",
    ]
    missing = [str(OUT / name) for name in expected if not (OUT / name).is_file() or (OUT / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing/empty outputs: {missing}")
    manifest = [
        {"path": str(OUT / name), "bytes": (OUT / name).stat().st_size, "sha256": sha256(OUT / name)}
        for name in expected
    ]
    pd.DataFrame(manifest).to_csv(OUT / "OUTPUT_MANIFEST.csv", index=False)
    log = [
        f"command: python {Path(__file__)}",
        f"started_utc: {started.isoformat()}",
        f"finished_utc: {environment['finished_utc']}",
        f"train_shape: {train.shape}",
        f"target_rate: {y.mean():.8f}",
        f"fold_sizes: {json.dumps({int(f): int((folds == f).sum()) for f in np.unique(folds)}, sort_keys=True)}",
        f"elapsed_seconds: {environment['elapsed_seconds']:.2f}",
        "status: SUCCESS",
        "errors: none",
        "claude_solution_source: not found locally",
    ]
    (OUT / "EXECUTION_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(blend_rows).to_string(index=False), flush=True)
    print(f"output_directory={OUT}", flush=True)
    print(f"elapsed_seconds={environment['elapsed_seconds']:.2f}", flush=True)


if __name__ == "__main__":
    main()
