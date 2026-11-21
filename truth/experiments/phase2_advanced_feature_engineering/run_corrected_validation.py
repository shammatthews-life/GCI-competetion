#!/usr/bin/env python3
"""Re-establish the raw baseline and test existing domain formulas correctly."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

ROOT = Path(r"D:\gci\truth")
DATA_DIR = Path(r"D:\gci\inpuy")
OUTPUT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)

spec = importlib.util.spec_from_file_location("preserved_baseline", BASELINE_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load preserved baseline: {BASELINE_SOURCE}")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

FORMULAS = [
    "LOAN_TO_GOODS_PRICE",
    "ANNUITY_CREDIT_ANNUALIZED",
    "DEBT_SERVICE_GAP",
    "FAMILY_PAYMENT_TO_INCOME",
    "INCOME_PER_MEMBER",
    "CREDIT_TO_INCOME_PER_MEMBER",
    "EXT_DISPERSION_RELATIVE",
    "EMPLOYMENT_SENTINEL",
    "BUREAU_HOUR",
    "BUREAU_MONTH",
    "BUREAU_QUARTER",
    "BUREAU_YEAR",
    "BUREAU_RECENT_COUNT",
    "BUREAU_OLDER_COUNT",
    "BUREAU_RECENT_SHARE",
    "REGION_RATING_PER_POPULATION",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_formula_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create only row-wise, label-free candidate features."""
    get = lambda name: raw[name].to_numpy(dtype=np.float64)
    annuity = get("AMT_ANNUITY")
    credit = get("AMT_CREDIT")
    income = get("AMT_INCOME_TOTAL")
    goods = get("AMT_GOODS_PRICE")
    family = get("CNT_FAM_MEMBERS")
    employment = get("DAYS_EMPLOYED")
    ext = np.column_stack(
        [get("EXT_SOURCE_1"), get("EXT_SOURCE_2"), get("EXT_SOURCE_3")]
    )
    count = np.isfinite(ext).sum(axis=1)
    mean = np.divide(
        np.where(np.isfinite(ext), ext, 0.0).sum(axis=1),
        count,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=count > 0,
    )
    variance = np.divide(
        np.where(np.isfinite(ext), (ext - mean[:, None]) ** 2, 0.0).sum(axis=1),
        count - 1,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=count > 1,
    )
    std = np.sqrt(variance)
    hour = get("AMT_REQ_CREDIT_BUREAU_HOUR")
    month = get("AMT_REQ_CREDIT_BUREAU_MON")
    quarter = get("AMT_REQ_CREDIT_BUREAU_QRT")
    year = get("AMT_REQ_CREDIT_BUREAU_YEAR")
    recent = hour + month
    older = quarter + year
    total = recent + older
    recent_share = np.divide(
        recent,
        total,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=np.isfinite(total) & (total != 0),
    )

    return pd.DataFrame(
        {
            "LOAN_TO_GOODS_PRICE": _safe_ratio(credit, goods),
            # A scalar multiple of baseline ANNUITY_TO_CREDIT: included as a
            # deliberate redundancy check, not as a distinct hypothesis.
            "ANNUITY_CREDIT_ANNUALIZED": _safe_ratio(12.0 * annuity, credit),
            # A scalar multiple/shift of baseline ANNUITY_TO_INCOME.
            "DEBT_SERVICE_GAP": _safe_ratio(12.0 * annuity, income) - 0.25,
            "FAMILY_PAYMENT_TO_INCOME": _safe_ratio(
                annuity * family, income
            ),
            "INCOME_PER_MEMBER": _safe_ratio(income, family),
            "CREDIT_TO_INCOME_PER_MEMBER": _safe_ratio(
                credit, _safe_ratio(income, family)
            ),
            "EXT_DISPERSION_RELATIVE": _safe_ratio(std, mean),
            "EMPLOYMENT_SENTINEL": (employment == 365243.0).astype(np.float64),
            "BUREAU_HOUR": hour,
            "BUREAU_MONTH": month,
            "BUREAU_QUARTER": quarter,
            "BUREAU_YEAR": year,
            "BUREAU_RECENT_COUNT": recent,
            "BUREAU_OLDER_COUNT": older,
            "BUREAU_RECENT_SHARE": recent_share,
            "REGION_RATING_PER_POPULATION": _safe_ratio(
                get("REGION_RATING_CLIENT"), get("REGION_POPULATION_RELATIVE")
            ),
        },
        index=raw.index,
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def make_features(raw: pd.DataFrame, include_formulas: bool) -> pd.DataFrame:
    target_free = raw.drop(
        columns=[baseline.TARGET_COLUMN], errors="ignore"
    ).reset_index(drop=True)
    features = baseline.build_rowwise_features(target_free)
    if include_formulas:
        extras = build_formula_features(target_free)
        if not features.index.equals(extras.index):
            raise AssertionError("Raw and formula feature rows do not align")
        features = pd.concat([features, extras], axis=1)
    if features.columns.duplicated().any():
        raise ValueError("Duplicate feature columns found")
    numeric = features.select_dtypes(include=[np.number]).to_numpy(
        dtype=np.float64, copy=False
    )
    if np.isinf(numeric).any():
        raise ValueError("Feature builder produced infinity")
    return features


def model_parameters(fold: int) -> dict[str, object]:
    config = baseline.MODEL_CONFIG["lightgbm"]
    params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    params["seed"] = baseline.FOLD_SEED + fold
    return params


def select_iterations(
    outer_fit_raw: pd.DataFrame,
    outer_fit_labels: np.ndarray,
    fold: int,
    include_formulas: bool,
) -> int:
    """Select iteration count only from an inner split of outer-training rows."""
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=20261102 + fold,
    )
    inner_fit_pos, inner_stop_pos = next(
        splitter.split(np.zeros(len(outer_fit_labels)), outer_fit_labels)
    )
    inner_fit_raw = outer_fit_raw.iloc[inner_fit_pos]
    inner_stop_raw = outer_fit_raw.iloc[inner_stop_pos]
    fit_features = make_features(inner_fit_raw, include_formulas)
    stop_features = make_features(inner_stop_raw, include_formulas)
    fit_matrix, stop_matrix = baseline.make_lightgbm_frames(
        fit_features, stop_features
    )
    train_set = lgb.Dataset(
        fit_matrix,
        label=outer_fit_labels[inner_fit_pos],
    )
    stop_set = lgb.Dataset(
        stop_matrix,
        label=outer_fit_labels[inner_stop_pos],
        reference=train_set,
    )
    settings = baseline.MODEL_CONFIG["lightgbm"]
    selection_model = lgb.train(
        model_parameters(fold),
        train_set,
        num_boost_round=settings["num_boost_round"],
        valid_sets=[stop_set],
        callbacks=[
            lgb.early_stopping(
                settings["early_stopping_rounds"], verbose=False
            )
        ],
    )
    selected = int(
        selection_model.best_iteration or selection_model.current_iteration()
    )
    if selected < 1:
        raise RuntimeError(f"No valid iteration selected for fold {fold}")
    return selected


def run_variant(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
    include_formulas: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = train[baseline.TARGET_COLUMN].to_numpy(dtype=np.int8)
    oof = np.full(len(train), np.nan, dtype=np.float64)
    records: list[dict[str, object]] = []
    settings = baseline.MODEL_CONFIG["lightgbm"]
    for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
        fit_raw = train.iloc[fit_idx]
        valid_raw = train.iloc[valid_idx]
        fit_labels = labels[fit_idx]
        valid_labels = labels[valid_idx]

        best_iteration = select_iterations(
            fit_raw, fit_labels, fold, include_formulas
        )
        fit_features = make_features(fit_raw, include_formulas)
        valid_features = make_features(valid_raw, include_formulas)
        fit_matrix, valid_matrix = baseline.make_lightgbm_frames(
            fit_features, valid_features
        )
        train_set = lgb.Dataset(fit_matrix, label=fit_labels)
        model = lgb.train(
            model_parameters(fold),
            train_set,
            num_boost_round=best_iteration,
        )
        prediction = model.predict(valid_matrix, num_iteration=best_iteration)
        oof[valid_idx] = prediction
        auc = float(roc_auc_score(valid_labels, prediction))
        records.append(
            {
                "variant": variant,
                "fold": fold,
                "n_fit": len(fit_idx),
                "n_valid": len(valid_idx),
                "best_iteration": best_iteration,
                "auc": auc,
            }
        )
        print(
            f"{variant} fold={fold} auc={auc:.8f} "
            f"best_iteration={best_iteration}",
            flush=True,
        )
        del model, train_set

    if not np.isfinite(oof).all():
        raise AssertionError(f"{variant} OOF predictions are incomplete")
    records.append(
        {
            "variant": variant,
            "fold": 0,
            "n_fit": len(train),
            "n_valid": len(train),
            "best_iteration": np.nan,
            "auc": float(roc_auc_score(labels, oof)),
        }
    )
    return pd.DataFrame(records), oof


def main() -> None:
    started = time.time()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    labels = train[baseline.TARGET_COLUMN].to_numpy(dtype=np.int8)

    expected = set(test.columns)
    if set(train.columns) - {baseline.TARGET_COLUMN} != expected:
        raise ValueError("Train and test raw column schemas do not match")
    if test.columns.tolist() != [
        c for c in train.columns if c != baseline.TARGET_COLUMN
    ]:
        raise ValueError("Train/test raw column order differs")
    if train[baseline.ID_COLUMN].duplicated().any():
        raise ValueError("Training IDs are not unique")

    splitter = StratifiedKFold(
        n_splits=baseline.N_SPLITS,
        shuffle=True,
        random_state=baseline.FOLD_SEED,
    )
    folds = [
        (fit_idx, valid_idx)
        for fit_idx, valid_idx in splitter.split(train, labels)
    ]
    fold_rows = []
    for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
        fold_rows.extend(
            {"SK_ID_CURR": int(train.iloc[i][baseline.ID_COLUMN]), "fold": fold}
            for i in valid_idx
        )
    pd.DataFrame(fold_rows).to_csv(
        OUTPUT_DIR / "FOLD_ASSIGNMENTS.csv", index=False
    )

    baseline_results, baseline_oof = run_variant(
        train, folds, "raw_baseline", include_formulas=False
    )
    formula_results, formula_oof = run_variant(
        train, folds, "baseline_plus_existing_formulas", include_formulas=True
    )
    fold_results = pd.concat(
        [baseline_results, formula_results], ignore_index=True
    )
    fold_results.to_csv(OUTPUT_DIR / "FOLD_LEVEL_CV_RESULTS.csv", index=False)

    oof_frame = pd.DataFrame(
        {
            baseline.ID_COLUMN: train[baseline.ID_COLUMN].to_numpy(),
            "TARGET": labels,
            "OOF_RAW_BASELINE": baseline_oof,
            "OOF_PLUS_FORMULAS": formula_oof,
        }
    )
    oof_frame.to_csv(OUTPUT_DIR / "OOF_BASELINE_PREDICTIONS.csv", index=False)
    summary_rows = []
    for variant, frame in fold_results[fold_results["fold"] > 0].groupby(
        "variant", sort=False
    ):
        aucs = frame["auc"].to_numpy(dtype=np.float64)
        pooled = float(
            roc_auc_score(
                labels,
                baseline_oof if variant == "raw_baseline" else formula_oof,
            )
        )
        summary_rows.append(
            {
                "variant": variant,
                "mean_fold_auc": float(aucs.mean()),
                "std_fold_auc": float(aucs.std(ddof=1)),
                "pooled_oof_auc": pooled,
            }
        )
    summary = pd.DataFrame(summary_rows)
    baseline_pooled = float(summary.loc[
        summary["variant"] == "raw_baseline", "pooled_oof_auc"
    ].iloc[0])
    summary["delta_pooled_vs_baseline"] = (
        summary["pooled_oof_auc"] - baseline_pooled
    )
    summary.to_csv(OUTPUT_DIR / "BASELINE_VALIDATION_RESULTS.csv", index=False)

    # Save rowwise candidate diagnostic information without using it to select
    # a feature set or tune model decisions.
    formula_train = build_formula_features(
        train.drop(columns=[baseline.TARGET_COLUMN])
    )
    formula_descriptions = {
        "LOAN_TO_GOODS_PRICE": "AMT_CREDIT / AMT_GOODS_PRICE; inverse of baseline CREDIT_TO_GOODS if goods are complete.",
        "ANNUITY_CREDIT_ANNUALIZED": "12 * AMT_ANNUITY / AMT_CREDIT; redundant scalar multiple of baseline ANNUITY_TO_CREDIT.",
        "DEBT_SERVICE_GAP": "12 * AMT_ANNUITY / AMT_INCOME_TOTAL - 0.25; scalar shift of the baseline income-payment ratio.",
        "FAMILY_PAYMENT_TO_INCOME": "AMT_ANNUITY * CNT_FAM_MEMBERS / AMT_INCOME_TOTAL; assumes household-wide allocation and is exploratory.",
        "INCOME_PER_MEMBER": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS; approximately duplicates baseline INCOME_PER_FAMILY except missing-family handling.",
        "CREDIT_TO_INCOME_PER_MEMBER": "AMT_CREDIT / (AMT_INCOME_TOTAL / CNT_FAM_MEMBERS); algebraically CREDIT_TO_INCOME * CNT_FAM_MEMBERS.",
        "EXT_DISPERSION_RELATIVE": "sample_std(available EXT_SOURCE_1..3; at least two observed) / mean(available EXT_SOURCE_1..3).",
        "EMPLOYMENT_SENTINEL": "1[DAYS_EMPLOYED == 365243]; duplicates baseline DAYS_EMPLOYED_SENTINEL.",
        "BUREAU_HOUR": "AMT_REQ_CREDIT_BUREAU_HOUR, preserved separately.",
        "BUREAU_MONTH": "AMT_REQ_CREDIT_BUREAU_MON, preserved separately.",
        "BUREAU_QUARTER": "AMT_REQ_CREDIT_BUREAU_QRT, preserved separately.",
        "BUREAU_YEAR": "AMT_REQ_CREDIT_BUREAU_YEAR, preserved separately.",
        "BUREAU_RECENT_COUNT": "HOUR + MON; interpretable only as two supplied window counts, not a standardized rate.",
        "BUREAU_OLDER_COUNT": "QRT + YEAR; interpretable only as two supplied window counts, not a standardized rate.",
        "BUREAU_RECENT_SHARE": "(HOUR + MON) / (HOUR + MON + QRT + YEAR); missing for missing fields or zero total.",
        "REGION_RATING_PER_POPULATION": "REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE; exploratory, weak semantic basis.",
    }
    diagnostics = []
    for feature in FORMULAS:
        values = formula_train[feature].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        diagnostics.append(
            {
                "feature": feature,
                "formula_and_interpretation": formula_descriptions[feature],
                "valid": int(np.isfinite(values).sum()),
                "missing_or_infinite": int((~np.isfinite(values)).sum()),
                "min": float(finite.min()) if finite.size else np.nan,
                "p01": float(np.quantile(finite, 0.01)) if finite.size else np.nan,
                "median": float(np.median(finite)) if finite.size else np.nan,
                "p99": float(np.quantile(finite, 0.99)) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
            }
        )
    pd.DataFrame(diagnostics).to_csv(
        OUTPUT_DIR / "FORMULA_DIAGNOSTICS.csv", index=False
    )

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = {
        "timestamp_utc": timestamp,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_sha256": sha256(TRAIN_PATH),
        "test_sha256": sha256(TEST_PATH),
        "baseline_source": str(BASELINE_SOURCE),
        "baseline_source_sha256": sha256(BASELINE_SOURCE),
        "folds": baseline.N_SPLITS,
        "outer_seed": baseline.FOLD_SEED,
        "inner_stopping_split": "StratifiedShuffleSplit 10%, seed=20261102+fold",
        "lightgbm_config": baseline.MODEL_CONFIG["lightgbm"],
        "runtime_seconds": round(time.time() - started, 2),
        "python": sys.version.replace("\n", " "),
        "lightgbm_version": lgb.__version__,
        "target_used_in_feature_builder": False,
    }
    (OUTPUT_DIR / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "requirements.txt").write_text(
        "numpy\npandas\nscikit-learn\nlightgbm\n",
        encoding="utf-8",
    )

    rules = [
        "# Validation leakage audit",
        "",
        "- Outer fold membership is generated once with `StratifiedKFold`, seed 42, and saved.",
        "- Each outer training partition is split again for iteration selection; outer validation labels are not passed to LightGBM during stopping.",
        "- Selected iteration is used to refit on the complete outer training partition, then score the untouched outer validation rows.",
        "- The rowwise baseline feature builder is reused unchanged; TARGET is removed before calling it.",
        "- Candidate formulas use rowwise supplied columns only; no test rows or labels are used to fit features.",
        "- LightGBM handles numeric missing values; category levels are generated by the existing baseline helper from the relevant fitting partition.",
        "- No target encoding, pooled train/test statistic, feature selection, or threshold tuning is performed.",
        "- This is a LightGBM-only corrected comparison, not the historical fixed LightGBM/CatBoost blend.",
    ]
    (OUTPUT_DIR / "VALIDATION_LEAKAGE_AUDIT.md").write_text(
        "\n".join(rules), encoding="utf-8"
    )

    source_audit = [
        "# Existing code and metric audit",
        "",
        "## Baseline implementation",
        "",
        f"- Source: `{BASELINE_SOURCE}`",
        "- Reuses `build_rowwise_features`, `make_lightgbm_frames`, `MODEL_CONFIG['lightgbm']`, `N_SPLITS=5`, and `FOLD_SEED=42` from the preserved baseline source.",
        "- The baseline is the raw-only Phase 2 snapshot, not the stronger Phase 2 fee/target-encoding feature stack.",
        "- Model: LightGBM binary GBDT, learning rate 0.03, 31 leaves, min child samples 80, feature fraction 0.80, bagging fraction 0.80, L1 0.10, L2 5.0, max bin 127, 3,000-round cap, 150-round early stopping.",
        "- Categorical frame construction is performed by the existing helper from each relevant fit partition; missing numeric values use LightGBM native handling.",
        "",
        "## Prior domain-formula execution audit",
        "",
        "- `domain_formula_probe/domain_feature_formulas.py` computes formulas row-wise and calculates each univariate AUC on the full training labels after median imputation. Those AUCs are descriptive in-sample diagnostics, not CV performance.",
        "- `domain_formula_execution_audit_01/run_domain_formula_audit.py` used the outer validation fold as LightGBM's early-stopping set. Its fold AUCs are contaminated by iteration selection and are not accepted as unbiased OOF results.",
        "- The earlier “top 3” were selected from full-training TARGET AUC, which leaks labels into feature selection. That variant is not repeated.",
        "- That previous runner also rebuilt categorical dummy columns using the whole training feature matrix rather than the baseline helper's fit-partition category handling, so its baseline was not an exact reproduction of the preserved baseline pipeline.",
        "- Existing deliverables remain untouched as historical records; this corrected run is the new validation reference.",
        "",
        "## Formula overlap / semantics",
        "",
        "- `ANNUITY_CREDIT_ANNUALIZED` is a constant multiple of baseline `ANNUITY_TO_CREDIT`.",
        "- `DEBT_SERVICE_GAP` is a linear scaling and shift of baseline `ANNUITY_TO_INCOME`.",
        "- `INCOME_PER_MEMBER` approximates baseline `INCOME_PER_FAMILY`; their missing-family handling differs.",
        "- `EMPLOYMENT_SENTINEL` duplicates baseline `DAYS_EMPLOYED_SENTINEL`.",
        "- `LOAN_TO_GOODS_PRICE` reverses baseline `CREDIT_TO_GOODS` and is a monotone reciprocal for positive goods values; tree splits may not be identical under finite binning, but it does not add new raw information.",
        "- `CREDIT_TO_INCOME_PER_MEMBER` algebraically equals `CREDIT_TO_INCOME * CNT_FAM_MEMBERS` when the family denominator is valid.",
        "- Bureau recent/older sums combine distinct supplied windows and are exploratory counts, not standardized rates. Individual window fields are also retained.",
        "- Annualizing `AMT_ANNUITY` is a mathematical rescaling. The data dictionary calls it “Loan annuity” but does not state its period; therefore interpreting it as an annual payment is not supported. The feature name here makes no annual-payment claim.",
        "- Multiplying annuity by household size assumes equal financial responsibility and is retained only for an ablation test, not endorsed as a domain fact.",
        "",
        "## Corrected metrics",
        "",
        summary.to_string(index=False),
    ]
    (OUTPUT_DIR / "SOURCE_CODE_AUDIT.md").write_text(
        "\n".join(source_audit), encoding="utf-8"
    )

    framework = [
        "# Validation framework",
        "",
        "The raw baseline feature builder and LightGBM configuration are imported from the preserved EXP85 source snapshot. The comparison uses the same fixed outer folds for both variants. Each fold selects a boosting iteration count on a stratified inner split of the outer training rows, refits on all outer training rows, and predicts the untouched outer validation rows.",
        "",
        f"- Execution timestamp UTC: {timestamp}",
        f"- Train shape: {train.shape}",
        f"- Test shape: {test.shape}",
        f"- Outer folds: {baseline.N_SPLITS}, shuffle=True, seed={baseline.FOLD_SEED}",
        "- Inner split: stratified 90/10, seed 20261102 + fold number",
        "- Preprocessing: existing raw-row builder and existing LightGBM frame helper; category levels fitted on the training partition only",
        "- Missing numeric values: retained for LightGBM's native missing-value handling",
        "- Validation rows are excluded from early stopping and all feature selection.",
        "- Test data is schema-checked but not used to fit or select this experiment.",
        "",
        "## Results",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "The prior run that stopped on outer folds is retained only as historical output. Its metrics are not considered valid OOF evidence. The previously named top-three set is not repeated because it was selected using full-training labels. This run tests the full predeclared formula list only.",
    ]
    (OUTPUT_DIR / "VALIDATION_FRAMEWORK.md").write_text(
        "\n".join(framework), encoding="utf-8"
    )

    report = [
        "# Phase 2 execution report — corrected validation gate",
        "",
        f"- Status: completed at {timestamp}",
        f"- Train/test dimensions: {train.shape} / {test.shape}",
        f"- Runtime seconds: {metadata['runtime_seconds']}",
        f"- Input train SHA-256: {metadata['train_sha256']}",
        f"- Input test SHA-256: {metadata['test_sha256']}",
        "",
        "## Corrected comparison",
        "",
        summary.to_string(index=False),
        "",
        "The numbers above are the valid result of this experiment; prior OOF-like metrics from outer-fold early stopping are invalid for unbiased comparison. These results are local OOF measurements, not leaderboard scores.",
        "",
        "## Formula interpretation cautions",
        "",
        "- Annualizing annuity is a constant rescaling of an existing baseline feature, not a new domain concept.",
        "- Family-size multiplication assumes equal allocation of the payment burden and is not retained as a preferred feature hypothesis.",
        "- Bureau windows are retained separately; their meanings do not justify treating them as rates over equivalent time intervals.",
        "- Relative external-score dispersion is undefined with fewer than two observed scores and can be unstable when its mean is close to zero; missingness and distribution are recorded for follow-up.",
        "- REGION_RATING_CLIENT / REGION_POPULATION_RELATIVE has uncertain semantics and is exploratory only.",
        "",
        "## Next gate",
        "",
        "Only after reviewing the corrected baseline and this predeclared formula comparison should feature-discovery screening begin. The full 500-hypothesis request is not treated as a quota and has not been started.",
    ]
    (OUTPUT_DIR / "PHASE2_EXECUTION_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"Execution complete: {timestamp}")
    print(f"Train={train.shape} Test={test.shape}")
    print(summary.to_string(index=False))
    print(f"Runtime seconds={metadata['runtime_seconds']}")
    print(f"Output directory={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
