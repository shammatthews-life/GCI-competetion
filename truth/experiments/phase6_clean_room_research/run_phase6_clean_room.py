from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.cluster.hierarchy import linkage
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE = ROOT / "experiments" / "phase6_clean_room_research"
RUNS = PHASE / "runs"
TRAIN_PATH = DATA / "train.csv"
TEST_PATH = DATA / "test.csv"
RULES_PATH = ROOT / "reports" / "final_competition_audit" / "OFFICIAL_RULES_AUDIT.md"
DICT_PATH = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition\HomeCredit_columns_description.xlsx"
)
LEDGER_PATH = ROOT / "GCI_MASTER_VERIFICATION_LEDGER_UPDATED_2026-11-04.md"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
PHASE2_RUNNER = (
    ROOT
    / "experiments"
    / "phase2_advanced_feature_engineering"
    / "run_corrected_validation.py"
)
EXISTING_PHASE2 = ROOT / "experiments" / "phase2_advanced_feature_engineering"
PHASE2B_RUNNER = ROOT / "experiments" / "phase2b_feature_intelligence" / "run_feature_intelligence.py"
PHASE3_RUNNER = ROOT / "experiments" / "phase3_hidden_pattern_discovery" / "run_phase3_hidden_pattern_discovery.py"
PHASE3B_RUNNER = ROOT / "experiments" / "phase3b_deep_validation" / "phase3b_ablation.py"
PHASE3C_RUNNER = ROOT / "experiments" / "phase3c_model_intelligence" / "phase3c_initial_benchmark.py"
PHASE3C_ADVANCED = ROOT / "experiments" / "phase3c_model_intelligence" / "phase3c_advanced_discovery.py"
PHASE3D_RUNNER = ROOT / "experiments" / "phase3d_competition_compliance_and_domain_research" / "phase3d_formula_experiment.py"
EXP02_VERIFY = ROOT / "experiments" / "exp02_financial_behaviour_replication_seed20261105" / "verify_exp02_seed_replication.py"
EXP02_REPORT = ROOT / "experiments" / "exp02_financial_behaviour_replication_seed20261105" / "REPLICATION_AUDIT_REPORT.md"
PHASE2B_REPORT = ROOT / "experiments" / "phase2b_feature_intelligence" / "PHASE2B_FINAL_REPORT.md"
PHASE3_REPORT = ROOT / "experiments" / "phase3_hidden_pattern_discovery" / "PHASE3_EXECUTION_REPORT.md"
PHASE3B_REPORT = ROOT / "experiments" / "phase3b_deep_validation" / "PHASE3B_EXECUTION_REPORT.md"
PHASE3C_REPORT = ROOT / "experiments" / "phase3c_model_intelligence" / "PHASE3C_EXECUTION_REPORT.md"
PHASE3D_REPORT = ROOT / "experiments" / "phase3d_competition_compliance_and_domain_research" / "FINAL_AUDIT_REPORT.md"
PHASE4_ALIAS_REPORT = ROOT / "experiments" / "phase4_notebook_recovery_gpu_discovery" / "EXP02_PHASE3C_RATIO_ALIAS_AUDIT.md"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SEED = 42
N_SPLITS = 5

CORE_DICTIONARY = {
    "AMT_INCOME_TOTAL": "Income of the client",
    "AMT_CREDIT": "Credit amount of the loan",
    "AMT_ANNUITY": "Loan annuity; payment period is not specified in the local dictionary",
    "AMT_GOODS_PRICE": "For consumer loans, price of the goods for which the loan is given",
    "DAYS_BIRTH": "Client's age in days at the time of application",
    "DAYS_EMPLOYED": "How many days before application the person started current employment",
    "CNT_FAM_MEMBERS": "How many family members the client has",
    "EXT_SOURCE_1": "Normalized score from external data source",
    "EXT_SOURCE_2": "Normalized score from external data source",
    "EXT_SOURCE_3": "Normalized score from external data source",
    "TARGET": "Client with payment difficulties (1) versus all other cases (0), per dictionary",
}

AUDIT_SOURCES = [
    Path(__file__).resolve(),
    BASELINE_SOURCE,
    PHASE2_RUNNER,
    PHASE2B_RUNNER,
    PHASE3_RUNNER,
    PHASE3B_RUNNER,
    PHASE3C_RUNNER,
    PHASE3C_ADVANCED,
    PHASE3D_RUNNER,
    EXP02_VERIFY,
    RULES_PATH,
    DICT_PATH,
    PHASE2B_REPORT,
    PHASE3_REPORT,
    PHASE3B_REPORT,
    PHASE3C_REPORT,
    PHASE3D_REPORT,
    PHASE4_ALIAS_REPORT,
    EXP02_REPORT,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = a[mask] / b[mask]
    return out


def finite_auc(y: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    mask = np.isfinite(score)
    if mask.sum() < 10 or np.unique(y[mask]).size != 2:
        return float("nan"), int(mask.sum())
    return float(roc_auc_score(y[mask], score[mask])), int(mask.sum())


def fold_auc_summary(
    y: np.ndarray, score: np.ndarray, folds: np.ndarray
) -> tuple[float, float, str]:
    rows = []
    for fold in sorted(np.unique(folds)):
        mask = (folds == fold) & np.isfinite(score)
        if mask.sum() and np.unique(y[mask]).size == 2:
            rows.append((int(fold), float(roc_auc_score(y[mask], score[mask]))))
    values = np.asarray([value for _, value in rows], dtype=np.float64)
    return (
        float(np.nanmean(values)) if len(values) else float("nan"),
        float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),
        json.dumps({str(fold): value for fold, value in rows}, sort_keys=True),
    )


def process_peak_bytes() -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("Psapi.dll")
    kernel32 = ctypes.WinDLL("Kernel32.dll")
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return None
    return int(counters.PeakWorkingSetSize)


def dictionary_rows() -> list[dict[str, str]]:
    from openpyxl import load_workbook

    workbook = load_workbook(DICT_PATH, read_only=True, data_only=True)
    result = []
    for row in workbook.active.iter_rows(values_only=True):
        if row and row[0] in CORE_DICTIONARY:
            result.append(
                {
                    "column": str(row[0]),
                    "official_definition": str(row[1] or "").strip(),
                }
            )
    workbook.close()
    found = {row["column"] for row in result}
    missing = set(CORE_DICTIONARY) - found
    if missing:
        raise ValueError(f"Dictionary missing required definitions: {sorted(missing)}")
    return result


def numeric_fingerprint(train: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    predictors = [c for c in train.columns if c not in {ID, TARGET}]
    for column in predictors:
        values = train[column]
        row = {
            "feature": column,
            "dtype": str(values.dtype),
            "n_rows": len(values),
            "missing_n": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()),
            "unique_nonmissing": int(values.nunique(dropna=True)),
            "most_common_value_share": (
                float(values.value_counts(normalize=True, dropna=True).iloc[0])
                if values.notna().any()
                else float("nan")
            ),
        }
        if pd.api.types.is_numeric_dtype(values):
            arr = values.to_numpy(dtype=np.float64)
            valid = np.isfinite(arr)
            x = arr[valid]
            row.update(
                {
                    "min": float(np.min(x)) if len(x) else float("nan"),
                    "p01": float(np.quantile(x, 0.01)) if len(x) else float("nan"),
                    "median": float(np.median(x)) if len(x) else float("nan"),
                    "p99": float(np.quantile(x, 0.99)) if len(x) else float("nan"),
                    "max": float(np.max(x)) if len(x) else float("nan"),
                    "integer_share_tol_1e-8": (
                        float(np.mean(np.isclose(x, np.round(x), atol=1e-8)))
                        if len(x)
                        else float("nan")
                    ),
                    "round_2_share_tol_1e-6": (
                        float(np.mean(np.isclose(x, np.round(x, 2), atol=1e-6)))
                        if len(x)
                        else float("nan")
                    ),
                    "multiple_10_share": (
                        float(np.mean(np.isclose(x / 10.0, np.round(x / 10.0), atol=1e-8)))
                        if len(x)
                        else float("nan")
                    ),
                    "multiple_100_share": (
                        float(np.mean(np.isclose(x / 100.0, np.round(x / 100.0), atol=1e-8)))
                        if len(x)
                        else float("nan")
                    ),
                    "multiple_1000_share": (
                        float(np.mean(np.isclose(x / 1000.0, np.round(x / 1000.0), atol=1e-8)))
                        if len(x)
                        else float("nan")
                    ),
                }
            )
            auc, n_auc = finite_auc(y, arr)
            row["univariate_auc_descriptive_only"] = auc
            row["univariate_auc_n"] = n_auc
        else:
            row.update(
                {
                    "min": "",
                    "p01": "",
                    "median": "",
                    "p99": "",
                    "max": "",
                    "integer_share_tol_1e-8": "",
                    "round_2_share_tol_1e-6": "",
                    "multiple_10_share": "",
                    "multiple_100_share": "",
                    "multiple_1000_share": "",
                    "univariate_auc_descriptive_only": "",
                    "univariate_auc_n": "",
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def financial_relationships(
    train: pd.DataFrame, y: np.ndarray, folds: np.ndarray, dictionary: list[dict[str, str]]
) -> pd.DataFrame:
    get = lambda name: train[name].to_numpy(dtype=np.float64)
    credit = get("AMT_CREDIT")
    annuity = get("AMT_ANNUITY")
    goods = get("AMT_GOODS_PRICE")
    income = get("AMT_INCOME_TOTAL")
    family = get("CNT_FAM_MEMBERS")
    features = {
        "CREDIT_TO_INCOME": safe_ratio(credit, income),
        "ANNUITY_TO_INCOME": safe_ratio(annuity, income),
        "CREDIT_TO_GOODS": safe_ratio(credit, goods),
        "CREDIT_TO_ANNUITY": safe_ratio(credit, annuity),
        "GOODS_TO_INCOME": safe_ratio(goods, income),
        "CREDIT_MINUS_GOODS": credit - goods,
        "CREDIT_GOODS_GAP_TO_INCOME": safe_ratio(credit - goods, income),
        "INCOME_PER_FAMILY_MEMBER": safe_ratio(income, family),
        "ANNUITY_PER_FAMILY_MEMBER": safe_ratio(annuity, family),
        "CREDIT_PER_FAMILY_MEMBER": safe_ratio(credit, family),
        "AGE_YEARS_DERIVED": -get("DAYS_BIRTH") / 365.25,
    }
    definitions = {
        "CREDIT_TO_INCOME": "AMT_CREDIT / AMT_INCOME_TOTAL",
        "ANNUITY_TO_INCOME": "AMT_ANNUITY / AMT_INCOME_TOTAL; payment period unspecified",
        "CREDIT_TO_GOODS": "AMT_CREDIT / AMT_GOODS_PRICE",
        "CREDIT_TO_ANNUITY": "AMT_CREDIT / AMT_ANNUITY; not a validated term estimate",
        "GOODS_TO_INCOME": "AMT_GOODS_PRICE / AMT_INCOME_TOTAL",
        "CREDIT_MINUS_GOODS": "AMT_CREDIT - AMT_GOODS_PRICE; not a down-payment interpretation",
        "CREDIT_GOODS_GAP_TO_INCOME": "(AMT_CREDIT - AMT_GOODS_PRICE) / AMT_INCOME_TOTAL",
        "INCOME_PER_FAMILY_MEMBER": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
        "ANNUITY_PER_FAMILY_MEMBER": "AMT_ANNUITY / CNT_FAM_MEMBERS; equal allocation unsupported",
        "CREDIT_PER_FAMILY_MEMBER": "AMT_CREDIT / CNT_FAM_MEMBERS",
        "AGE_YEARS_DERIVED": "-DAYS_BIRTH / 365.25",
    }
    rows = []
    for name, score in features.items():
        auc, n = finite_auc(y, score)
        mean_fold, sd_fold, fold_json = fold_auc_summary(y, score, folds)
        valid = np.isfinite(score)
        corr = spearmanr(score[valid], y[valid]).statistic if valid.sum() > 2 else np.nan
        rows.append(
            {
                "feature": name,
                "formula": definitions[name],
                "n_valid": n,
                "missing_n": int((~valid).sum()),
                "unique_valid": int(pd.Series(score[valid]).nunique()) if n else 0,
                "spearman_vs_target_descriptive": float(corr),
                "univariate_auc_descriptive": auc,
                "fold_mean_univariate_auc": mean_fold,
                "fold_sd_univariate_auc": sd_fold,
                "fold_auc_json": fold_json,
                "incremental_model_gain": "NOT TESTED",
                "interpretation_caution": (
                    "Association only; reused labels/folds, no causal or domain-mechanism claim."
                ),
            }
        )
    arr = safe_ratio(credit, goods)
    cash = train["NAME_CONTRACT_TYPE"].eq("Cash loans").to_numpy()
    for step in (0.0066, 0.0132):
        coord = (arr - 1.0) / step
        valid = cash & np.isfinite(coord)
        residual = np.abs(coord[valid] - np.round(coord[valid]))
        rows.append(
            {
                "feature": f"PRIOR_LATTICE_HYPOTHESIS_{step}",
                "formula": (
                    f"(AMT_CREDIT / AMT_GOODS_PRICE - 1) / {step}; "
                    "cash-loan rows only; compare distance to integer"
                ),
                "n_valid": int(valid.sum()),
                "missing_n": int((cash & ~np.isfinite(coord)).sum()),
                "unique_valid": "",
                "spearman_vs_target_descriptive": "",
                "univariate_auc_descriptive": "",
                "fold_mean_univariate_auc": "",
                "fold_sd_univariate_auc": "",
                "fold_auc_json": "",
                "candidate_integer_coordinate_share": float(np.mean(residual < 0.01)),
                "candidate_residual_below_0_1_share": float(np.mean(residual < 0.1)),
                "incremental_model_gain": "NOT TESTED",
                "interpretation_caution": (
                    "Matches the prior decoder's (credit/goods - 1)/step hypothesis; "
                    "training-set support is descriptive, not a verified contract increment."
                ),
            }
        )
    dict_map = {r["column"]: r["official_definition"] for r in dictionary}
    for row in rows:
        row["dictionary_reference"] = dict_map.get(
            row["feature"].replace("_DERIVED", ""), ""
        )
    return pd.DataFrame(rows)


def external_score_discovery(train: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    arr = train[cols].to_numpy(dtype=np.float64)
    count = np.isfinite(arr).sum(axis=1)
    mean = np.divide(
        np.nansum(arr, axis=1),
        count,
        out=np.full(len(train), np.nan),
        where=count > 0,
    )
    std = np.full(len(train), np.nan)
    score_range = np.full(len(train), np.nan)
    for i, row in enumerate(arr):
        vals = row[np.isfinite(row)]
        if len(vals) >= 2:
            std[i] = float(np.std(vals, ddof=1))
            score_range[i] = float(np.max(vals) - np.min(vals))
    candidates: dict[str, np.ndarray] = {
        **{c: train[c].to_numpy(dtype=np.float64) for c in cols},
        "EXT_MEAN_AVAILABLE": mean,
        "EXT_SD_AVAILABLE": std,
        "EXT_RANGE_AVAILABLE": score_range,
        "EXT_SCORE_COUNT": count.astype(np.float64),
    }
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = arr[:, i], arr[:, j]
            candidates[f"ABS_DIFF_{i+1}_{j+1}"] = np.abs(a - b)
            candidates[f"SIGNED_DIFF_{i+1}_{j+1}"] = a - b
    credit_income = safe_ratio(
        train["AMT_CREDIT"].to_numpy(dtype=np.float64),
        train["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64),
    )
    annuity_income = safe_ratio(
        train["AMT_ANNUITY"].to_numpy(dtype=np.float64),
        train["AMT_INCOME_TOTAL"].to_numpy(dtype=np.float64),
    )
    candidates["EXT_MEAN_X_CREDIT_TO_INCOME"] = mean * credit_income
    candidates["EXT_SD_X_CREDIT_TO_INCOME"] = std * credit_income
    candidates["EXT_MEAN_X_ANNUITY_TO_INCOME"] = mean * annuity_income
    missing_code = np.zeros(len(train), dtype=np.int8)
    for i in range(3):
        missing_code |= (~np.isfinite(arr[:, i])).astype(np.int8) << i
    candidates["EXT_MISSING_MASK_CODE"] = missing_code.astype(np.float64)
    rows = []
    for name, score in candidates.items():
        auc, n = finite_auc(y, score)
        mean_fold, sd_fold, fold_json = fold_auc_summary(y, score, folds)
        vals = score[np.isfinite(score)]
        rows.append(
            {
                "feature": name,
                "valid_n": n,
                "missing_n": len(train) - n,
                "univariate_auc_descriptive": auc,
                "fold_mean_auc": mean_fold,
                "fold_sd_auc": sd_fold,
                "fold_auc_json": fold_json,
                "p01": float(np.quantile(vals, 0.01)) if len(vals) else np.nan,
                "median": float(np.median(vals)) if len(vals) else np.nan,
                "p99": float(np.quantile(vals, 0.99)) if len(vals) else np.nan,
                "incremental_model_gain": "NOT TESTED",
                "warning": "Exploratory association on reused training labels; not feature-selection evidence.",
            }
        )
    return pd.DataFrame(rows).sort_values(
        "univariate_auc_descriptive", key=lambda s: (s - 0.5).abs(), ascending=False
    )


def missingness_audit(train: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    predictors = [c for c in train.columns if c not in {ID, TARGET}]
    masks = train[predictors].isna()
    rows = []
    for column in predictors:
        mask = masks[column].to_numpy()
        if mask.any() and (~mask).any():
            rate = float(y[mask].mean())
            valid_rates = [
                float(y[mask & (folds == fold)].mean())
                for fold in sorted(np.unique(folds))
                if np.any(mask & (folds == fold))
            ]
            indicator = mask.astype(np.float64)
            auc, _ = finite_auc(y, indicator)
            rows.append(
                {
                    "pattern_type": "single_column",
                    "pattern": column,
                    "n": int(mask.sum()),
                    "share": float(mask.mean()),
                    "default_rate": rate,
                    "fold_default_rate_min": min(valid_rates),
                    "fold_default_rate_max": max(valid_rates),
                    "fold_default_rate_sd": float(np.std(valid_rates, ddof=1)) if len(valid_rates) > 1 else np.nan,
                    "missing_indicator_auc": auc,
                    "interpretation": "Descriptive association; not incremental model gain.",
                }
            )
    row_mask = masks.to_numpy(dtype=np.uint8)
    codes = pd.Series(
        ["|".join(np.asarray(predictors)[row].tolist()) if row.any() else "NO_MISSING" for row in row_mask]
    )
    counts = codes.value_counts()
    for pattern, n in counts.head(30).items():
        if pattern == "NO_MISSING":
            continue
        mask = codes.to_numpy() == pattern
        rates = [
            float(y[mask & (folds == fold)].mean())
            for fold in sorted(np.unique(folds))
            if np.any(mask & (folds == fold))
        ]
        rows.append(
            {
                "pattern_type": "joint_missing_columns",
                "pattern": pattern,
                "n": int(mask.sum()),
                "share": float(mask.mean()),
                "default_rate": float(y[mask].mean()),
                "fold_default_rate_min": min(rates),
                "fold_default_rate_max": max(rates),
                "fold_default_rate_sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else np.nan,
                "missing_indicator_auc": "",
                "interpretation": "Top-frequency exact missingness pattern; sparse-pattern rates are noisy.",
            }
        )
    return pd.DataFrame(rows).sort_values(["pattern_type", "n"], ascending=[True, False])


def borrower_segments(train: pd.DataFrame, y: np.ndarray, out_dir: Path) -> tuple[pd.DataFrame, str]:
    from sklearn.decomposition import PCA

    numeric = [
        c for c in train.select_dtypes(include=[np.number]).columns if c not in {ID, TARGET}
    ]
    matrix = train[numeric].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(matrix)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed).astype(np.float32, copy=False)
    pca = PCA(n_components=min(10, scaled.shape[1]), svd_solver="randomized", random_state=SEED)
    pcs = pca.fit_transform(scaled)
    loadings = pd.DataFrame(
        pca.components_[: min(5, len(pca.components_))],
        columns=numeric,
    )
    loadings.insert(0, "component", [f"PC{i+1}" for i in range(len(loadings))])
    loadings.to_csv(out_dir / "PCA_TOP_LOADINGS.csv", index=False)
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(len(train), size=min(30000, len(train)), replace=False))
    kmeans = KMeans(n_clusters=4, random_state=SEED, n_init=5, max_iter=100)
    kmeans.fit(pcs[sample_idx, : min(8, pcs.shape[1])])
    clusters = kmeans.predict(pcs[:, : min(8, pcs.shape[1])])
    rows = []
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster
        record: dict[str, object] = {
            "method": "KMeans_K4_fit_on_random_train_sample_PCA",
            "cluster": int(cluster),
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "default_rate_descriptive": float(y[mask].mean()),
        }
        for name in [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ]:
            if name in train:
                record[f"median_{name}"] = float(train.loc[mask, name].median())
        rows.append(record)
    segment_frame = pd.DataFrame(rows)
    ari_rows = []
    for seed in (42, 43):
        model = KMeans(n_clusters=4, random_state=seed, n_init=5, max_iter=100).fit(
            pcs[sample_idx, : min(8, pcs.shape[1])]
        )
        ari_rows.append(model.labels_)
    ari = float(adjusted_rand_score(ari_rows[0], ari_rows[1]))
    bic_rows = []
    gm_sample = pcs[sample_idx[: min(8000, len(sample_idx))], : min(6, pcs.shape[1])]
    for k in (2, 3, 4):
        model = GaussianMixture(
            n_components=k, covariance_type="diag", random_state=SEED, reg_covar=1e-4
        ).fit(gm_sample)
        bic_rows.append({"components": k, "bic": float(model.bic(gm_sample)), "sample_n": len(gm_sample)})
    pd.DataFrame(bic_rows).to_csv(out_dir / "GMM_BIC_DIAGNOSTICS.csv", index=False)
    hier_n = min(1500, len(sample_idx))
    hier_sample = pcs[sample_idx[:hier_n], : min(6, pcs.shape[1])]
    if hier_n >= 3:
        hierarchy = linkage(hier_sample, method="ward")
        pd.DataFrame(
            [{"sample_n": hier_n, "linkage_method": "ward", "cophenetic_structure": "linkage matrix computed; exploratory only"}]
        ).to_csv(out_dir / "HIERARCHICAL_SAMPLE_DIAGNOSTIC.csv", index=False)
        del hierarchy
    else:
        pd.DataFrame().to_csv(out_dir / "HIERARCHICAL_SAMPLE_DIAGNOSTIC.csv", index=False)
    explained = pca.explained_variance_ratio_
    text = (
        f"# Borrower segment discovery\n\n"
        f"## Scope and safeguards\n\n"
        f"Training data only; ID and TARGET excluded from imputation, scaling, PCA, K-means, and GMM fitting. "
        f"TARGET is used only after fitting to describe segment default rates. This is exploratory structure, "
        f"not evidence that clusters are real borrower classes and not a validation feature. Any cluster feature "
        f"for a predictive model must be refit within each outer training fold.\n\n"
        f"- Numeric input predictors: {len(numeric)}.\n"
        f"- PCA explained variance by first components: {json.dumps([round(float(v), 6) for v in explained[:10]])}.\n"
        f"- K-means: k=4; fit on a fixed random sample of {len(sample_idx):,} training rows in PCA space; "
        f"ARI across two initializations/seeds on that sample: {ari:.6f}.\n"
        f"- GMM diagonal-covariance BIC diagnostics on {len(gm_sample):,} rows and first six PCs: "
        f"see `GMM_BIC_DIAGNOSTICS.csv`.\n"
        f"- Hierarchical Ward linkage: computed on {hier_n:,} sampled training rows; no population-level "
        f"claim is made.\n"
        f"- UMAP unavailable in the current Python environment; not installed or run.\n"
        f"- Interpret clusters as geometric partitions of standardized numerical predictors; not validated "
        f"real-world segments.\n\n"
        f"## K-means descriptive profile\n\n"
        f"```text\n{segment_frame.to_string(index=False)}\n```\n\n"
        f"## Decision\n\n"
        f"Retain segment analysis as a hypothesis generator only. Segment default rates use already-exposed "
        f"training labels and must not be used to select a segment-specific model without a prespecified "
        f"fold-local experiment and independent confirmation.\n"
    )
    return segment_frame, text


def duplicate_similarity(train: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    predictors = [c for c in train.columns if c not in {ID, TARGET}]
    signature = pd.util.hash_pandas_object(train[predictors], index=False).to_numpy()
    counts = pd.Series(signature).value_counts()
    duplicate_hashes = counts[counts > 1].index
    duplicate_rows = np.isin(signature, duplicate_hashes)
    duplicate_groups = int((counts > 1).sum())
    duplicate_group_y_agree = []
    for value in duplicate_hashes:
        idx = np.flatnonzero(signature == value)
        duplicate_group_y_agree.append(len(np.unique(y[idx])) == 1)

    numeric = [
        c for c in train.select_dtypes(include=[np.number]).columns if c not in {ID, TARGET}
    ]
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(len(train), size=min(10000, len(train)), replace=False))
    sampled = train.iloc[sample_idx][numeric].replace([np.inf, -np.inf], np.nan)
    imputed = SimpleImputer(strategy="median").fit_transform(sampled)
    scaled = StandardScaler().fit_transform(imputed).astype(np.float32, copy=False)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean", n_jobs=-1)
    nn.fit(scaled)
    distances, neighbors = nn.kneighbors(scaled)
    nearest_distance = distances[:, 1]
    nearest_sample = neighbors[:, 1]
    sampled_y = y[sample_idx]
    match_rate = float(np.mean(sampled_y == sampled_y[nearest_sample]))
    exact_numeric_near = int(np.sum(nearest_distance <= 1e-8))
    return pd.DataFrame(
        [
            {
                "audit": "exact_full_predictor_signature",
                "n_rows": len(train),
                "unique_signatures": int(pd.Series(signature).nunique()),
                "duplicate_signature_groups": duplicate_groups,
                "rows_in_duplicate_groups": int(duplicate_rows.sum()),
                "duplicate_signature_target_agreement_group_share": (
                    float(np.mean(duplicate_group_y_agree))
                    if duplicate_group_y_agree
                    else np.nan
                ),
                "sample_n": len(sample_idx),
                "nearest_neighbor_distance_median": "",
                "nearest_neighbor_distance_p01": "",
                "nearest_neighbor_distance_p99": "",
                "nearest_neighbor_target_agreement": "",
                "exact_numeric_nearest_pairs": "",
                "notes": "ID/TARGET excluded from exact predictor signature.",
            },
            {
                "audit": "numeric_standardized_nearest_neighbor_sample",
                "n_rows": len(train),
                "unique_signatures": "",
                "duplicate_signature_groups": "",
                "rows_in_duplicate_groups": "",
                "duplicate_signature_target_agreement_group_share": "",
                "sample_n": len(sample_idx),
                "nearest_neighbor_distance_median": float(np.median(nearest_distance)),
                "nearest_neighbor_distance_p01": float(np.quantile(nearest_distance, 0.01)),
                "nearest_neighbor_distance_p99": float(np.quantile(nearest_distance, 0.99)),
                "nearest_neighbor_target_agreement": match_rate,
                "exact_numeric_nearest_pairs": exact_numeric_near,
                "notes": (
                    "Numeric-only standardized nearest neighbours in a fixed sample; "
                    "not entity linkage and not a predictive feature."
                ),
            },
        ]
    )


def nonlinear_discovery(train: pd.DataFrame, y: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    numeric = [
        c for c in train.select_dtypes(include=[np.number]).columns if c not in {ID, TARGET}
    ]
    rows = []
    per_feature_bins = []
    for feature in numeric:
        x = train[feature].to_numpy(dtype=np.float64)
        log_x = np.full(len(x), np.nan)
        valid_positive = np.isfinite(x) & (x > -1)
        log_x[valid_positive] = np.log1p(x[valid_positive])
        raw_oof = np.full(len(x), np.nan)
        log_oof = np.full(len(x), np.nan)
        quantile_oof = np.full(len(x), np.nan)
        for fold in sorted(np.unique(folds)):
            fit_mask = folds != fold
            val_mask = folds == fold
            fit_vals = x[fit_mask & np.isfinite(x)]
            val_idx = np.flatnonzero(val_mask & np.isfinite(x))
            if len(fit_vals) < 20 or len(val_idx) == 0:
                continue
            edges = np.unique(np.quantile(fit_vals, np.linspace(0, 1, 11)))
            if len(edges) < 2:
                quantile_oof[val_idx] = 0.0
            else:
                quantile_oof[val_idx] = np.digitize(x[val_idx], edges[1:-1], right=True)
            raw_oof[val_idx] = x[val_idx]
            log_oof[val_idx] = log_x[val_idx]
            bin_ids = np.digitize(x[val_idx], edges[1:-1], right=True)
            for bin_id in np.unique(bin_ids):
                group = val_idx[bin_ids == bin_id]
                per_feature_bins.append(
                    {
                        "feature": feature,
                        "fold": int(fold),
                        "bin": int(bin_id),
                        "fit_quantile_edges": json.dumps(edges.tolist()),
                        "validation_n": int(len(group)),
                        "validation_mean_value": float(np.mean(x[group])),
                        "validation_default_rate_descriptive": float(np.mean(y[group])),
                        "cutpoints_fit_on_outer_train_only": True,
                    }
                )
        raw_auc, raw_n = finite_auc(y, raw_oof)
        log_auc, log_n = finite_auc(y, log_oof)
        bin_auc, bin_n = finite_auc(y, quantile_oof)
        rows.append(
            {
                "feature": feature,
                "raw_univariate_oof_auc_descriptive": raw_auc,
                "raw_valid_n": raw_n,
                "log1p_univariate_oof_auc_descriptive": log_auc,
                "log_valid_n": log_n,
                "fold_local_quantile_bin_auc_descriptive": bin_auc,
                "bin_valid_n": bin_n,
                "raw_fold_mean_auc": fold_auc_summary(y, raw_oof, folds)[0],
                "binned_fold_mean_auc": fold_auc_summary(y, quantile_oof, folds)[0],
                "binned_fold_sd_auc": fold_auc_summary(y, quantile_oof, folds)[1],
                "interpretation": (
                    "Univariate discovery only; transformations are not model-level gains. "
                    "Quantile cutpoints are fitted per outer fit fold."
                ),
            }
        )
    pd.DataFrame(per_feature_bins).to_csv(
        PHASE_RUN_OUTPUT / "NONLINEAR_VALIDATION_BIN_PROFILES.csv", index=False
    )
    return pd.DataFrame(rows).sort_values(
        "raw_univariate_oof_auc_descriptive", key=lambda s: (s - 0.5).abs(), ascending=False
    )


def generate_experiment_plan() -> pd.DataFrame:
    entries = [
        ("EXP01", "Raw structure and fingerprints", "Values contain heaping, unusual precision, or informative discontinuities.", "Train-only descriptive audit; preregister fingerprints; no test covariate fitting.", "ALLOWED", "No model gain; false numeric-pattern stories.", "Low / low", "Run first; hypothesis requires stable source-supported pattern."),
        ("EXP02", "Financial quantization", "Credit/goods/annuity values contain stable lattice-like structure beyond ordinary rounding.", "Fold-local fit of candidate grid parameters; compare preregistered null/random grids; same folds.", "ALLOWED if rowwise/fold-fit; semantic claims unverified", "Arbitrary step looks plausible; multiple testing.", "Low / low", "Only after EXP01; require robust fold support and negative controls."),
        ("EXP03", "Financial interactions", "Financial burdens carry different risk at different applicant/score strata.", "Predefined rowwise products/splines; common folds; nested stopping; paired OOF comparison.", "ALLOWED for supplied fields; unsupported contract meaning excluded", "Redundant ratios or adaptive interaction search.", "Medium / medium", "Candidate only if >=4/5 positive folds and preregistered lift."),
        ("EXP04", "Borrower segmentation", "Unsupervised predictor geometry contains stable segments with different model errors.", "Fit imputation/scaler/PCA/clusters inside each outer fit fold; apply to validation; labels never fit clusters.", "ALLOWED_WITH_RESTRICTIONS", "Cluster instability; researcher selection on exposed labels.", "Medium / high", "Do not use current full-train exploratory clusters as features."),
        ("EXP05", "External-score disagreement", "Score disagreement adds signal conditional on average score and financial burden.", "Pair differences/range/dispersion plus same-fold ablation and residual model; no target encoding.", "ALLOWED for supplied columns", "Missingness confounding; duplicate representation.", "Low-medium / low", "Preregister small family; use paired baseline residual ranking checks."),
        ("EXP06", "Missingness", "Patterns of unavailable fields add stable predictive information.", "Fold-local missing indicators/pattern counts; cap rare patterns; common-fold ablation.", "ALLOWED for supplied columns", "Sparse pattern overfitting; unstable rates.", "Low / low", "Prior missingness ablations showed negligible lift; test only targeted groups."),
        ("EXP07", "Numerical representation", "Log/rank/bin/spline representations improve tree/linear ranking.", "Every fitted cutpoint/transform inside outer fold; fit transforms only on fit rows.", "ALLOWED_WITH_RESTRICTIONS", "Redundant monotone transforms; global quantile leakage.", "Medium / medium", "Test log and predeclared spline families on same folds."),
        ("EXP08", "Model diversity", "Linear, LightGBM, CatBoost, XGBoost errors are complementary.", "OOF models on identical folds; fixed configurations; correlation/error-overlap before fixed blends.", "ALLOWED_WITH_RESTRICTIONS", "Tuning/blend selection on reused OOF labels.", "High / high", "First after feature candidates; only fixed blends preregistered."),
        ("EXP09", "Neural representations", "Entity-free tabular embeddings capture nonlinear signal missed by boosted trees.", "Fold-local preprocessing and training; fixed architecture/early stopping inner fold.", "ALLOWED_WITH_RESTRICTIONS", "Compute cost, unstable results, label overfit.", "High / high", "Defer until simpler methods show unsolved error structure."),
        ("EXP10", "Nonlinear interactions", "Smooth nonlinear and pairwise interactions explain stable residual ranking errors.", "GAM/spline/EBM fitted in outer fold; discovery candidates from fit fold only.", "ALLOWED_WITH_RESTRICTIONS", "Multiple testing, interpretability overclaim.", "Medium-high / medium", "Prioritize after EXP05 and EXP07; EBM unavailable here."),
        ("EXP11", "Duplicates / similarity", "Repeated predictor profiles create reliable difficulty strata.", "Train-only signatures; any similarity feature fit and queried inside folds; ID excluded.", "ALLOWED_WITH_RESTRICTIONS", "Identity inference/test coupling and leakage.", "Medium / high", "Audit only; no applicant linkage or test neighbor features."),
        ("EXP12", "Robustness/generalization", "Candidate lifts survive independent split/seed and perturbation.", "Freeze code/config; untouched new cohort preferred; seed replicates on same data are not independent.", "ALLOWED", "Confirmation set becomes adaptive through repeated peeking.", "Medium / medium", "Run last; do not tune against confirmation seed."),
    ]
    columns = [
        "experiment_id", "research_question", "hypothesis", "method_and_validation",
        "competition_eligibility", "failure_modes", "runtime_memory_estimate",
        "promotion_criteria_and_order",
    ]
    return pd.DataFrame(entries, columns=columns)


PHASE_RUN_OUTPUT: Path


def main() -> int:
    global PHASE_RUN_OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("T", "").replace("Z", "").replace("_", "").isalnum():
        raise ValueError("run-id may contain only alphanumeric characters, T, Z, and underscore")
    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    PHASE_RUN_OUTPUT = RUNS / f"phase6_{args.run_id}"
    PHASE_RUN_OUTPUT.mkdir(parents=True, exist_ok=False)

    required = [TRAIN_PATH, TEST_PATH, RULES_PATH, DICT_PATH, LEDGER_PATH]
    missing_paths = [str(path) for path in required if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required audit inputs missing: {missing_paths}")
    input_hashes = {str(path): sha256_file(path) for path in required}
    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    train_hash = input_hashes[str(TRAIN_PATH)]
    if ID not in train or TARGET not in train:
        raise ValueError("Training data lacks required ID/TARGET columns")
    if train[ID].isna().any() or train[ID].duplicated().any():
        raise ValueError("Training applicant IDs are missing or duplicated")
    if train[TARGET].isna().any() or set(train[TARGET].unique()) != {0, 1}:
        raise ValueError("Training target must contain exactly binary 0/1 labels")
    if len(train) != 171202:
        raise ValueError(f"Unexpected training rows: {len(train)}")

    test_header = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID])[ID]
    if len(test_ids) != 61500 or test_ids.isna().any() or test_ids.duplicated().any():
        raise ValueError("Test ID integrity check failed")
    train_predictors = [c for c in train.columns if c != TARGET]
    if test_header != train_predictors:
        raise ValueError("Train/test column order or schema mismatch")
    if TARGET in test_header:
        raise ValueError("Test schema unexpectedly contains TARGET")
    del test_ids

    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    folds = np.empty(len(train), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_pairs = list(splitter.split(np.zeros(len(train)), y))
    for fold_no, (_, valid_idx) in enumerate(fold_pairs, start=1):
        folds[valid_idx] = fold_no
    fold_frame = pd.DataFrame({ID: ids, "fold": folds})
    fold_frame.to_csv(PHASE_RUN_OUTPUT / "FOLD_ASSIGNMENTS.csv", index=False)
    fold_hash = sha256_file(PHASE_RUN_OUTPUT / "FOLD_ASSIGNMENTS.csv")

    saved_folds_path = EXISTING_PHASE2 / "FOLD_ASSIGNMENTS.csv"
    saved_fold_alignment = "not checked"
    if saved_folds_path.is_file():
        saved = pd.read_csv(saved_folds_path)
        if saved[ID].duplicated().any():
            raise ValueError("Historical fold file has duplicate IDs")
        saved_map = saved.set_index(ID)["fold"].reindex(ids)
        if saved_map.isna().any():
            raise ValueError("Historical fold IDs do not cover the current train IDs")
        saved_fold_alignment = str(
            np.array_equal(saved_map.to_numpy(dtype=np.int8), folds)
        )

    dictionary = dictionary_rows()
    pd.DataFrame(dictionary).to_csv(PHASE_RUN_OUTPUT / "CORE_DICTIONARY_DEFINITIONS.csv", index=False)
    fingerprints = numeric_fingerprint(train, y)
    fingerprints.to_csv(PHASE_RUN_OUTPUT / "NUMERICAL_FINGERPRINT_AUDIT.csv", index=False)
    financial = financial_relationships(train, y, folds, dictionary)
    financial.to_csv(PHASE_RUN_OUTPUT / "FINANCIAL_RELATIONSHIP_AUDIT.csv", index=False)
    missingness = missingness_audit(train, y, folds)
    missingness.to_csv(PHASE_RUN_OUTPUT / "MISSINGNESS_PATTERN_AUDIT.csv", index=False)
    external = external_score_discovery(train, y, folds)
    external.to_csv(PHASE_RUN_OUTPUT / "EXTERNAL_SCORE_DISCOVERY.csv", index=False)
    segments, segment_report = borrower_segments(train, y, PHASE_RUN_OUTPUT)
    write_text(PHASE_RUN_OUTPUT / "BORROWER_SEGMENT_DISCOVERY.md", segment_report)
    duplicate = duplicate_similarity(train, y)
    duplicate.to_csv(PHASE_RUN_OUTPUT / "DUPLICATE_SIMILARITY_AUDIT.csv", index=False)
    nonlinear = nonlinear_discovery(train, y, folds)
    nonlinear.to_csv(PHASE_RUN_OUTPUT / "NONLINEAR_PATTERN_DISCOVERY.csv", index=False)
    generate_experiment_plan().to_csv(PHASE_RUN_OUTPUT / "EXPERIMENT_MASTER_PLAN.csv", index=False)
    lattice_observations = financial[
        financial["feature"].astype(str).str.startswith("PRIOR_LATTICE_HYPOTHESIS_")
    ]
    lattice_summary = "; ".join(
        f"{r['feature']}: integer-within-0.01={r['candidate_integer_coordinate_share']:.4f}, "
        f"residual<0.1={r['candidate_residual_below_0_1_share']:.4f}, n={int(r['n_valid'])}"
        for _, r in lattice_observations.iterrows()
    )
    external_scores = external[
        external["feature"].isin(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_MEAN_AVAILABLE"])
    ]
    external_summary = "; ".join(
        f"{r['feature']} AUC={r['univariate_auc_descriptive']:.4f}"
        for _, r in external_scores.iterrows()
    )
    missing_top = missingness[missingness["pattern_type"] == "single_column"].sort_values(
        "missing_indicator_auc",
        key=lambda s: abs(pd.to_numeric(s, errors="coerce").fillna(0.5) - 0.5),
        ascending=False,
    ).head(4)
    missing_summary = "; ".join(
        f"{r['pattern']} missing={int(r['n'])} ({r['share']:.1%}), "
        f"default={r['default_rate']:.3%}, indicator AUC={r['missing_indicator_auc']:.4f}"
        for _, r in missing_top.iterrows()
    )
    duplicate_exact = duplicate.iloc[0]
    cluster_rates = segments["default_rate_descriptive"].to_numpy(dtype=np.float64)
    nonlinear_top = nonlinear.head(3)
    amount_integer_rows = fingerprints[
        fingerprints["feature"].isin(
            ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]
        )
    ]
    amount_integer_summary = "; ".join(
        f"{r['feature']}={r['integer_share_tol_1e-8']:.4%}"
        for _, r in amount_integer_rows.iterrows()
    )
    nonlinear_summary = "; ".join(
        f"{r['feature']} raw={r['raw_univariate_oof_auc_descriptive']:.4f}, "
        f"fold-local bins={r['fold_local_quantile_bin_auc_descriptive']:.4f}"
        for _, r in nonlinear_top.iterrows()
    )

    eligibility = pd.DataFrame(
        [
            ("Supplied train predictors/labels", "Explicitly permitted", "Train-only fitting and evaluation; target is available only in training."),
            ("Supplied test file", "Permitted for final inference and schema/ID checks", "This run reads only test header and IDs; no predictor statistics or test TARGET."),
            ("External datasets/labels", "Explicitly prohibited", "Not used."),
            ("Manual labels/prediction overrides", "Explicitly prohibited", "Not used."),
            ("Rowwise transforms from supplied predictors", "Permitted", "Use dictionary-supported columns; semantic interpretations still need evidence."),
            ("Fold-local unsupervised transforms", "Permitted subject to official review", "Fit on outer fit only; apply unchanged to validation."),
            ("Pooled train/test statistics", "Unresolved; excluded by clean-room policy", "Requires written organizer clarification."),
            ("Target encoding/supervised embeddings", "Unresolved; excluded", "Nested validation does not itself settle eligibility."),
            ("Test-to-test relational coupling", "Ambiguous/high risk; excluded", "No applicant linking, cross-test feature building, or rank coupling."),
            ("Train-only group statistics", "Allowed with restrictions", "Fit on each outer fit fold; no full-data maps in OOF."),
            ("Clustering", "Allowed with restrictions", "Exploration only here; predictive use must be fold-local."),
            ("Public leaderboard tuning", "Not valid as independent validation", "No leaderboard score used in this task."),
            ("OOF and fixed blend", "Permitted as validation method", "OOF must be complete/aligned; ensemble selection must not reuse reported labels."),
            ("Code reuse / third-party licenses", "Citation required; license status unverified", "No new third-party solution code adopted."),
        ],
        columns=["method", "status", "clean_room_control"],
    )
    eligibility.to_csv(PHASE_RUN_OUTPUT / "COMPETITION_ELIGIBILITY_REGISTER.csv", index=False)
    write_text(
        PHASE_RUN_OUTPUT / "CLEAN_ROOM_RULES.md",
        """# Phase 6 clean-room rules

Authority: supplied course README/tutorial/dictionary and existing local official-rules audit. This is a conservative research policy, not an organizer ruling.

- Use only supplied training data for feature discovery, label association, segmentation, and validation.
- Test file is limited to final inference, schema/ID integrity checks, and byte hashing. This run reads only its header and ID column.
- Exclude external datasets, outside labels, hand labeling, manual score edits, and test-to-test entity coupling.
- Do not fit pooled train+test population statistics. They remain unresolved under the local rule text.
- Do not use target encoding/supervised representations until the organizer clarifies eligibility.
- Fit any learned preprocessing, feature selection, quantiles, clusters, encoders, and model stopping decisions within the corresponding outer training fold.
- OOF predictions must be exactly once per training ID, aligned to TARGET/fold, and generated by models that excluded that row's label.
- Fix the candidate and blend procedure before reading confirmation labels. Repeated random splits of the known labels do not create an independent cohort.
- Do not use leaderboard feedback as a feature-selection/confirmation signal.
- Preserve historical Phase 2/3 outputs; do not change champions or create submissions from this task.
""",
    )
    write_text(
        PHASE_RUN_OUTPUT / "RULES_BLOCKER_REGISTER.md",
        """# Rules blocker register

| Blocker | Local evidence | Phase 6 disposition | Closure needed |
|---|---|---|---|
| Pooled train+test covariates | Local rules do not explicitly permit/forbid fitting unlabeled pooled statistics. | Excluded. | Written organizer clarification. |
| Target encoding / supervised representation | No explicit clause; leakage safety does not determine eligibility. | Excluded. | Written organizer clarification. |
| Test-to-test entity or score coupling | Hand-labeling clause creates ambiguity for feature-driven individual score changes. | Excluded. | Written organizer clarification; safest is never to use. |
| External data | Explicit no-external-data rule. | Prohibited and not used. | None; do not add external sources. |
| Manual labels / prediction edits | Explicit hand-labeling restriction. | Prohibited and not used. | None. |
| Citation/license disclosure | Source citation required; licenses not specified locally. | New third-party models/code not imported. | Ask organizers on disclosure/license format as needed. |
| Independent confirmation | Existing labeled train set and folds repeatedly examined; no unseen labels. | All new associations are exploratory, not independent confirmation. | New previously unseen labeled cohort or organizer-approved external validation protocol. |
| Competition rule currency | Existing audit did not check a live portal or later amendment. | No eligibility certification. | Verify current organizer rules before submission. |
""",
    )

    audit_rows = [
        ("Raw baseline / corrected Phase 2", str(BASELINE_SOURCE), "CONFIRMED SAFE for fold-local OOF design; not independent confirmation", "Rowwise supplied columns; nested inner stopping; train-fold frame fitting; no TE/pooling.", "Prior exact OOF 0.760658; fresh run below."),
        ("Phase 2B peer-relative", str(PHASE2B_RUNNER), "CONFIRMED SAFE by source/report for validation; exploratory only", "Peer map uses explicit fit-only reference rows; no test pool; prior pooled lift +0.000034 is negligible.", "External interactions negative; no promotion."),
        ("Phase 3 product decoder", str(PHASE3_RUNNER), "CONFIRMED SAFE by reported fold controls; domain meaning unverified", "Fee/lattice/key params fit on outer train; no labels used in feature fitting; no test pool.", "Discovery OOF 0.761353; reused folds."),
        ("Phase 3B product model", str(PHASE3B_RUNNER), "CONFIRMED SAFE for validation design; exploratory only", "Fixed folds; inner raw-feature stopping; fit-only parameters; negative controls.", "In-run baseline 0.760658; product 0.762444."),
        ("Phase 3C LightGBM/CatBoost/XGB benchmark", str(PHASE3C_RUNNER), "CONFIRMED SAFE for fold-local model fitting; adaptive candidate search limits evidence", "Fold-local preprocessing; fixed configurations; validated ID/target/fold OOF.", "Best fixed equal blend 0.763957; selected after inspecting reused folds."),
        ("Phase 3C advanced discovery", str(PHASE3C_ADVANCED), "CONFIRMED SAFE for validation transformations; exploratory only", "Quantile cutpoints, clusters, and preprocessing fit on outer train; no target encoding.", "Cluster/nonlinear/missingness lifts negligible; one candidate-selection surface."),
        ("Phase 3D formula probe", str(PHASE3D_RUNNER), "CONFIRMED SAFE for rowwise training validation; exploratory only", "Rowwise supplied training columns; no test pool, target encoding, or outer-label stopping.", "No new formula gain; historical folds exposed."),
        ("EXP02 independent-seed replication", str(EXP02_VERIFY), "RULES UNRESOLVED for some interpretations; validation leakage not found in audited verifier", "Supplied train data, seed 20261105, fold-scoped model; test used for schema only.", "Joint block +0.006336 vs its own raw comparator, but ratios alias Phase 3C and annuity-period semantics unresolved."),
        ("Historical V16 OOF claim", "Phase 5 Task 3 cache provenance investigation", "INSUFFICIENT EVIDENCE; RULES UNRESOLVED; not verified", "Original gci_cache_v16 predictions/fold map/input binding are absent; V15 cache diagnostic is not V16; pooled features and target encoding remain unresolved.", "Saved 0.77649 is an unverified claim. The 0.79467683 bundle is invalid in-sample OOF evidence and is not accepted."),
        ("V5 pooled/target-encoded notebook", str(PHASE3D_REPORT), "RULES UNRESOLVED; not eligible in clean-room pipeline", "Historical audit documented train/test pooling and nested target encoding; source lineage gaps.", "Retain only as forensic evidence; not used as candidate."),
        ("V16/EXP85 historical pipelines", "Phase 5 Task 3 audit", "RULES UNRESOLVED / excluded", "Pooled transforms, target encoding, test-to-test methods unresolved; caches/provenance incomplete.", "Not used as Phase 6 baseline or feature source."),
    ]
    pd.DataFrame(
        audit_rows,
        columns=["experiment", "source", "classification", "static_audit", "evidence_and_limit"],
    ).to_csv(PHASE_RUN_OUTPUT / "HISTORICAL_EXPERIMENT_ELIGIBILITY.csv", index=False)

    failures = [
        ("VAL-01", "Historical raw baseline", "No material leakage found in corrected runner", "Historical folds reused; research exposure prevents independent confirmation.", "CONFIRMED SAFE / INSUFFICIENT EVIDENCE for independence"),
        ("VAL-02", "Earlier domain formula probe", "Outer-fold early stopping used outer validation AUC; features selected by full-data target AUC.", "Contaminated OOF and target-driven selection.", "CONFIRMED LEAKAGE"),
        ("VAL-03", "Earlier dummy-column formula runner", "Whole-training category/dummy schema fit outside folds.", "Preprocessing leakage and non-identical baseline.", "CONFIRMED LEAKAGE"),
        ("VAL-04", "V5 saved-score notebook", "Pooled train/test statistics and target encodings; matching cache/output absent.", "Rule scope unresolved; OOF/source/cache provenance incomplete.", "RULES UNRESOLVED / INSUFFICIENT EVIDENCE"),
        ("VAL-05", "Blend/benchmark selection on exposed folds", "Several model/feature/blend candidates selected after observing common OOF labels.", "Selection optimism; scores remain discovery, not independent confirmation.", "CONFIRMED LIMITATION"),
        ("VAL-06", "Historical OOF artifacts", "Some outputs lack fold maps, row IDs, or full source/input binding.", "Cannot establish exact validation exclusion/alignment from AUC alone.", "INSUFFICIENT EVIDENCE"),
        ("VAL-07", "EXP02 score interpretation", "Ratio block overlaps baseline; annuity period/family allocation not in dictionary.", "Joint predictive association is not a new domain mechanism or champion lift.", "RULES UNRESOLVED for semantic assertions"),
        ("VAL-08", "Target encoding in Phase 6", "No Phase 6 target encoding implemented.", "Excluded until eligibility clarification, regardless of nested safety.", "RULES UNRESOLVED / DISABLED"),
        ("VAL-09", "Historical V16 OOF claim", "No source-bound gci_cache_v16 arrays, original fold map, or row-alignment sidecar recovered; anomalous bundle arrays trace to a full-fit train-prediction scaffold.", "0.77649 cannot be called verified OOF; 0.79467683 is rejected as OOF evidence. No V16 reproduction was performed.", "INSUFFICIENT EVIDENCE; RULES UNRESOLVED"),
    ]
    pd.DataFrame(
        failures,
        columns=["failure_id", "scope", "finding", "effect", "classification"],
    ).to_csv(PHASE_RUN_OUTPUT / "VALIDATION_FAILURE_REGISTER.csv", index=False)

    # The existing corrected baseline implementation is imported without invoking its CLI.
    spec = importlib.util.spec_from_file_location("phase6_corrected_baseline", PHASE2_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import corrected raw baseline runner: {PHASE2_RUNNER}")
    baseline_runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline_runner
    spec.loader.exec_module(baseline_runner)
    model_start = time.perf_counter()
    _, raw_oof = baseline_runner.run_variant(
        train,
        fold_pairs,
        variant="phase6_raw_baseline",
        include_formulas=False,
    )
    model_seconds = time.perf_counter() - model_start
    if not np.isfinite(raw_oof).all():
        raise AssertionError("Raw baseline did not produce exactly one finite OOF prediction per row")
    oof_auc = float(roc_auc_score(y, raw_oof))
    fold_metrics = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": int(fold),
                "n_validation": int(mask.sum()),
                "positive_n": int(y[mask].sum()),
                "auc": float(roc_auc_score(y[mask], raw_oof[mask])),
            }
        )
    pd.DataFrame(fold_metrics).to_csv(
        PHASE_RUN_OUTPUT / "BASELINE_FOLD_METRICS.csv", index=False
    )
    oof_frame = pd.DataFrame({ID: ids, TARGET: y, "fold": folds, "oof_raw_baseline": raw_oof})
    oof_frame.to_csv(PHASE_RUN_OUTPUT / "RAW_BASELINE_OOF.csv", index=False)
    oof_hash = sha256_file(PHASE_RUN_OUTPUT / "RAW_BASELINE_OOF.csv")
    hist_auc = 0.760658
    base_report = f"""# Baseline reproducibility report

## Result

- Fresh raw-only five-fold OOF ROC-AUC: **{oof_auc:.9f}**.
- Historical corrected Phase 2 reported OOF: `0.760658` (existing reproducibility evidence).
- Fresh minus historical rounded claim: `{oof_auc - hist_auc:+.9f}`.
- Fold AUCs and validation counts: `BASELINE_FOLD_METRICS.csv`.
- OOF file: `RAW_BASELINE_OOF.csv`; SHA-256 `{oof_hash}`; exactly one finite prediction per training row, keyed by ID and joined back to TARGET/fold.
- Fold assignments SHA-256: `{fold_hash}`; match to preserved Phase 2 fold mapping by ID: `{saved_fold_alignment}`.
- Training runtime: `{model_seconds:.1f}` seconds. No feature-family search or large model sweep was run.

## Protocol and restrictions

The raw-only runner is the preserved corrected Phase 2 implementation: five stratified folds, seed 42, nested early stopping on a stratified inner split of each outer training partition, fit-partition feature framing, and validation prediction only after refitting on all outer-fit rows. The model uses raw/rowwise fields from the supplied training data. No test predictor values were read; test header/IDs were used only for schema and ID checks. No external data, target encoding, pooled train/test statistics, manually curated prediction rule, or submission was used.

This confirms the baseline under the current available code and data hashes; it does not make the reused five folds an untouched estimate. The model/configuration was inherited from prior research and the training labels have been repeatedly examined. The reported historical Phase 2 score is numerically reproduced only if the fresh value agrees within numerical tolerance; otherwise the discrepancy remains explicit and must be traced, not rounded away.

## Recommendation

Use this as the Phase 6 clean-room development reference, not as a champion claim. Require identical folds for initial paired discovery, and acquire genuinely unseen labeled data for independent confirmation.
"""
    write_text(PHASE_RUN_OUTPUT / "BASELINE_REPRODUCIBILITY_REPORT.md", base_report)

    # Static audit source hashes and fixed findings.
    source_inventory = []
    for path in AUDIT_SOURCES:
        if path.is_file():
            source_inventory.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "role": "source_or_evidence",
                }
            )
        else:
            source_inventory.append(
                {"path": str(path), "bytes": "", "sha256": "", "role": "MISSING"}
            )
    pd.DataFrame(source_inventory).to_csv(
        PHASE_RUN_OUTPUT / "AUDITED_SOURCE_INVENTORY.csv", index=False
    )
    code_audit = f"""# Phase 2/3 code audit

## Scope and evidence

Audited preserved raw baseline source, corrected Phase 2 validation runner, Phase 2B peer runner, Phase 3 decoder runner, Phase 3B ablation runner, Phase 3C benchmark and advanced discovery runners, Phase 3D formula runner, and EXP02 independent-seed verifier. The code/source hashes and file sizes are in `AUDITED_SOURCE_INVENTORY.csv`. Historical execution reports and validation governance were cross-checked. This is a source/code-and-artifact audit, not a byte-for-byte reproduction of every old model execution.

## Main findings

1. **Corrected raw baseline:** nested inner stopping; outer validation not used in feature selection or stopping; transformation and categorical frame fit on model-training rows. No target encoding, pooled train/test fit, or external data in the corrected run. Raw baseline is the clean comparison reference.
2. **Phase 2B peer-relative features:** code accepts `peer_fit_raw` separately from transform rows and fits peer statistics on the fit reference. Historical report says no test rows/labels and no target encoding. Peer-only lift `+0.000034` is too small to promote; external interactions and combined family were negative.
3. **Phase 3 product decoder:** implementation/report describe fit-only fee/lattice/key parameters and inner early stopping; no TARGET is used to construct decoder features. Product structure is predictive evidence, not proof of actual loan terms/fees. Original folds are reused.
4. **Phase 3B controlled product model:** governance preregistered candidate families and negative controls; preprocessing and parameters are fit inside outer fit partitions; same fold-specific stopping rounds are applied to candidates. Best product OOF `0.762444` versus `0.760658` in-run raw reference, 5/5 fold improvement, but exposed-fold discovery only.
5. **Phase 3C model/ensemble:** source/report validate ID, TARGET, fold, finite prediction alignment and use fold-local preprocessing. Equal-weight blend has fixed weights, but the best of multiple variants/blends was chosen after seeing reused OOF labels, so reported maximum is adaptively selected.
6. **Phase 3C advanced:** quantile cutpoints, cluster assignments, and missingness representation are fit in each outer training fold for model use. Reported additions gave tiny/negative changes (e.g. k=4 segment `-0.000059`, all-missingness `+0.000031`, binned annuity `+0.000174`); discovery-only.
7. **Phase 3D formula probe:** rowwise supplied-training transformations; nested early stopping; no pooled test transformations or target encoding in the tested probe. Candidate formulas largely duplicate existing ratios and unsupported annuity-period semantics remain.
8. **EXP02 seed 20261105:** independent verifier confirms five-fold OOF alignment, seed-specific folds, finite predictions, and test use limited to schema; joint ratio block improves its matched raw-only comparator by `+0.006336142` pooled / 5-of-5 folds. However ratios alias Phase 3C baseline and this is not incremental champion lift. New split seed on the same known labels is not independent confirmation. Household allocation/annuity-period interpretations remain unsupported.
9. **Historical V16 OOF:** not verified. The executed notebook's configured `gci_cache_v16` arrays, original fold map, and source/input-to-array manifest are missing. The `0.77404329` candidate result uses V15 caches and is not a V16 reproduction. The saved `0.77649` remains an unverified notebook-output claim. The `0.79467683` bundle is invalid OOF evidence because its arrays are identical full-fit training predictions under the inspected scaffold. V16 pooled features/target encoding also remain unresolved under local rules; no reproduction is attempted in this phase.

## Confirmed defects/failures retained in history

- Earlier domain-formula probe used outer validation for early stopping and used full-training target AUC to choose formulas: **CONFIRMED LEAKAGE**; discard its OOF/selection claim.
- Earlier formula comparison fit dummy/category columns from full training rather than from each fold's fit rows: **CONFIRMED PREPROCESSING LEAKAGE** and not an exact baseline comparison.
- Several older cached OOFs lack source/input/fold/ID binding; AUC alone cannot prove OOF. **INSUFFICIENT EVIDENCE**.
- V5/V16 methods that pool train/test, use target encoding, or couple test applicants are excluded because rule status is unresolved; their historical outputs are not Phase 6 candidate evidence.
- Model and blend selection across reused historical folds introduces adaptive selection optimism even when each individual OOF generation is fold-safe.

## Eligibility classification

Historical raw Phase 2, Phase 2B, Phase 3, 3B, 3C and 3D rowwise/fold-local experiments are **methodologically eligible for continued training-only research**, subject to their specific feature definitions and fresh source/data/OOF integrity checks. They are **not independently confirmed** and not automatic champions. EXP02 is eligible only as a rowwise predictive hypothesis after semantic caveats; do not describe it as a new Phase 3C ratio discovery. V5/V16/EXP85 pooled/target-encoded/test-coupled paths are **excluded pending rule-owner approval**.

## Boundary

No historical model or output was edited. This audit did not evaluate a submission, use leaderboard feedback, inspect test predictors beyond schema/IDs, or assert compliance beyond the local rules.
"""
    write_text(PHASE_RUN_OUTPUT / "PHASE2_PHASE3_CODE_AUDIT.md", code_audit)

    val_spec = f"""# Validation framework specification

## Common discovery protocol

- Primary metric: ROC-AUC.
- Initial comparison folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, saved by ID in `FOLD_ASSIGNMENTS.csv` (SHA-256 `{fold_hash}`).
- All candidates in a comparison receive the same row/fold assignments.
- Every learned transform, imputer, category vocabulary, bin boundary, feature selection, clusterer, and supervised fit is learned only on the current outer fit data. No test covariates participate.
- Inner early stopping uses an inner stratified split of outer-training labels; refit on all outer-fit rows for exactly the selected iterations; score each outer validation row once.
- OOF file must contain unique ID, target, fold, one finite prediction per row, and exact current data ID/label/fold alignment. Hash predictions and source.
- Report pooled OOF AUC, five fold AUCs, mean/std fold AUC, fold deltas, positive-fold count, prediction correlation and ranking disagreement for model comparisons.
- Training validation labels may be used for descriptive error analysis only after prediction generation; they cannot select features/weights for the same reported score.
- Fixed blends must have weights frozen before evaluation. Any blend selection requires nested selection or a separate untouched labeled cohort.
- Confirmation seed/cohort is kept sealed; no tuning after inspecting its results. A different random seed over these same labels is not independent confirmation.

## Current exposure and limitations

The saved seed-42 folds and every label in this training table have been examined in earlier research. The Phase 6 baseline is reproducible and usable for paired exploratory comparisons, but not a fresh independent estimate. A truly independent confirmation requires a previously unseen labeled cohort or an organizer-approved validation split that has not informed feature/model design.

## Prohibited or disabled

No test-to-test coupling, pooled train/test fitting, unresolved target encoding, external data, manual label/score edits, or leaderboard-driven tuning. Test data are read only for header/ID integrity here.
"""
    write_text(PHASE_RUN_OUTPUT / "VALIDATION_FRAMEWORK_SPEC.md", val_spec)

    hidden_summary = f"""# Hidden-pattern discovery report

## Scope

All analyses below use the `{len(train):,}` supplied training rows only. `TARGET` is used for descriptive rates/associations, not for transforming features or fitting unsupervised clusters. The test file was used only for raw schema, ID uniqueness/count, and file hashing. These are exploratory findings on previously inspected labels and do not establish independent predictive lift or causal mechanisms.

## Observations

- The official dictionary defines the core financial fields as client income, loan credit amount, loan annuity, and consumer-loan goods price. It does **not** specify annuity payment period, actual payment duration, interest rate, or household allocation of debt.
- Raw numeric fingerprints, repeated-value shares, heaping, decimal precision, and outlier quantiles are in `NUMERICAL_FINGERPRINT_AUDIT.csv`. Integer-like shares for the four key financial amount fields are: {amount_integer_summary}. This describes source encoding, not an actual contract price grid.
- The prior product decoder's actual lattice hypothesis is `(AMT_CREDIT / AMT_GOODS_PRICE - 1) / step` for cash-loan rows, not the unadjusted credit/goods ratio divided by step. This run's training-only diagnostic: {lattice_summary}. The pattern is worth a carefully controlled ablation, but remains a numerical regularity, not proof of actual financing tiers.
- Ordinary ratio AUCs are descriptive and already represented in historical features. For example, `CREDIT_TO_GOODS` univariate AUC is {financial.loc[financial.feature == 'CREDIT_TO_GOODS', 'univariate_auc_descriptive'].iloc[0]:.4f}; it is not incremental gain. `FINANCIAL_RELATIONSHIP_AUDIT.csv` records the tested definitions and semantic caveats.
- Leading external-score raw univariate AUCs: {external_summary}. Pairwise disagreements, range, dispersion, interactions, and missingness are in `EXTERNAL_SCORE_DISCOVERY.csv`; these numbers do not establish incremental model gain. Prior Phase 3C interaction/missingness ablations were near zero or negative.
- Strongest single-field missingness indicators from the train-only descriptive screen: {missing_summary}. The four credit-bureau request fields share the same missing rows in this dataset; this is one missingness block rather than four independent findings. Sparse-pattern rates remain noisy.
- PCA/K-means/GMM/hierarchical numerical exploration is in `BORROWER_SEGMENT_DISCOVERY.md` and companion CSVs. Clusters are geometric partitions, not validated borrower classes. Cluster features must be learned inside each model fold.
- Exact duplicate signatures excluding ID and TARGET: {int(duplicate_exact['duplicate_signature_groups'])} duplicate groups among {int(duplicate_exact['n_rows']):,} rows. Numeric nearest-neighbor results are sample-only and are not entity linkage. See `DUPLICATE_SIMILARITY_AUDIT.csv`.
- Leading per-feature nonlinear screens (raw / fold-local bins): {nonlinear_summary}. Cutpoints are fit per outer training fold. These are univariate screens, not model-level improvement evidence; the monotone log transform cannot create new rank information by itself.
- Four-cluster training-only default rates span {np.min(cluster_rates):.3%}–{np.max(cluster_rates):.3%}; K-means seed agreement and PCA profiles are exploratory, not proof of stable real borrower types.

## Pre-existing hidden-pattern evidence

Prior Phase 3 evidence found a product-decoder feature block at `0.762826` discovery OOF and a controlled Phase 3B product model at `0.762444` versus `0.760658` raw; the decoder's purported lattice/fee meaning remains unproven, and seed-42 folds are exposed. Phase 3C equal-weight blend was `0.763957` but selected after inspecting multiple candidates on reused folds. Phase 2B peer deviations gave only `+0.000034`. EXP02 ratio block showed `+0.006336` against its own raw CatBoost baseline on a different seed, but aliases existing Phase 3C ratios; it does not show incremental lift over that stronger baseline.

## Conclusions

1. The most credible signal already observed is combined supplied-field structure around loan/product amounts, but it requires a preregistered incremental ablation against the raw-plus-existing-product baseline.
2. External-score disagreement, missingness, clustering, and generic interactions have not shown convincing incremental lift in prior controlled reports.
3. No new hidden pattern found here is promoted as predictive until it beats an aligned baseline in fold-safe model ablation and survives a genuinely untouched labeled confirmation.
4. Numerical regularities do not prove loan terms, fees, causal risk mechanisms, or applicant identity.
"""
    write_text(PHASE_RUN_OUTPUT / "HIDDEN_PATTERN_REPORT.md", hidden_summary)

    plan = generate_experiment_plan()
    first_actions = (
        "1) Freeze/audit this raw baseline and the feature manifest. "
        "2) Run one predeclared product-feature incremental ablation against raw-plus-existing-product on common folds, with negative controls and no retuning. "
        "3) Test a small external-score disagreement block against that same baseline. "
        "4) Only then evaluate targeted missingness/nonlinear representations. "
        "5) Model diversity and blends follow only after candidate OOF arrays pass integrity checks. "
        "6) Reserve confirmation for genuinely unseen labeled data."
    )
    write_text(
        PHASE_RUN_OUTPUT / "NEXT_PHASE_RECOMMENDATION.md",
        f"""# Next phase recommendation

## First experiment

Run **EXP02-P6: a preregistered, fold-safe incremental product-structure ablation**. Compare the clean raw-only baseline with raw plus the nonredundant Phase 3 product block using the same fixed five folds. Include one random-key/permuted-feature negative control. Freeze exact features, fixed model parameters, fold-local parameter fitting, and decision rules before reading OOF results. Do not copy test-pool, target-encoded, or test-coupled code. This uses already-known labels and is a discovery replication, not independent confirmation.

## Sequence

{first_actions}

## Gates before doing model work

- Confirm current organizer rules and maintain conservative exclusions until written clarification.
- Obtain genuinely unseen labeled confirmation data before any champion promotion.
- Do not try to recover or reproduce V16/EXP85 as an eligible pipeline.
- Do not optimize blend weights on the same OOF labels used to quote final performance.
- No AUC target, including 0.78 or 0.79, is promised.
""",
    )

    experiment_ids = plan["experiment_id"].tolist()
    rule_rows = len(eligibility)
    fail_rows = len(failures)
    elapsed = time.perf_counter() - wall_start
    started_utc_text = started.isoformat()
    log_lines = [
        f"status=COMPLETED",
        f"run_id={args.run_id}",
        f"started_utc={started_utc_text}",
        f"finished_utc={datetime.now(timezone.utc).isoformat()}",
        f"runtime_seconds={elapsed:.3f}",
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}; pandas={pd.__version__}; scipy={scipy.__version__}; sklearn={sklearn.__version__}",
        f"cpu_count={os.cpu_count()}",
        f"train_shape={train.shape}; train_sha256={train_hash}",
        f"test_sha256={input_hashes[str(TEST_PATH)]}; test predictors read=no; header/id checks only=yes",
        f"official_rules_sha256={input_hashes[str(RULES_PATH)]}",
        f"dictionary_sha256={input_hashes[str(DICT_PATH)]}",
        f"ledger_latest_before_run=2026-11-04 Phase 5 Task 3 V16 cache provenance investigation",
        f"phase6_source_sha256={sha256_file(Path(__file__).resolve())}",
        f"fold_sha256={fold_hash}; saved_seed42_fold_map_match={saved_fold_alignment}",
        f"baseline_pooled_auc={oof_auc:.9f}; baseline_runtime_seconds={model_seconds:.3f}",
        f"baseline_oof_sha256={oof_hash}",
        f"process_peak_working_set_bytes={process_peak_bytes()}",
        f"eligibility_rows={rule_rows}; validation_failure_rows={fail_rows}; experiments_planned={len(experiment_ids)}",
        "large_model_training_sweeps=NOT RUN",
        "test_TARGET_read=NO; external_data=NO; pooled_train_test_fit=NO; target_encoding=NO; test_to_test_coupling=NO",
        "submission_created=NO; champion_modified=NO; historical_artifacts_modified=NO",
        f"outputs={PHASE_RUN_OUTPUT}",
    ]
    write_text(PHASE_RUN_OUTPUT / "EXECUTION_LOG.txt", "\n".join(log_lines))

    # The final manifest intentionally excludes itself.
    manifest_rows = []
    for path in sorted(PHASE_RUN_OUTPUT.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(
        PHASE_RUN_OUTPUT / "SHA256_MANIFEST.csv", index=False
    )

    report = f"""# Phase 6 execution report

## Status

**Clean-room foundation and training-only discovery completed.** Historical Phase 2/3 source/evidence was audited; a fresh raw-only baseline was run. No large model sweeps, submissions, or champion changes.

## Inputs and validation

- Training rows/columns: `{train.shape[0]:,} x {train.shape[1]}`; SHA-256 `{train_hash}`.
- Test rows: `61,500`; SHA-256 `{input_hashes[str(TEST_PATH)]}`. Test predictors were not read; header/ID checks only.
- Saved seed-42 fold alignment: `{saved_fold_alignment}`; new fold file SHA-256 `{fold_hash}`.
- Fresh raw-only OOF AUC: **{oof_auc:.9f}**, compared with historical corrected Phase 2 `0.760658`.
- Fold metrics and ID/TARGET/fold-aligned OOF predictions are preserved.

## Completed deliverables

The timestamped folder contains the requested eligibility register, clean-room rules, blocker register, Phase 2/3 code audit, experiment eligibility and validation-failure registers, raw baseline report, hidden-pattern reports and CSVs, experiment master plan, validation specification, manifest, execution log, and next-step recommendation.

## Important findings

- Confirmed leakage remains in the older domain-formula probe (outer validation used for stopping and full-label feature selection) and earlier global dummy-column preprocessing. Those older metrics are rejected as unbiased OOF.
- Corrected Phase 2 and later Phase 3 fold-local runners are suitable for **exploratory research** under the clean-room restrictions; selection over repeatedly viewed folds prevents independent confirmation.
- Existing controlled results do not establish new incremental lift for generic peer deviations, external-score interactions, missingness, or clustering. Product-structure signal is the strongest candidate family but needs a frozen incremental ablation.
- EXP02's seven finance ratios overlap Phase 3C features; its `+0.006336` compares to its own raw-only baseline, not to Phase 3C/champion.
- The historical V16 OOF score `0.77649` remains unverified. Missing V16 caches/folds/source binding prevent reproduction; the `0.79467683` duplicate bundle is invalid OOF evidence. Neither result is used as a baseline or candidate.
- Pooled train/test statistics, target encoding, and test-to-test coupling remain excluded pending rule clarification.
- No actual loan term, rate, fee, or household allocation is inferred from numeric regularity.

## Remaining blockers

The local official materials are not an organizer ruling on unresolved methods, the research folds/labels are exposed, and no unseen labeled confirmation cohort is available. Model promotion and competition eligibility for any pipeline relying on those methods remain blocked.

## Execution

Run ID `{args.run_id}`; UTC start `{started_utc_text}`; runtime `{elapsed:.1f}s`. The detailed environment, hashes, and no-training/submission assertions are in `EXECUTION_LOG.txt`.
"""
    write_text(PHASE_RUN_OUTPUT / "PHASE6_EXECUTION_REPORT.md", report)

    # Update the manifest after the primary report has been written.
    manifest_rows = []
    for path in sorted(PHASE_RUN_OUTPUT.iterdir()):
        if path.is_file() and path.name != "SHA256_MANIFEST.csv":
            manifest_rows.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    pd.DataFrame(manifest_rows).to_csv(PHASE_RUN_OUTPUT / "SHA256_MANIFEST.csv", index=False)
    print("\n".join(log_lines))
    print(f"OUTPUT_DIRECTORY={PHASE_RUN_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
