from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


NUMERIC_FIELDS = (
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_LAST_PHONE_CHANGE",
)

LOW_CARDINALITY_FIELDS = (
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_MOBIL",
    "REGION_RATING_CLIENT",
    "CNT_CHILDREN",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    category_codes: np.ndarray
    family: str
    unique_categories: int


def significant_integer_codes(values: np.ndarray, digits: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    codes = np.full(len(values), -1, dtype=np.int64)
    valid = np.isfinite(values)
    nonzero = valid & (values != 0)
    exponent = np.zeros(len(values), dtype=np.int32)
    exponent[nonzero] = np.floor(np.log10(np.abs(values[nonzero]))).astype(np.int32)
    scale = np.power(10.0, digits - 1 - exponent[nonzero], dtype=np.float64)
    mantissa = np.rint(values[nonzero] * scale).astype(np.int64)
    if np.any(np.abs(mantissa) >= 1_000_000):
        raise OverflowError("Significant-digit mantissa exceeds code layout")
    codes[valid] = (
        (exponent[valid].astype(np.int64) + 1000) * 2_000_001
        + 1_000_000
    )
    codes[nonzero] += mantissa
    return codes


def _factorize_codes(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    if len(parts) == 1:
        codes, _ = pd.factorize(
            parts[0], sort=False, use_na_sentinel=False
        )
    else:
        codes, _ = pd.MultiIndex.from_arrays(parts).factorize(
            sort=False, use_na_sentinel=False
        )
    if np.any(codes < 0):
        raise AssertionError("Candidate factorization left missing category codes")
    return codes.astype(np.int32)


def _candidate(
    name: str,
    parts: tuple[np.ndarray, ...],
    family: str,
) -> Candidate:
    codes = _factorize_codes(parts)
    return Candidate(
        name=name,
        category_codes=codes,
        family=family,
        unique_categories=int(codes.max()) + 1,
    )


def generate_candidates(raw: pd.DataFrame) -> Iterator[Candidate]:
    missing_numeric = set(NUMERIC_FIELDS).difference(raw.columns)
    missing_low_card = set(LOW_CARDINALITY_FIELDS).difference(raw.columns)
    if missing_numeric or missing_low_card:
        raise ValueError(
            f"Missing candidate inputs: numeric={sorted(missing_numeric)}, "
            f"low_cardinality={sorted(missing_low_card)}"
        )

    numeric = {
        column: pd.to_numeric(raw[column], errors="raise").to_numpy(dtype=np.float64)
        for column in NUMERIC_FIELDS
    }
    low_card = {
        column: pd.factorize(
            raw[column].astype("string").fillna("MISSING"), sort=True
        )[0].astype(np.int32)
        for column in LOW_CARDINALITY_FIELDS
    }
    seen: set[bytes] = set()

    def unique(candidate: Candidate) -> bool:
        digest = hashlib.blake2b(
            candidate.category_codes.tobytes(), digest_size=12
        ).digest()
        if digest in seen:
            return False
        seen.add(digest)
        return True

    for column, values in numeric.items():
        for digits in (3, 4, 5):
            candidate = _candidate(
                f"numeric:{column}:sig{digits}",
                (significant_integer_codes(values, digits),),
                "numeric_single",
            )
            if unique(candidate):
                yield candidate
        finite = np.isfinite(values)
        if finite.any() and np.unique(values[finite]).size > 1:
            quantiles = np.unique(
                np.quantile(values[finite], np.linspace(0, 1, 11))
            )
            if len(quantiles) > 2:
                bins = np.full(len(values), -1, dtype=np.int32)
                bins[finite] = np.searchsorted(
                    quantiles[1:-1], values[finite], side="right"
                )
                candidate = _candidate(
                    f"smooth:{column}:decile_bins",
                    (bins,),
                    "smooth_bin_control",
                )
                if unique(candidate):
                    yield candidate

    for left_name, right_name in itertools.combinations(NUMERIC_FIELDS, 2):
        left, right = numeric[left_name], numeric[right_name]
        operations = (
            ("sum", left + right),
            ("difference", left - right),
            ("ratio", _safe_ratio(left, right)),
        )
        for operation, values in operations:
            for digits in (3, 4, 5):
                candidate = _candidate(
                    f"pair:{left_name}:{operation}:{right_name}:sig{digits}",
                    (significant_integer_codes(values, digits),),
                    f"numeric_pair_{operation}",
                )
                if unique(candidate):
                    yield candidate

    low_names = tuple(LOW_CARDINALITY_FIELDS)
    for width in (2, 3):
        for columns in itertools.combinations(low_names, width):
            candidate = _candidate(
                "cross:" + "*".join(columns),
                tuple(low_card[column] for column in columns),
                f"low_cardinality_{width}_way",
            )
            if unique(candidate):
                yield candidate

    # A small preregistered check for category-by-numeric-bin structure omitted
    # from the broad pair/triple scan, especially organization type.
    for numeric_name in (
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_LAST_PHONE_CHANGE",
    ):
        values = numeric[numeric_name]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        edges = np.unique(np.quantile(values[finite], np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        bins = np.full(len(values), -1, dtype=np.int32)
        bins[finite] = np.searchsorted(edges[1:-1], values[finite], side="right")
        for category in (
            "ORGANIZATION_TYPE",
            "OCCUPATION_TYPE",
            "NAME_INCOME_TYPE",
        ):
            candidate = _candidate(
                f"cross:{numeric_name}:deciles*{category}",
                (bins, low_card[category]),
                "numeric_bin_category_cross",
            )
            if unique(candidate):
                yield candidate


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def crossfit_smoothed_target_rate(
    codes: np.ndarray,
    y: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    smoothing: float,
) -> np.ndarray:
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be strictly positive")
    codes = np.asarray(codes, dtype=np.int64)
    y = np.asarray(y)
    if codes.ndim != 1 or y.ndim != 1 or len(codes) != len(y):
        raise ValueError("codes and y must be aligned one-dimensional arrays")
    if not len(codes) or np.any(codes < 0):
        raise ValueError("codes must be nonempty and nonnegative")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only binary labels")
    output = np.full(len(y), np.nan, dtype=np.float32)
    n_categories = int(codes.max()) + 1
    applied = np.zeros(len(y), dtype=bool)
    for fit_idx, apply_idx in folds:
        fit_idx = np.asarray(fit_idx, dtype=np.int64)
        apply_idx = np.asarray(apply_idx, dtype=np.int64)
        if (
            fit_idx.ndim != 1
            or apply_idx.ndim != 1
            or not len(fit_idx)
            or not len(apply_idx)
            or np.any(fit_idx < 0)
            or np.any(apply_idx < 0)
            or np.any(fit_idx >= len(y))
            or np.any(apply_idx >= len(y))
        ):
            raise ValueError("fold indices must be nonempty valid row indices")
        fit_mask = np.zeros(len(y), dtype=bool)
        fit_mask[fit_idx] = True
        if fit_mask[apply_idx].any():
            raise ValueError("fit and apply rows must be disjoint")
        if applied[apply_idx].any():
            raise ValueError("each row must be applied exactly once")
        prior = float(y[fit_idx].mean())
        fit_codes = codes[fit_idx]
        sums = np.bincount(
            fit_codes,
            weights=y[fit_idx].astype(np.float64),
            minlength=n_categories,
        )
        counts = np.bincount(fit_codes, minlength=n_categories)
        rates = (sums + smoothing * prior) / (counts + smoothing)
        output[apply_idx] = rates[codes[apply_idx]]
        applied[apply_idx] = True
    if not applied.all() or not np.isfinite(output).all():
        raise AssertionError("Candidate cross-fit encoding is incomplete")
    return output


def score_candidate(
    candidate: Candidate,
    y: np.ndarray,
    baseline_oof: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    smoothing: float = 30.0,
    blend_weights: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, object]:
    encoded = crossfit_smoothed_target_rate(
        candidate.category_codes, y, folds, smoothing
    )
    baseline_auc = float(roc_auc_score(y, baseline_oof))
    encoded_auc = float(roc_auc_score(y, encoded))
    result: dict[str, object] = {
        "candidate": candidate.name,
        "family": candidate.family,
        "unique_categories": candidate.unique_categories,
        "smoothing": smoothing,
        "baseline_oof_auc": baseline_auc,
        "target_rate_oof_auc": encoded_auc,
        "target_rate_delta_vs_baseline": encoded_auc - baseline_auc,
        "target_rate_oof_spearman_vs_baseline": float(
            pd.Series(encoded).corr(pd.Series(baseline_oof), method="spearman")
        ),
    }
    for weight in blend_weights:
        blended = (1.0 - weight) * baseline_oof + weight * encoded
        result[f"fixed_blend_{weight:.2f}_auc"] = float(
            roc_auc_score(y, blended)
        )
        result[f"fixed_blend_{weight:.2f}_delta"] = float(
            result[f"fixed_blend_{weight:.2f}_auc"] - baseline_auc
        )
    return result
