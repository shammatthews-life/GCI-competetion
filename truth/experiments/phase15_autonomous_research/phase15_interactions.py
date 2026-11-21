from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET"
ORGANIZATION_COLUMN = "ORGANIZATION_TYPE"
EXTERNAL_COLUMNS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")


def build_organization_external_bin_cross(
    raw: pd.DataFrame,
    fit_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> tuple[pd.Series, np.ndarray]:
    """Cross organization with fit-fold quartiles of rowwise external score mean."""
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    required = {ORGANIZATION_COLUMN, *EXTERNAL_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    apply_indices = np.asarray(apply_indices, dtype=np.int64)
    if (
        len(np.unique(fit_indices)) != len(fit_indices)
        or len(np.unique(apply_indices)) != len(apply_indices)
        or np.intersect1d(fit_indices, apply_indices).size
        or np.any(fit_indices < 0)
        or np.any(apply_indices < 0)
        or np.any(fit_indices >= len(raw))
        or np.any(apply_indices >= len(raw))
    ):
        raise ValueError("Fit/apply indices must be valid, unique, and disjoint")

    scores = raw.loc[:, EXTERNAL_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=np.float64)
    observed = np.isfinite(scores)
    counts = observed.sum(axis=1)
    score_mean = np.divide(
        np.where(observed, scores, 0.0).sum(axis=1),
        counts,
        out=np.full(len(raw), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    fit_scores = score_mean[fit_indices]
    fit_scores = fit_scores[np.isfinite(fit_scores)]
    if not len(fit_scores):
        raise ValueError("No finite external-score means exist in fit rows")
    cuts = np.unique(np.quantile(fit_scores, [0.25, 0.5, 0.75]))

    organization = (
        raw[ORGANIZATION_COLUMN]
        .astype("string")
        .fillna("MISSING")
        .to_numpy(dtype=str)
    )
    cross = np.full(len(raw), "NOT_IN_FIT_OR_APPLY", dtype=object)
    for indices in (fit_indices, apply_indices):
        values = score_mean[indices]
        bins = np.full(len(indices), "MISSING", dtype=object)
        finite = np.isfinite(values)
        bins[finite] = np.digitize(values[finite], cuts, right=True).astype(str)
        cross[indices] = np.char.add(
            np.char.add(organization[indices], "|EXT_Q"),
            bins.astype(str),
        )
    return pd.Series(cross, name="ORGANIZATION_X_EXT_SOURCE_QUARTILE"), cuts
