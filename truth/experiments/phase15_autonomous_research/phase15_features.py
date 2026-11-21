from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET"
SECONDS_PER_YEAR = 365.25
TIME_COLUMNS = {
    "DAYS_BIRTH",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "DAYS_EMPLOYED",
}
EMPLOYMENT_SENTINEL = 365243.0


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def build_time_lifecycle_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive elapsed-age features from supplied application date offsets."""
    if TARGET_COLUMN in raw.columns:
        raise ValueError("TARGET must be separated before feature construction")
    missing = TIME_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Required source columns are missing: {sorted(missing)}")

    age_days = np.abs(_numeric(raw, "DAYS_BIRTH"))
    registration_days = np.abs(_numeric(raw, "DAYS_REGISTRATION"))
    id_publish_days = np.abs(_numeric(raw, "DAYS_ID_PUBLISH"))
    phone_change_days = np.abs(_numeric(raw, "DAYS_LAST_PHONE_CHANGE"))
    employed_days = np.abs(_numeric(raw, "DAYS_EMPLOYED"))
    employed_days[employed_days == EMPLOYMENT_SENTINEL] = np.nan

    def years_at_event(elapsed_days: np.ndarray) -> np.ndarray:
        result = np.full(len(age_days), np.nan, dtype=np.float64)
        valid = np.isfinite(age_days) & np.isfinite(elapsed_days)
        result[valid] = (age_days[valid] - elapsed_days[valid]) / SECONDS_PER_YEAR
        return result

    def fraction_of_age(elapsed_days: np.ndarray) -> np.ndarray:
        result = np.full(len(age_days), np.nan, dtype=np.float64)
        valid = (
            np.isfinite(age_days)
            & np.isfinite(elapsed_days)
            & (age_days != 0)
        )
        result[valid] = elapsed_days[valid] / age_days[valid]
        return result

    return pd.DataFrame(
        {
            "AGE_YEARS_AT_REGISTRATION": years_at_event(registration_days),
            "AGE_YEARS_AT_ID_PUBLISH": years_at_event(id_publish_days),
            "AGE_YEARS_AT_LAST_PHONE_CHANGE": years_at_event(phone_change_days),
            "AGE_YEARS_AT_EMPLOYMENT_START": years_at_event(employed_days),
            "REGISTRATION_TO_AGE_FRACTION": fraction_of_age(registration_days),
            "ID_PUBLISH_TO_AGE_FRACTION": fraction_of_age(id_publish_days),
            "PHONE_CHANGE_TO_AGE_FRACTION": fraction_of_age(phone_change_days),
        },
        index=raw.index,
    ).replace([np.inf, -np.inf], np.nan)
