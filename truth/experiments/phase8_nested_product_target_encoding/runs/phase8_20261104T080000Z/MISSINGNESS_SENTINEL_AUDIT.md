# Missingness and employment-sentinel audit

The executable baseline `run_independent_baseline.py` creates the employment sentinel indicator, replaces the 365243 sentinel with missing in clean employment-duration features, emits an `NA_` indicator for every numeric source column, computes `RAW_MISSING_COUNT`, and includes `EXT_COUNT` for the number of available external scores. Therefore Claude's proposed sentinel and external-score missing-count ideas are already represented in the matched baseline. The four bureau inquiry fields also have individual NA flags plus the global missing count. Adding duplicate versions would not test new information. No separate missingness model ablation was run.

This conclusion is a source audit, not a claim that the existing flags are optimal. Prior Phase 6 screening reported little or no incremental missingness lift; the Phase 8 target-encoding comparison keeps the existing raw baseline unchanged.

Source confirmation: DAYS_EMPLOYED_SENTINEL, DAYS_EMPLOYED_CLEAN, per-column NA_* flags, RAW_MISSING_COUNT, and EXT_COUNT (count of available EXT_SOURCE_1..3) are already present in the raw baseline feature builder. The candidate does not duplicate these features.
