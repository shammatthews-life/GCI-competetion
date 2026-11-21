# Phase 8 nested product target-encoding results

## Outcome

The study completed on `train.csv` only. It repeated five-fold stratified OOF modeling at seeds `(42, 7, 2026)` with the same LightGBM family and shared, inner-selected boosting rounds. It trained 90 outer-fold models (plus 15 inner stopping models). No test rows, submission, new holdout, external data, or model-family/hyperparameter sweep was used.

## Main claim test

- Mean seed-pooled AUC for raw + Phase 3 product + nested 8-key target means: `0.767622665` (seed SD `0.000290390`).
- Mean paired pooled delta vs raw + Phase 3 product: `+0.005422344`.
- Fold improvements vs raw + Phase 3 product: `15/15`; positive seed-pooled deltas: `3/3`.
- 10-key means: mean pooled delta `+0.005233080`, `15/15` positive folds.
- Eight target means plus log-count features: mean pooled delta `+0.005569627`, `15/15` positive folds.

See `SEED_METRICS.csv` for every seed and `FOLD_RESULTS.csv` for all 90 outer-fold fits. The primary metric is the mean of three seed-specific pooled OOF AUCs, not an aggregate over three duplicate predictions per applicant.

## Encoding integrity

Training-row encodings are inner-fold OOF within each outer fit fold. An inner row's target is excluded from its target map. Outer-validation encodings use target maps built only from the complete outer-fit labels. Smoothing alpha is fixed at `30.0`; unseen keys fall back to that fit partition's target prior. No key definition uses TARGET, and no fit/apply indices overlap. Details and key cardinality/singleton/unseen rates are in `TARGET_ENCODING_INTEGRITY.csv` and `PRODUCT_KEY_SUPPORT.csv`.

The product key catalog is a reconstruction of Claude's text, **not an exact source reproduction**: raw A/C ratios are rounded to 3/4/5 decimals; KQ keys combine the rounded A/C representation with nearest integer `(C/G - 1)/0.0066` on cash loans; P keys use the unverified fixed offset 130.5; AG3 is rounded A/G. Eight keys are `Q3,Q4,Q5,KQ3,KQ4,P4,P5,AG3`; the 10-key candidate adds `P3,KQ5`. These constants are hypotheses only, not verified loan terms, fees, or product codes.

## Missingness and sentinel proposals

The matched raw baseline already contains per-numeric-column missing indicators, a raw missing count, external-score availability count, and employment-sentinel handling. These proposals were audited as already present rather than redundantly re-added. See `MISSINGNESS_SENTINEL_AUDIT.md`.

## Rule and evidence status

The local official rule audit finds target encoding **unresolved in rule scope** (not expressly prohibited in the supplied text), while the former implementation failed validation nesting. The current encodings are nested correctly for research, but this run does not resolve eligibility. Do not submit a target-encoded model without organizer clarification.

All three seeds reuse the same labeled dataset and prior project research has exposed these labels. Repeated folds assess sensitivity to partition seeds but are not independent confirmations; the many candidate keys also create selection risk. No 0.77 claim or champion promotion follows from this run. Claude's HistGradientBoosting AUCs and these LightGBM AUCs are not direct comparisons because the learner, feature set, baseline, and exact key code differ.

## Final status

Interpret `AGGREGATE_COMPARISON.csv` as evidence about whether supervised product-key signal survives proper nested encoding on repeated CV seeds. Any observed gain remains exploratory and conditional on rule clarification. Preserve all candidates and the per-row OOF vectors; do not tune a final blend on these same predictions.
