# Phase 12 execution report

## Executive status

- Run folder: `D:\gci\truth\experiments\phase12_label_free_product_representation\runs\phase12_20261104T173534Z`
- Execution window (UTC): 2026-11-04T17:35:34.990669+00:00 to 2026-11-04T18:25:59.709616+00:00
- Input: `D:\gci\inpuy\train.csv`; SHA-256 `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`; 171,202 rows and 34 columns.
- Primary seed: `20261105`; one fixed stratified five-fold assignment shared by all primary variants.
- Raw numeric OOF AUC: **0.743257**.
- Raw numeric plus eight-key one-hot OOF AUC: **0.758729** (paired change +0.015473).
- Eight-key one-hot positive folds: **5/5**.
- All key formulas are label-free, but the historical 130.5 offset and 0.0066 step have unresolved provenance; eligibility is therefore not conclusively established.
- CPU LightGBM was used. No test data, target encoding, target-derived category statistics, external data, or submission was used.

## Representation results

| Representation | Pooled OOF AUC | Delta vs raw numeric | Positive folds |
|---|---:|---:|---:|
| raw_numeric | 0.743257 | +0.000000 | 0/5 |
| raw_plus_keys_ohe | 0.758729 | +0.015473 | 5/5 |
| raw_plus_keys_frequency_ohe | 0.759816 | +0.016559 | 5/5 |
| raw_plus_decompositions | 0.753826 | +0.010569 | 5/5 |
| raw_plus_interactions | 0.742791 | -0.000466 | 1/5 |
| raw_plus_keys_decompositions_interactions | 0.758422 | +0.015165 | 5/5 |
| raw_plus_keys_hash | 0.759298 | +0.016042 | 5/5 |

## Statistical reporting

Paired fold differences and the conditional stratified bootstrap intervals are in `FOLD_LEVEL_METRICS.csv` and `PAIRED_BOOTSTRAP_REPORT.md`. The bootstrap resamples the OOF rows conditional on these fitted models and is not independent confirmation.

## Ablation and negative-control interpretation

- The strongest single key by pooled OOF AUC was `single_Q3` (0.755227); the full fold-level `PRODUCT_KEY_INDIVIDUAL_ABLATION.csv` remains exploratory.
- Individual and leave-one-out ablations reuse the exposed outer folds and share the all-key-selected configuration/rounds. They are diagnostic, not independent key selection evidence.
- Mean negative-control pooled AUCs by control type: `[{"control": "frequency_preserving_shuffle", "pooled_oof_auc": 0.7411837768919256}, {"control": "randomized_assignment", "pooled_oof_auc": 0.7413842745281484}]`. Every fixed-seed repetition and fold is retained in `NEGATIVE_CONTROL_RESULTS.csv`.

## Secondary-seed replication

- `raw_numeric` seed 20261106: pooled OOF AUC 0.743681; paired change vs that seed's raw numeric baseline +0.000000; 0/5 positive folds.
- `raw_plus_keys_ohe` seed 20261106: pooled OOF AUC 0.759590; paired change vs that seed's raw numeric baseline +0.015909; 5/5 positive folds.

This is robustness replication on a new seed/fold assignment, not independent confirmation.

## Competition eligibility gate

No model in this run uses target-derived feature statistics or test-fitted transforms. The 130.5 fee offset and 0.0066 financing step are preserved solely to reproduce the Phase 11 definitions; the available audit does not establish their permitted-data derivation. Accordingly, P4/P5/KQ3/KQ4 and decompositions containing these constants remain eligibility-pending. Formula-level label freedom does not prove organizer approval.

## Integrity, runtime, and limits

OOF row identity, target alignment, fold coverage, finite prediction values, row counts, source hashes, and artifact hashes are checked in `OOF_INTEGRITY_REPORT.md` and `SHA256_MANIFEST.csv`. Per-fit CPU runtime, sparse feature count, nonzero count, and process peak working set are recorded in the comparison outputs.

The primary comparison is a research result on previously examined competition training labels. The OOF metrics and bootstrap do not provide an untouched holdout, external confirmation, official score, competition approval, or a champion promotion.

## Required final decisions

1. **Can label-free representations preserve most of the historical eight-key gain?** This run shows strong label-free key signal against its own numeric-only baseline, but does **not** answer the historical preservation question: the Phase 11 LightGBM baseline included raw categorical one-hot and rowwise-derived features, which are absent from this numeric-only baseline. The absolute AUC and lift are therefore not comparable. Add a matched raw-categorical/rowwise baseline before making that claim.
2. **Which individual keys are strongest?** Q3 ranked first by pooled OOF AUC at 0.755227, narrowly ahead of P4 at 0.755052. All single keys were below the all-key representation. Leaving out KQ3 produced only a +0.000206 pooled difference and 3/5 fold wins; do not remove keys from these exposed-fold diagnostics.
3. **Is the improvement stable across folds and seeds?** Eight-key OHE and fit-fold frequency-supported OHE improved all five primary folds. The second-seed key OHE comparison improved all five folds (+0.015909 against its own numeric baseline). This is robustness replication on the same labels, not independent confirmation.
4. **Do genuine keys outperform matched controls?** Yes, in this within-run comparison: genuine key OHE scored 0.758729, versus mean pooled AUC 0.741184 for frequency-preserving shuffles and 0.741384 for randomized assignments. Controls are fixed-seed diagnostics, not independent validation.
5. **Which model is demonstrably eligible?** None is certified as competition-eligible. No target-dependent statistics or test fitting were used, but the provenance of the historical 130.5 fee offset and 0.0066 financing step remains unresolved, and organizer approval is not documented.
6. **Single best next experiment:** resolve those constants' provenance or ask the organizer, then compare raw numeric plus raw categorical/rowwise features against the same fold-fitted one-hot keys on identical preregistered folds and a secondary seed.
