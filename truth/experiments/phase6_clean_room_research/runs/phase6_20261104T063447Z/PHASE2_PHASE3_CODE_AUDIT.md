# Phase 2/3 code audit

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
