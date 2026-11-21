# Phase 19 eligibility audit

## Track A — label-free supplied-data candidates

| Feature/model family | Label-derived? | Test fitting/coupling? | Status |
|---|---|---|---|
| Original supplied predictors with LightGBM / CatBoost / MLP | No feature target statistics; CatBoost native CTR is an exception and remains unresolved | No test data used in these OOF runs | Raw-only feature content is the conservative benchmark; no organizer certification is asserted |
| Deterministic raw/rowwise features (age, ratios, missing counts, employment-sentinel treatment) | No | Computed per row from supplied columns | Candidate Track A, subject to the local "applicable to other data" rule and competition interpretation |
| Q3/Q4/Q5/AG3 integerized ratios | No; no 130.5 or 0.0066 constants | No test data used; deterministic rowwise construction | Track A candidate, but derived-feature permission is not independently certified |
| Phase 19 fold-fitted medians/scales/category vocabularies | No labels used in fitting the transformations | Fit on each outer training fold only | Fold-safe; unknown categories map to an explicit unknown index |

## Track B — exploratory / unresolved

| Method | Reason |
|---|---|
| Phase 9/11/13/15/16/18 all-eight keys | P4/P5 and KQ3/KQ4 include unresolved 130.5/0.0066 hypotheses; CatBoost native target statistics/CTR rule status is unresolved |
| Phase 17 target rates and counts | Explicitly label-derived; organizer eligibility unresolved |
| Phase 18 blend | Inherits Phase 15 and Phase 17 unresolved methods |
| Any post-hoc OOF blend or selected candidate | Repeated exposure/adaptive selection on the same labels; not independent validation |

## Governance

No test rows, sample-submission values, external data, submissions, or champion files were used or changed in Phase 19. EXP85 was not imported. The Phase 19 neural model uses only train labels for supervised model fitting; preprocessing vocabularies, imputations, and scaling are fit inside each outer training fold. OOF labels have nevertheless been examined throughout prior project phases, so this is not clean independent confirmation. Local competition materials and absent related history tables constrain further progress.
