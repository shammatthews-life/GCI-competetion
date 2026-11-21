# Next phase recommendation

## Current outcome

The Phase 7 primary candidate `raw_plus_all_product` has development delta `+0.002441` and the one-time holdout delta `+0.000919`. However, the holdout is not independent because its labels were included in prior project cross-validation outputs. Therefore no promotion criteria are fully satisfied.

## Single best next step

Obtain a genuinely new, organizer-approved labeled confirmation cohort (or a truly untouched course-provided validation set, if one exists and was not previously inspected). Freeze the Phase 7 primary code, exact formula block, data contract, and model configuration before exposing those labels. If no such cohort is available, retain the Phase 7 result as exploratory and do not claim independent improvement.

Do not continue broad feature search or tune on the locked holdout. If new labels cannot be obtained, the next development-only study should be a narrow predeclared external-score disagreement ablation against the raw-plus-product development baseline, with the same folds and no holdout use; this still cannot authorize promotion.

## Compliance blockers

Pooled train/test transforms, target encoding, and test-to-test coupling remain excluded pending organizer clarification. Product decoder fee/step features are permitted here only as fit-fold numerical hypotheses; they are not validated contract semantics.
