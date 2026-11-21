# Next phase recommendation

## First experiment

Run **EXP02-P6: a preregistered, fold-safe incremental product-structure ablation**. Compare the clean raw-only baseline with raw plus the nonredundant Phase 3 product block using the same fixed five folds. Include one random-key/permuted-feature negative control. Freeze exact features, fixed model parameters, fold-local parameter fitting, and decision rules before reading OOF results. Do not copy test-pool, target-encoded, or test-coupled code. This uses already-known labels and is a discovery replication, not independent confirmation.

## Sequence

1) Freeze/audit this raw baseline and the feature manifest. 2) Run one predeclared product-feature incremental ablation against raw-plus-existing-product on common folds, with negative controls and no retuning. 3) Test a small external-score disagreement block against that same baseline. 4) Only then evaluate targeted missingness/nonlinear representations. 5) Model diversity and blends follow only after candidate OOF arrays pass integrity checks. 6) Reserve confirmation for genuinely unseen labeled data.

## Gates before doing model work

- Confirm current organizer rules and maintain conservative exclusions until written clarification.
- Obtain genuinely unseen labeled confirmation data before any champion promotion.
- Do not try to recover or reproduce V16/EXP85 as an eligible pipeline.
- Do not optimize blend weights on the same OOF labels used to quote final performance.
- No AUC target, including 0.78 or 0.79, is promised.
