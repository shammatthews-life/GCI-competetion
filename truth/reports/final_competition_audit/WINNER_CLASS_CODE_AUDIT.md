# Previous winner class material audit

## Scope

Reviewed `D:\gci\rules and info\past year winner class\tutorial_model_ensembling.ipynb`
and its adjacent downloaded HTML/image teaching materials. The notebook is a
general educational class resource about model evaluation and ensemble
families. It is not the previous winner's full competition implementation,
does not serve as the official competition rulebook, and does not establish
that a technique is allowed by current rules.

## Technical summary

The material explains standard supervised models, validation, and ways to
combine model predictions. The repeatable concepts potentially useful for a
future TRUTH implementation are:

- Compare models using validation data rather than training scores.
- Keep preprocessing and model fitting tied to the training partition.
- Treat ensembling as a measured experiment; retain a simple baseline.
- Record the method and source when adapting code.

These are educational lessons, not a verified recipe for the GCI submission.
The lesson does not provide provenance for the friend's prediction caches,
feature matrices, or particular blend coefficients, and its reported
illustrations (if any) are not independent validation of those artifacts.

## Decision

**Inspected; no method is automatically endorsed.** Reuse only generic
concepts that are independently implemented with the provided GCI data,
proper outer validation, and rule-compliant provenance. Consult
[`OFFICIAL_RULES_AUDIT.md`](./OFFICIAL_RULES_AUDIT.md) for rule decisions.
