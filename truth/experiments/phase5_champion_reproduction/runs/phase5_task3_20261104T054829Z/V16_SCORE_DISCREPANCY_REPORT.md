# V16 score discrepancy report

The saved V16 notebook outputs claim fixed-formula AUC `0.77649` and component blend AUCs old `0.77412`, E1 `0.77512`, E1ET `0.77608`, E1AET `0.77618`. They remain saved-output claims, not independently reproduced results.

The recovered candidate model caches yield old `0.77202437`, E1 `0.77257636`, E1ET `0.77356771`, and E1AET `0.77368572`; the reconstructed formula yields `0.77404329`. It differs from the saved final claim by `-0.00244671`.

## Explanation

The candidate predictions are verified as byte-identical members of the downloaded V15 cache archive. They are not verified V16 outputs. The V16 notebook configures a distinct `gci_cache_v16`; that directory was not found. The candidate final result is close to the known V15 reference because it reuses V15 sets. The lower score is explained primarily by **cache-generation mismatch**, not a demonstrated V16 regression.

Remaining uncertainty: we do not have the actual `gci_cache_v16` folder or a manifest that joins its arrays to the exact input hashes, row order, fold map, seeds, packages, and feature schema. Array length and AUC are not proof of alignment or OOF validity. `V16_SCORE_DISCREPANCY_REPORT.csv` contains the component-by-component values.
