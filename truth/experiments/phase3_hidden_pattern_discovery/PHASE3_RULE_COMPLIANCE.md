# Phase 3 rule compliance gate

Authority reviewed: supplied course README and official data dictionary, summarized in the pre-existing local official rules audit. No live organizer portal or later amendment was checked. Lack of a specific ban is not treated as approval.

| Method | Status | Phase 3 control |
|---|---|---|
| Row-wise transformations of supplied columns | ALLOWED | Use only official input columns; no external facts/data. |
| Statistics fit on each training fold | ALLOWED_WITH_RESTRICTIONS | Fit only on the current fit split; apply unchanged to validation rows. |
| Pooled train/test unsupervised statistics | UNRESOLVED | Disabled. |
| Frequency encoding | ALLOWED_WITH_RESTRICTIONS | Training-fold-only map; unseen keys map to zero; no test rows in fit. |
| Target encoding | UNRESOLVED | Disabled. |
| Repeat-applicant inference | UNRESOLVED | Disabled. |
| Derived applicant fingerprints | UNRESOLVED | Disabled. |
| Cross-validation target statistics | UNRESOLVED | Disabled with target encoding; no labels in formula construction. |
| Clustering | ALLOWED_WITH_RESTRICTIONS | Not used in Batch 1; if tried, fit clusters within each training fold only. |
| External information/data | PROHIBITED | Not used. |

Batch 1 uses only row-wise supplied columns and unsupervised fee/step/key-frequency estimates fitted on outer training partitions. No target encoding, pooled train/test stats, inferred person links, external data, or hand-created labels are used.
