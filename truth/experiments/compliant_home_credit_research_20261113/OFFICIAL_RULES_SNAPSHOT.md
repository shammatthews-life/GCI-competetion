# Official GCI World 2026 rules snapshot

## Source and retrieval

- Repository: <https://github.com/Hareen-Liyanage/GCI_World_Competition>
- Revision checked: `8de63b276524ca7b067e35a6e8b14f5be64a11d8`
- Source file: `README.ipynb`, cells 2-4, fetched from the repository's
  `main` branch on 2026-11-13.
- This record summarizes the exact published text checked; the organizer and
  Omnicampus platform remain the final authority for eligibility.

## Published requirements

1. **Permitted data:** The README says no external data may be used at any
   stage and instructs participants to rely solely on the provided
   `input/train.csv` and `input/test.csv`. The sample submission supplies
   output schema/IDs.
2. **Hand-labeling:** The README prohibits manual predictions for all or part
   of test data, including manual decisions for part of predictions based on
   EDA-derived rules/conditions. Predictions must be automatically generated
   by a reproducible model applicable to similar data.
3. **Reproducibility:** Reproduce predictions as much as possible and set
   random seeds. The README expressly notes exact reproducibility may not be
   guaranteed and does not require downgrading framework versions.
4. **Collaboration:** The ordinary private-sharing prohibition is waived for
   this tutorial competition; this does not change data/prediction rules.
5. **Attribution:** Cited reference code is allowed; code reuse should name
   its original source. Top-ranking participants are required to disclose
   code.
6. **Metric/schema:** The metric is ROC-AUC; submission follows the sample
   schema `SK_ID_CURR,TARGET`.
7. **Leaderboard:** Public scoring is a test subset; final private scoring is
   the full test set. Only the last submission determines the final ranking.
8. **Reproduction:** Honors/Outstanding consideration requires code capable
   of reproducing the submitted CSV.

## Interpretation used for this work

The README does not explicitly discuss target encoding, fitting unsupervised
preprocessing on train plus test covariates, internal CV requirements, or
model caching. These points are not represented as explicit prohibitions in
this snapshot. To use the most defensible interpretation:

- No external sources, cached predictions, or applicant lookup tables.
- No matching applicants to transfer/propagate any training target labels.
- No per-row score edits or hand-coded test decisions.
- No train/test pooling in any fit, vocabulary, statistics, residual model,
  scaler, or representation.
- Any supervised feature such as target encoding is allowed in a candidate
  only if generated strictly within the active training folds; otherwise it
  is rejected as validation leakage regardless of formal eligibility.
- Any technique whose rule status remains uncertain is omitted unless written
  organizer approval is available.

The no-pooling and nested-CV rules above are conservative implementation
choices for methodological defensibility, not additional verbatim rules from
the README.

## Source citation register

- Official rules/data description: GCI World 2026 competition
  `README.ipynb`, cells 2-4, revision above.
- Model implementations to be cited in final delivery if retained:
  LightGBM (Ke et al., 2017), XGBoost (Chen & Guestrin, 2016), CatBoost
  (Prokhorenkova et al., 2018), scikit-learn (Pedregosa et al., 2011).
- No copied participant solution or external applicant dataset is used by the
  new baseline.
