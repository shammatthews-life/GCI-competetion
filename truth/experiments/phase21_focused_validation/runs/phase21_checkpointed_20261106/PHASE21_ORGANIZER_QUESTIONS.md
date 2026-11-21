# Organizer questions — eligibility clarification

These are prepared questions, not messages sent to organizers.

1. **Nested target statistics:** Are supervised encodings (category means,
   smoothed rates, counts conditioned on TARGET, and hierarchical/shrunk
   product-key encodings) allowed when every statistic is computed solely from
   supplied training labels and is nested/cross-fitted so each scored row's
   label is excluded?
2. **Data-derived constants:** Are deterministic constants inferred only from
   the supplied training columns (for example a repeated fee gap or financing
   increment) allowed in rowwise feature formulas when the complete estimator
   is reproducible and is applied without applicant-specific manual decisions?
   Are constants estimated inside each training fold treated differently from
   constants fixed once from all supplied training rows?
3. **Train/test covariate pooling:** May unsupervised transforms such as
   category-frequency maps or quantile bins be fit on the combined supplied
   train and test covariates, without using test labels, or must every transform
   be fit on training rows only?

Until written answers are received, the runner treats target-derived features
as unresolved Track B and avoids all test-covariate fitting.
