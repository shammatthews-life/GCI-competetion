# Append-only correction addendum

This addendum does not alter or remove any original file in this run.

The original `FINANCIAL_RELATIONSHIP_AUDIT.csv` and `HIDDEN_PATTERN_REPORT.md` in this run used the incorrect diagnostic formula `(AMT_CREDIT / AMT_GOODS_PRICE) / step` for the historical product-lattice hypothesis. The Phase 3 decoder source uses `(AMT_CREDIT / AMT_GOODS_PRICE - 1) / step` on cash-loan rows. Therefore, the original run's lattice-support values are invalid and must not be cited.

Use the corrected, authoritative Phase 6 run `..\phase6_20261104T063150Z\`, which records the source equation and training-only support: for step `0.0066`, 96.2939% of 154,988 cash-loan rows have distance below 0.01 to an integer coordinate; for `0.0132`, 92.6388%. These are descriptive numerical regularities, not proof of actual loan terms or financing tiers.

The first run's independently repeated raw baseline and other train-only summaries are retained as historical artifacts, but the corrected run is the authoritative set of Phase 6 deliverables. The corrected run also adds the Phase 6 runner's source hash to its source inventory.
