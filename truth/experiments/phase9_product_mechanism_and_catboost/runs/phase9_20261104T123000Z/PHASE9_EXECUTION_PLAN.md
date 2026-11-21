# Phase 9 pre-registered experiment plan

## Workstreams

**A. Product-mechanism discrimination.** Compare annuity/credit identity keys at 3/4/5 decimals against exact loan amount keys (C,A), (G,C), (G,C,A), financing multiplier K, cash-only ratio identity, and Q4+K. All target means are nested inner-OOF within each outer-fit fold; smoothing alpha=30.0. Three five-fold seeds: (42, 7, 2026). The baseline is the exact raw+Phase-3 product feature block.

**B. Model-family/encoding.** On the same seeds/folds, compare CatBoost CPU raw+product baseline against native categorical Q4, eight product keys, and eight keys plus the predeclared organization×occupation and income-type×education crosses. CatBoost iterations are selected on an inner split of outer-fit rows on the baseline, shared across CatBoost candidates; outer validation never selects iterations.

## Hypotheses

- H1 product identity: ratio-key encodings should outperform exact amount-pair/triple and K-only encodings. Predictive gain does not establish risk-based pricing or causation.
- H2 smooth term vs discrete identity: compare Q3/Q4/Q5 key resolution. Finer-key gain is consistent with higher-resolution identity, but not proof that keys are contractual products.
- H3 histogram resolution: compare high-resolution unsupervised ordinal Q4 against supervised target means and CatBoost native categorical keys. A target-mean advantage indicates useful supervised group-risk estimates beyond simple ordered precision; it cannot rule out every histogram/representation explanation.
- Segment claims: use saved cross-fit Phase 8 OOF predictions for cash/revolving, age quartiles, and pensioner subgroup diagnostics. These are exploratory subgroup summaries with no causal interpretation.
- Annuity fee claim: verify repeated exact (goods,credit) cells' adjacent annuity gaps and compare the 130.5 gap to nearby placebo widths; do not infer multiplicity or a true fee.

## Restrictions and reporting

Only the supplied training CSV is opened. No test data, external data, or test-to-test coupling. Target encoding is nested correctly but remains unresolved by the local competition rules; results are research-only pending written organizer guidance. No holdout is scored, no submission is generated, and no champion is replaced. All seeds reuse historically examined labels and are sensitivity checks, not independent confirmation.
