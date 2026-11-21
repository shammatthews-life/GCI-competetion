# Independent feature-builder tests

The independent row-wise and product-key builders passed all formula-level
tests from `test_phase10_independent_features.py`.

Tests cover:

- Hand-calculated ratios, age conversion, missingness counters, employment
  sentinel handling, category missing-value treatment, product-key rounding,
  KQ eligibility, and invalid sentinels.
- Rejection of TARGET-bearing inputs.
- Full supplied training-column equality with the official dictionary.
- Full-row feature construction, unique columns, finite count/indicator fields,
  numeric ratio types, and nonempty categorical strings.

Test command: `python -m unittest discover -s <Phase 10 directory> -p test_phase10_independent_features.py -v`.

Captured test output:

```text
test_feature_formulas_and_target_exclusion (test_phase10_independent_features.IndependentFeatureBuilderTests.test_feature_formulas_and_target_exclusion) ... ok
test_product_key_formulas_and_invalid_sentinels (test_phase10_independent_features.IndependentFeatureBuilderTests.test_product_key_formulas_and_invalid_sentinels) ... ok
test_target_column_is_rejected (test_phase10_independent_features.IndependentFeatureBuilderTests.test_target_column_is_rejected) ... ok
test_training_data_matches_dictionary_and_builds_finitely (test_phase10_independent_features.IndependentFeatureBuilderTests.test_training_data_matches_dictionary_and_builds_finitely) ... ok

----------------------------------------------------------------------
Ran 4 tests in 2.293s

OK
```
