from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import run_phase13

from phase13_features import (
    PRODUCT_KEYS,
    build_label_free_decompositions,
    build_product_keys,
    build_raw_feature_blocks,
)


class Phase13FeatureTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", None],
                "CNT_CHILDREN": [0, 1, 0],
                "DAYS_EMPLOYED": [10, 365243, -20],
                "DAYS_BIRTH": [-10000, -15000, -30000],
                "AMT_CREDIT": [1000.0, 1000.0, 0.0],
                "AMT_ANNUITY": [100.0, 100.0, 50.0],
                "AMT_GOODS_PRICE": [900.0, 900.0, 0.0],
                "AMT_INCOME_TOTAL": [500.0, 1000.0, 250.0],
                "REGION_POPULATION_RELATIVE": [0.1, 0.2, 0.3],
                "EXT_SOURCE_1": [0.1, np.nan, 0.4],
                "EXT_SOURCE_2": [0.2, 0.3, np.nan],
                "EXT_SOURCE_3": [np.nan, 0.7, 0.8],
            }
        )

    def test_raw_blocks_preserve_categories_and_create_documented_rowwise(self) -> None:
        numeric, categorical, rowwise = build_raw_feature_blocks(self.frame())
        self.assertIn("CNT_CHILDREN", numeric)
        self.assertEqual(categorical.loc[2, "NAME_CONTRACT_TYPE"], "MISSING")
        self.assertEqual(rowwise.loc[1, "DAYS_EMPLOYED_IS_365243"], 1)
        self.assertTrue(np.isnan(rowwise.loc[1, "DAYS_EMPLOYED_CLEAN"]))
        self.assertEqual(rowwise.loc[1, "EXT_SOURCE_MISSING_COUNT"], 1)
        self.assertAlmostEqual(rowwise.loc[0, "ANNUITY_TO_CREDIT"], 0.1)

    def test_eight_formula_keys_retain_precision_and_safe_invalids(self) -> None:
        keys = build_product_keys(self.frame())
        self.assertEqual(tuple(keys.columns), PRODUCT_KEYS)
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q3"], "100")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q4"], "1000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_KQ3"], "17|100")
        self.assertEqual(keys.loc[1, "PRODUCT_KEY_KQ4"], "NONCASH_OR_INVALID")
        self.assertEqual(keys.loc[2, "PRODUCT_KEY_Q4"], "MISSING")

    def test_eligible_decompositions_use_no_unverified_constants(self) -> None:
        values = build_label_free_decompositions(self.frame())
        self.assertAlmostEqual(values.loc[0, "KEY_COMPONENT_ANNUITY_TO_GOODS"], 1 / 9)
        self.assertEqual(values.loc[0, "KEY_COMPONENT_CREDIT_MINUS_GOODS"], 100)
        self.assertEqual(values.loc[0, "KEY_COMPONENT_ANNUITY_MINUS_CREDIT"], -900)

    def test_target_is_rejected_by_all_feature_builders(self) -> None:
        frame = self.frame().assign(TARGET=[0, 1, 0])
        for builder in (
            build_raw_feature_blocks,
            build_product_keys,
            build_label_free_decompositions,
        ):
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(ValueError, "TARGET"):
                    builder(frame)

    def test_runner_matrices_align_noncontiguous_rows(self) -> None:
        numeric, categorical, rowwise = build_raw_feature_blocks(self.frame())
        keys = build_product_keys(self.frame())
        decompositions = build_label_free_decompositions(self.frame())
        blocks = (numeric, categorical, rowwise, keys, decompositions)

        for variant in (
            "C_complete_raw_rowwise",
            "D_C_plus_keys_ohe",
            "E_C_plus_keys_frequency_ohe",
            "F_C_plus_supported_numeric_decompositions",
        ):
            with self.subTest(variant=variant):
                fit, apply, details = run_phase13.make_matrices(
                    np.array([0, 2]),
                    np.array([1]),
                    *blocks,
                    variant,
                )
                self.assertEqual(fit.shape[0], 2)
                self.assertEqual(apply.shape[0], 1)
                self.assertEqual(fit.shape[1], apply.shape[1])
                self.assertEqual(details["feature_count"], fit.shape[1])


if __name__ == "__main__":
    unittest.main()
