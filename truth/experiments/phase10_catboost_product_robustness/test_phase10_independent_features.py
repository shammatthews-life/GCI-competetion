from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from phase10_independent_feature_builder import (
    build_product_keys,
    build_rowwise_features,
    categorical_columns,
)


TRAIN_PATH = Path(r"D:\gci\inpuy\train.csv")
DICTIONARY_PATH = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition"
    r"\HomeCredit_columns_description.xlsx"
)


def small_predictors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
            "DAYS_BIRTH": [-3652.5, -7305.0, -365.25],
            "DAYS_EMPLOYED": [365243.0, -365.25, -10.0],
            "AMT_CREDIT": [200.0, 120.0, 0.0],
            "AMT_ANNUITY": [20.0, 12.0, 10.0],
            "AMT_GOODS_PRICE": [180.0, 120.0, 100.0],
            "AMT_INCOME_TOTAL": [100.0, 0.0, 50.0],
            "EXT_SOURCE_1": [np.nan, 0.1, 0.2],
            "EXT_SOURCE_2": [0.2, 0.3, np.nan],
            "EXT_SOURCE_3": [0.5, 0.4, np.nan],
            "CAT": ["x", None, "z"],
        }
    )


class IndependentFeatureBuilderTests(unittest.TestCase):
    def test_feature_formulas_and_target_exclusion(self) -> None:
        raw = small_predictors()
        features = build_rowwise_features(raw)
        self.assertNotIn("SK_ID_CURR", features.columns)
        self.assertNotIn("TARGET", features.columns)
        self.assertEqual(features.loc[0, "DAYS_EMPLOYED_IS_365243"], 1)
        self.assertTrue(np.isnan(features.loc[0, "DAYS_EMPLOYED_CLEAN"]))
        self.assertEqual(features.loc[1, "DAYS_EMPLOYED_IS_365243"], 0)
        self.assertEqual(features.loc[0, "AGE_YEARS_FROM_DAYS_BIRTH"], 10.0)
        self.assertEqual(features.loc[0, "CREDIT_TO_INCOME"], 2.0)
        self.assertEqual(features.loc[0, "ANNUITY_TO_INCOME"], 0.2)
        self.assertEqual(features.loc[0, "GOODS_TO_CREDIT"], 0.9)
        self.assertEqual(features.loc[0, "CREDIT_TO_ANNUITY"], 10.0)
        self.assertEqual(features.loc[0, "EXT_SOURCE_MISSING_COUNT"], 1)
        self.assertEqual(features.loc[1, "EXT_SOURCE_MISSING_COUNT"], 0)
        self.assertEqual(features.loc[2, "EXT_SOURCE_MISSING_COUNT"], 2)
        self.assertEqual(features.loc[1, "CAT"], "MISSING")
        self.assertTrue(np.isnan(features.loc[1, "CREDIT_TO_INCOME"]))
        self.assertIn("CAT", categorical_columns(features))

    def test_product_key_formulas_and_invalid_sentinels(self) -> None:
        keys = build_product_keys(small_predictors())
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q3"], "100")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q4"], "1000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q5"], "10000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_KQ3"], "17|100")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_KQ4"], "17|1000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_P4"], "-5525")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_P5"], "-55250")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_AG3"], "111")
        self.assertEqual(keys.loc[1, "PRODUCT_KEY_KQ3"], "NONCASH_OR_INVALID")
        self.assertEqual(keys.loc[2, "PRODUCT_KEY_Q3"], "MISSING")
        self.assertEqual(keys.loc[2, "PRODUCT_KEY_KQ4"], "NONCASH_OR_INVALID")

    def test_target_column_is_rejected(self) -> None:
        raw = small_predictors().assign(TARGET=[0, 1, 0])
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_rowwise_features(raw)
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_product_keys(raw)

    def test_training_data_matches_dictionary_and_builds_finitely(self) -> None:
        self.assertTrue(TRAIN_PATH.is_file())
        self.assertTrue(DICTIONARY_PATH.is_file())
        train = pd.read_csv(TRAIN_PATH, low_memory=False)
        dictionary = pd.read_excel(DICTIONARY_PATH)
        dictionary_fields = set(dictionary["Row"].astype(str))
        self.assertEqual(set(train.columns), dictionary_fields)
        raw = train.drop(columns=["TARGET"])
        features = build_rowwise_features(raw)
        keys = build_product_keys(raw)
        self.assertEqual(len(features), len(train))
        self.assertEqual(len(keys), len(train))
        self.assertNotIn("TARGET", features.columns)
        self.assertNotIn("SK_ID_CURR", features.columns)
        self.assertTrue(features.columns.is_unique)
        self.assertTrue(keys.columns.is_unique)
        self.assertEqual(keys.shape[1], 8)
        self.assertTrue(
            np.isfinite(
                features[
                    [
                        "DAYS_EMPLOYED_IS_365243",
                        "RAW_MISSING_COUNT",
                        "EXT_SOURCE_MISSING_COUNT",
                        "AGE_YEARS_FROM_DAYS_BIRTH",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        )
        self.assertTrue(
            all(
                pd.api.types.is_numeric_dtype(features[column].dtype)
                for column in (
                    "CREDIT_TO_INCOME",
                    "ANNUITY_TO_INCOME",
                    "CREDIT_TO_ANNUITY",
                    "ANNUITY_TO_CREDIT",
                    "GOODS_TO_CREDIT",
                    "CREDIT_TO_GOODS",
                )
            )
        )
        self.assertTrue(
            all(
                features[column].astype(str).str.len().gt(0).all()
                for column in categorical_columns(features)
            )
        )


if __name__ == "__main__":
    unittest.main()
