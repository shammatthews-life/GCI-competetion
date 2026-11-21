from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase30_native_feature_scratch as phase30


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving", None],
            "AMT_INCOME_TOTAL": [1000.0, 1200.0, 900.0, 1500.0],
            "AMT_CREDIT": [2000.0, 2400.0, 1800.0, 3000.0],
            "AMT_ANNUITY": [100.0, 150.0, 90.0, 200.0],
            "AMT_GOODS_PRICE": [1800.0, 2200.0, 1700.0, 2900.0],
            "CNT_FAM_MEMBERS": [1.0, 2.0, 1.0, 3.0],
            "DAYS_BIRTH": [-9000.0, -12000.0, -15000.0, -18000.0],
            "DAYS_EMPLOYED": [-1000.0, 365243.0, -5000.0, -9000.0],
            "DAYS_REGISTRATION": [-2000.0, -3000.0, -4000.0, -5000.0],
            "DAYS_ID_PUBLISH": [-3000.0, -5000.0, -6000.0, -7000.0],
            "DAYS_LAST_PHONE_CHANGE": [-500.0, -600.0, -700.0, -800.0],
            "EXT_SOURCE_1": [0.2, 0.4, np.nan, 0.6],
            "EXT_SOURCE_2": [0.3, np.nan, 0.7, 0.8],
            "EXT_SOURCE_3": [0.4, 0.5, 0.8, 0.9],
        }
    )


class Phase30ScratchTests(unittest.TestCase):
    def test_features_use_only_supplied_rows_and_match_schema(self) -> None:
        raw = fixture()
        train, train_cats = phase30.build_features(
            raw, ("EXT_SOURCE_1", "EXT_SOURCE_2")
        )
        test, test_cats = phase30.build_features(
            raw.iloc[:2].copy(), ("EXT_SOURCE_1", "EXT_SOURCE_2")
        )
        self.assertEqual(train_cats, test_cats)
        self.assertEqual(train.columns.tolist(), test.columns.tolist())
        self.assertNotIn("TARGET", train)
        self.assertNotIn("SK_ID_CURR", train)
        self.assertIn("P30_KEY_ANNUITY_CREDIT_4", train_cats)
        self.assertAlmostEqual(train.loc[0, "P30_CREDIT_TO_INCOME"], 2.0)
        self.assertEqual(train.loc[0, "P30_KEY_ANNUITY_CREDIT_3"], "50")

    def test_ratio_keys_do_not_use_empirical_constants(self) -> None:
        ratio = phase30._ratio_key(
            np.array([100.0, 150.0]),
            np.array([2000.0, 2500.0]),
            decimals=3,
        )
        np.testing.assert_array_equal(ratio, np.array(["50", "60"]))


if __name__ == "__main__":
    unittest.main()
