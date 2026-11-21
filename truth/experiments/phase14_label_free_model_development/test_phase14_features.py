import unittest

import numpy as np
import pandas as pd

from phase14_features import build_numeric_enhancements


class NumericEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": [100.0, 0.0, 400.0],
                "AMT_CREDIT": [200.0, 50.0, 800.0],
                "AMT_ANNUITY": [20.0, 5.0, 80.0],
                "AMT_GOODS_PRICE": [180.0, 40.0, 700.0],
                "CNT_FAM_MEMBERS": [2.0, 1.0, np.nan],
                "DAYS_BIRTH": [-3652.5, -7305.0, -10957.5],
                "DAYS_EMPLOYED": [-365.25, 365243.0, -1826.25],
                "EXT_SOURCE_1": [0.2, np.nan, 0.8],
                "EXT_SOURCE_2": [0.4, 0.6, np.nan],
                "EXT_SOURCE_3": [0.6, np.nan, 0.4],
                "FLAG_DOCUMENT_2": [0, 1, 0],
                "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0, np.nan, 2.0],
                "AMT_REQ_CREDIT_BUREAU_DAY": [1.0, np.nan, 0.0],
            },
            index=[10, 30, 70],
        )

    def test_features_are_row_aligned_and_finite_or_missing(self):
        result = build_numeric_enhancements(self.raw)
        self.assertEqual(list(result.index), [10, 30, 70])
        self.assertEqual(len(result), len(self.raw))
        self.assertTrue(np.isfinite(result.to_numpy()[np.isfinite(result)]).all())
        self.assertTrue(np.isnan(result.loc[30, "LOG1P_EMPLOYED_DAYS_CLEAN"]))
        self.assertAlmostEqual(result.loc[10, "EXT_SOURCE_MEAN"], 0.4)
        self.assertAlmostEqual(result.loc[10, "CREDIT_PER_FAMILY_MEMBER"], 100.0)
        self.assertEqual(result.loc[30, "EXT_SOURCE_OBSERVED_COUNT"], 1.0)

    def test_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_numeric_enhancements(self.raw.assign(TARGET=[0, 1, 0]))

    def test_zero_and_missing_denominators_stay_missing(self):
        result = build_numeric_enhancements(self.raw)
        self.assertEqual(result.loc[30, "INCOME_PER_FAMILY_MEMBER"], 0.0)
        self.assertTrue(np.isnan(result.loc[70, "CREDIT_PER_FAMILY_MEMBER"]))


if __name__ == "__main__":
    unittest.main()
