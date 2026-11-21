import unittest

import numpy as np
import pandas as pd

from run_phase52_gpu_archive_port import feature_groups, make_features


class Phase52FeatureTests(unittest.TestCase):
    def test_features_are_predictor_only_aligned_and_finite_or_missing(self):
        raw = pd.DataFrame(
            {
                "SK_ID_CURR": [1, 2, 3],
                "TARGET": [0, 1, 0],
                "NAME_CONTRACT_TYPE": ["Cash loans", "Cash loans", "Revolving loans"],
                "CODE_GENDER": ["F", "M", "F"],
                "DAYS_EMPLOYED": [365243, -365, -730],
                "DAYS_BIRTH": [-12000, -15000, -20000],
                "DAYS_REGISTRATION": [-1000, -2000, -3000],
                "DAYS_ID_PUBLISH": [-500, -1000, -1500],
                "DAYS_LAST_PHONE_CHANGE": [-20, -40, -60],
                "OWN_CAR_AGE": [np.nan, 2.0, 8.0],
                "AMT_INCOME_TOTAL": [30000.0, 60000.0, 45000.0],
                "AMT_CREDIT": [100000.0, 200000.0, 80000.0],
                "AMT_ANNUITY": [6000.0, 10000.0, 5000.0],
                "AMT_GOODS_PRICE": [90000.0, 180000.0, 75000.0],
                "CNT_FAM_MEMBERS": [1.0, 3.0, np.nan],
                "EXT_SOURCE_1": [0.2, np.nan, 0.8],
                "EXT_SOURCE_2": [0.4, 0.6, 0.9],
                "EXT_SOURCE_3": [0.3, 0.7, np.nan],
                "ORGANIZATION_TYPE": ["Business", "XNA", "School"],
                "OCCUPATION_TYPE": ["Laborers", "Managers", "Drivers"],
                "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0, 1.0, np.nan],
                "AMT_REQ_CREDIT_BUREAU_MON": [0.0, 2.0, 1.0],
                "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0, 3.0, 2.0],
            }
        )

        base, expanded = make_features(raw)

        self.assertEqual(len(base), len(raw))
        self.assertEqual(len(expanded), len(raw))
        self.assertNotIn("TARGET", base.columns)
        self.assertNotIn("SK_ID_CURR", expanded.columns)
        self.assertTrue(set(base.columns).issubset(expanded.columns))
        self.assertIn("DAYS_EMPLOYED_CLEAN", expanded.columns)
        self.assertIn("EXT_PRODUCT", expanded.columns)
        self.assertIn("CREDIT_INC", expanded.columns)
        self.assertIn("EMP_YEARS", expanded.columns)
        self.assertFalse(
            np.isinf(expanded.select_dtypes(include="number").to_numpy()).any()
        )
        groups = feature_groups(base, expanded)
        grouped = [column for columns in groups.values() for column in columns]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), set(expanded.columns) - set(base.columns))

    def test_missing_and_zero_denominators_do_not_create_infinity(self):
        raw = pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": [0.0, np.nan],
                "AMT_CREDIT": [0.0, 1.0],
                "AMT_ANNUITY": [0.0, 1.0],
                "AMT_GOODS_PRICE": [0.0, np.nan],
                "CNT_FAM_MEMBERS": [0.0, np.nan],
                "DAYS_BIRTH": [-1.0, -2.0],
                "DAYS_EMPLOYED": [365243.0, -1.0],
            }
        )
        _, expanded = make_features(raw)
        values = expanded.select_dtypes(include="number").to_numpy()
        self.assertFalse(np.isinf(values).any())


if __name__ == "__main__":
    unittest.main()
