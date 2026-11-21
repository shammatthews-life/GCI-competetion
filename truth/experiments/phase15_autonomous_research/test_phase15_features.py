import unittest

import numpy as np
import pandas as pd

from phase15_features import build_time_lifecycle_features
from phase15_interactions import build_organization_external_bin_cross
from run_phase15_xgboost_category_preflight import align_categories


class TimeLifecycleFeatureTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "DAYS_BIRTH": [-36525.0, -73050.0, -109575.0],
                "DAYS_REGISTRATION": [-3652.5, -36525.0, np.nan],
                "DAYS_ID_PUBLISH": [-1826.25, -14610.0, -36525.0],
                "DAYS_LAST_PHONE_CHANGE": [-365.25, -1826.25, -3652.5],
                "DAYS_EMPLOYED": [-3652.5, 365243.0, -7305.0],
            },
            index=[3, 5, 9],
        )

    def test_features_are_rowwise_aligned_and_expected(self):
        result = build_time_lifecycle_features(self.raw)
        self.assertEqual(list(result.index), [3, 5, 9])
        self.assertAlmostEqual(result.loc[3, "AGE_YEARS_AT_REGISTRATION"], 90.0)
        self.assertAlmostEqual(result.loc[3, "REGISTRATION_TO_AGE_FRACTION"], 0.1)
        self.assertAlmostEqual(result.loc[3, "AGE_YEARS_AT_EMPLOYMENT_START"], 90.0)
        self.assertTrue(np.isnan(result.loc[5, "AGE_YEARS_AT_EMPLOYMENT_START"]))
        self.assertTrue(np.isnan(result.loc[9, "AGE_YEARS_AT_REGISTRATION"]))
        self.assertTrue(np.isfinite(result.to_numpy()[np.isfinite(result)]).all())

    def test_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_time_lifecycle_features(self.raw.assign(TARGET=[0, 1, 0]))

    def test_required_columns_are_checked(self):
        with self.assertRaisesRegex(ValueError, "DAYS_REGISTRATION"):
            build_time_lifecycle_features(self.raw.drop(columns="DAYS_REGISTRATION"))

    def test_category_vocabulary_comes_from_fit_rows(self):
        fit = pd.DataFrame({"category": ["a", "b", "a"]})
        apply = pd.DataFrame({"category": ["b", "unseen"]})
        fit_aligned, apply_aligned, support = align_categories(
            fit, apply, ("category",)
        )
        self.assertEqual(support, {"category": 2})
        self.assertEqual(list(fit_aligned["category"].cat.categories), ["a", "b"])
        self.assertEqual(str(apply_aligned["category"].iloc[0]), "b")
        self.assertTrue(pd.isna(apply_aligned["category"].iloc[1]))


class CategoricalInteractionTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame(
            {
                "ORGANIZATION_TYPE": ["A", "B", "A", "C", "D", "E"],
                "EXT_SOURCE_1": [0.1, 0.2, 0.3, 0.4, 0.9, np.nan],
                "EXT_SOURCE_2": [np.nan] * 6,
                "EXT_SOURCE_3": [np.nan] * 6,
            }
        )

    def test_external_score_bins_fit_on_fit_rows_only(self):
        cross, cuts = build_organization_external_bin_cross(
            self.raw, np.array([0, 1, 2, 3]), np.array([4, 5])
        )
        self.assertEqual(len(cuts), 3)
        self.assertEqual(cross.iloc[4], "D|EXT_Q3")
        self.assertEqual(cross.iloc[5], "E|EXT_QMISSING")
        changed = self.raw.copy()
        changed.loc[4, "EXT_SOURCE_1"] = 0.99
        _, changed_cuts = build_organization_external_bin_cross(
            changed, np.array([0, 1, 2, 3]), np.array([4, 5])
        )
        np.testing.assert_array_equal(cuts, changed_cuts)

    def test_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_organization_external_bin_cross(
                self.raw.assign(TARGET=[0, 0, 1, 0, 1, 0]),
                np.array([0, 1, 2, 3]),
                np.array([4, 5]),
            )


if __name__ == "__main__":
    unittest.main()
