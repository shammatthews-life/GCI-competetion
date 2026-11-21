import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER_PATH = Path(__file__).with_name("run_phase3_hidden_pattern_discovery.py")
SPEC = importlib.util.spec_from_file_location("phase3_tested_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load feature runner: {RUNNER_PATH}")
phase3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase3)


class ProductFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = pd.read_csv(phase3.TRAIN_PATH, nrows=4000)
        cls.raw = source.drop(columns=[phase3.TARGET]).reset_index(drop=True)
        cls.fit = cls.raw.iloc[:3000].copy()
        cls.query = cls.raw.iloc[3000:].copy()
        cls.params = phase3.fit_fee_and_step(cls.fit)
        cls.maps = phase3.fit_product_maps(cls.fit, cls.params)

    def test_safe_ratio_masks_zero_and_missing_denominators(self):
        result = phase3.safe_ratio(
            np.array([4.0, 5.0, np.nan, 8.0]),
            np.array([2.0, 0.0, 2.0, np.nan]),
        )
        self.assertEqual(result[0], 2.0)
        self.assertTrue(np.isnan(result[1:]).all())

    def test_feature_transform_has_expected_columns_and_no_infinities(self):
        result = phase3.make_product_features(self.query, self.params, self.maps)
        self.assertEqual(len(result), len(self.query))
        self.assertEqual(result.shape[1], len(phase3.FEATURE_DEFINITIONS))
        self.assertFalse(np.isinf(result.to_numpy(dtype=np.float64)).any())

    def test_feature_transform_resets_row_index(self):
        result = phase3.make_product_features(self.query, self.params, self.maps)
        self.assertTrue(result.index.equals(pd.RangeIndex(len(self.query))))

    def test_target_column_does_not_change_fitted_decoders_or_features(self):
        raw_with_target = self.fit.copy()
        raw_with_target[phase3.TARGET] = np.arange(len(raw_with_target)) % 2
        params_with_target = phase3.fit_fee_and_step(raw_with_target)
        self.assertEqual(self.params, params_with_target)
        maps_with_target = phase3.fit_product_maps(raw_with_target, params_with_target)
        left = phase3.make_product_features(self.query, self.params, self.maps)
        right = phase3.make_product_features(self.query, params_with_target, maps_with_target)
        pd.testing.assert_frame_equal(left, right)

    def test_lattice_coordinate_is_missing_for_non_cash_contract(self):
        query = self.query.copy()
        revolving_positions = np.flatnonzero(query["NAME_CONTRACT_TYPE"].eq("Revolving loans").to_numpy())
        if len(revolving_positions) == 0:
            self.skipTest("Sample does not contain a revolving contract.")
        row_position = int(revolving_positions[0])
        features = phase3.make_product_features(query, self.params, self.maps)
        self.assertTrue(np.isnan(features.loc[row_position, "PROD_K_DECIMAL"]))
        self.assertEqual(features.loc[row_position, "PROD_K_VALID"], 0.0)


if __name__ == "__main__":
    unittest.main()
