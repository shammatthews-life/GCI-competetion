from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase27_feature_composition as phase27


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [1000.0, 1200.0, 1500.0, 1700.0],
            "AMT_CREDIT": [2000.0, 2200.0, 3000.0, 3500.0],
            "AMT_ANNUITY": [100.0, 150.0, 210.0, 250.0],
            "AMT_GOODS_PRICE": [1800.0, 2100.0, 2800.0, 3300.0],
            "CNT_FAM_MEMBERS": [1.0, 2.0, 2.0, 3.0],
            "DAYS_BIRTH": [-9000.0, -12000.0, -15000.0, -18000.0],
            "DAYS_EMPLOYED": [-1000.0, 365243.0, -5000.0, -9000.0],
            "EXT_SOURCE_1": [0.2, 0.4, np.nan, 0.6],
            "EXT_SOURCE_2": [0.3, np.nan, 0.7, 0.8],
            "EXT_SOURCE_3": [0.4, np.nan, 0.8, 0.9],
        }
    )


class Phase27FeatureTests(unittest.TestCase):
    def test_context_interactions_are_finite_or_missing_and_target_free(self) -> None:
        raw = fixture()
        features = phase27.context_interaction_features(raw)
        self.assertEqual(len(features), len(raw))
        self.assertNotIn("TARGET", features.columns)
        values = features.to_numpy(dtype=np.float64)
        self.assertTrue((np.isfinite(values) | np.isnan(values)).all())
        self.assertTrue(
            np.isnan(features.loc[1, "P27_EXT_SPREAD_X_CREDIT_TO_INCOME"])
        )
        self.assertAlmostEqual(
            features.loc[0, "P27_FAMILY_ADJUSTED_ANNUITY_BURDEN"], 0.1
        )

    def test_one_hot_encoder_is_fold_fitted_and_ignores_unseen_categories(self) -> None:
        fit = pd.DataFrame({"category": ["A", "B", "A"]})
        valid = pd.DataFrame({"category": ["C"]})
        encoder = phase27.OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=np.float32
        )
        encoded_fit = encoder.fit_transform(fit)
        encoded_valid = encoder.transform(valid)
        self.assertEqual(encoded_fit.shape, (3, 2))
        self.assertEqual(encoded_valid.shape, (1, 2))
        np.testing.assert_array_equal(encoded_valid, np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
