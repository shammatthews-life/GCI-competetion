from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase29_clean_scratch as phase29


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


class Phase29ScratchTests(unittest.TestCase):
    def test_features_are_deterministic_target_free_rowwise_transforms(self) -> None:
        raw = fixture()
        numeric_a, categorical_a = phase29.make_rowwise_features(raw)
        numeric_b, categorical_b = phase29.make_rowwise_features(raw)
        pd.testing.assert_frame_equal(numeric_a, numeric_b)
        pd.testing.assert_frame_equal(categorical_a, categorical_b)
        self.assertNotIn("SK_ID_CURR", numeric_a)
        self.assertNotIn("TARGET", numeric_a)
        self.assertTrue(
            np.isfinite(numeric_a.to_numpy()[np.isfinite(numeric_a.to_numpy())]).all()
        )
        self.assertEqual(
            numeric_a.loc[0, "SCRATCH_INCOME_PER_FAMILY_MEMBER"], 1000
        )
        self.assertTrue(np.isnan(numeric_a.loc[1, "SCRATCH_EMPLOYMENT_YEARS"]))

    def test_onehot_encoder_is_fit_only_on_training_rows(self) -> None:
        raw = fixture()
        numeric, categorical = phase29.make_rowwise_features(raw)
        fit_rows = np.array([0, 1, 2])
        valid_rows = np.array([3])
        test = (numeric.iloc[3:], categorical.iloc[3:])
        fit, valid, _, info = phase29.make_fold_matrices(
            numeric, categorical, fit_rows, valid_rows, test
        )
        self.assertEqual(fit.shape[1], valid.shape[1])
        self.assertGreater(info["onehot_feature_count"], 0)
        self.assertEqual(fit.shape[0], 3)
        self.assertEqual(valid.shape[0], 1)

    def test_target_column_is_rejected_from_features(self) -> None:
        raw = fixture()
        raw["TARGET"] = [0, 1, 0, 1]
        with self.assertRaises(ValueError):
            phase29.make_rowwise_features(raw)


if __name__ == "__main__":
    unittest.main()
