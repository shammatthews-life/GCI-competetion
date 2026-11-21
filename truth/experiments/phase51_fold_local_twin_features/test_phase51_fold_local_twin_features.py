from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase51_fold_local_twin_features as phase51


class Phase51Tests(unittest.TestCase):
    def test_incomplete_date_keys_are_row_unique(self) -> None:
        raw = pd.DataFrame(
            {
                "DAYS_BIRTH": [-10000, -10000, np.nan],
                "DAYS_REGISTRATION": [-1000, -1000, np.nan],
                "DAYS_ID_PUBLISH": [-500, -500, np.nan],
                "CODE_GENDER": ["F", "F", "F"],
                "NAME_EDUCATION_TYPE": ["Higher", "Higher", "Higher"],
                "REGION_RATING_CLIENT": [2, 2, 2],
            }
        )
        keys = phase51.build_twin_keys(raw)
        self.assertEqual(keys.iloc[0]["birth_registration_gap"], 9000.0)
        self.assertEqual(keys.iloc[0]["birth_id_publication_gap"], 9500.0)
        self.assertTrue(keys.iloc[0].equals(keys.iloc[1]))
        self.assertNotEqual(
            keys.iloc[2]["unique_missing_key"],
            keys.iloc[0]["unique_missing_key"],
        )

    def test_group_statistics_use_only_source_labels(self) -> None:
        keys = pd.DataFrame(
            {
                "birth_registration_gap": [10.0, 10.0, 20.0],
                "birth_id_publication_gap": [30.0, 30.0, 40.0],
                "gender": ["F", "F", "M"],
                "education": ["Higher", "Higher", "Secondary"],
                "region_rating": ["2", "2", "3"],
                "unique_missing_key": ["", "", ""],
            }
        )
        applied = phase51._apply_group_statistics(
            keys.iloc[:1].reset_index(drop=True),
            np.array([1], dtype=np.int8),
            keys.iloc[1:].reset_index(drop=True),
        )
        self.assertAlmostEqual(applied[0, 0], (1 + 10 * 1.0) / 11)
        self.assertAlmostEqual(float(applied[0, 1]), float(np.log1p(1)))
        self.assertEqual(applied[0, 2], 1.0)
        self.assertEqual(applied[1, 2], 0.0)

    def test_crossfit_group_features_are_complete_and_bounded(self) -> None:
        raw = pd.DataFrame(
            {
                "DAYS_BIRTH": np.arange(60) * -10,
                "DAYS_REGISTRATION": np.arange(60) * -10 + np.arange(60),
                "DAYS_ID_PUBLISH": np.arange(60) * -10 + 2 * np.arange(60),
                "CODE_GENDER": ["F"] * 60,
                "NAME_EDUCATION_TYPE": ["Higher"] * 60,
                "REGION_RATING_CLIENT": [2] * 60,
            }
        )
        keys = phase51.build_twin_keys(raw)
        y = np.tile(np.array([0, 1], dtype=np.int8), 30)
        features = phase51.crossfit_group_features(keys, y, seed=3)
        self.assertEqual(features.shape, (60, 3))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all((features[:, 0] >= 0) & (features[:, 0] <= 1)))
        self.assertTrue(np.all(features[:, 2] == 0))


if __name__ == "__main__":
    unittest.main()
