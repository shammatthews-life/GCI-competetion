from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase25_exp02_compliant_features as phase25


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AMT_CREDIT": [1000, 1200, 1500, 1700, 2000, 2300, 2600, 3000],
            "AMT_ANNUITY": [100, 140, 180, 200, 250, 280, 320, 350],
            "DAYS_BIRTH": [-9000, -10000, -12000, -13000, -15000, -16000, -18000, -19000],
            "EXT_SOURCE_1": [0.1, 0.3, 0.5, 0.2, 0.4, 0.6, 0.7, 0.8],
            "EXT_SOURCE_2": [0.2, 0.4, 0.6, 0.3, 0.5, 0.7, 0.8, 0.9],
            "EXT_SOURCE_3": [0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.9, 0.1],
            "OCCUPATION_TYPE": ["A", "A", "B", "B", "A", "B", "C", "C"],
            "CODE_GENDER": ["F", "F", "M", "M", "F", "M", "F", "M"],
        }
    )


class Phase25FeatureTests(unittest.TestCase):
    def test_formula_peer_features_use_only_fit_groups(self) -> None:
        raw = fixture()
        fit = np.arange(6)
        apply = np.array([6, 7])
        fit_features, apply_features = phase25.formula_peer_features(raw, fit, apply)
        changed = raw.copy()
        changed.loc[6:, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = 0.99
        changed_fit, changed_apply = phase25.formula_peer_features(
            changed, fit, apply
        )
        pd.testing.assert_frame_equal(fit_features, changed_fit)
        self.assertEqual(fit_features.shape, (6, 3))
        self.assertEqual(apply_features.shape, (2, 3))
        self.assertNotIn("P25_IMPLIED_RATE", fit_features.columns)
        self.assertNotIn("P25_IMPLIED_TERM", fit_features.columns)
        self.assertTrue(np.isfinite(fit_features.to_numpy()).all())

    def test_ratio_target_encoding_is_oof_and_uses_fit_labels_only(self) -> None:
        raw = fixture()
        fit = np.arange(8)
        apply = np.array([], dtype=np.int64)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        encoded, _ = phase25.ratio_target_encoding_features(
            raw, y, fit, apply, seed=10
        )
        changed_y = y.copy()
        changed_y[0] = 1 - changed_y[0]
        changed, _ = phase25.ratio_target_encoding_features(
            raw, changed_y, fit, apply, seed=10
        )
        self.assertEqual(encoded.shape, (8, 2))
        self.assertTrue(np.isfinite(encoded.to_numpy()).all())
        np.testing.assert_array_equal(
            encoded.iloc[0].to_numpy(), changed.iloc[0].to_numpy()
        )

    def test_peer_percentile_unseen_group_has_neutral_fallback(self) -> None:
        fit, apply = phase25._peer_percentile(
            np.array([0.2, 0.4]),
            np.array(["A", "A"]),
            np.array([0.9]),
            np.array(["unseen"]),
        )
        self.assertTrue(np.isfinite(fit).all())
        np.testing.assert_array_equal(apply, np.array([0.5]))

    def test_integerized_rejects_values_outside_int64(self) -> None:
        with self.assertRaises(ValueError):
            phase25._integerized(np.array([1e14]), decimals=6)


if __name__ == "__main__":
    unittest.main()
