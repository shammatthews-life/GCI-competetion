import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_core_v2 import (  # noqa: E402
    _smooth_target_map,
    build_target_keys,
    compute_fold_target_encoding,
)
from test_pipeline_core_v1 import small_frame  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402


class TargetEncodingTests(unittest.TestCase):
    def test_key_construction_does_not_read_labels_or_other_rows(self):
        frame = small_frame()
        keys_before = build_target_keys(frame)
        changed = frame.copy(deep=True)
        changed["TARGET"] = 1 - changed["TARGET"]
        keys_after = build_target_keys(changed)
        pd.testing.assert_frame_equal(keys_before, keys_after)

    def test_unknown_query_key_uses_training_prior_without_count(self):
        fit_key = pd.Series(["known", "known", "other"])
        fit_y = np.array([0.0, 1.0, 0.0])
        query_key = pd.Series(["unseen"])
        encoded, log_count = _smooth_target_map(
            fit_key, fit_y, query_key, prior=1.0 / 3.0, smoothing=10.0
        )
        self.assertAlmostEqual(encoded[0], 1.0 / 3.0)
        self.assertEqual(log_count[0], 0.0)

    def test_training_encodings_are_oof_and_finite(self):
        frame = small_frame(size=20)
        target = np.tile([0, 1], 10).astype(np.int8)
        seed = 10
        encoded_fit, encoded_query = compute_fold_target_encoding(
            frame, target, frame.iloc[:3].copy(), seed=seed, inner_folds=2
        )
        self.assertEqual(len(encoded_fit), len(frame))
        self.assertEqual(len(encoded_query), 3)
        self.assertTrue(np.isfinite(encoded_fit.to_numpy()).all())
        self.assertTrue(np.isfinite(encoded_query.to_numpy()).all())
        self.assertTrue((encoded_fit.filter(like="TE_LOGCOUNT_") >= 0).all().all())
        keys = build_target_keys(frame).reset_index(drop=True)
        splits = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=seed
        ).split(np.zeros(len(target)), target)
        for inner_fit, inner_valid in splits:
            prior = float(target[inner_fit].mean())
            for column in keys:
                expected, expected_count = _smooth_target_map(
                    keys[column].iloc[inner_fit],
                    target[inner_fit],
                    keys[column].iloc[inner_valid],
                    prior,
                    smoothing=30.0,
                )
                np.testing.assert_allclose(
                    encoded_fit.loc[inner_valid, f"TE_{column}"].to_numpy(),
                    expected,
                )
                np.testing.assert_allclose(
                    encoded_fit.loc[
                        inner_valid, f"TE_LOGCOUNT_{column}"
                    ].to_numpy(),
                    expected_count,
                )


if __name__ == "__main__":
    unittest.main()
