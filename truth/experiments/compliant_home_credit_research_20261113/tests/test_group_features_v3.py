import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_core_v3 import compute_fold_group_features  # noqa: E402
from test_pipeline_core_v1 import small_frame  # noqa: E402


class FoldLocalGroupFeatureTests(unittest.TestCase):
    def test_group_features_are_aligned_finite_and_target_independent(self):
        frame = small_frame(size=20)
        fit = frame.iloc[:16].reset_index(drop=True)
        query = frame.iloc[16:].reset_index(drop=True)
        original_query = query.copy(deep=True)
        fit_features, query_features = compute_fold_group_features(
            fit, query, seed=18, inner_folds=2
        )
        query["TARGET"] = 1 - query["TARGET"]
        fit_features_after, query_features_after = compute_fold_group_features(
            fit, query, seed=18, inner_folds=2
        )
        self.assertEqual(len(fit_features), len(fit))
        self.assertEqual(len(query_features), len(query))
        self.assertGreater(fit_features.shape[1], 0)
        self.assertFalse(np.isinf(fit_features.to_numpy()).any())
        self.assertFalse(np.isinf(query_features.to_numpy()).any())
        pd.testing.assert_frame_equal(fit_features, fit_features_after)
        pd.testing.assert_frame_equal(query_features, query_features_after)
        pd.testing.assert_frame_equal(
            query.drop(columns=["TARGET"]), original_query.drop(columns=["TARGET"])
        )


if __name__ == "__main__":
    unittest.main()
