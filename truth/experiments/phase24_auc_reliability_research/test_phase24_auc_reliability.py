from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase24_auc_reliability as phase24


class Phase24ReliabilityTests(unittest.TestCase):
    def test_oof_summary_reports_within_fold_and_cross_fold_pairs(self) -> None:
        target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        prediction = np.array(
            [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.05, 0.06]
        )
        folds = np.repeat(np.arange(1, 6), 2)

        result = phase24.oof_auc_summary(target, prediction, folds)

        self.assertEqual(len(result["per_fold"]), 5)
        self.assertGreater(result["cross_fold_pair_fraction"], 0.7)
        self.assertEqual(result["mean_fold_auc"], 1.0)
        self.assertLess(result["pooled_auc"], result["mean_fold_auc"])

    def test_duplicate_profile_summary_counts_cross_fold_groups(self) -> None:
        frame = pd.DataFrame({"profile": ["a", "a", "b", "c", "c"]})
        target = np.array([0, 1, 0, 1, 1], dtype=np.int8)
        folds = np.array([1, 2, 1, 2, 3], dtype=np.int8)

        result = phase24._group_diagnostics(
            frame, target, folds, ["profile"], "fixture"
        )

        self.assertEqual(result["duplicate_groups"], 2)
        self.assertEqual(result["mixed_target_duplicate_groups"], 1)
        self.assertEqual(result["duplicate_groups_crossing_random_folds"], 2)
        self.assertEqual(result["rows_in_duplicate_groups_crossing_random_folds"], 4)

    def test_fold_ids_must_cover_all_five_folds(self) -> None:
        with self.assertRaises(ValueError):
            phase24._validated_fold_ids(
                np.array([0, 1, 0, 1, 0, 1]),
                np.array([1, 1, 2, 2, 3, 3]),
            )


if __name__ == "__main__":
    unittest.main()
