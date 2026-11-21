from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase46_nested_expert_weight as phase46


class Phase46WeightSelectionTests(unittest.TestCase):
    def test_selects_expert_weight_from_inner_oof(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int8)
        global_prediction = np.array([0.6, 0.7, 0.4, 0.3, 0.5, 0.2])
        expert_prediction = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        weight, metrics = phase46.choose_expert_weight(
            y, global_prediction, expert_prediction
        )
        self.assertEqual(weight, 1.0)
        self.assertEqual(metrics["inner_oof_auc"], 1.0)

    def test_invalid_oof_vectors_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            phase46.choose_expert_weight(
                np.array([0, 1]),
                np.array([0.1, np.nan]),
                np.array([0.2, 0.8]),
            )


if __name__ == "__main__":
    unittest.main()
