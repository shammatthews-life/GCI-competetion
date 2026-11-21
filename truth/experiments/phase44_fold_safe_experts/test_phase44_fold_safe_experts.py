from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase44_fold_safe_experts as phase44


class Phase44GateTests(unittest.TestCase):
    def test_feature_only_gates_are_deterministic(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                    "Revolving loans",
                ],
                "EXT_SOURCE_1": [0.1, 0.2, np.nan, 0.4],
                "EXT_SOURCE_2": [0.2, 0.3, 0.5, 0.6],
                "EXT_SOURCE_3": [0.3, 0.4, 0.6, np.nan],
            }
        )
        gates = phase44.make_gate_memberships(raw)
        np.testing.assert_array_equal(
            gates["contract_type"], np.array([0, 1, 0, 1], dtype=np.int8)
        )
        np.testing.assert_array_equal(
            gates["external_completeness"],
            np.array([1, 1, 0, 0], dtype=np.int8),
        )

    def test_unknown_contract_type_is_rejected(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": ["unexpected"],
                "EXT_SOURCE_1": [0.1],
                "EXT_SOURCE_2": [0.2],
                "EXT_SOURCE_3": [0.3],
            }
        )
        with self.assertRaises(ValueError):
            phase44.make_gate_memberships(raw)


if __name__ == "__main__":
    unittest.main()
