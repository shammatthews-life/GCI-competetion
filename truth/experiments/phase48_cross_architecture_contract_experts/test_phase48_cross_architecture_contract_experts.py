from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase48_cross_architecture_contract_experts as phase48


class Phase48Tests(unittest.TestCase):
    def test_contract_gate_is_deterministic(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                    "Revolving loans",
                ]
            }
        )
        gate = phase48.make_contract_gate(raw)
        np.testing.assert_array_equal(gate, np.array([0, 1, 0, 1], dtype=np.int8))

    def test_unknown_contract_type_is_rejected(self) -> None:
        raw = pd.DataFrame({"NAME_CONTRACT_TYPE": ["Mortgage"]})
        with self.assertRaises(ValueError):
            phase48.make_contract_gate(raw)

    def test_cross_architecture_blend_arithmetic(self) -> None:
        p_lgb = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        p_xgb = np.array([0.4, 0.6, 0.8], dtype=np.float32)
        blend = phase48.equal_blend(p_lgb, p_xgb)
        np.testing.assert_allclose(blend, np.array([0.3, 0.5, 0.7], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

