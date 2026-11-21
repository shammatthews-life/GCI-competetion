from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase47_evidence_partition_experts as phase47


class Phase47GateTests(unittest.TestCase):
    def test_feature_only_gates_are_deterministic(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                    "Cash loans",
                ],
                "AMT_GOODS_PRICE": [100000.0, np.nan, 80000.0, 95000.0],
                "AMT_CREDIT": [100000.0, 50000.0, 100000.0, 100000.0],
                "DAYS_EMPLOYED": [-1500.0, -300.0, 365243.0, -4200.0],
            }
        )
        gates = phase47.make_gate_memberships(raw)
        np.testing.assert_array_equal(
            gates["contract_type"], np.array([0, 1, 0, 0], dtype=np.int8)
        )
        # goods/credit ratio: 1.0 (>=0.90 -> 0), nan (<0.90 -> 1), 0.80 (<0.90 -> 1), 0.95 (>=0.90 -> 0)
        np.testing.assert_array_equal(
            gates["goods_credit_ratio"], np.array([0, 1, 1, 0], dtype=np.int8)
        )
        # pensioner: -1500 -> 0, -300 -> 0, 365243 -> 1, -4200 -> 0
        np.testing.assert_array_equal(
            gates["pensioner_status"], np.array([0, 0, 1, 0], dtype=np.int8)
        )
        # 3-way contract_goods:
        # row 0: Cash & ratio 1.0 >= 0.90 -> group 1
        # row 1: Revolving -> group 0
        # row 2: Cash & ratio 0.80 < 0.90 -> group 2
        # row 3: Cash & ratio 0.95 >= 0.90 -> group 1
        np.testing.assert_array_equal(
            gates["contract_goods_bivariate"], np.array([1, 0, 2, 1], dtype=np.int8)
        )

    def test_unknown_contract_type_is_rejected(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": ["Unknown loan type"],
                "AMT_GOODS_PRICE": [100000.0],
                "AMT_CREDIT": [100000.0],
                "DAYS_EMPLOYED": [-1000.0],
            }
        )
        with self.assertRaises(ValueError):
            phase47.make_gate_memberships(raw)


if __name__ == "__main__":
    unittest.main()

