from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase49_unified_contract_pricing_specialists as phase49


class Phase49Tests(unittest.TestCase):
    def test_pricing_block_construction_no_leakage(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
                "AMT_CREDIT": [100000.0, 50000.0],
                "AMT_ANNUITY": [5000.0, 2500.0],
                "AMT_GOODS_PRICE": [100000.0, np.nan],
                "AMT_INCOME_TOTAL": [150000.0, 200000.0],
                "EXT_SOURCE_1": [0.5, np.nan],
                "EXT_SOURCE_2": [0.6, 0.7],
                "EXT_SOURCE_3": [np.nan, 0.8],
            }
        )
        pricing_df = phase49.build_pricing_block(raw, m1=0.5, m2=0.6, m3=0.7)
        self.assertIn("Q0_FEE", pricing_df.columns)
        self.assertIn("TARIFF_STEP_0066_RESIDUAL", pricing_df.columns)
        self.assertIn("EXT_PRODUCT", pricing_df.columns)
        self.assertIn("MANIFOLD_SOLVENCY_CONSENSUS", pricing_df.columns)
        self.assertEqual(len(pricing_df), 2)
        self.assertTrue(np.isfinite(pricing_df.to_numpy()).all())

    def test_contract_gate_is_deterministic(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                ]
            }
        )
        gate = phase49.make_contract_gate(raw)
        np.testing.assert_array_equal(gate, np.array([0, 1, 0], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()

