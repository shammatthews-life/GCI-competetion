from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase45_nested_expert_calibration as phase45


class Phase45CalibrationTests(unittest.TestCase):
    def test_contract_gate_is_covariate_only(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                ]
            }
        )
        np.testing.assert_array_equal(
            phase45.contract_membership(raw), np.array([0, 1, 0], dtype=np.int8)
        )

    def test_platt_calibration_preserves_finite_probability_shape(self) -> None:
        probabilities = np.linspace(0.01, 0.99, 100)
        targets = (probabilities > 0.5).astype(np.int8)
        calibrator = phase45.LogisticRegression(
            C=phase45.CALIBRATION_C,
            solver="lbfgs",
            max_iter=1000,
            random_state=1,
        ).fit(
            phase45.logit(probabilities).reshape(-1, 1), targets
        )
        calibrated = phase45.apply_platt(calibrator, probabilities)
        self.assertEqual(calibrated.shape, probabilities.shape)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())


if __name__ == "__main__":
    unittest.main()
