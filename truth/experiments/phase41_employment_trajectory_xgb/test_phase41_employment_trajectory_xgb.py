from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase41_employment_trajectory_xgb as phase41


class Phase41EmploymentFeatureTests(unittest.TestCase):
    def test_sentinel_is_missing_and_regular_rows_use_documented_formula(self) -> None:
        raw = pd.DataFrame(
            {
                "DAYS_BIRTH": [-3652.5, -7305.0],
                "DAYS_EMPLOYED": [-365.25, 365243.0],
            }
        )
        rowwise = pd.DataFrame(index=raw.index)
        output = phase41.add_employment_features(raw, rowwise)
        self.assertAlmostEqual(
            output.loc[0, "SAFE_EXT_EMPLOYMENT_AGE_RATIO"], 0.1
        )
        self.assertAlmostEqual(
            output.loc[0, "SAFE_EXT_ESTIMATED_PREEMPLOYMENT_AGE"], 9.0
        )
        self.assertTrue(
            np.isnan(output.loc[1, "SAFE_EXT_EMPLOYMENT_AGE_RATIO"])
        )
        self.assertTrue(
            np.isnan(output.loc[1, "SAFE_EXT_ESTIMATED_PREEMPLOYMENT_AGE"])
        )


if __name__ == "__main__":
    unittest.main()
