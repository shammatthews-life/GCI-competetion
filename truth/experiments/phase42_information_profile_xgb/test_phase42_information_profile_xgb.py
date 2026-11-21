from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase42_information_profile_xgb as phase42


class Phase42InformationProfileTests(unittest.TestCase):
    def test_profile_encodes_missingness_and_employment_sentinel(self) -> None:
        raw = pd.DataFrame(
            {
                "EXT_SOURCE_1": [0.2, None],
                "EXT_SOURCE_2": [None, 0.4],
                "EXT_SOURCE_3": [0.3, None],
                "OCCUPATION_TYPE": ["Laborers", None],
                "ORGANIZATION_TYPE": ["Business", None],
                "NAME_INCOME_TYPE": ["Working", "Pensioner"],
                "AMT_REQ_CREDIT_BUREAU_YEAR": [0, None],
                "AMT_REQ_CREDIT_BUREAU_QRT": [1, None],
                "DAYS_EMPLOYED": [-365.25, 365243.0],
            }
        )
        keys = pd.DataFrame({"existing_key": ["a", "b"]})
        result = phase42.add_information_profile(raw, keys)
        self.assertEqual(result.loc[0, phase42.PROFILE_COLUMN], "2")
        self.assertEqual(result.loc[1, phase42.PROFILE_COLUMN], "477")


if __name__ == "__main__":
    unittest.main()
