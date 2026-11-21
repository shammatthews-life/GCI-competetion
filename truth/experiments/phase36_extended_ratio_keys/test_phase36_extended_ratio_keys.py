from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase36_extended_ratio_keys as phase36


class Phase36RatioKeyTests(unittest.TestCase):
    def test_extra_keys_are_rowwise_and_exclude_target_and_id(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"])
        keys = phase36.build_extra_keys(raw)
        self.assertEqual(len(keys), len(raw))
        self.assertEqual(tuple(keys.columns), phase36.EXTRA_KEYS)
        self.assertNotIn("TARGET", keys.columns)
        self.assertNotIn("SK_ID_CURR", keys.columns)

    def test_all_extra_keys_have_finite_category_values(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"])
        keys = phase36.build_extra_keys(raw)
        self.assertFalse(keys.isna().any().any())
        self.assertTrue(keys.apply(lambda column: column.astype(str).str.len().gt(0).all()).all())


if __name__ == "__main__":
    unittest.main()
