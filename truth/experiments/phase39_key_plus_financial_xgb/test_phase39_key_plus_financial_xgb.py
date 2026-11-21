from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase39_key_plus_financial_xgb as phase39


class Phase39FinancialFeatureTests(unittest.TestCase):
    def test_financial_features_are_appended_without_changing_base_rows(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"]).reset_index(drop=True)
        blocks = phase39.phase33.build_safe_blocks(raw)
        fit = np.arange(700)
        apply = np.arange(700, 1000)
        base_fit, base_apply = phase39.phase38.make_matrices(
            blocks, fit, apply, phase39.KEY_MIN_FREQUENCY
        )
        fit_matrix, apply_matrix = phase39.make_matrices(blocks, fit, apply)
        self.assertEqual(fit_matrix.shape[0], len(fit))
        self.assertEqual(apply_matrix.shape[0], len(apply))
        self.assertEqual(
            fit_matrix.shape[1], base_fit.shape[1] + blocks[3].shape[1]
        )
        self.assertEqual(
            apply_matrix.shape[1], base_apply.shape[1] + blocks[3].shape[1]
        )
        np.testing.assert_array_equal(
            fit_matrix[:, : base_fit.shape[1]].indptr, base_fit.indptr
        )
        np.testing.assert_array_equal(
            fit_matrix[:, : base_fit.shape[1]].indices, base_fit.indices
        )
        np.testing.assert_array_equal(
            fit_matrix[:, : base_fit.shape[1]].data, base_fit.data
        )


if __name__ == "__main__":
    unittest.main()
