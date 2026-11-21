from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase43_bureau_formula_ablation as phase43


class Phase43BureauFormulaTests(unittest.TestCase):
    def test_formulas_preserve_missingness_and_match_definitions(self) -> None:
        raw = pd.DataFrame(
            {
                "EXT_SOURCE_1": [0.2, 0.3, 0.5],
                "EXT_SOURCE_2": [0.4, 0.6, np.nan],
                "EXT_SOURCE_3": [0.8, 0.9, 0.7],
                "AMT_INCOME_TOTAL": [120.0, 100.0, 80.0],
                "AMT_ANNUITY": [12.0, 20.0, 10.0],
            }
        )
        result = phase43.add_bureau_formulas(raw, pd.DataFrame(index=raw.index))
        self.assertAlmostEqual(result.loc[0, "SAFE_EXT_PRODUCT"], 0.064)
        self.assertAlmostEqual(
            result.loc[0, "SAFE_EXT_HARMONIC_MEAN"],
            3.0 / (1.0 / 0.2 + 1.0 / 0.4 + 1.0 / 0.8),
        )
        self.assertAlmostEqual(
            result.loc[0, "SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ"],
            120.0 / 12.0 * (0.4 * 0.8) ** 2,
        )
        self.assertTrue(np.isnan(result.loc[2, "SAFE_EXT_PRODUCT"]))
        self.assertTrue(np.isnan(result.loc[2, "SAFE_EXT_HARMONIC_MEAN"]))
        self.assertTrue(
            np.isnan(result.loc[2, "SAFE_EXT_INCOME_ANNUITY_X_EXT23_SQ"])
        )

    def test_empty_formula_set_matches_phase38_baseline_matrix(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"]).reset_index(drop=True)
        blocks = phase43.phase33.build_safe_blocks(raw)
        formula_features = phase43.add_bureau_formulas(
            raw, pd.DataFrame(index=raw.index)
        )
        baseline = phase43.phase38.make_matrices(
            blocks, np.arange(700), np.arange(700, 1000), 50
        )
        actual = phase43.make_matrices(
            blocks, formula_features, np.arange(700), np.arange(700, 1000), ()
        )
        for left, right in zip(actual, baseline):
            self.assertEqual(left.shape, right.shape)
            np.testing.assert_array_equal(left.indptr, right.indptr)
            np.testing.assert_array_equal(left.indices, right.indices)
            self.assertTrue(np.array_equal(left.data, right.data, equal_nan=True))

    def test_selected_formula_is_appended_without_leaking_into_base(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"]).reset_index(drop=True)
        blocks = phase43.phase33.build_safe_blocks(raw)
        formula_features = phase43.add_bureau_formulas(
            raw, pd.DataFrame(index=raw.index)
        )
        base_fit, base_apply = phase43.phase38.make_matrices(
            blocks, np.arange(700), np.arange(700, 1000), 50
        )
        actual_fit, actual_apply = phase43.make_matrices(
            blocks,
            formula_features,
            np.arange(700),
            np.arange(700, 1000),
            ("SAFE_EXT_PRODUCT",),
        )
        self.assertEqual(actual_fit.shape[1], base_fit.shape[1] + 1)
        self.assertEqual(actual_apply.shape[1], base_apply.shape[1] + 1)


if __name__ == "__main__":
    unittest.main()
