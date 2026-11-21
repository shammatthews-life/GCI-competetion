from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase38_key_frequency_thresholds as phase38


class Phase38FrequencyTests(unittest.TestCase):
    def test_frequency_threshold_changes_only_fit_fold_encoding(self) -> None:
        raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"]).reset_index(drop=True)
        blocks = phase38.phase33.build_safe_blocks(raw)
        fit = np.arange(700)
        apply = np.arange(700, 1000)
        low_fit, low_apply = phase38.make_matrices(blocks, fit, apply, 2)
        reference_fit, reference_apply = phase38.phase33.make_matrices(
            raw, blocks, fit, apply, "keys_frequency"
        )
        high_fit, high_apply = phase38.make_matrices(blocks, fit, apply, 50)
        for actual, expected in (
            (low_fit, reference_fit),
            (low_apply, reference_apply),
        ):
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_array_equal(actual.indptr, expected.indptr)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            self.assertTrue(
                np.array_equal(actual.data, expected.data, equal_nan=True)
            )
        self.assertEqual(low_fit.shape[1], low_apply.shape[1])
        self.assertEqual(high_fit.shape[1], high_apply.shape[1])
        self.assertLessEqual(high_fit.shape[1], low_fit.shape[1])
        self.assertEqual(low_fit.shape[0], len(fit))
        self.assertEqual(high_apply.shape[0], len(apply))


if __name__ == "__main__":
    unittest.main()
