from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase33_rules_first_features as phase33


class Phase33FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=1000
        ).drop(columns=["TARGET"])
        cls.blocks = phase33.build_safe_blocks(cls.raw)

    def test_feature_blocks_are_row_aligned_and_do_not_use_target_or_id(self) -> None:
        numeric, categorical, rowwise, financial, keys, peer = self.blocks
        for frame in (numeric, categorical, rowwise, financial, keys, peer):
            self.assertEqual(len(frame), len(self.raw))
            self.assertNotIn("TARGET", frame.columns)
            self.assertNotIn("SK_ID_CURR", frame.columns)
        self.assertEqual(len(keys.columns), 4)
        self.assertEqual(len(financial.columns), 7)

    def test_financial_features_are_guarded_ratios(self) -> None:
        _, _, _, financial, _, _ = self.blocks
        self.assertTrue(
            np.allclose(
                financial["SAFE_FIN_CREDIT_TO_INCOME"].to_numpy(),
                self.raw["AMT_CREDIT"].to_numpy()
                / self.raw["AMT_INCOME_TOTAL"].to_numpy(),
                equal_nan=True,
            )
        )

    def test_one_hot_vocabulary_is_fit_only_on_fit_rows(self) -> None:
        fit_rows = np.arange(600)
        apply_rows = np.arange(600, 1000)
        fit, apply = phase33.make_matrices(
            self.raw, self.blocks, fit_rows, apply_rows, "keys_frequency"
        )
        self.assertEqual(fit.shape[1], apply.shape[1])
        self.assertEqual(fit.shape[0], len(fit_rows))
        self.assertEqual(apply.shape[0], len(apply_rows))

    def test_peer_percentile_uses_fold_fit_groups_and_handles_unseen_group(self) -> None:
        fit, apply = phase33._peer_percentile(
            np.array([0.1, 0.3, 0.2]),
            np.array(["A", "A", "B"]),
            np.array([0.2, 0.5]),
            np.array(["A", "C"]),
        )
        self.assertAlmostEqual(float(fit[0]), 0.5)
        self.assertAlmostEqual(float(apply[0]), 0.5)
        self.assertAlmostEqual(float(apply[1]), 0.5)


if __name__ == "__main__":
    unittest.main()
