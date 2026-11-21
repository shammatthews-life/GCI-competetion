from __future__ import annotations

import unittest

import pandas as pd

from phase11_key_ablation import _variant_frames
from phase11_nested_ensemble import _one_hot_matrix


class Phase11ModelPipelineTests(unittest.TestCase):
    def test_one_hot_encoder_ignores_categories_unseen_in_fit_rows(self) -> None:
        fit = pd.DataFrame(
            {"CATEGORY": ["a", "a", "b"], "VALUE": [0.0, 1.0, 2.0]}
        )
        apply = pd.DataFrame(
            {"CATEGORY": ["unseen"], "VALUE": [3.0]}
        )

        fit_matrix, apply_matrix = _one_hot_matrix(
            fit, apply, ["CATEGORY"], ["VALUE"]
        )

        self.assertEqual(fit_matrix.shape[0], 3)
        self.assertEqual(apply_matrix.shape[0], 1)
        self.assertEqual(fit_matrix.shape[1], apply_matrix.shape[1])
        self.assertEqual(apply_matrix[0].nnz, 1)

    def test_ablation_grid_contains_single_and_leave_one_out_variants(self) -> None:
        row_features = pd.DataFrame({"RAW": [0.0, 1.0]})
        keys = pd.DataFrame(
            {f"KEY_{index}": ["a", "b"] for index in range(8)}
        )

        variants = _variant_frames(row_features, keys)

        self.assertEqual(len(variants), 18)
        self.assertEqual(variants["single__KEY_0"].shape[1], 2)
        self.assertEqual(variants["leave_out__KEY_0"].shape[1], 8)
        self.assertEqual(variants["all_keys"].shape[1], 9)


if __name__ == "__main__":
    unittest.main()
