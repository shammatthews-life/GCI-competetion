from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_pairwise_auc_diagnostics as phase22


class PairwiseAucDiagnosticsTests(unittest.TestCase):
    def test_decile_pair_decomposition_matches_auc_with_ties(self) -> None:
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        baseline = np.array([0.1, 0.1, 0.5, 0.4, 0.7, 0.6, 0.9, 0.8])
        candidate = np.array([0.2, 0.1, 0.4, 0.5, 0.8, 0.6, 0.9, 0.7])
        cells, _, summary = phase22.make_diagnostics(
            labels, baseline, {"candidate": candidate}
        )
        expected_pairs = int(np.sum(labels == 1) * np.sum(labels == 0))
        self.assertEqual(int(cells["pair_count"].sum()), expected_pairs)
        self.assertAlmostEqual(
            float(summary["baseline_auc"]), roc_auc_score(labels, baseline)
        )
        self.assertAlmostEqual(
            float(summary["candidate_aucs"]["candidate"]),
            roc_auc_score(labels, candidate),
        )
        delta_from_pairs = cells["net_pair_change"].sum() / expected_pairs
        self.assertAlmostEqual(
            float(delta_from_pairs),
            roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline),
        )

    def test_score_deciles_are_stable_and_bounded(self) -> None:
        scores = np.array([0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        bands = phase22.score_deciles(scores)
        self.assertTrue(np.all((bands >= 0) & (bands < phase22.N_BANDS)))
        np.testing.assert_array_equal(bands, phase22.score_deciles(scores))

    def test_invalid_nonfinite_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            phase22.score_deciles(np.array([0.1, np.nan]))


if __name__ == "__main__":
    unittest.main()
