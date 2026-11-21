from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase50_multiseed_pricing_synthesis as phase50


class Phase50Tests(unittest.TestCase):
    def test_oof_alignment_is_enforced(self) -> None:
        reference = pd.DataFrame(
            {"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}
        )
        aligned = reference.copy()
        phase50.validate_oof(aligned, reference, "test")
        with self.assertRaises(ValueError):
            phase50.validate_oof(aligned.iloc[::-1].reset_index(drop=True), reference, "test")

    def test_metric_record_rejects_nonfinite_predictions(self) -> None:
        with self.assertRaises(ValueError):
            phase50.metric_record(
                np.array([0, 1], dtype=np.int8),
                np.array([0.2, np.nan]),
                np.array([1, 1], dtype=np.int8),
            )

    def test_metric_record_reports_auc_and_log_loss(self) -> None:
        result = phase50.metric_record(
            np.array([0, 0, 1, 1], dtype=np.int8),
            np.array([0.1, 0.4, 0.6, 0.9]),
            np.array([1, 1, 1, 1], dtype=np.int8),
        )
        self.assertEqual(result["pooled_auc"], 1.0)
        self.assertGreater(result["pooled_log_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
