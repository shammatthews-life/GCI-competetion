from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase21_focused_gpu as phase21


class Phase21FocusedValidationTests(unittest.TestCase):
    def test_learning_curve_samples_are_nested_and_stratified(self) -> None:
        y = np.array([0] * 800 + [1] * 200, dtype=np.int8)
        fit = np.arange(len(y))
        subsets = phase21.nested_stratified_subsets(fit, y, fold=1)
        prior = set()
        for fraction in phase21.LEARNING_FRACTIONS:
            current = set(subsets[fraction].tolist())
            self.assertTrue(prior.issubset(current))
            prior = current
            labels = y[subsets[fraction]]
            self.assertEqual(len(np.unique(labels)), 2)
            self.assertAlmostEqual(float(labels.mean()), 0.2, delta=0.005)

    def test_learning_curve_rebuilds_frequency_features_from_each_subset(self) -> None:
        rows = 4
        blocks = (
            pd.DataFrame({"AMT_CREDIT": [10.0, 20.0, 30.0, 40.0]}),
            pd.DataFrame({"CATEGORY": ["a", "b", "a", "c"]}),
            pd.DataFrame({"ROW_FEATURE": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame(
                {
                    key: ["a", "a", "b", "c"]
                    for key in phase21.features13.PRODUCT_KEYS
                }
            ),
            pd.DataFrame({"DECOMPOSITION": [0.1, 0.2, 0.3, 0.4]}),
            pd.DataFrame({"ENHANCEMENT": [1.0, 2.0, 3.0, 4.0]}),
            pd.DataFrame(index=np.arange(rows)),
        )
        full_fit, _, _ = phase21.learning_curve_matrix(
            blocks, np.array([0, 1, 2]), np.array([3])
        )
        subset_fit, subset_apply, _ = phase21.learning_curve_matrix(
            blocks, np.array([0, 2]), np.array([3])
        )
        self.assertEqual(full_fit.loc[0, "FIT_COUNT_PRODUCT_KEY_Q3"], 2)
        self.assertEqual(subset_fit.loc[0, "FIT_COUNT_PRODUCT_KEY_Q3"], 1)
        self.assertEqual(
            subset_apply.loc[0, "FIT_COUNT_PRODUCT_KEY_Q3"], 0
        )

    def test_fold_checkpoint_round_trip_and_identity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary)
            valid_rows = np.array([2, 7, 11], dtype=np.int64)
            identity = {"train_sha256": "abc"}
            predictions = {
                "candidate_a": np.array([0.1, 0.2, 0.3]),
                "candidate_b": np.array([0.4, 0.5, 0.6]),
            }
            seeds = {
                1: np.array([0.11, 0.21, 0.31]),
                2: np.array([0.12, 0.22, 0.32]),
            }
            path = phase21.save_fold_checkpoint(
                checkpoint_dir,
                1,
                valid_rows,
                np.array([0.15, 0.25, 0.35]),
                seeds,
                predictions,
                [{"fit_seconds": 1.0}],
                [{"gate": "contract_type"}],
                identity,
            )
            loaded = phase21.load_fold_checkpoint(
                path,
                1,
                valid_rows,
                list(predictions),
                list(seeds),
                identity,
            )
            np.testing.assert_array_equal(
                loaded["global_prediction"], np.array([0.15, 0.25, 0.35])
            )
            np.testing.assert_array_equal(
                loaded["predictions"]["candidate_b"], predictions["candidate_b"]
            )
            np.testing.assert_array_equal(
                loaded["seed_predictions"][2], seeds[2]
            )
            with self.assertRaises(ValueError):
                phase21.load_fold_checkpoint(
                    path,
                    1,
                    valid_rows,
                    list(predictions),
                    list(seeds),
                    {"train_sha256": "different"},
                )

    def test_routes_are_feature_only_and_group_codes_are_stable(self) -> None:
        raw = pd.DataFrame(
            {
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                    "Revolving loans",
                ],
                "EXT_SOURCE_1": [0.1, 0.2, np.nan, np.nan],
                "EXT_SOURCE_2": [0.1, np.nan, 0.2, np.nan],
                "EXT_SOURCE_3": [0.1, 0.2, 0.2, np.nan],
            }
        )
        np.testing.assert_array_equal(
            phase21.contract_route(raw), np.array([True, False, True, False])
        )
        np.testing.assert_array_equal(
            phase21.external_completeness_route(raw),
            np.array([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            phase21.combined_route(raw), np.array([3, 0, 2, 0])
        )

    def test_platt_calibration_is_bounded_and_monotonic(self) -> None:
        scores = np.linspace(0.02, 0.98, 1000)
        labels = (scores + np.random.default_rng(42).normal(0, 0.12, len(scores)) > 0.5).astype(
            np.int8
        )
        parameters = phase21.fit_platt_oof(scores, labels)
        self.assertGreaterEqual(parameters[0], 0.01)
        calibrated = phase21.apply_platt(scores, parameters)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue(((calibrated > 0) & (calibrated < 1)).all())
        self.assertTrue(np.all(np.diff(calibrated) >= 0))

    def test_feature_group_ablation_drops_only_requested_columns(self) -> None:
        x_fit = pd.DataFrame(
            {
                "EXT_SOURCE_1": [0.1, 0.2],
                "EXT_SOURCE_MEAN": [0.1, 0.2],
                "AMT_CREDIT": [100.0, 200.0],
                "PRODUCT_KEY_Q3": ["1", "2"],
                "FIT_COUNT_PRODUCT_KEY_Q3": [1.0, 2.0],
            }
        )
        x_apply = x_fit.copy()
        fit, apply, cats = phase21.drop_feature_group(
            x_fit, x_apply, ["PRODUCT_KEY_Q3"], "external_scores"
        )
        self.assertEqual(
            fit.columns.tolist(),
            [
                "AMT_CREDIT",
                "PRODUCT_KEY_Q3",
                "FIT_COUNT_PRODUCT_KEY_Q3",
            ],
        )
        self.assertEqual(fit.columns.tolist(), apply.columns.tolist())
        self.assertEqual(cats, ["PRODUCT_KEY_Q3"])

    def test_gpu_model_is_explicit_and_uses_frozen_phase15_iterations(self) -> None:
        model = phase21.make_model(12345)
        params = model.get_params()
        self.assertEqual(params["task_type"], "GPU")
        self.assertEqual(params["iterations"], 700)
        self.assertEqual(params["depth"], 6)
        self.assertEqual(params["learning_rate"], 0.05)


if __name__ == "__main__":
    unittest.main()
