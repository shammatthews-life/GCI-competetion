import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "exp85_phase2_inner_early_stopping"))

import run_experiment as experiment  # noqa: E402


def loan_rows(count=40, include_target=True):
    credit = np.arange(10_000.0, 10_000.0 + 1_000.0 * count, 1_000.0)
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(count, dtype=np.int64),
            "NAME_CONTRACT_TYPE": ["Cash loans"] * count,
            "CODE_GENDER": ["F"] * count,
            "NAME_INCOME_TYPE": ["Working"] * count,
            "NAME_EDUCATION_TYPE": ["Higher education"] * count,
            "NAME_FAMILY_STATUS": ["Married"] * count,
            "NAME_HOUSING_TYPE": ["House / apartment"] * count,
            "OCCUPATION_TYPE": ["Laborers"] * count,
            "ORGANIZATION_TYPE": ["Business Entity Type 3"] * count,
            "CNT_CHILDREN": np.zeros(count),
            "AMT_INCOME_TOTAL": credit * 0.2,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": credit * 0.1 + 129.5,
            "AMT_GOODS_PRICE": credit / (1.0 + 0.0132 * 5.0),
            "REGION_POPULATION_RELATIVE": np.full(count, 0.02),
            "DAYS_BIRTH": -np.arange(30_000, 30_000 + count),
            "DAYS_EMPLOYED": -np.arange(1_000, 1_000 + count),
            "DAYS_REGISTRATION": -np.arange(2_000, 2_000 + count),
            "DAYS_ID_PUBLISH": -np.arange(500, 500 + count),
            "OWN_CAR_AGE": np.full(count, np.nan),
            "FLAG_MOBIL": np.ones(count),
            "FLAG_EMP_PHONE": np.ones(count),
            "FLAG_WORK_PHONE": np.zeros(count),
            "CNT_FAM_MEMBERS": np.full(count, 2.0),
            "REGION_RATING_CLIENT": np.ones(count),
            "EXT_SOURCE_1": np.linspace(0.1, 0.9, count),
            "EXT_SOURCE_2": np.linspace(0.2, 0.8, count),
            "EXT_SOURCE_3": np.linspace(0.3, 0.7, count),
            "DAYS_LAST_PHONE_CHANGE": -np.arange(100, 100 + count),
            "AMT_REQ_CREDIT_BUREAU_HOUR": np.zeros(count),
            "AMT_REQ_CREDIT_BUREAU_MON": np.zeros(count),
            "AMT_REQ_CREDIT_BUREAU_QRT": np.zeros(count),
            "AMT_REQ_CREDIT_BUREAU_YEAR": np.zeros(count),
        }
    )
    frame["ORGANIZATION_TYPE"] = "Business Entity Type 3"
    if include_target:
        frame["TARGET"] = (np.arange(count) % 2).astype(np.int8)
    return frame


class InnerEarlyStoppingTests(unittest.TestCase):
    def test_inner_stop_is_disjoint_and_inside_outer_fit(self):
        labels = np.arange(200) % 2
        outer_fit = np.arange(160)
        outer_valid = np.arange(160, 200)

        fit_a, stop_a = experiment._inner_stop_split(outer_fit, labels, 1)
        fit_b, stop_b = experiment._inner_stop_split(outer_fit, labels, 1)

        np.testing.assert_array_equal(fit_a, fit_b)
        np.testing.assert_array_equal(stop_a, stop_b)
        self.assertEqual(len(np.intersect1d(fit_a, stop_a)), 0)
        self.assertTrue(set(fit_a).issubset(set(outer_fit)))
        self.assertTrue(set(stop_a).issubset(set(outer_fit)))
        self.assertEqual(len(np.intersect1d(stop_a, outer_valid)), 0)

    def test_fold_auc_summary_scores_the_supplied_blend_predictions(self):
        labels = np.tile([0, 1], 5)
        fold_ids = np.repeat(np.arange(1, 6), 2)
        predictions = np.tile([0.1, 0.9], 5)

        scores = experiment._fold_auc_scores(labels, fold_ids, predictions)

        self.assertEqual(scores, [1.0] * 5)

    def test_fee_counts_use_reference_rows_not_apply_rows(self):
        reference = loan_rows(20)
        apply_first = loan_rows(4, include_target=False)
        apply_second = apply_first.copy()
        apply_second["AMT_ANNUITY"] += 317.0
        apply_second["AMT_CREDIT"] *= 1.13

        fit_a, _, _ = experiment.build_fold_local_fee_features(
            reference, apply_first
        )
        fit_b, _, _ = experiment.build_fold_local_fee_features(
            reference, apply_second
        )

        pd.testing.assert_frame_equal(fit_a, fit_b)

    def test_fee_features_ignore_target_labels(self):
        first = loan_rows(20, include_target=True)
        second = first.copy()
        second["TARGET"] = 1 - second["TARGET"]
        apply = loan_rows(5, include_target=False)

        first_fit, first_apply, _ = experiment.build_fold_local_fee_features(
            first, apply
        )
        second_fit, second_apply, _ = experiment.build_fold_local_fee_features(
            second, apply
        )

        pd.testing.assert_frame_equal(first_fit, second_fit)
        pd.testing.assert_frame_equal(first_apply, second_apply)

    def test_apply_labels_cannot_change_target_encodings(self):
        keys = {
            "FEE_Q3": np.asarray(["a", "a", "b", "b", "c", "c", "a", "z"]),
            "FEE_Q4": np.asarray(["a", "a", "b", "b", "c", "c", "a", "z"]),
            "FEE_Q5": np.asarray(["a", "a", "b", "b", "c", "c", "a", "z"]),
            "PRODUCT_K": np.asarray(["1", "1", "2", "2", "3", "3", "1", "z"]),
        }
        labels = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int8)
        changed = labels.copy()
        changed[6:] = 1 - changed[6:]
        fit = np.arange(6)
        apply = np.arange(6, 8)

        expected = experiment.phase_02.apply_training_encodings(
            keys, labels, fit, apply
        )
        actual = experiment.phase_02.apply_training_encodings(
            keys, changed, fit, apply
        )

        pd.testing.assert_frame_equal(expected, actual)

    def test_phase2_iteration_selection_and_refit_build_separate_encodings(self):
        train = loan_rows(100)
        test = loan_rows(7, include_target=False)
        test["SK_ID_CURR"] += 1_000
        labels = train["TARGET"].to_numpy(dtype=np.int8)
        outer_fit = np.arange(80)
        outer_valid = np.arange(80, 100)
        inner_fit = outer_fit[:70]
        inner_stop = outer_fit[70:]
        captured = {}

        def fake_fit(**kwargs):
            captured.update(kwargs)
            return (
                np.full(len(kwargs["outer_apply_features"]), 0.5, dtype=np.float32),
                12,
                pd.DataFrame(),
            )

        with patch.object(experiment, "_fit_with_inner_stopping", side_effect=fake_fit):
            valid, test_predictions, iterations = experiment._fit_predict_pair(
                model_name="lightgbm",
                outer_fit_indices=outer_fit,
                outer_valid_indices=outer_valid,
                inner_fit_indices=inner_fit,
                inner_stop_indices=inner_stop,
                train_raw=train,
                test_raw=test,
                labels=labels,
                fold_number=1,
                phase2=True,
            )

        changed_labels = labels.copy()
        changed_labels[outer_valid] = 1 - changed_labels[outer_valid]
        changed_capture = {}

        def capture_changed_labels(**kwargs):
            changed_capture.update(kwargs)
            return (
                np.full(len(kwargs["outer_apply_features"]), 0.5, dtype=np.float32),
                12,
                pd.DataFrame(),
            )

        with patch.object(
            experiment,
            "_fit_with_inner_stopping",
            side_effect=capture_changed_labels,
        ):
            experiment._fit_predict_pair(
                model_name="lightgbm",
                outer_fit_indices=outer_fit,
                outer_valid_indices=outer_valid,
                inner_fit_indices=inner_fit,
                inner_stop_indices=inner_stop,
                train_raw=train,
                test_raw=test,
                labels=changed_labels,
                fold_number=1,
                phase2=True,
            )

        self.assertEqual(len(valid), len(outer_valid))
        self.assertEqual(len(test_predictions), len(test))
        self.assertEqual(iterations, 12)
        self.assertEqual(len(captured["fit_features"]), len(inner_fit))
        self.assertEqual(len(captured["stop_features"]), len(inner_stop))
        self.assertEqual(len(captured["outer_fit_features"]), len(outer_fit))
        self.assertEqual(
            len(captured["outer_apply_features"]), len(outer_valid) + len(test)
        )
        self.assertIn("TE_FEE_Q4", captured["fit_features"])
        self.assertIn("TE_FEE_Q4", captured["stop_features"])
        self.assertNotIn("SK_ID_CURR", captured["outer_fit_features"])
        self.assertNotIn("TARGET", captured["outer_fit_features"])
        pd.testing.assert_frame_equal(
            captured["outer_fit_features"],
            changed_capture["outer_fit_features"],
        )
        pd.testing.assert_frame_equal(
            captured["outer_apply_features"],
            changed_capture["outer_apply_features"],
        )

    def test_feature_transform_uses_only_raw_supplied_columns(self):
        raw = loan_rows(12)
        raw_features, _ = experiment._raw_feature_frames(
            raw.iloc[:8], raw.iloc[8:].drop(columns=["TARGET"])
        )

        self.assertNotIn("SK_ID_CURR", raw_features.columns)
        self.assertNotIn("TARGET", raw_features.columns)
        self.assertTrue(set(experiment.baseline.CATEGORICAL_COLUMNS).issubset(
            raw_features.columns
        ))

    def test_catboost_early_stops_then_refits_without_outer_eval_set(self):
        calls = []

        class FakeCatBoost:
            def __init__(self, **params):
                self.params = params
                self.best_iteration_ = 10
                calls.append({"params": params})

            def fit(self, features, labels, **kwargs):
                calls[-1].update(
                    {
                        "rows": len(features),
                        "labels": len(labels),
                        "kwargs": kwargs,
                    }
                )

            def predict_proba(self, features):
                positive = np.full(len(features), 0.25, dtype=np.float64)
                return np.column_stack([1.0 - positive, positive])

            def get_feature_importance(self):
                return np.zeros(len(fit_features.columns), dtype=np.float64)

        fit_features = experiment.baseline.build_rowwise_features(
            loan_rows(20).drop(columns=["TARGET"])
        )
        stop_features = experiment.baseline.build_rowwise_features(
            loan_rows(5, include_target=False)
        )
        outer_fit_features = experiment.baseline.build_rowwise_features(
            loan_rows(30).drop(columns=["TARGET"])
        )
        outer_apply_features = experiment.baseline.build_rowwise_features(
            loan_rows(7, include_target=False)
        )
        fit_labels = np.arange(20) % 2
        outer_labels = np.arange(30) % 2

        with patch.object(experiment, "CatBoostClassifier", FakeCatBoost):
            predictions, iterations, _ = experiment._fit_with_inner_stopping(
                "catboost",
                fit_features,
                fit_labels,
                stop_features,
                np.arange(5) % 2,
                outer_fit_features,
                outer_labels,
                outer_apply_features,
                1,
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("eval_set", calls[0]["kwargs"])
        self.assertIs(calls[0]["kwargs"]["eval_set"][0], stop_features)
        self.assertNotIn("eval_set", calls[1]["kwargs"])
        self.assertEqual(calls[1]["rows"], len(outer_fit_features))
        self.assertEqual(calls[1]["labels"], len(outer_labels))
        self.assertEqual(iterations, 11)
        self.assertEqual(len(predictions), len(outer_apply_features))

    def test_lightgbm_early_stops_then_refits_with_selected_iterations(self):
        dataset_calls = []
        train_calls = []

        class FakeDataset:
            def __init__(self, data, label, reference=None):
                self.data = data
                self.label = label
                self.reference = reference
                dataset_calls.append(self)

        class FakeModel:
            best_iteration = 7

            def current_iteration(self):
                return 7

            def predict(self, data, num_iteration=None):
                return np.full(len(data), 0.4, dtype=np.float64)

            def feature_importance(self, importance_type):
                return np.zeros(len(final_fit.columns), dtype=np.float64)

        def fake_train(params, train_set, **kwargs):
            train_calls.append((train_set, kwargs))
            return FakeModel()

        fit_features = experiment.baseline.build_rowwise_features(
            loan_rows(20).drop(columns=["TARGET"])
        )
        stop_features = experiment.baseline.build_rowwise_features(
            loan_rows(5, include_target=False)
        )
        outer_fit_features = experiment.baseline.build_rowwise_features(
            loan_rows(30).drop(columns=["TARGET"])
        )
        outer_apply_features = experiment.baseline.build_rowwise_features(
            loan_rows(7, include_target=False)
        )
        final_fit, _ = experiment.baseline.make_lightgbm_frames(
            outer_fit_features, outer_apply_features
        )

        with (
            patch.object(experiment.lgb, "Dataset", FakeDataset),
            patch.object(experiment.lgb, "train", side_effect=fake_train),
            patch.object(experiment.lgb, "early_stopping", return_value="stop"),
        ):
            predictions, iterations, _ = experiment._fit_with_inner_stopping(
                "lightgbm",
                fit_features,
                np.arange(20) % 2,
                stop_features,
                np.arange(5) % 2,
                outer_fit_features,
                np.arange(30) % 2,
                outer_apply_features,
                1,
            )

        self.assertEqual(len(train_calls), 2)
        self.assertIn("valid_sets", train_calls[0][1])
        self.assertNotIn("valid_sets", train_calls[1][1])
        self.assertEqual(train_calls[1][1]["num_boost_round"], 7)
        self.assertEqual(len(dataset_calls[-1].label), 30)
        self.assertEqual(iterations, 7)
        self.assertEqual(len(predictions), len(outer_apply_features))


if __name__ == "__main__":
    unittest.main()
