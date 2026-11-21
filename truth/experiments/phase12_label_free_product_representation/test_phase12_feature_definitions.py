from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_phase12 import _paired_bootstrap, make_matrices
from phase12_feature_definitions import (
    PRODUCT_KEYS,
    build_decomposed_features,
    build_product_keys,
    build_selected_interactions,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Cash loans",
                "Revolving loans",
                None,
            ],
            "AMT_CREDIT": [1000.0, 0.0, 1000.0, np.nan],
            "AMT_ANNUITY": [100.0, 100.0, 100.0, np.nan],
            "AMT_GOODS_PRICE": [900.0, 0.0, 900.0, 0.0],
            "REGION_POPULATION_RELATIVE": [0.1, 0.2, 0.3, 0.4],
            "EXT_SOURCE_1": [0.5, np.nan, 0.7, 0.1],
            "EXT_SOURCE_2": [0.4, 0.3, np.nan, 0.2],
            "EXT_SOURCE_3": [0.2, 0.1, 0.8, np.nan],
        }
    )


class ProductKeyFormulaTests(unittest.TestCase):
    def test_exact_precision_and_cross_formulas(self) -> None:
        keys = build_product_keys(sample_frame())
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q3"], "100")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q4"], "1000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q5"], "10000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_P4"], "-305")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_P5"], "-3050")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_AG3"], "111")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_KQ3"], "17|100")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_KQ4"], "17|1000")

    def test_missing_and_invalid_denominators_are_explicit(self) -> None:
        keys = build_product_keys(sample_frame())
        self.assertEqual(keys.loc[1, "PRODUCT_KEY_Q3"], "MISSING")
        self.assertEqual(keys.loc[1, "PRODUCT_KEY_KQ3"], "NONCASH_OR_INVALID")
        self.assertEqual(keys.loc[2, "PRODUCT_KEY_KQ3"], "NONCASH_OR_INVALID")
        self.assertEqual(keys.loc[3, "PRODUCT_KEY_P4"], "MISSING")
        self.assertEqual(keys.loc[3, "PRODUCT_KEY_AG3"], "MISSING")

    def test_target_is_rejected(self) -> None:
        frame = sample_frame().assign(TARGET=[0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_product_keys(frame)
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_decomposed_features(frame)
        with self.assertRaisesRegex(ValueError, "TARGET"):
            build_selected_interactions(frame)

    def test_numeric_components_and_interactions_are_label_free(self) -> None:
        frame = sample_frame()
        components = build_decomposed_features(frame)
        interactions = build_selected_interactions(frame)
        self.assertAlmostEqual(
            components.loc[0, "KEY_COMPONENT_ANNUITY_TO_CREDIT"], 0.1
        )
        self.assertTrue(
            np.isnan(
                components.loc[2, "KEY_COMPONENT_FINANCING_COORDINATE"]
            )
        )
        self.assertAlmostEqual(
            interactions.loc[0, "INTERACTION_CREDIT_X_REGION_POPULATION"],
            100.0,
        )
        self.assertTrue(
            np.isnan(interactions.loc[1, "INTERACTION_EXT_SOURCE_1_X_2"])
        )

    def test_frequency_encoder_is_fit_only_and_unknowns_are_safe(self) -> None:
        fit_keys = pd.DataFrame(
            {key: ["common", "common", "rare"] for key in PRODUCT_KEYS}
        )
        apply_keys = pd.DataFrame({key: ["unseen"] for key in PRODUCT_KEYS})
        numeric_fit = pd.DataFrame({"n": [1.0, 2.0, 3.0]})
        numeric_apply = pd.DataFrame({"n": [4.0]})
        empty_fit = pd.DataFrame(index=range(3))
        empty_apply = pd.DataFrame(index=range(1))
        fit_matrix, apply_matrix, details = make_matrices(
            numeric_fit,
            numeric_apply,
            fit_keys,
            apply_keys,
            empty_fit,
            empty_apply,
            empty_fit,
            empty_apply,
            "raw_plus_keys_frequency_ohe",
        )
        self.assertEqual(fit_matrix.shape[0], 3)
        self.assertEqual(apply_matrix.shape[0], 1)
        self.assertEqual(
            details["encoded_key_feature_count"], 2 * len(PRODUCT_KEYS)
        )
        frequency_start = 1 + details["encoded_key_feature_count"]
        np.testing.assert_array_equal(
            apply_matrix[
                :, frequency_start : frequency_start + len(PRODUCT_KEYS)
            ].toarray(),
            np.zeros((1, len(PRODUCT_KEYS)), dtype=np.float32),
        )
        self.assertTrue(np.isfinite(apply_matrix.data).all())

    def test_paired_bootstrap_matches_direct_auc_resampling(self) -> None:
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        baseline = np.array([0.1, 0.8, 0.2, 0.4, 0.4, 0.9, 0.3, 0.7])
        candidate = np.array([0.2, 0.9, 0.1, 0.5, 0.4, 0.8, 0.3, 0.6])
        repetitions = 40
        seed = 73
        actual = _paired_bootstrap(
            y,
            baseline,
            {"candidate": candidate},
            seed,
            repetitions,
        ).iloc[0]
        rng = np.random.default_rng(seed)
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        direct = []
        for _ in range(repetitions):
            sample = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            direct.append(
                roc_auc_score(y[sample], candidate[sample])
                - roc_auc_score(y[sample], baseline[sample])
            )
        self.assertAlmostEqual(
            actual["paired_stratified_bootstrap_95_low"],
            np.quantile(direct, 0.025),
        )
        self.assertAlmostEqual(
            actual["paired_stratified_bootstrap_95_high"],
            np.quantile(direct, 0.975),
        )


if __name__ == "__main__":
    unittest.main()
