from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase20_segmentation_study as phase20


class Phase20RegistryTests(unittest.TestCase):
    def test_registry_has_exact_experiment_and_placebo_counts(self) -> None:
        registry = phase20.build_registry()
        real = registry.loc[registry["kind"] == "real"]
        placebos = registry.loc[registry["kind"] == "placebo"]
        self.assertEqual(len(real), 150)
        self.assertEqual(real["experiment_id"].nunique(), 150)
        self.assertEqual(len(placebos), 30)
        self.assertEqual(real.groupby("family_id").size().to_dict(), {
            family["id"]: 10 for family in phase20.FAMILIES
        })
        self.assertTrue(registry["device"].eq("GPU").all())
        self.assertTrue(registry["one_hot_max_size"].eq(256).all())

    def test_configs_are_not_seed_only_variants(self) -> None:
        configs = pd.DataFrame(phase20.CONFIGS)
        self.assertEqual(configs["id"].nunique(), 10)
        distinct = configs[
            [
                "model",
                "feature_set",
                "depth",
                "iterations",
                "learning_rate",
                "l2_leaf_reg",
            ]
        ].drop_duplicates()
        self.assertEqual(len(distinct), 10)

    def test_routes_are_feature_only_and_income_cut_is_fit_local(self) -> None:
        registry = phase20.build_registry()
        routes = registry.loc[registry["kind"] == "real", "route_definition"]
        self.assertFalse(routes.str.contains("TARGET", case=False).any())

        raw = pd.DataFrame(
            {
                "AMT_INCOME_TOTAL": [10.0, 20.0, 30.0, 40.0],
                "NAME_CONTRACT_TYPE": [
                    "Cash loans",
                    "Revolving loans",
                    "Cash loans",
                    "Revolving loans",
                ],
            }
        )
        median_family = next(
            family for family in phase20.FAMILIES if family["id"] == "F06"
        )
        route = phase20.make_route(median_family, raw, np.array([0, 1]))
        np.testing.assert_array_equal(route, np.array([False, True, True, True]))

    def test_cluster_families_are_explicitly_gpu_blocked(self) -> None:
        registry = phase20.build_registry()
        blocked = registry.loc[
            registry["family_id"].isin(["F13", "F14"])
            & (registry["kind"] == "real")
        ]
        self.assertEqual(len(blocked), 20)
        self.assertTrue(
            blocked["status"].eq("GPU_BLOCKED_NO_CUDA_CLUSTERING_LIBRARY").all()
        )

    def test_constant_free_key_builder_uses_only_direct_ratios(self) -> None:
        raw = pd.DataFrame(
            {
                "AMT_CREDIT": [1000.0, 0.0],
                "AMT_ANNUITY": [50.0, 30.0],
                "AMT_GOODS_PRICE": [900.0, 400.0],
            }
        )
        keys = phase20.build_constant_free_keys(raw)
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q3"], "50")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q4"], "500")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_Q5"], "5000")
        self.assertEqual(keys.loc[0, "PRODUCT_KEY_AG3"], "56")
        self.assertEqual(keys.loc[1, "PRODUCT_KEY_Q3"], "MISSING")

    def test_exact_route_uses_matching_expert_for_every_row(self) -> None:
        group_a = np.array([True, False, True, False])
        a = np.array([0.1, np.nan, 0.3, np.nan])
        b = np.array([np.nan, 0.2, np.nan, 0.4])
        np.testing.assert_array_equal(
            phase20.exact_route_predictions(a, b, group_a),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )

    def test_id_parity_placebo_route_is_detected(self) -> None:
        row = pd.Series(
            {"route_definition": "SK_ID_CURR parity; placebo diagnostic only"}
        )
        ids = np.array([101, 102, 103, 104])
        np.testing.assert_array_equal(
            phase20.make_placebo(row, ids, len(ids)),
            np.array([False, True, False, True]),
        )


if __name__ == "__main__":
    unittest.main()
