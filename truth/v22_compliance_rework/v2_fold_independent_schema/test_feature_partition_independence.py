from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_clean_v2 as pipeline


class FeaturePartitionIndependenceTests(unittest.TestCase):
    def test_test_covariate_changes_do_not_change_train_features(self) -> None:
        train = pd.read_csv(
            r"D:\gci\inpuy\train.csv", nrows=500
        ).drop(columns=[pipeline.TARGET])
        test = pd.read_csv(r"D:\gci\inpuy\test.csv", nrows=200)
        original_train, original_train_cat, _, _ = (
            pipeline.make_feature_partitions(train, test)
        )

        perturbed_test = test.copy()
        for column in perturbed_test.columns:
            if column == pipeline.ID:
                continue
            perturbed_test[column] = perturbed_test[column].iloc[::-1].to_numpy()
        if not any(
            np.any(test[column].to_numpy() != perturbed_test[column].to_numpy())
            for column in test.columns
            if column != pipeline.ID
        ):
            self.fail("Test perturbation did not change any covariate")

        repeated_train, repeated_train_cat, _, _ = (
            pipeline.make_feature_partitions(train, perturbed_test)
        )
        pd.testing.assert_frame_equal(original_train, repeated_train)
        pd.testing.assert_frame_equal(original_train_cat, repeated_train_cat)


if __name__ == "__main__":
    unittest.main()
