from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_phase23_auc_feature_research as phase23


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "OWN_CAR_AGE": [1.0, np.nan, 3.0, np.nan, 5.0, 1.0, 2.0, 4.0, 8.0],
            "CNT_FAM_MEMBERS": [1, 2, 1, 3, 2, 1, 2, 2, 1],
            "EXT_SOURCE_1": [0.1, 0.2, np.nan, 0.4, 0.5, 0.1, 0.3, 0.4, 0.5],
            "EXT_SOURCE_2": [0.2, np.nan, 0.3, 0.5, 0.4, 0.2, 0.4, 0.6, 0.8],
            "EXT_SOURCE_3": [0.3, 0.4, np.nan, 0.6, 0.7, 0.3, 0.5, 0.8, 0.9],
            "AMT_REQ_CREDIT_BUREAU_HOUR": [np.nan] * 4 + [0, 1, 0, 2, 1],
            "AMT_REQ_CREDIT_BUREAU_MON": [np.nan] * 4 + [0, 1, 1, 2, 1],
            "AMT_REQ_CREDIT_BUREAU_QRT": [np.nan] * 4 + [0, 0, 1, 2, 1],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [np.nan] * 4 + [0, 1, 1, 1, 1],
            "DAYS_EMPLOYED": [100, 200, 365243, 400, 500, 300, 450, 700, 800],
            "DAYS_BIRTH": [-10000, -12000, -14000, -16000, -18000, -20000, -22000, -24000, -26000],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"] * 4 + ["Cash loans"],
            "AMT_CREDIT": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
            "AMT_ANNUITY": [100, 130, 170, 190, 220, 260, 280, 310, 340],
            "AMT_GOODS_PRICE": [900, 1400, 1900, 2300, 2800, 3300, 3800, 4200, 4800],
            "AMT_INCOME_TOTAL": [800, 900, 1100, 1200, 1400, 1600, 1700, 1900, 2100],
            "OCCUPATION_TYPE": ["A", "A", "B", "B", "C", "A", "B", "C", "D"],
            "NAME_EDUCATION_TYPE": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X"],
            "NAME_INCOME_TYPE": ["Employee", "Pensioner"] * 4 + ["Employee"],
            "ORGANIZATION_TYPE": ["Org1", "Org1", "Org2", "Org2", "Org3", "Org1", "Org2", "Org3", "Org4"],
            "NAME_FAMILY_STATUS": ["Single", "Married"] * 4 + ["Single"],
            "NAME_HOUSING_TYPE": ["House", "Rent"] * 4 + ["House"],
        }
    )


class Phase23FeatureTests(unittest.TestCase):
    def test_application_process_features_are_aligned_and_have_unseen_pattern_flag(self) -> None:
        raw = fixture()
        raw.loc[8, "CNT_FAM_MEMBERS"] = np.nan
        fit = np.arange(6)
        apply = np.array([6, 7, 8])
        train_features, apply_features = phase23.application_process_features(
            raw, fit, apply
        )
        self.assertEqual(train_features.columns.tolist(), apply_features.columns.tolist())
        self.assertIn("P23_MISSING_PROFILE_UNSEEN", train_features)
        self.assertEqual(int(apply_features["P23_MISSING_PROFILE_UNSEEN"].sum()), 1)
        self.assertTrue(
            all(
                column.startswith("P23_")
                for column in train_features.columns
            )
        )

    def test_external_disagreement_handles_one_or_no_available_scores(self) -> None:
        raw = fixture()
        output = phase23.external_disagreement_features(raw)
        self.assertTrue(np.isnan(output.loc[2, "P23_EXT_STD"]))
        self.assertTrue(np.isnan(output.loc[2, "P23_EXT_RANGE"]))
        self.assertEqual(output.loc[2, "P23_EXT_AVAILABLE_COUNT"], 1)
        self.assertTrue(np.isfinite(output.loc[0, "P23_EXT_STD"]))

    def test_financial_residuals_are_cross_fitted_and_finite(self) -> None:
        raw = fixture()
        fit = np.arange(6)
        apply = np.array([6, 7, 8])
        fit_features, apply_features = phase23.financial_residual_features(
            raw, fit, apply, seed=7
        )
        self.assertEqual(fit_features.shape, (6, 2))
        self.assertEqual(apply_features.shape, (3, 2))
        self.assertTrue(np.isfinite(fit_features.to_numpy()).all())
        self.assertTrue(np.isfinite(apply_features.to_numpy()).all())

    def test_applicant_frequency_features_are_fold_local_and_finite(self) -> None:
        raw = fixture()
        fit = np.arange(6)
        apply = np.array([6, 7, 8])
        fit_features, apply_features = phase23.applicant_consistency_features(
            raw, fit, apply, seed=11
        )
        self.assertEqual(fit_features.shape, (6, 8))
        self.assertEqual(apply_features.shape, (3, 8))
        self.assertTrue(np.isfinite(fit_features.to_numpy()).all())
        self.assertTrue(np.isfinite(apply_features.to_numpy()).all())
        self.assertTrue((fit_features.to_numpy() >= 0).all())


if __name__ == "__main__":
    unittest.main()
