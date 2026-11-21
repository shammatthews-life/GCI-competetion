import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_core_v1 import (  # noqa: E402
    CAT_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    _lgb_frame,
    _xgb_matrix,
    make_features,
)


def small_frame(size=8):
    data = {
        "SK_ID_CURR": np.arange(size),
        "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"] * (size // 2),
        "CODE_GENDER": ["F", "M"] * (size // 2),
        "NAME_INCOME_TYPE": ["Working", "Pensioner"] * (size // 2),
        "NAME_EDUCATION_TYPE": ["Higher education", "Secondary"] * (size // 2),
        "NAME_FAMILY_STATUS": ["Married", "Single"] * (size // 2),
        "NAME_HOUSING_TYPE": ["House", "Rented"] * (size // 2),
        "REGION_RATING_CLIENT": [1, 2] * (size // 2),
        "OCCUPATION_TYPE": ["Staff", "Other"] * (size // 2),
        "ORGANIZATION_TYPE": ["School", "Business"] * (size // 2),
        "CNT_FAM_MEMBERS": [2.0] * size,
        "DAYS_EMPLOYED": [-500, 365243] * (size // 2),
        "DAYS_BIRTH": [-10000] * size,
        "DAYS_REGISTRATION": [-2000] * size,
        "DAYS_ID_PUBLISH": [-3000] * size,
        "DAYS_LAST_PHONE_CHANGE": [-500] * size,
        "AMT_INCOME_TOTAL": [100000.0] * size,
        "AMT_CREDIT": [200000.0] * size,
        "AMT_ANNUITY": [10000.0] * size,
        "AMT_GOODS_PRICE": [180000.0] * size,
        "EXT_SOURCE_1": [0.2, np.nan] * (size // 2),
        "EXT_SOURCE_2": [0.4] * size,
        "EXT_SOURCE_3": [0.6] * size,
        "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0] * size,
        "AMT_REQ_CREDIT_BUREAU_MON": [1.0] * size,
        "AMT_REQ_CREDIT_BUREAU_QRT": [0.0] * size,
        "AMT_REQ_CREDIT_BUREAU_YEAR": [2.0] * size,
        "TARGET": [0, 1] * (size // 2),
    }
    return pd.DataFrame(data)


class PipelineCoreTests(unittest.TestCase):
    def test_rowwise_features_are_deterministic_and_do_not_mutate_input(self):
        frame = small_frame()
        original = frame.copy(deep=True)
        first = make_features(frame, "rowwise_v1")
        second = make_features(frame, "rowwise_v1")
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(first, second)
        self.assertNotIn(ID_COLUMN, first.columns)
        self.assertNotIn(TARGET_COLUMN, first.columns)
        self.assertTrue(np.isfinite(first.select_dtypes(include=[np.number]).to_numpy()[~np.isnan(
            first.select_dtypes(include=[np.number]).to_numpy()
        )]).all())
        self.assertEqual(first["DAYS_EMPLOYED_CLEAN"].isna().sum(), 4)

    def test_category_vocabulary_is_fit_only_on_training_rows(self):
        frame = small_frame()
        x = make_features(frame, "raw")
        query_input = x.iloc[6:].copy()
        query_input.loc[query_input.index[0], "ORGANIZATION_TYPE"] = "Unseen organization"
        fit, query = _lgb_frame(x.iloc[:6], query_input)
        for column in CAT_COLUMNS:
            self.assertIsInstance(fit[column].dtype, pd.CategoricalDtype)
            self.assertEqual(len(fit[column].cat.categories), len(pd.unique(x.iloc[:6][column])))
            self.assertTrue(query[column].cat.categories.equals(fit[column].cat.categories))
        self.assertTrue(pd.isna(query["ORGANIZATION_TYPE"].iloc[0]))

    def test_xgboost_one_hot_unknown_categories_and_row_alignment(self):
        frame = small_frame()
        x = make_features(frame, "raw")
        train = x.iloc[:6].copy()
        query = x.iloc[6:].copy()
        query.loc[query.index[0], "ORGANIZATION_TYPE"] = "Unseen organization"
        matrix_fit, matrix_query = _xgb_matrix(train, query)
        self.assertEqual(matrix_fit.shape[0], len(train))
        self.assertEqual(matrix_query.shape[0], len(query))
        self.assertEqual(matrix_fit.shape[1], matrix_query.shape[1])


if __name__ == "__main__":
    unittest.main()
