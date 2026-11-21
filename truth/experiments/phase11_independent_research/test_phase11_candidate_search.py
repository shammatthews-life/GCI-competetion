from __future__ import annotations

import unittest

import numpy as np

from phase11_candidate_search import (
    _candidate,
    crossfit_smoothed_target_rate,
    significant_integer_codes,
)


class CandidateSearchTests(unittest.TestCase):
    def test_missing_values_have_a_real_category(self) -> None:
        source = np.array([np.nan, 1.0, np.inf, 1.0])
        candidate = _candidate(
            "test", (significant_integer_codes(source, 3),), "test"
        )

        self.assertTrue(np.all(candidate.category_codes >= 0))
        self.assertEqual(candidate.category_codes[0], candidate.category_codes[2])
        self.assertNotEqual(candidate.category_codes[0], candidate.category_codes[1])
        self.assertEqual(candidate.category_codes[1], candidate.category_codes[3])

    def test_missing_category_is_not_merged_with_valid_numeric_code(self) -> None:
        source = np.array([np.nan, 0.0, 1.0])
        candidate = _candidate(
            "test", (significant_integer_codes(source, 4),), "test"
        )
        self.assertEqual(candidate.unique_categories, 3)

    def test_multi_field_keys_are_nonnegative_and_deterministic(self) -> None:
        parts = (
            np.array([-1, 0, 0, -1], dtype=np.int32),
            np.array([3, 3, 4, 3], dtype=np.int32),
        )
        first = _candidate("cross", parts, "test")
        second = _candidate("cross", parts, "test")
        self.assertTrue(np.all(first.category_codes >= 0))
        self.assertEqual(first.unique_categories, 3)
        np.testing.assert_array_equal(first.category_codes, second.category_codes)

    def test_crossfit_encoding_uses_fit_rows_only(self) -> None:
        codes = np.array([0, 0, 1, 1], dtype=np.int32)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        changed_held_out_labels = np.array([0, 1, 1, 0], dtype=np.int8)
        folds = (
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        )

        encoded = crossfit_smoothed_target_rate(codes, y, folds, smoothing=1.0)
        changed = crossfit_smoothed_target_rate(
            codes, changed_held_out_labels, folds, smoothing=1.0
        )

        np.testing.assert_array_equal(encoded, changed)


if __name__ == "__main__":
    unittest.main()
