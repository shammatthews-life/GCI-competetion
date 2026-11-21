import ast
import tempfile
import unittest
from pathlib import Path

from src.audit_python_code import _literal_paths


class LiteralPathTests(unittest.TestCase):
    def test_extracts_only_static_reader_paths(self) -> None:
        tree = ast.parse(
            """
import numpy as np
import pandas as pd
pd.read_csv("train.csv")
np.load(f"model_{fold}.npz")
"""
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _literal_paths(Path(temporary_directory), (), tree)

        self.assertEqual([item["literal"] for item in paths], ["train.csv"])

    def test_resolves_static_path_join(self) -> None:
        tree = ast.parse(
            """
import os
import pandas as pd
pd.read_csv(os.path.join("data", "train.csv"))
"""
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _literal_paths(Path(temporary_directory), (), tree)

        self.assertEqual(len(paths), 1)
        self.assertEqual(Path(paths[0]["literal"]), Path("data") / "train.csv")

    def test_ignores_write_only_open(self) -> None:
        tree = ast.parse('open("result.csv", "w")')

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = _literal_paths(Path(temporary_directory), (), tree)

        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
