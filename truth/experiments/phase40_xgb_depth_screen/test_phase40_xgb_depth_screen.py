from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase40_xgb_depth_screen as phase40


class Phase40ConfigurationTests(unittest.TestCase):
    def test_depth_screen_changes_tree_complexity_and_regularization(self) -> None:
        shallow = phase40.CONFIGS["depth3_child20"]
        deep = phase40.CONFIGS["depth5_child60"]
        self.assertLess(shallow["max_depth"], deep["max_depth"])
        self.assertLess(shallow["min_child_weight"], deep["min_child_weight"])
        self.assertNotEqual(shallow["learning_rate"], deep["learning_rate"])
        self.assertNotEqual(shallow["reg_lambda"], deep["reg_lambda"])


if __name__ == "__main__":
    unittest.main()
