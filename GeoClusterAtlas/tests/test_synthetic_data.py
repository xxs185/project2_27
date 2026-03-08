from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.synthetic_data import SyntheticDataConfig, generate_synthetic_dataset


class SyntheticDataTests(unittest.TestCase):
    def test_generate_default_dataset_shape(self) -> None:
        dataframe = generate_synthetic_dataset(SyntheticDataConfig())
        feature_columns = [column for column in dataframe.columns if str(column).startswith("特征")]

        self.assertEqual(len(dataframe), 2000)
        self.assertEqual(len(feature_columns), 10)
        self.assertIn("目标值", dataframe.columns)
        self.assertIn("单元编号", dataframe.columns)


if __name__ == "__main__":
    unittest.main()
