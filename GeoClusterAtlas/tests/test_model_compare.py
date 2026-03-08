from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.clustering import ClusteringConfig, run_clustering
from arcgis_rf_shap.model_compare import ModelCompareConfig, run_model_compare
from arcgis_rf_shap.synthetic_data import SyntheticDataConfig, save_synthetic_dataset


class ModelCompareTests(unittest.TestCase):
    def test_model_compare_creates_best_model_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "demo.csv"
            cluster_dir = temp_path / "cluster_outputs"
            compare_dir = temp_path / "compare_outputs"
            save_synthetic_dataset(input_csv, SyntheticDataConfig(n_samples=320, random_state=11))

            run_clustering(
                ClusteringConfig(
                    input_csv=input_csv,
                    output_dir=cluster_dir,
                    feature_cols=[f"特征{i}" for i in range(1, 11)],
                    k_min=3,
                    k_max=5,
                    fixed_k=4,
                    eval_sample_size=120,
                    random_state=42,
                )
            )

            result = run_model_compare(
                ModelCompareConfig(
                    input_csv=input_csv,
                    cluster_csv=cluster_dir / "best_cluster_assignments.csv",
                    output_dir=compare_dir,
                    feature_cols=[f"特征{i}" for i in range(1, 11)],
                    target_col="cluster_target",
                    join_col="单元编号",
                )
            )

            self.assertIn("best_model_name", result)
            self.assertTrue((compare_dir / "model_compare_results.csv").exists())
            self.assertTrue((compare_dir / "best_model_bundle.joblib").exists())
            self.assertTrue((compare_dir / "best_model_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
