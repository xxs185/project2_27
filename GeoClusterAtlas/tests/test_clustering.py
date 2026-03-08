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
from arcgis_rf_shap.synthetic_data import SyntheticDataConfig, save_synthetic_dataset


class ClusteringTests(unittest.TestCase):
    def test_clustering_creates_comparison_and_visual_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_csv = temp_path / "demo.csv"
            output_dir = temp_path / "cluster_outputs"
            save_synthetic_dataset(input_csv, SyntheticDataConfig(n_samples=260, random_state=9))

            result = run_clustering(
                ClusteringConfig(
                    input_csv=input_csv,
                    output_dir=output_dir,
                    feature_cols=[f"特征{i}" for i in range(1, 11)],
                    k_min=3,
                    k_max=5,
                    fixed_k=4,
                    eval_sample_size=120,
                    random_state=42,
                )
            )

            self.assertIn("best_algorithm", result)
            self.assertTrue((output_dir / "kmeans_selection_metrics.csv").exists())
            self.assertTrue((output_dir / "cluster_algorithm_metrics.csv").exists())
            self.assertTrue((output_dir / "best_cluster_assignments.csv").exists())
            self.assertTrue((output_dir / "pca_2d.png").exists())
            self.assertTrue((output_dir / "tsne_2d.png").exists())
            self.assertTrue((output_dir / "pca_3d_visualization.html").exists())


if __name__ == "__main__":
    unittest.main()
