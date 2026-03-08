from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.clustering import load_clustering_config_from_json, run_clustering
from arcgis_rf_shap.model_compare import load_model_compare_config_from_json, run_model_compare
from arcgis_rf_shap.shap_analysis import load_shap_config_from_json, run_shap_analysis
from arcgis_rf_shap.synthetic_data import SyntheticDataConfig, save_synthetic_dataset


def main() -> None:
    sample_csv = ROOT / "data" / "sample" / "demo_geodata.csv"
    if not sample_csv.exists():
        save_synthetic_dataset(sample_csv, SyntheticDataConfig(n_samples=2000, random_state=42))

    clustering_results = run_clustering(load_clustering_config_from_json(ROOT / "configs" / "clustering_config.json"))
    compare_results = run_model_compare(load_model_compare_config_from_json(ROOT / "configs" / "model_compare_config.json"))
    shap_results = run_shap_analysis(load_shap_config_from_json(ROOT / "configs" / "shap_config.json"))

    print(f"聚类流程已完成，输出目录：{clustering_results['output_dir']}")
    print(f"模型对比已完成，输出目录：{compare_results['output_dir']}")
    print(f"SHAP 分析已完成，输出目录：{shap_results['output_dir']}")


if __name__ == "__main__":
    main()
