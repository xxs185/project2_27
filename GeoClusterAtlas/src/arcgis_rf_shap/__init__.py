"""ArcGIS + clustering + model comparison + SHAP workflow template."""

from .clustering import ClusteringConfig, run_clustering
from .model_compare import ModelCompareConfig, run_model_compare
from .shap_analysis import ShapAnalysisConfig, run_shap_analysis
from .synthetic_data import SyntheticDataConfig, generate_synthetic_dataset, save_synthetic_dataset

__all__ = [
    "ClusteringConfig",
    "ModelCompareConfig",
    "ShapAnalysisConfig",
    "SyntheticDataConfig",
    "generate_synthetic_dataset",
    "run_clustering",
    "run_model_compare",
    "run_shap_analysis",
    "save_synthetic_dataset",
]

__version__ = "0.2.0"
