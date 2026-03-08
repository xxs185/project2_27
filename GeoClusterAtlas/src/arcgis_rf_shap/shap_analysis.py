from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    configure_matplotlib,
    read_csv_with_fallback,
    resolve_config_path,
    sanitize_filename,
    write_json,
)

try:
    import shap
except Exception:  # noqa: BLE001
    shap = None


configure_matplotlib()


@dataclass(slots=True)
class ShapAnalysisConfig:
    input_csv: Path
    cluster_csv: Path
    model_bundle: Path
    output_dir: Path = Path("outputs/demo_shap")
    target_col: str = "cluster_target"
    join_col: str = "单元编号"
    use_all_samples: bool = True
    shap_sample_size: int = 0
    background_size: int = 100
    interaction_features: list[str] = field(default_factory=list)


def load_shap_config_from_json(config_path: str | Path) -> ShapAnalysisConfig:
    config_file = Path(config_path)
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = config_file.parent
    return ShapAnalysisConfig(
        input_csv=resolve_config_path(payload["input_csv"], base_dir),
        cluster_csv=resolve_config_path(payload["cluster_csv"], base_dir),
        model_bundle=resolve_config_path(payload["model_bundle"], base_dir),
        output_dir=resolve_config_path(payload.get("output_dir", "../outputs/demo_shap"), base_dir),
        target_col=payload.get("target_col", "cluster_target"),
        join_col=payload.get("join_col", "单元编号"),
        use_all_samples=bool(payload.get("use_all_samples", True)),
        shap_sample_size=int(payload.get("shap_sample_size", 0)),
        background_size=int(payload.get("background_size", 100)),
        interaction_features=list(payload.get("interaction_features", [])),
    )


def _merge_target_labels(config: ShapAnalysisConfig) -> pd.DataFrame:
    feature_frame = read_csv_with_fallback(config.input_csv)
    cluster_frame = read_csv_with_fallback(config.cluster_csv)
    return feature_frame.merge(
        cluster_frame[[config.join_col, config.target_col]],
        on=config.join_col,
        how="left",
        validate="one_to_one",
    )


def _extract_class_shap_values(shap_values: Any, class_index: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_index])

    values_array = np.asarray(shap_values)
    if values_array.ndim == 3:
        return values_array[:, :, class_index]
    raise ValueError("无法解析多分类 SHAP 输出结构。")


def _save_summary_plot(class_values: np.ndarray, feature_frame: pd.DataFrame, output_path: Path) -> None:
    plt.close("all")
    plt.figure()
    shap.summary_plot(class_values, feature_frame, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _save_dependence_plot(feature_name: str, class_values: np.ndarray, feature_frame: pd.DataFrame, output_path: Path) -> None:
    plt.close("all")
    plt.figure()
    shap.dependence_plot(feature_name, class_values, feature_frame, interaction_index="auto", show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _build_explainer(bundle: dict[str, Any], x_eval: pd.DataFrame, config: ShapAnalysisConfig):
    family = bundle["model_family"]
    model = bundle["model"]
    if family in {"rf", "xgboost", "lightgbm"}:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            warnings.filterwarnings("ignore", message="X has feature names.*")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_eval)
        return shap_values

    x_eval_array = x_eval.to_numpy()
    background = x_eval.sample(n=min(config.background_size, len(x_eval)), random_state=42).to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
        warnings.filterwarnings("ignore", message="X has feature names.*")
        explainer = shap.Explainer(model.predict_proba, background)
        try:
            explanation = explainer(x_eval_array, silent=True)
        except TypeError:
            explanation = explainer(x_eval_array)
    return explanation.values


def run_shap_analysis(config: ShapAnalysisConfig) -> dict[str, Any]:
    if shap is None:
        raise RuntimeError("当前环境未安装 shap，无法运行 SHAP 分析。")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = output_dir / "summary_plots"
    interaction_dir = output_dir / "interaction_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    interaction_dir.mkdir(parents=True, exist_ok=True)

    dataframe = _merge_target_labels(config)
    bundle = joblib.load(config.model_bundle)
    feature_cols = list(bundle["feature_cols"])

    eval_ids = set(bundle["test_ids"])
    eval_frame = dataframe[dataframe[config.join_col].isin(eval_ids)].copy()
    if not config.use_all_samples and config.shap_sample_size > 0 and len(eval_frame) > config.shap_sample_size:
        eval_frame = eval_frame.sample(n=config.shap_sample_size, random_state=42)

    x_frame = eval_frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    transformed = bundle["preprocessor"].transform(x_frame)
    x_eval = pd.DataFrame(transformed, columns=feature_cols, index=eval_frame.index)
    shap_values = _build_explainer(bundle, x_eval, config)

    label_encoder = bundle["label_encoder"]
    class_labels = list(label_encoder.classes_)
    interaction_features = config.interaction_features or feature_cols
    summary_records: list[dict[str, Any]] = []

    for class_index, class_label in enumerate(class_labels):
        class_values = _extract_class_shap_values(shap_values, class_index)
        summary_frame = pd.DataFrame(
            {
                "cluster_label": class_label,
                "feature": feature_cols,
                "mean_abs_shap": np.abs(class_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        summary_records.extend(summary_frame.to_dict("records"))

        cluster_name = f"cluster_{class_label}"
        _save_summary_plot(class_values, x_eval, summary_dir / f"{cluster_name}_summary.png")

        cluster_interaction_dir = interaction_dir / cluster_name
        cluster_interaction_dir.mkdir(parents=True, exist_ok=True)
        for feature_name in interaction_features:
            _save_dependence_plot(
                feature_name,
                class_values,
                x_eval,
                cluster_interaction_dir / f"{sanitize_filename(feature_name)}_interaction.png",
            )

    pd.DataFrame(summary_records).to_csv(output_dir / "shap_summary_by_cluster.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "shap_run_summary.json",
        {
            "model_name": bundle["model_name"],
            "model_family": bundle["model_family"],
            "target_col": bundle["target_col"],
            "cluster_count": len(class_labels),
            "feature_count": len(feature_cols),
            "use_all_samples": config.use_all_samples,
            "sample_count": len(x_eval),
        },
    )

    return {
        "output_dir": str(output_dir),
        "cluster_count": len(class_labels),
        "feature_count": len(feature_cols),
    }
