from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .common import configure_matplotlib, read_csv_with_fallback, resolve_feature_columns

try:
    import shap
except Exception:  # noqa: BLE001
    shap = None


configure_matplotlib()


@dataclass(slots=True)
class WorkflowConfig:
    input_csv: Path
    output_dir: Path = Path("outputs/demo_run")
    target_col: str = "目标值"
    feature_cols: list[str] = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int | None = 12
    min_samples_leaf: int = 2
    shap_sample_size: int = 400
    cluster_csv: Path | None = None
    cluster_join_col: str = "单元编号"
    cluster_feature_cols: list[str] = field(default_factory=list)
    cluster_encode: str = "onehot"
    group_explain_col: str | None = None


def load_config_from_json(config_path: str | Path) -> WorkflowConfig:
    config_file = Path(config_path)
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = config_file.parent
    return WorkflowConfig(
        input_csv=_resolve_config_path(payload["input_csv"], base_dir),
        output_dir=_resolve_config_path(payload.get("output_dir", "../outputs/demo_run"), base_dir),
        target_col=payload.get("target_col", "目标值"),
        feature_cols=list(payload.get("feature_cols", [])),
        test_size=float(payload.get("test_size", 0.2)),
        random_state=int(payload.get("random_state", 42)),
        n_estimators=int(payload.get("n_estimators", 300)),
        max_depth=payload.get("max_depth", 12),
        min_samples_leaf=int(payload.get("min_samples_leaf", 2)),
        shap_sample_size=int(payload.get("shap_sample_size", 400)),
        cluster_csv=_resolve_optional_config_path(payload.get("cluster_csv"), base_dir),
        cluster_join_col=payload.get("cluster_join_col", "单元编号"),
        cluster_feature_cols=list(payload.get("cluster_feature_cols", [])),
        cluster_encode=payload.get("cluster_encode", "onehot"),
        group_explain_col=payload.get("group_explain_col"),
    )


def _resolve_config_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_config_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    return _resolve_config_path(path_value, base_dir)


def _serialize_config(config: WorkflowConfig, input_path: Path, output_dir: Path, feature_cols: list[str]) -> dict[str, Any]:
    config_payload = asdict(config)
    serialized: dict[str, Any] = {}
    for key, value in config_payload.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    serialized["input_csv"] = str(input_path)
    serialized["output_dir"] = str(output_dir)
    serialized["feature_cols"] = feature_cols
    if config.cluster_csv is not None:
        serialized["cluster_csv"] = str(config.cluster_csv)
    return serialized


def _save_bar_plot(dataframe: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path) -> None:
    sorted_frame = dataframe.sort_values(by=x_col, ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(sorted_frame[y_col], sorted_frame[x_col], color="#3B82F6")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _build_shap_summary(
    model: RandomForestRegressor,
    feature_frame: pd.DataFrame,
    config: WorkflowConfig,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if shap is None:
        return None, None

    sample_size = min(config.shap_sample_size, len(feature_frame))
    sample_frame = feature_frame.sample(n=sample_size, random_state=config.random_state)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample_frame)
    shap_values = np.asarray(explanation.values if hasattr(explanation, "values") else explanation)
    if shap_values.ndim == 3:
        shap_values = shap_values.mean(axis=-1)

    shap_frame = pd.DataFrame(shap_values, columns=sample_frame.columns, index=sample_frame.index)
    summary_frame = pd.DataFrame(
        {
            "feature": sample_frame.columns,
            "mean_abs_shap": shap_frame.abs().mean(axis=0).to_numpy(),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    return summary_frame, shap_frame


def _merge_cluster_assignments(dataframe: pd.DataFrame, config: WorkflowConfig) -> pd.DataFrame:
    if config.cluster_csv is None:
        return dataframe

    cluster_frame = read_csv_with_fallback(config.cluster_csv)
    join_col = config.cluster_join_col
    if join_col not in dataframe.columns:
        raise ValueError(f"原始数据缺少聚类连接列：{join_col}")
    if join_col not in cluster_frame.columns:
        raise ValueError(f"聚类结果缺少连接列：{join_col}")

    duplicate_columns = [column for column in cluster_frame.columns if column != join_col and column in dataframe.columns]
    if duplicate_columns:
        cluster_frame = cluster_frame.drop(columns=duplicate_columns)

    merged = dataframe.merge(cluster_frame, on=join_col, how="left", validate="one_to_one")
    return merged


def _prepare_feature_frame(dataframe: pd.DataFrame, config: WorkflowConfig) -> tuple[pd.DataFrame, list[str]]:
    base_feature_cols = resolve_feature_columns(dataframe, config.feature_cols)
    feature_frame = dataframe[base_feature_cols].apply(pd.to_numeric, errors="coerce")

    cluster_feature_names: list[str] = []
    if config.cluster_feature_cols:
        missing_columns = [column for column in config.cluster_feature_cols if column not in dataframe.columns]
        if missing_columns:
            joined = "、".join(missing_columns)
            raise ValueError(f"缺少聚类特征列：{joined}")

        cluster_source = dataframe[config.cluster_feature_cols].copy()
        if config.cluster_encode == "numeric":
            cluster_feature_frame = cluster_source.apply(pd.to_numeric, errors="coerce")
        else:
            cluster_feature_frame = pd.get_dummies(
                cluster_source.astype("string"),
                prefix=config.cluster_feature_cols,
                dtype=int,
            )
        cluster_feature_names = cluster_feature_frame.columns.tolist()
        feature_frame = pd.concat([feature_frame, cluster_feature_frame], axis=1)

    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
    return feature_frame, base_feature_cols + cluster_feature_names


def _build_group_metrics(prediction_frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_value, group_frame in prediction_frame.groupby(group_col):
        actual = group_frame["真实值"].to_numpy()
        predicted = group_frame["预测值"].to_numpy()
        records.append(
            {
                group_col: group_value,
                "sample_count": int(len(group_frame)),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r2": float(r2_score(actual, predicted)) if len(group_frame) > 1 else np.nan,
            }
        )
    return pd.DataFrame(records).sort_values(group_col)


def _build_group_shap_summary(shap_frame: pd.DataFrame, group_series: pd.Series, group_col: str) -> pd.DataFrame:
    grouped_frame = shap_frame.copy()
    grouped_frame[group_col] = group_series.loc[shap_frame.index].to_numpy()
    records: list[dict[str, Any]] = []
    for group_value, group_data in grouped_frame.groupby(group_col):
        mean_abs_values = group_data.drop(columns=[group_col]).abs().mean(axis=0)
        for feature_name, shap_value in mean_abs_values.items():
            records.append(
                {
                    group_col: group_value,
                    "feature": feature_name,
                    "mean_abs_shap": float(shap_value),
                }
            )
    return pd.DataFrame(records).sort_values([group_col, "mean_abs_shap"], ascending=[True, False])


def run_pipeline(config: WorkflowConfig) -> dict[str, Any]:
    input_path = Path(config.input_csv)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = read_csv_with_fallback(input_path)
    dataframe = _merge_cluster_assignments(dataframe, config)
    feature_frame, feature_cols = _prepare_feature_frame(dataframe, config)
    if config.target_col not in dataframe.columns:
        raise ValueError(f"目标列不存在：{config.target_col}")
    target_series = dataframe[config.target_col].copy()
    group_series = None
    if config.group_explain_col:
        if config.group_explain_col not in dataframe.columns:
            raise ValueError(f"分组解释列不存在：{config.group_explain_col}")
        group_series = dataframe[config.group_explain_col].copy()

    all_index = dataframe.index.to_numpy()
    train_index, test_index = train_test_split(
        all_index,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    x_train = feature_frame.loc[train_index]
    x_test = feature_frame.loc[test_index]
    y_train = target_series.loc[train_index]
    y_test = target_series.loc[test_index]

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    prediction = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, prediction)),
        "mae": float(mean_absolute_error(y_test, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
        "sample_count": int(len(dataframe)),
        "feature_count": int(len(feature_cols)),
    }

    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    prediction_frame = pd.DataFrame(
        {
            "样本索引": x_test.index,
            "真实值": y_test.to_numpy(),
            "预测值": prediction,
        }
    ).sort_values("样本索引")

    if group_series is not None:
        prediction_frame[config.group_explain_col] = group_series.loc[prediction_frame["样本索引"]].to_numpy()

    shap_summary, shap_frame = _build_shap_summary(model, x_test, config)
    group_metrics = None
    group_shap_summary = None
    if group_series is not None:
        group_metrics = _build_group_metrics(prediction_frame, config.group_explain_col)
        if shap_frame is not None:
            group_shap_summary = _build_group_shap_summary(shap_frame, group_series, config.group_explain_col)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(_serialize_config(config, input_path, output_dir, feature_cols), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    joblib.dump(model, output_dir / "rf_model.joblib")

    _save_bar_plot(feature_importance, "importance", "feature", "RF Feature Importance", output_dir / "feature_importance.png")

    if shap_summary is not None:
        shap_summary.to_csv(output_dir / "shap_summary.csv", index=False, encoding="utf-8-sig")
        _save_bar_plot(shap_summary, "mean_abs_shap", "feature", "SHAP Importance", output_dir / "shap_importance.png")
    else:
        (output_dir / "shap_status.txt").write_text("当前环境未安装 shap，已跳过 SHAP 结果导出。", encoding="utf-8")

    if group_metrics is not None:
        group_metrics.to_csv(output_dir / "metrics_by_group.csv", index=False, encoding="utf-8-sig")
    if group_shap_summary is not None:
        group_shap_summary.to_csv(output_dir / "shap_summary_by_group.csv", index=False, encoding="utf-8-sig")

    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "shap_summary": shap_summary,
        "group_metrics": group_metrics,
        "group_shap_summary": group_shap_summary,
    }
