from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans, MeanShift, MiniBatchKMeans, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import (
    configure_matplotlib,
    read_csv_with_fallback,
    resolve_config_path,
    resolve_feature_columns,
    write_json,
)


configure_matplotlib()


@dataclass(slots=True)
class ClusteringConfig:
    input_csv: Path
    output_dir: Path = Path("outputs/demo_clustering")
    feature_cols: list[str] = field(default_factory=list)
    k_min: int = 3
    k_max: int = 6
    fixed_k: int = 4
    eval_sample_size: int = 600
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300
    dbscan_eps: float = 1.15
    dbscan_min_samples: int = 12
    density_quantile: float = 0.2


def load_clustering_config_from_json(config_path: str | Path) -> ClusteringConfig:
    config_file = Path(config_path)
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = config_file.parent
    return ClusteringConfig(
        input_csv=resolve_config_path(payload["input_csv"], base_dir),
        output_dir=resolve_config_path(payload.get("output_dir", "../outputs/demo_clustering"), base_dir),
        feature_cols=list(payload.get("feature_cols", [])),
        k_min=int(payload.get("k_min", 3)),
        k_max=int(payload.get("k_max", 6)),
        fixed_k=int(payload.get("fixed_k", 4)),
        eval_sample_size=int(payload.get("eval_sample_size", 600)),
        random_state=int(payload.get("random_state", 42)),
        n_init=int(payload.get("n_init", 10)),
        max_iter=int(payload.get("max_iter", 300)),
        dbscan_eps=float(payload.get("dbscan_eps", 1.15)),
        dbscan_min_samples=int(payload.get("dbscan_min_samples", 12)),
        density_quantile=float(payload.get("density_quantile", 0.2)),
    )


def _prepare_matrix(dataframe: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    numeric_frame = dataframe[feature_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    imputed_array = imputer.fit_transform(numeric_frame)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(imputed_array)
    return scaled_array, feature_cols


def _build_eval_slice(data_array: np.ndarray, labels: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, np.ndarray]:
    sample_size = min(config.eval_sample_size, len(data_array))
    if sample_size >= len(data_array):
        return data_array, labels
    rng = np.random.default_rng(config.random_state)
    sample_index = rng.choice(len(data_array), size=sample_size, replace=False)
    return data_array[sample_index], labels[sample_index]


def _save_kmeans_selection_plots(metrics_frame: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_frame["k"], metrics_frame["inertia"], marker="o", color="#2563EB")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("KMeans Elbow Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "kmeans_elbow_curve.png", dpi=200)
    plt.close()

    figure, axes = plt.subplots(3, 1, figsize=(8, 10))
    score_columns = [
        ("silhouette", "Silhouette Score", "#16A34A"),
        ("calinski_harabasz", "Calinski-Harabasz Score", "#7C3AED"),
        ("davies_bouldin", "Davies-Bouldin Score", "#DC2626"),
    ]
    for axis, (column, title, color) in zip(axes, score_columns, strict=True):
        axis.plot(metrics_frame["k"], metrics_frame[column], marker="o", color=color)
        axis.set_title(title)
        axis.set_xlabel("K")
        axis.set_ylabel(column)
    figure.tight_layout()
    figure.savefig(output_dir / "kmeans_selection_scores.png", dpi=200)
    plt.close(figure)


def _evaluate_labels(data_array: np.ndarray, labels: np.ndarray, algorithm_name: str, config: ClusteringConfig) -> dict[str, Any]:
    cluster_values = labels[labels != -1]
    n_clusters = int(len(np.unique(cluster_values))) if len(cluster_values) else 0
    noise_ratio = float((labels == -1).mean()) if np.any(labels == -1) else 0.0
    valid_mask = labels != -1

    if n_clusters < 2 or valid_mask.sum() < 5:
        return {
            "algorithm": algorithm_name,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    eval_array, eval_labels = _build_eval_slice(data_array[valid_mask], labels[valid_mask], config)
    if len(np.unique(eval_labels)) < 2:
        return {
            "algorithm": algorithm_name,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
    return {
        "algorithm": algorithm_name,
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "silhouette": float(silhouette_score(eval_array, eval_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(eval_array, eval_labels)),
        "davies_bouldin": float(davies_bouldin_score(eval_array, eval_labels)),
    }


def _run_kmeans_selection(data_array: np.ndarray, config: ClusteringConfig) -> pd.DataFrame:
    metrics_records: list[dict[str, Any]] = []
    for k in range(config.k_min, config.k_max + 1):
        model = KMeans(
            n_clusters=k,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )
        labels = model.fit_predict(data_array)
        metric_record = _evaluate_labels(data_array, labels, f"KMeans_k{k}", config)
        metric_record["k"] = k
        metric_record["inertia"] = float(model.inertia_)
        metrics_records.append(metric_record)
    return pd.DataFrame(metrics_records)


def _run_algorithm_comparison(data_array: np.ndarray, config: ClusteringConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    bandwidth = estimate_bandwidth(
        data_array,
        quantile=config.density_quantile,
        n_samples=min(len(data_array), 500),
        random_state=config.random_state,
    )
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0

    algorithms = {
        f"KMeans_k{config.fixed_k}": KMeans(
            n_clusters=config.fixed_k,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        ),
        f"MiniBatchKMeans_k{config.fixed_k}": MiniBatchKMeans(
            n_clusters=config.fixed_k,
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
            batch_size=min(256, max(64, len(data_array) // 10)),
        ),
        "Density_MeanShift": MeanShift(bandwidth=bandwidth, bin_seeding=True),
        "DBSCAN": DBSCAN(eps=config.dbscan_eps, min_samples=config.dbscan_min_samples),
    }

    labels_map: dict[str, np.ndarray] = {}
    metrics_records: list[dict[str, Any]] = []
    for algorithm_name, estimator in algorithms.items():
        labels = estimator.fit_predict(data_array)
        labels_map[algorithm_name] = labels
        metrics_records.append(_evaluate_labels(data_array, labels, algorithm_name, config))
    return pd.DataFrame(metrics_records), labels_map


def _select_best_algorithm(metrics_frame: pd.DataFrame) -> str:
    valid_frame = metrics_frame.dropna(subset=["silhouette"]).copy()
    if valid_frame.empty:
        raise ValueError("没有可用于选择最佳聚类模型的有效指标。")
    ranked = valid_frame.sort_values(
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    )
    return str(ranked.iloc[0]["algorithm"])


def _save_embeddings(
    data_array: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    random_state: int,
) -> None:
    label_frame = pd.DataFrame({"cluster_target": labels})

    pca_2d = PCA(n_components=2, random_state=random_state).fit_transform(data_array)
    pca_3d = PCA(n_components=3, random_state=random_state).fit_transform(data_array)
    tsne_perplexity = min(30, max(5, (len(data_array) - 1) // 3))
    tsne_2d = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    ).fit_transform(data_array)

    pca_frame = pd.DataFrame({"pca1": pca_2d[:, 0], "pca2": pca_2d[:, 1]}).join(label_frame)
    tsne_frame = pd.DataFrame({"tsne1": tsne_2d[:, 0], "tsne2": tsne_2d[:, 1]}).join(label_frame)
    pca_3d_frame = pd.DataFrame(
        {
            "pca1": pca_3d[:, 0],
            "pca2": pca_3d[:, 1],
            "pca3": pca_3d[:, 2],
        }
    ).join(label_frame)

    pca_frame.to_csv(output_dir / "pca_embedding.csv", index=False, encoding="utf-8-sig")
    tsne_frame.to_csv(output_dir / "tsne_embedding.csv", index=False, encoding="utf-8-sig")
    pca_3d_frame.to_csv(output_dir / "pca_3d_embedding.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 6))
    plt.scatter(pca_frame["pca1"], pca_frame["pca2"], c=pca_frame["cluster_target"], cmap="tab10", s=16)
    plt.title("PCA 2D Visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "pca_2d.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(tsne_frame["tsne1"], tsne_frame["tsne2"], c=tsne_frame["cluster_target"], cmap="tab10", s=16)
    plt.title("t-SNE 2D Visualization")
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_2d.png", dpi=200)
    plt.close()

    figure = px.scatter_3d(
        pca_3d_frame,
        x="pca1",
        y="pca2",
        z="pca3",
        color=pca_3d_frame["cluster_target"].astype(str),
        title="Best Clustering PCA 3D",
    )
    figure.write_html(output_dir / "pca_3d_visualization.html")


def run_clustering(config: ClusteringConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = read_csv_with_fallback(config.input_csv)
    feature_cols = resolve_feature_columns(dataframe, config.feature_cols)
    data_array, _ = _prepare_matrix(dataframe, feature_cols)

    kmeans_selection = _run_kmeans_selection(data_array, config)
    kmeans_selection.to_csv(output_dir / "kmeans_selection_metrics.csv", index=False, encoding="utf-8-sig")
    _save_kmeans_selection_plots(kmeans_selection, output_dir)

    algorithm_metrics, labels_map = _run_algorithm_comparison(data_array, config)
    best_algorithm = _select_best_algorithm(algorithm_metrics)
    best_labels = labels_map[best_algorithm]

    algorithm_metrics.to_csv(output_dir / "cluster_algorithm_metrics.csv", index=False, encoding="utf-8-sig")

    identifier_cols = [column for column in ["单元编号", "X坐标", "Y坐标", "分区"] if column in dataframe.columns]
    assignments_all = dataframe[identifier_cols].copy() if identifier_cols else pd.DataFrame({"row_index": np.arange(len(dataframe))})
    for algorithm_name, labels in labels_map.items():
        assignments_all[algorithm_name] = labels
    assignments_all.to_csv(output_dir / "cluster_assignments_all.csv", index=False, encoding="utf-8-sig")

    best_assignments = dataframe[identifier_cols].copy() if identifier_cols else pd.DataFrame({"row_index": np.arange(len(dataframe))})
    best_assignments["cluster_target"] = best_labels
    best_assignments["cluster_algorithm"] = best_algorithm
    best_assignments.to_csv(output_dir / "best_cluster_assignments.csv", index=False, encoding="utf-8-sig")

    _save_embeddings(data_array, best_labels, output_dir, config.random_state)

    write_json(
        output_dir / "best_cluster_summary.json",
        {
            "selected_k": config.fixed_k,
            "best_algorithm": best_algorithm,
            "target_col": "cluster_target",
            "feature_cols": feature_cols,
        },
    )

    return {
        "output_dir": str(output_dir),
        "best_algorithm": best_algorithm,
        "target_col": "cluster_target",
    }
