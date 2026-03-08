# GeoClusterAtlas

空间聚类、模型预测与 SHAP 解释的一体化工作流模板。

这个仓库现在按下面这条主线组织：

1. 先跑 KMeans，保留原来的选 K 图  
2. 固定 `K=4`  
3. 在 `K=4` 基础上对比 `KMeans`、`MiniBatchKMeans`、`Density_MeanShift`、`DBSCAN`  
4. 输出聚类评价指标总表，自动保留效果最好的聚类结果作为下游预测目标  
5. 基于最佳聚类结果输出 `PCA`、`t-SNE` 和 `3D HTML` 可视化  
6. 用 `RF`、`XGBoost`、`LightGBM`、`MLP` 做分类实验  
7. 输出模型评价表，按准确率保留最优模型  
8. 对最优模型做 SHAP，并默认对全部样本绘制交互图

## 快速开始

```bash
pip install -e .
python scripts/generate_demo_data.py
python scripts/run_demo.py
```

## 输出目录

### 聚类阶段

`outputs/demo_clustering/`

- `kmeans_selection_metrics.csv`
- `kmeans_elbow_curve.png`
- `kmeans_selection_scores.png`
- `cluster_algorithm_metrics.csv`
- `cluster_assignments_all.csv`
- `best_cluster_assignments.csv`
- `pca_embedding.csv`
- `tsne_embedding.csv`
- `pca_2d.png`
- `tsne_2d.png`
- `pca_3d_visualization.html`
- `best_cluster_summary.json`

### 模型实验阶段

`outputs/demo_model_compare/`

- `model_compare_results.csv`
- `best_model_summary.json`
- `best_model_bundle.joblib`
- `best_model_predictions.csv`
- `best_model_confusion_matrix.csv`
- `reports/`

### SHAP 阶段

`outputs/demo_shap/`

- `shap_summary_by_cluster.csv`
- `shap_run_summary.json`
- `summary_plots/`
- `interaction_plots/`

其中：

- `summary_plots/` 下是每个 cluster 的 summary plot
- `interaction_plots/cluster_x/` 下是该 cluster 的全部特征交互图

## 关键配置

- 聚类配置：`configs/clustering_config.json`
- 模型实验配置：`configs/model_compare_config.json`
- SHAP 配置：`configs/shap_config.json`

当前默认就是：

- 聚类目标固定看 `K=4`
- 下游预测目标取最佳聚类算法的 `cluster_target`
- SHAP 默认使用全部测试样本绘图
