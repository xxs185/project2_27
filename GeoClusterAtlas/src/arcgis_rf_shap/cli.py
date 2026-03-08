from __future__ import annotations

import argparse
from pathlib import Path

from .clustering import load_clustering_config_from_json, run_clustering
from .model_compare import load_model_compare_config_from_json, run_model_compare
from .shap_analysis import load_shap_config_from_json, run_shap_analysis
from .synthetic_data import SyntheticDataConfig, save_synthetic_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ArcGIS + 聚类 + 模型对比 + SHAP 工作流")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-data", help="生成演示数据")
    generate_parser.add_argument("--output-csv", type=Path, required=True, help="输出 CSV 路径")
    generate_parser.add_argument("--n-samples", type=int, default=2000, help="样本量")
    generate_parser.add_argument("--random-state", type=int, default=42, help="随机种子")

    clustering_parser = subparsers.add_parser("run-clustering", help="运行聚类")
    clustering_parser.add_argument("--config", type=Path, required=True, help="聚类配置路径")

    compare_parser = subparsers.add_parser("run-model-compare", help="运行模型对比")
    compare_parser.add_argument("--config", type=Path, required=True, help="模型对比配置路径")

    shap_parser = subparsers.add_parser("run-shap", help="运行 SHAP 分析")
    shap_parser.add_argument("--config", type=Path, required=True, help="SHAP 配置路径")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-data":
        save_synthetic_dataset(
            args.output_csv,
            SyntheticDataConfig(n_samples=args.n_samples, random_state=args.random_state),
        )
        print(f"示例数据已写入：{args.output_csv}")
        return

    if args.command == "run-clustering":
        results = run_clustering(load_clustering_config_from_json(args.config))
        print(f"聚类流程完成，输出目录：{results['output_dir']}")
        return

    if args.command == "run-model-compare":
        results = run_model_compare(load_model_compare_config_from_json(args.config))
        print(f"模型对比完成，输出目录：{results['output_dir']}")
        return

    results = run_shap_analysis(load_shap_config_from_json(args.config))
    print(f"SHAP 分析完成，输出目录：{results['output_dir']}")


if __name__ == "__main__":
    main()
