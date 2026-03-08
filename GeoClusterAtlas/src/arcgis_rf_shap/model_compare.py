from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from .common import read_csv_with_fallback, resolve_config_path, resolve_feature_columns, write_json


@dataclass(slots=True)
class ModelCompareConfig:
    input_csv: Path
    cluster_csv: Path
    output_dir: Path = Path("outputs/demo_model_compare")
    feature_cols: list[str] = field(default_factory=list)
    target_col: str = "cluster_target"
    join_col: str = "单元编号"
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1


def load_model_compare_config_from_json(config_path: str | Path) -> ModelCompareConfig:
    config_file = Path(config_path)
    payload = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = config_file.parent
    return ModelCompareConfig(
        input_csv=resolve_config_path(payload["input_csv"], base_dir),
        cluster_csv=resolve_config_path(payload["cluster_csv"], base_dir),
        output_dir=resolve_config_path(payload.get("output_dir", "../outputs/demo_model_compare"), base_dir),
        feature_cols=list(payload.get("feature_cols", [])),
        target_col=payload.get("target_col", "cluster_target"),
        join_col=payload.get("join_col", "单元编号"),
        test_size=float(payload.get("test_size", 0.2)),
        random_state=int(payload.get("random_state", 42)),
        n_jobs=int(payload.get("n_jobs", -1)),
    )


def _merge_target_labels(config: ModelCompareConfig) -> pd.DataFrame:
    feature_frame = read_csv_with_fallback(config.input_csv)
    cluster_frame = read_csv_with_fallback(config.cluster_csv)
    if config.join_col not in feature_frame.columns:
        raise ValueError(f"输入数据缺少连接列：{config.join_col}")
    if config.join_col not in cluster_frame.columns:
        raise ValueError(f"聚类结果缺少连接列：{config.join_col}")
    if config.target_col not in cluster_frame.columns:
        raise ValueError(f"聚类结果缺少目标列：{config.target_col}")

    merged = feature_frame.merge(
        cluster_frame[[config.join_col, config.target_col]],
        on=config.join_col,
        how="left",
        validate="one_to_one",
    )
    if merged[config.target_col].isna().any():
        raise ValueError(f"目标列 {config.target_col} 合并后存在缺失值。")
    return merged


def _build_experiments(config: ModelCompareConfig, n_classes: int) -> list[tuple[str, str, Pipeline]]:
    return [
        (
            "RandomForest",
            "rf",
            Pipeline(
                [
                    ("preprocess", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=600,
                            random_state=config.random_state,
                            n_jobs=config.n_jobs,
                        ),
                    ),
                ]
            ),
        ),
        (
            "XGBoost",
            "xgboost",
            Pipeline(
                [
                    ("preprocess", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBClassifier(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="multi:softprob",
                            num_class=n_classes,
                            eval_metric="mlogloss",
                            tree_method="hist",
                            random_state=config.random_state,
                            n_jobs=config.n_jobs,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        ),
        (
            "LightGBM",
            "lightgbm",
            Pipeline(
                [
                    ("preprocess", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMClassifier(
                            n_estimators=300,
                            learning_rate=0.08,
                            max_depth=-1,
                            objective="multiclass",
                            num_class=n_classes,
                            random_state=config.random_state,
                            n_jobs=config.n_jobs,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "MLP",
            "mlp",
            Pipeline(
                [
                    ("preprocess", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            learning_rate_init=0.001,
                            max_iter=400,
                            early_stopping=True,
                            random_state=config.random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]


def run_model_compare(config: ModelCompareConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataframe = _merge_target_labels(config)
    feature_cols = resolve_feature_columns(dataframe, config.feature_cols)
    x_frame = dataframe[feature_cols].apply(pd.to_numeric, errors="coerce")
    label_encoder = LabelEncoder()
    y_series = pd.Series(label_encoder.fit_transform(dataframe[config.target_col]), index=dataframe.index)

    train_index, test_index = train_test_split(
        dataframe.index.to_numpy(),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_series,
    )
    x_train = x_frame.loc[train_index]
    x_test = x_frame.loc[test_index]
    y_train = y_series.loc[train_index]
    y_test = y_series.loc[test_index]
    x_train_input = x_train.to_numpy()
    x_test_input = x_test.to_numpy()

    records: list[dict[str, Any]] = []
    trained_models: dict[str, dict[str, Any]] = {}
    n_classes = int(y_series.nunique())

    for model_name, model_family, pipeline in _build_experiments(config, n_classes):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
            warnings.filterwarnings("ignore", message="X has feature names.*")
            pipeline.fit(x_train_input, y_train)
            prediction = pipeline.predict(x_test_input)
        report_text = classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(prediction),
            digits=4,
            zero_division=0,
        )
        (reports_dir / f"{model_name}.txt").write_text(report_text, encoding="utf-8")
        records.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, prediction)),
                "macro_f1": float(f1_score(y_test, prediction, average="macro")),
                "weighted_f1": float(f1_score(y_test, prediction, average="weighted")),
            }
        )
        trained_models[model_name] = {"family": model_family, "pipeline": pipeline}

    results_frame = pd.DataFrame(records).sort_values(["accuracy", "macro_f1"], ascending=False)
    best_model_name = str(results_frame.iloc[0]["model_name"])
    best_model_entry = trained_models[best_model_name]
    best_pipeline = best_model_entry["pipeline"]
    best_prediction = best_pipeline.predict(x_test)
    decoded_prediction = label_encoder.inverse_transform(best_prediction)
    decoded_truth = label_encoder.inverse_transform(y_test)

    results_frame.to_csv(output_dir / "model_compare_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        confusion_matrix(decoded_truth, decoded_prediction, labels=label_encoder.classes_),
        index=[f"true_{label}" for label in label_encoder.classes_],
        columns=[f"pred_{label}" for label in label_encoder.classes_],
    ).to_csv(output_dir / "best_model_confusion_matrix.csv", encoding="utf-8-sig")
    pd.DataFrame(
        {
            config.join_col: dataframe.loc[test_index, config.join_col].tolist(),
            "真实类别": decoded_truth,
            "预测类别": decoded_prediction,
        }
    ).to_csv(output_dir / "best_model_predictions.csv", index=False, encoding="utf-8-sig")

    write_json(
        output_dir / "best_model_summary.json",
        {
            "best_model_name": best_model_name,
            "best_model_family": best_model_entry["family"],
            "selection_metric": "accuracy",
            "target_col": config.target_col,
            "feature_cols": feature_cols,
        },
    )

    bundle = {
        "model_name": best_model_name,
        "model_family": best_model_entry["family"],
        "model": best_pipeline.named_steps["model"],
        "preprocessor": best_pipeline.named_steps["preprocess"],
        "feature_cols": feature_cols,
        "target_col": config.target_col,
        "join_col": config.join_col,
        "test_ids": dataframe.loc[test_index, config.join_col].tolist(),
        "class_labels": label_encoder.classes_.tolist(),
        "label_encoder": label_encoder,
    }
    joblib.dump(bundle, output_dir / "best_model_bundle.joblib")

    return {
        "output_dir": str(output_dir),
        "results": results_frame,
        "best_model_name": best_model_name,
    }
