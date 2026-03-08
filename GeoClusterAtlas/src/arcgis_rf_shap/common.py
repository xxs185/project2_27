from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def configure_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]


def read_csv_with_fallback(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    encodings = (None, "utf-8-sig", "utf-8", "gbk", "gb18030")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            if encoding is None:
                return pd.read_csv(csv_path)
            return pd.read_csv(csv_path, encoding=encoding)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"读取数据失败：{csv_path}") from last_error


def resolve_feature_columns(dataframe: pd.DataFrame, feature_cols: list[str] | None = None) -> list[str]:
    explicit_columns = list(feature_cols or [])
    if explicit_columns:
        missing_columns = [column for column in explicit_columns if column not in dataframe.columns]
        if missing_columns:
            joined = "、".join(missing_columns)
            raise ValueError(f"缺少特征列：{joined}")
        return explicit_columns

    inferred_columns = [column for column in dataframe.columns if str(column).startswith("特征")]
    if not inferred_columns:
        raise ValueError("未找到特征列，请显式传入 feature_cols。")
    return inferred_columns


def resolve_config_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_optional_config_path(path_value: str | Path | None, base_dir: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    return resolve_config_path(path_value, base_dir)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_filename(value: str) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in value)
    sanitized = sanitized.strip("_")
    return sanitized or "item"
