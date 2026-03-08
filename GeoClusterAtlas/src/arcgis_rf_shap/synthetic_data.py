from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticDataConfig:
    n_samples: int = 2000
    n_features: int = 10
    random_state: int = 42


def _build_zone_codes(x_coord: np.ndarray, y_coord: np.ndarray) -> np.ndarray:
    conditions = [
        (x_coord > 60) & (y_coord > 60),
        (x_coord <= 40) & (y_coord > 55),
        (x_coord > 55) & (y_coord <= 40),
        (x_coord <= 35) & (y_coord <= 35),
    ]
    choices = np.array(["核心区", "北部组团", "东南片区", "老城区"], dtype=object)
    return np.select(conditions, choices, default="过渡区")


def generate_synthetic_dataset(config: SyntheticDataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)
    sample_count = config.n_samples

    x_coord = rng.uniform(0, 100, sample_count)
    y_coord = rng.uniform(0, 100, sample_count)
    zone_code = _build_zone_codes(x_coord, y_coord)

    zone_effect_map = {
        "核心区": 0.9,
        "北部组团": 0.4,
        "东南片区": 0.2,
        "老城区": -0.2,
        "过渡区": 0.0,
    }
    zone_effect = np.array([zone_effect_map[item] for item in zone_code], dtype=float)

    base_wave = np.sin(x_coord / 12.0) + np.cos(y_coord / 15.0)
    radial_term = np.sqrt((x_coord - 50.0) ** 2 + (y_coord - 50.0) ** 2) / 50.0

    feature_data: dict[str, np.ndarray] = {}
    for idx in range(1, config.n_features + 1):
        noise = rng.normal(0, 0.6, sample_count)
        slope_x = (idx * 0.04) * (x_coord / 100.0)
        slope_y = ((config.n_features - idx + 1) * 0.03) * (y_coord / 100.0)
        cross_term = 0.15 * np.sin((idx + 2) * x_coord / 30.0) + 0.12 * np.cos((idx + 1) * y_coord / 25.0)
        feature_value = noise + slope_x - slope_y + 0.2 * base_wave - 0.1 * radial_term + 0.18 * zone_effect + cross_term
        feature_data[f"特征{idx}"] = np.round(feature_value, 6)

    target_value = (
        1.8 * feature_data["特征1"]
        - 1.3 * np.square(feature_data["特征3"])
        + 0.9 * feature_data["特征5"] * feature_data["特征6"]
        + 1.1 * np.sin(feature_data["特征2"])
        + 0.7 * feature_data["特征8"]
        - 0.4 * feature_data["特征10"]
        + 0.5 * zone_effect
        + 0.2 * (x_coord / 100.0)
        - 0.15 * (y_coord / 100.0)
        + rng.normal(0, 0.35, sample_count)
    )

    return pd.DataFrame(
        {
            "单元编号": np.arange(1, sample_count + 1, dtype=int),
            "X坐标": np.round(x_coord, 4),
            "Y坐标": np.round(y_coord, 4),
            "分区": zone_code,
            **feature_data,
            "目标值": np.round(target_value, 6),
        }
    )


def save_synthetic_dataset(output_csv: str | Path, config: SyntheticDataConfig | None = None) -> pd.DataFrame:
    effective_config = config or SyntheticDataConfig()
    dataframe = generate_synthetic_dataset(effective_config)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return dataframe
