from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.synthetic_data import SyntheticDataConfig, save_synthetic_dataset


def main() -> None:
    output_csv = ROOT / "data" / "sample" / "demo_geodata.csv"
    config = SyntheticDataConfig(n_samples=2000, random_state=42)
    save_synthetic_dataset(output_csv, config)
    print(f"示例数据已生成：{output_csv}")


if __name__ == "__main__":
    main()
