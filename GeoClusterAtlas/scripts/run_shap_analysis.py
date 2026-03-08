from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.shap_analysis import load_shap_config_from_json, run_shap_analysis


def main() -> None:
    results = run_shap_analysis(load_shap_config_from_json(ROOT / "configs" / "shap_config.json"))
    print(f"SHAP 分析已完成，输出目录：{results['output_dir']}")


if __name__ == "__main__":
    main()
