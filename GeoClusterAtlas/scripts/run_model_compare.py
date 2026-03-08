from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arcgis_rf_shap.model_compare import load_model_compare_config_from_json, run_model_compare


def main() -> None:
    results = run_model_compare(load_model_compare_config_from_json(ROOT / "configs" / "model_compare_config.json"))
    print(f"模型对比已完成，输出目录：{results['output_dir']}")


if __name__ == "__main__":
    main()
