from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 JSON 配置文件。"""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """保存 JSON 配置，方便复现实验。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """递归更新配置字典，用于命令行参数覆盖默认配置。"""

    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
