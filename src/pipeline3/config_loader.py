from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_pipeline3_config_payload(config_path: str) -> dict[str, Any]:
    return _load_with_extends(Path(config_path).resolve())


def _load_with_extends(config_file: Path) -> dict[str, Any]:
    with config_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping: {config_file}")
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    parent_file = (config_file.parent / str(extends)).resolve()
    if not parent_file.exists():
        raise ValueError(f"Parent config not found for extends='{extends}': {parent_file}")
    return _deep_merge(_load_with_extends(parent_file), raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
