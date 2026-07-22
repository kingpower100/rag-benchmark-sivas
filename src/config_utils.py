from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CONFIG_DIR_NAMES = {"final_experiments", "experiments Orchestration LLM"}


class DuplicateKeyError(ValueError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: UniqueKeyLoader, node, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise DuplicateKeyError(
                f"Duplicate YAML key {key!r} in {mark.name} at line {mark.line + 1}, column {mark.column + 1}."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=UniqueKeyLoader) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping: {path}")
    return raw


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def is_official_config_path(config_file: Path) -> bool:
    return config_file.parent.name in OFFICIAL_CONFIG_DIR_NAMES


def official_config_files(pipeline: str) -> list[Path]:
    root = PROJECT_ROOT / "configs" / pipeline
    files: list[Path] = []
    for dir_name in OFFICIAL_CONFIG_DIR_NAMES:
        directory = root / dir_name
        if directory.exists():
            files.extend(sorted(directory.glob("*.yaml")))
    return files


def validate_unique_values(scope: str, values: Iterable[tuple[str, Path]]) -> None:
    by_value: dict[str, list[Path]] = defaultdict(list)
    for value, path in values:
        value = str(value or "").strip()
        if value:
            by_value[value].append(path.resolve())
    duplicates = {value: paths for value, paths in by_value.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{value}: {', '.join(path.name for path in paths)}"
            for value, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate official {scope}: {details}")
