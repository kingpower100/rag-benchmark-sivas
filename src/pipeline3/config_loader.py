from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import deep_merge, is_official_config_path, load_yaml_mapping, official_config_files, validate_unique_values


def load_pipeline3_config_payload(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    payload = _load_with_extends(config_file)
    _validate_official_pipeline3_ids(config_file)
    return payload


def _load_with_extends(config_file: Path) -> dict[str, Any]:
    raw = load_yaml_mapping(config_file)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    parent_file = (config_file.parent / str(extends)).resolve()
    if not parent_file.exists():
        raise ValueError(f"Parent config not found for extends='{extends}': {parent_file}")
    return deep_merge(_load_with_extends(parent_file), raw)


def _validate_official_pipeline3_ids(config_file: Path) -> None:
    if not is_official_config_path(config_file):
        return
    run_ids: list[tuple[str, Path]] = []
    input_run_ids: list[tuple[str, Path]] = []
    for candidate in official_config_files("pipeline3"):
        payload = _load_with_extends(candidate.resolve())
        pipeline3 = payload.get("pipeline3") if isinstance(payload, dict) else None
        inputs = payload.get("inputs") if isinstance(payload, dict) else None
        if isinstance(pipeline3, dict):
            run_ids.append((str(pipeline3.get("run_id") or ""), candidate))
        if isinstance(inputs, dict):
            run_path = str(inputs.get("pipeline1_results_path") or "")
            if run_path:
                input_run_ids.append((Path(run_path).parent.name, candidate))
    validate_unique_values("Pipeline 3 run_id", run_ids)
    validate_unique_values("Pipeline 3 input Pipeline 1 run", input_run_ids)
