from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_pipeline_config_payload(config_path: str, validate_unique_experiment_id: bool = True) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    payload = _load_with_extends(config_file)
    payload = _normalize_documents_config(payload)
    if validate_unique_experiment_id:
        _validate_experiment_id_matches_config_name(config_file, payload)
        _validate_unique_experiment_id(config_file, payload)
    return payload


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


def _normalize_documents_config(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload.pop("documents", None)
    if documents is None:
        return payload
    if not isinstance(documents, dict):
        raise ValueError("documents config must be a YAML mapping.")
    data = dict(payload.get("data") or {})
    field_map = {
        "path": "documents_path",
        "source_type": "documents_source_type",
        "file_glob": "documents_file_glob",
        "recursive": "documents_recursive",
        "text_field": "document_text_field",
    }
    for source_key, target_key in field_map.items():
        if source_key in documents:
            data[target_key] = documents[source_key]
    payload["data"] = data
    return payload


def _validate_unique_experiment_id(config_file: Path, payload: dict[str, Any]) -> None:
    exp = payload.get("experiment") if isinstance(payload, dict) else None
    experiment_id = exp.get("experiment_id") if isinstance(exp, dict) else None
    if not experiment_id or config_file.parent.name != "experiments":
        return
    same_ids: list[Path] = []
    for candidate in config_file.parent.glob("*.yaml"):
        try:
            candidate_payload = _load_with_extends(candidate.resolve())
            candidate_exp = candidate_payload.get("experiment", {})
            if isinstance(candidate_exp, dict) and candidate_exp.get("experiment_id") == experiment_id:
                same_ids.append(candidate.resolve())
        except Exception:
            continue
    if len(same_ids) > 1:
        files = ", ".join(p.name for p in same_ids)
        raise ValueError(f"Duplicate experiment_id '{experiment_id}' found in configs/pipeline1/experiments: {files}")


def _validate_experiment_id_matches_config_name(config_file: Path, payload: dict[str, Any]) -> None:
    exp = payload.get("experiment") if isinstance(payload, dict) else None
    experiment_id = exp.get("experiment_id") if isinstance(exp, dict) else None
    if not experiment_id or config_file.parent.name != "experiments":
        return
    if config_file.stem != experiment_id:
        raise ValueError(
            f"experiment.experiment_id '{experiment_id}' must match config filename stem '{config_file.stem}'"
        )
