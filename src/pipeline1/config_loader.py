from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import (
    deep_merge,
    is_official_config_path,
    load_yaml_mapping,
    official_config_files,
    validate_unique_values,
)


def load_pipeline_config_payload(config_path: str, validate_unique_experiment_id: bool = True) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    payload = _load_with_extends(config_file)
    payload = _normalize_documents_config(payload)
    payload = _normalize_retrieval_type_alias(payload)
    payload = _resolve_generation_prompt(payload)
    if validate_unique_experiment_id:
        _validate_experiment_id_matches_config_name(config_file, payload)
        _validate_unique_experiment_id(config_file, payload)
        _validate_official_experiment_ids(config_file)
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


def _normalize_retrieval_type_alias(payload: dict[str, Any]) -> dict[str, Any]:
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict) or "type" not in retrieval:
        return payload
    normalized = dict(retrieval)
    normalized["retriever_type"] = normalized.pop("type")
    payload["retrieval"] = normalized
    return payload


def _resolve_generation_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        return payload
    prompt_path = generation.get("prompt_path")
    if not prompt_path:
        return payload
    resolved = _resolve_project_path(str(prompt_path))
    if not resolved.is_file():
        raise ValueError(f"generation.prompt_path is missing or not a file: {resolved}")
    prompt = resolved.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"generation.prompt_path is empty: {resolved}")
    payload["generation"] = {**generation, "system_prompt": prompt}
    return payload


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / candidate).resolve()


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


def _validate_official_experiment_ids(config_file: Path) -> None:
    if not is_official_config_path(config_file):
        return
    values: list[tuple[str, Path]] = []
    for candidate in official_config_files("pipeline1"):
        payload = _load_with_extends(candidate.resolve())
        exp = payload.get("experiment") if isinstance(payload, dict) else None
        if isinstance(exp, dict):
            values.append((str(exp.get("experiment_id") or ""), candidate))
    validate_unique_values("Pipeline 1 experiment_id", values)
