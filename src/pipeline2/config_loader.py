from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import deep_merge, is_official_config_path, load_yaml_mapping, official_config_files, validate_unique_values


def load_eval_config_payload(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    payload = _load_with_extends(config_file)
    _validate_official_eval_ids(config_file)
    _validate_official_chunk_evaluation(config_file, payload)
    _validate_official_failure_threshold(config_file, payload)
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


def _validate_official_eval_ids(config_file: Path) -> None:
    if not is_official_config_path(config_file):
        return
    eval_ids: list[tuple[str, Path]] = []
    input_run_ids: list[tuple[str, Path]] = []
    for candidate in official_config_files("pipeline2"):
        payload = _load_with_extends(candidate.resolve())
        evaluation = payload.get("evaluation") if isinstance(payload, dict) else None
        inputs = payload.get("inputs") if isinstance(payload, dict) else None
        if isinstance(evaluation, dict):
            eval_ids.append((str(evaluation.get("eval_run_id") or ""), candidate))
        if isinstance(inputs, dict):
            run_path = str(inputs.get("pipeline1_results_path") or "")
            if run_path:
                input_run_ids.append((Path(run_path).parent.name, candidate))
    validate_unique_values("Pipeline 2 eval_run_id", eval_ids)
    validate_unique_values("Pipeline 2 input Pipeline 1 run", input_run_ids)


def _validate_official_chunk_evaluation(config_file: Path, payload: dict[str, Any]) -> None:
    if not is_official_config_path(config_file):
        return
    retrieval_evaluation = payload.get("retrieval_evaluation")
    if not isinstance(retrieval_evaluation, dict):
        raise ValueError(f"Official Pipeline 2 config must define retrieval_evaluation: {config_file}")
    chunk_level = retrieval_evaluation.get("chunk_level")
    if not isinstance(chunk_level, dict) or chunk_level.get("enabled") is not True:
        raise ValueError(f"Official Pipeline 2 config must enable retrieval_evaluation.chunk_level: {config_file}")
    if not str(chunk_level.get("ground_truth_path") or "").strip():
        raise ValueError(f"Official Pipeline 2 config must set chunk_level.ground_truth_path: {config_file}")
    if chunk_level.get("missing_question_policy") != "error":
        raise ValueError(f"Official Pipeline 2 config must use chunk_level.missing_question_policy=error: {config_file}")


def _validate_official_failure_threshold(config_file: Path, payload: dict[str, Any]) -> None:
    if not is_official_config_path(config_file):
        return
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Official Pipeline 2 config must define evaluation settings: {config_file}")
    if evaluation.get("strict_failure_threshold") is not True:
        raise ValueError(f"Official Pipeline 2 config must set evaluation.strict_failure_threshold=true: {config_file}")
    max_failure_rate = evaluation.get("max_generation_failure_rate")
    try:
        max_failure_rate_float = float(max_failure_rate)
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"Official Pipeline 2 config must set evaluation.max_generation_failure_rate=0.0: {config_file}"
        ) from ex
    if max_failure_rate_float != 0.0:
        raise ValueError(f"Official Pipeline 2 config must set evaluation.max_generation_failure_rate=0.0: {config_file}")
