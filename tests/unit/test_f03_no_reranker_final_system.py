"""Tests for F03 - Final System without the BGE-M3 reranker.

F03 is a strict single-variable ablation of F01. The only intended
Pipeline 1 resolved-config differences are:
  experiment.experiment_id: F01 -> F03
  reranker.enabled:        true -> false
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_mapping
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.config_loader import load_pipeline3_config_payload
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


F01_P1_PATH = "configs/pipeline1/final_experiments/F01_mistral_small_final_system.yaml"
F01_P2_PATH = "configs/pipeline2/final_experiments/F01_mistral_small_final_system_eval.yaml"
F01_P3_PATH = "configs/pipeline3/final_experiments/F01_mistral_small_final_system_eval.yaml"

F03_P1_PATH = "configs/pipeline1/final_experiments/F03_no_reranker_final_system.yaml"
F03_P2_PATH = "configs/pipeline2/final_experiments/F03_no_reranker_final_system_eval.yaml"
F03_P3_PATH = "configs/pipeline3/final_experiments/F03_no_reranker_final_system_eval.yaml"

MAPPING_PATH = "configs/official_experiment_mapping.yaml"


def _p1(path: str) -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(path))


def _p2(path: str) -> EvalConfig:
    return EvalConfig.model_validate(load_eval_config_payload(path))


def _p3(path: str) -> Pipeline3Config:
    return Pipeline3Config.model_validate(load_pipeline3_config_payload(path))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, child_prefix))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(_flatten(child, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _diff_paths(left: Any, right: Any) -> set[str]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    keys = set(left_flat) | set(right_flat)
    return {key for key in keys if left_flat.get(key) != right_flat.get(key)}


def test_f03_resolves_successfully():
    cfg = _p1(F03_P1_PATH)
    assert cfg.experiment.experiment_id == "F03"


def test_f03_disables_only_reranker_enabled_in_pipeline1_resolved_config():
    f01 = _p1(F01_P1_PATH)
    f03 = _p1(F03_P1_PATH)

    assert f01.reranker.enabled is True
    assert f03.reranker.enabled is False

    allowed = {"experiment.experiment_id", "reranker.enabled"}
    assert _diff_paths(f01.model_dump(), f03.model_dump()) == allowed


def test_f03_retrieval_parameters_are_unchanged():
    f01 = _p1(F01_P1_PATH)
    f03 = _p1(F03_P1_PATH)

    assert f03.retrieval == f01.retrieval
    assert f03.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert f03.retrieval.top_k == 5
    assert f03.retrieval.fetch_k == 20
    assert f03.retrieval.hybrid.dense_fetch_k == 20
    assert f03.retrieval.hybrid.bm25_fetch_k == 20
    assert f03.retrieval.hybrid.rrf_k == 60


def test_f03_generation_embeddings_chunking_and_orchestration_are_unchanged():
    f01 = _p1(F01_P1_PATH)
    f03 = _p1(F03_P1_PATH)

    assert f03.generation == f01.generation
    assert f03.embedding == f01.embedding
    assert f03.chunking == f01.chunking
    assert f03.orchestration == f01.orchestration
    assert f03.index == f01.index
    assert f03.data == f01.data
    assert f03.runtime == f01.runtime
    assert f03.parent_context == f01.parent_context
    assert f03.telemetry == f01.telemetry


def test_f03_preserves_bge_m3_reranker_metadata_but_disables_execution():
    f01 = _p1(F01_P1_PATH)
    f03 = _p1(F03_P1_PATH)

    assert f03.reranker.enabled is False
    assert f03.reranker.model_name == f01.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert f03.reranker.device == f01.reranker.device == "cuda"
    assert f03.reranker.rerank_top_k == f01.reranker.rerank_top_k == 20
    assert f03.reranker.final_top_k == f01.reranker.final_top_k


def test_f03_pipeline2_points_to_f03_outputs_and_keeps_f01_eval_settings():
    f01 = _p2(F01_P2_PATH)
    f03 = _p2(F03_P2_PATH)

    assert f03.evaluation.eval_run_id == "F03"
    assert f03.inputs.pipeline1_results_path == "data/runs/pipeline1/F03/results.jsonl"
    assert f03.inputs.rag_outputs == ["data/runs/pipeline1/F03/results.jsonl"]

    f01_dump = f01.model_dump()
    f03_dump = f03.model_dump()
    allowed = {
        "evaluation.eval_run_id",
        "inputs.pipeline1_results_path",
        "inputs.rag_outputs[0]",
        "source_validation.expected_experiment_id",
        "source_validation.expected_reranker_enabled",
        "source_validation.require_reranker_diagnostics",
    }
    assert _diff_paths(f01_dump, f03_dump) == allowed


def test_f03_pipeline2_source_validation_contract():
    cfg = _p2(F03_P2_PATH)
    sv = cfg.source_validation
    assert sv is not None

    assert sv.expected_experiment_id == "F03"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "ollama"
    assert sv.expected_generator_model == "mistral-small:latest"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is False
    assert sv.require_hybrid_diagnostics is True
    assert sv.require_reranker_diagnostics is False
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f03_pipeline3_points_to_f03_outputs_and_keeps_f01_eval_settings():
    f01 = _p3(F01_P3_PATH)
    f03 = _p3(F03_P3_PATH)

    assert f03.pipeline3.run_id == "F03"
    assert f03.inputs.pipeline1_results_path == "data/runs/pipeline1/F03/results.jsonl"

    allowed = {
        "pipeline3.run_id",
        "inputs.pipeline1_results_path",
        "source_validation.expected_experiment_id",
        "source_validation.expected_reranker_enabled",
        "source_validation.require_reranker_diagnostics",
    }
    assert _diff_paths(f01.model_dump(), f03.model_dump()) == allowed


def test_f03_pipeline3_source_validation_contract():
    cfg = _p3(F03_P3_PATH)
    sv = cfg.source_validation
    assert sv is not None

    assert sv.expected_experiment_id == "F03"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "ollama"
    assert sv.expected_generator_model == "mistral-small:latest"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is False
    assert sv.require_reranker_diagnostics is False
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f03_is_registered_for_official_pipeline4_discovery():
    mapping = load_yaml_mapping(Path(MAPPING_PATH))
    experiments = mapping["official_experiment_mapping"]

    assert "F03" in experiments
    assert experiments["F03"]["pipeline1"] == F03_P1_PATH
    assert experiments["F03"]["pipeline2"] == F03_P2_PATH
    assert experiments["F03"]["pipeline3"] == F03_P3_PATH
    assert Path(experiments["F03"]["pipeline1"]).exists()
    assert Path(experiments["F03"]["pipeline2"]).exists()
    assert Path(experiments["F03"]["pipeline3"]).exists()


def test_f00_and_f01_behavior_remain_unchanged():
    f01 = _p1(F01_P1_PATH)

    assert f01.experiment.experiment_id == "F01"
    assert f01.generation.provider == "ollama"
    assert f01.generation.model_name == "mistral-small:latest"
    assert f01.reranker.enabled is True
    assert f01.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert f01.retrieval.top_k == 5
    assert f01.orchestration.enabled is False
