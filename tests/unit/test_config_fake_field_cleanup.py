from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.pipeline1.orchestrator import _validate_configured_dense_dim
from src.pipeline1.schemas.config_schema import OrchestrationConfig, PipelineConfig
from src.pipeline2.schemas.eval_config_schema import EvalConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FIELD_USAGE_REGISTRY = {
    "retrieval.fallback_to_global": "operative",
    "index.dense_dim": "operative",
    "index.index_name": "operative",
    "orchestration.prompt_version": "operative_validation_label",
    "orchestration.tasks": "deprecated_fixed_workflow",
    "generation.configurable": "removed",
    "chunking.tokenizer_name": "operative",
    "chunking.max_chunk_chars": "operative_strategy_or_validation_limit",
    "chunking.max_chunk_tokens": "operative_strategy_or_validation_limit",
    "retrieval.bm25.enabled": "removed",
    "bert_score.max_length": "removed",
    "parent_context.mapping_policy": "removed_fixed_policy",
    "parent_context.score_policy": "removed_fixed_policy",
    "parent_context.preserve_child_provenance": "removed_mandatory",
    "parent_context.max_parent_tokens": "operative",
    "parent_context.oversized_parent_policy": "removed_fixed_policy",
}


def _minimal_pipeline1_payload() -> dict:
    return {
        "experiment": {"experiment_id": "test", "output_dir": "data/runs"},
        "data": {"documents_path": "docs.jsonl", "questions_path": "questions.jsonl"},
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
        "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
        "index": {"type": "faiss", "dense_dim": 2},
        "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
        "reranker": {"enabled": False},
        "orchestration": {"prompt_path": "src/pipeline1/prompts/orchestration_promptV4.txt", "prompt_version": "orchestration_promptV4"},
        "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
        "telemetry": {},
        "runtime": {},
    }


def test_dense_dim_mismatch_fails_before_index_use():
    cfg = PipelineConfig.model_validate(_minimal_pipeline1_payload())
    with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
        _validate_configured_dense_dim(cfg, np.ones((2, 3), dtype="float32"))


def test_dense_dim_match_passes():
    cfg = PipelineConfig.model_validate(_minimal_pipeline1_payload())
    _validate_configured_dense_dim(cfg, np.ones((2, 2), dtype="float32"))


def test_orchestration_prompt_version_must_match_prompt_path():
    with pytest.raises(ValueError, match="prompt_version must match"):
        OrchestrationConfig(
            prompt_path="src/pipeline1/prompts/orchestration_promptV4.txt",
            prompt_version="orchestration_promptV3",
        )


def test_removed_bm25_enabled_has_clear_migration_error():
    payload = _minimal_pipeline1_payload()
    payload["retrieval"]["bm25"] = {"enabled": False}
    with pytest.raises(ValueError, match="retrieval.bm25.enabled"):
        PipelineConfig.model_validate(payload)


def test_removed_parent_context_fields_have_clear_migration_error():
    payload = _minimal_pipeline1_payload()
    payload["parent_context"] = {
        "enabled": True,
        "score_policy": "best_child",
    }
    with pytest.raises(ValueError, match="parent_context.score_policy"):
        PipelineConfig.model_validate(payload)


def test_generation_configurable_has_clear_migration_error():
    payload = _minimal_pipeline1_payload()
    payload["generation"]["configurable"] = True
    with pytest.raises(ValueError, match="generation.configurable"):
        PipelineConfig.model_validate(payload)


def test_bert_score_max_length_has_clear_migration_error():
    with pytest.raises(ValueError, match="bert_score.max_length"):
        EvalConfig.model_validate(
            {
                "evaluation": {"eval_run_id": "eval"},
                "inputs": {"rag_outputs": ["results.jsonl"]},
                "bert_score": {"enabled": True, "max_length": 512},
            }
        )


def test_official_e00_configs_parse_without_deprecation_warnings():
    paths = [
        PROJECT_ROOT / "configs/pipeline1/final_experiments/E00-G_global_dense_baseline.yaml",
        PROJECT_ROOT / "configs/pipeline1/final_experiments/E00-C_category_aware_dense_baseline.yaml",
    ]
    for path in paths:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PipelineConfig.from_yaml(str(path))
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


def test_official_yaml_contains_no_known_fake_fields():
    forbidden_by_path = {
        ("retrieval", "bm25", "enabled"),
        ("generation", "configurable"),
        ("bert_score", "max_length"),
    }
    official_dirs = [
        PROJECT_ROOT / "configs/pipeline1/final_experiments",
        PROJECT_ROOT / "configs/pipeline2/final_experiments",
        PROJECT_ROOT / "configs/pipeline3/final_experiments",
    ]
    violations: list[str] = []
    for directory in official_dirs:
        for path in directory.glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for key_path in forbidden_by_path:
                if _has_path(payload, key_path):
                    violations.append(f"{path.relative_to(PROJECT_ROOT).as_posix()}: {'.'.join(key_path)}")
    assert violations == []


def test_scoped_fields_have_usage_registry_classification():
    expected = {
        "retrieval.fallback_to_global",
        "index.dense_dim",
        "index.index_name",
        "orchestration.prompt_version",
        "orchestration.tasks",
        "generation.configurable",
        "chunking.tokenizer_name",
        "chunking.max_chunk_chars",
        "chunking.max_chunk_tokens",
        "retrieval.bm25.enabled",
        "bert_score.max_length",
        "parent_context.mapping_policy",
        "parent_context.score_policy",
        "parent_context.preserve_child_provenance",
        "parent_context.max_parent_tokens",
        "parent_context.oversized_parent_policy",
    }
    assert set(CONFIG_FIELD_USAGE_REGISTRY) == expected
    assert all(classification for classification in CONFIG_FIELD_USAGE_REGISTRY.values())
    assert CONFIG_FIELD_USAGE_REGISTRY["parent_context.mapping_policy"] == "removed_fixed_policy"
    assert CONFIG_FIELD_USAGE_REGISTRY["parent_context.score_policy"] == "removed_fixed_policy"
    assert CONFIG_FIELD_USAGE_REGISTRY["parent_context.preserve_child_provenance"] == "removed_mandatory"
    assert CONFIG_FIELD_USAGE_REGISTRY["parent_context.oversized_parent_policy"] == "removed_fixed_policy"


def _has_path(payload: dict, key_path: tuple[str, ...]) -> bool:
    current = payload
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True
