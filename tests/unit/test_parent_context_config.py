"""Tests for ParentContextConfig schema."""
from __future__ import annotations

import pytest

from src.pipeline1.schemas.config_schema import ParentContextConfig, PipelineConfig


# ---------------------------------------------------------------------------
# ParentContextConfig unit tests
# ---------------------------------------------------------------------------

def test_default_disabled():
    cfg = ParentContextConfig()
    assert cfg.enabled is False


def test_defaults_match_spec():
    cfg = ParentContextConfig()
    assert cfg.parent_unit == "markdown_section"
    assert cfg.deduplicate is True
    assert cfg.missing_parent_policy == "use_child"
    assert cfg.unique_parent_top_k == 5
    assert cfg.max_parent_tokens == 1800


def test_valid_c03_config_parses():
    cfg = ParentContextConfig(
        enabled=True,
        parent_unit="markdown_section",
        deduplicate=True,
        missing_parent_policy="use_child",
        unique_parent_top_k=5,
        max_parent_tokens=1800,
    )
    assert cfg.enabled is True
    assert cfg.unique_parent_top_k == 5
    assert cfg.max_parent_tokens == 1800


@pytest.mark.parametrize(
    "field,value",
    [
        ("mapping_policy", "deepest_containing_then_overlap"),
        ("score_policy", "best_child"),
        ("preserve_child_provenance", True),
        ("oversized_parent_policy", "prefer_deeper_section"),
    ],
)
def test_removed_parent_context_policy_fields_rejected(field, value):
    with pytest.raises(Exception, match=f"parent_context.{field}"):
        ParentContextConfig(**{field: value})


def test_max_parent_tokens_must_be_positive():
    with pytest.raises(Exception):
        ParentContextConfig(max_parent_tokens=0)


def test_unknown_field_rejected():
    with pytest.raises(Exception):
        ParentContextConfig(enabled=False, unknown_field="oops")


def test_invalid_parent_unit_rejected():
    with pytest.raises(Exception):
        ParentContextConfig(parent_unit="whole_document")


def test_invalid_missing_parent_policy_rejected():
    with pytest.raises(Exception):
        ParentContextConfig(missing_parent_policy="skip")


def test_unique_parent_top_k_must_be_positive():
    with pytest.raises(Exception):
        ParentContextConfig(unique_parent_top_k=0)


def test_missing_parent_policy_error_accepted():
    cfg = ParentContextConfig(missing_parent_policy="error")
    assert cfg.missing_parent_policy == "error"


# ---------------------------------------------------------------------------
# PipelineConfig integration: parent_context field defaults
# ---------------------------------------------------------------------------

_MINIMAL_PIPELINE_CONFIG = {
    "experiment": {"experiment_id": "test_exp", "output_dir": "data/runs/pipeline1"},
    "data": {
        "documents_path": "data/raw/docs.jsonl",
        "questions_path": "data/raw/q.jsonl",
    },
    "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 200},
    "embedding": {"provider": "sentence_transformers", "model_name": "intfloat/multilingual-e5-small"},
    "index": {"type": "faiss"},
    "retrieval": {"retriever_type": "dense", "top_k": 5, "fetch_k": 20},
    "reranker": {"enabled": False},
    "generation": {
        "provider": "ollama",
        "model_name": "qwen2.5:7b-instruct",
        "system_prompt": "Answer: {question}\nContext: {context}",
    },
    "telemetry": {"estimate_cost": False},
    "runtime": {"save_csv": False},
}


def test_pipeline_config_parent_context_defaults_to_disabled():
    cfg = PipelineConfig.model_validate(_MINIMAL_PIPELINE_CONFIG)
    assert cfg.parent_context.enabled is False


def test_pipeline_config_parent_context_c03_block_accepted():
    payload = {
        **_MINIMAL_PIPELINE_CONFIG,
        "parent_context": {
            "enabled": True,
            "parent_unit": "markdown_section",
            "deduplicate": True,
            "missing_parent_policy": "use_child",
            "unique_parent_top_k": 5,
            "max_parent_tokens": 1800,
        },
    }
    cfg = PipelineConfig.model_validate(payload)
    assert cfg.parent_context.enabled is True
    assert cfg.parent_context.unique_parent_top_k == 5
    assert cfg.parent_context.max_parent_tokens == 1800


def test_pipeline_config_existing_configs_parse_without_parent_context():
    """Existing YAML configs without parent_context must still parse."""
    cfg = PipelineConfig.model_validate(_MINIMAL_PIPELINE_CONFIG)
    assert cfg.parent_context.enabled is False
