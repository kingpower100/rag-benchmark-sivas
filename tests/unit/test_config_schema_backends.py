"""Tests for multi-backend config schema additions: pgvector, BM25 analyzer, hybrid dense_backend."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline1.schemas.config_schema import (
    BM25Config,
    HybridConfig,
    IndexConfig,
    PgvectorConfig,
)


# ---------------------------------------------------------------------------
# PgvectorConfig
# ---------------------------------------------------------------------------

def test_pgvector_config_defaults():
    cfg = PgvectorConfig()
    assert cfg.dsn_env == "PGVECTOR_DSN"
    assert cfg.schema_name == "rag"
    assert cfg.table_name == "chunk_embeddings"
    assert cfg.index_type == "hnsw"
    assert cfg.rebuild_index is False
    assert cfg.hnsw_m == 16
    assert cfg.hnsw_ef_construction == 64
    assert cfg.hnsw_ef_search == 40
    assert cfg.ivfflat_lists == 100
    assert cfg.pool_min == 1
    assert cfg.pool_max == 5


def test_pgvector_config_custom_values():
    cfg = PgvectorConfig(
        dsn_env="MY_DSN",
        schema_name="custom",
        table_name="my_table",
        index_type="ivfflat",
        rebuild_index=True,
        hnsw_m=32,
        ivfflat_lists=200,
        pool_max=10,
    )
    assert cfg.dsn_env == "MY_DSN"
    assert cfg.index_type == "ivfflat"
    assert cfg.rebuild_index is True
    assert cfg.hnsw_m == 32
    assert cfg.ivfflat_lists == 200
    assert cfg.pool_max == 10


def test_pgvector_config_exact_index_type():
    cfg = PgvectorConfig(index_type="exact")
    assert cfg.index_type == "exact"


def test_pgvector_config_rejects_unknown_index_type():
    with pytest.raises(ValidationError):
        PgvectorConfig(index_type="unknown_type")


# ---------------------------------------------------------------------------
# IndexConfig — pgvector type
# ---------------------------------------------------------------------------

def test_index_config_pgvector_type_accepted():
    cfg = IndexConfig(type="pgvector", pgvector=PgvectorConfig())
    assert cfg.type == "pgvector"
    assert cfg.pgvector is not None


def test_index_config_faiss_type_still_works():
    cfg = IndexConfig(type="faiss")
    assert cfg.type == "faiss"
    assert cfg.pgvector is None


def test_index_config_elasticsearch_type_still_works():
    cfg = IndexConfig(type="elasticsearch")
    assert cfg.type == "elasticsearch"


def test_index_config_rejects_unknown_type():
    with pytest.raises(ValidationError):
        IndexConfig(type="mysql")


# ---------------------------------------------------------------------------
# BM25Config — host_env and analyzer
# ---------------------------------------------------------------------------

def test_bm25_config_default_analyzer_is_german():
    cfg = BM25Config()
    assert cfg.analyzer == "german"
    assert cfg.host_env is None


def test_bm25_config_host_env_set():
    cfg = BM25Config(host_env="ES_HOST_URL", analyzer="standard")
    assert cfg.host_env == "ES_HOST_URL"
    assert cfg.analyzer == "standard"


# ---------------------------------------------------------------------------
# HybridConfig — dense_backend
# ---------------------------------------------------------------------------

def test_hybrid_config_default_dense_backend_is_faiss():
    cfg = HybridConfig()
    assert cfg.dense_backend == "faiss"


def test_hybrid_config_pgvector_dense_backend():
    cfg = HybridConfig(dense_backend="pgvector")
    assert cfg.dense_backend == "pgvector"


def test_hybrid_config_rejects_unknown_dense_backend():
    with pytest.raises(ValidationError):
        HybridConfig(dense_backend="elasticsearch")


# ---------------------------------------------------------------------------
# PipelineConfig cross-validator
# ---------------------------------------------------------------------------

def _base_payload(**overrides):
    payload = {
        "experiment": {"experiment_id": "test", "random_seed": 42, "output_dir": "data/runs/test"},
        "data": {
            "documents_path": "data/raw/docs.jsonl",
            "questions_path": "data/raw/qs.jsonl",
        },
        "chunking": {"strategy": "fixed_token", "chunk_size": 256, "chunk_overlap": 32},
        "embedding": {"provider": "sentence_transformers", "model_name": "test-model"},
        "index": {"type": "faiss"},
        "retrieval": {"retriever_type": "dense", "top_k": 5, "fetch_k": 20},
        "reranker": {"enabled": False},
        "orchestration": {"fixed": True, "model_name": "mistral-small"},
        "generation": {
            "provider": "ollama",
            "model_name": "gemma3:1b",
            "system_prompt": "Answer.",
        },
        "telemetry": {"estimate_cost": False},
        "runtime": {"save_csv": False, "log_level": "INFO", "resume": False, "overwrite": True},
    }
    payload.update(overrides)
    return payload


def test_validator_pgvector_requires_pgvector_block():
    from src.pipeline1.schemas.config_schema import PipelineConfig

    payload = _base_payload(index={"type": "pgvector"})
    with pytest.raises((ValidationError, ValueError)):
        PipelineConfig.model_validate(payload)


def test_validator_pgvector_with_pgvector_block_passes():
    from src.pipeline1.schemas.config_schema import PipelineConfig

    payload = _base_payload(
        index={
            "type": "pgvector",
            "pgvector": {"dsn_env": "PGVECTOR_DSN"},
        },
        retrieval={"retriever_type": "dense", "top_k": 5, "fetch_k": 20},
    )
    cfg = PipelineConfig.model_validate(payload)
    assert cfg.index.type == "pgvector"


def test_validator_hybrid_pgvector_backend_requires_pgvector_index():
    from src.pipeline1.schemas.config_schema import PipelineConfig

    payload = _base_payload(
        index={"type": "faiss"},
        retrieval={
            "retriever_type": "hybrid_rrf",
            "top_k": 5,
            "fetch_k": 20,
            "hybrid": {"dense_backend": "pgvector"},
        },
    )
    with pytest.raises((ValidationError, ValueError)):
        PipelineConfig.model_validate(payload)
