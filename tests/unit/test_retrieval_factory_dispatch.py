"""Tests for retrieval/factory.py typed dispatch across FAISS, pgvector, Elasticsearch."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from src.pipeline1.schemas.config_schema import (
    BM25Config,
    HybridConfig,
    MetadataBoostingConfig,
    MetadataFilteringConfig,
    RetrievalConfig,
)


def _retrieval_config(**overrides) -> RetrievalConfig:
    defaults = {
        "retriever_type": "dense",
        "top_k": 5,
        "fetch_k": 20,
        "metadata_boosting": MetadataBoostingConfig(enabled=False),
        "metadata_filtering": MetadataFilteringConfig(enabled=False),
        "bm25": BM25Config(backend="local"),
        "hybrid": HybridConfig(),
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def _fake_chunks():
    chunk = MagicMock()
    chunk.chunk_id = "c1"
    chunk.metadata = {}
    chunk.text = "text"
    return [chunk]


def _faiss_index():
    from src.pipeline1.indexing.faiss_index import FaissIndex
    return FaissIndex(metric="cosine")


def _pgvector_index():
    from src.pipeline1.indexing.pgvector_index import PgvectorIndex
    pool = MagicMock()
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cur
    cur.fetchone.return_value = (0,)
    cur.fetchall.return_value = []
    pool.getconn.return_value = conn
    pool.putconn.return_value = None
    return PgvectorIndex(_pool=pool)


def _elasticsearch_index():
    from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex
    idx = MagicMock(spec=ElasticsearchIndex)
    idx.metric = "cosine"
    idx.uses_external_storage = True
    idx.__class__ = ElasticsearchIndex
    return idx


def test_faiss_index_dispatches_to_dense_retriever():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.dense_retriever import DenseRetriever

    cfg = _retrieval_config(retriever_type="dense")
    retriever = build_retriever(cfg, MagicMock(), _faiss_index(), _fake_chunks())
    assert isinstance(retriever, DenseRetriever)


def test_pgvector_index_dispatches_to_pgvector_dense_retriever():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever

    cfg = _retrieval_config(retriever_type="dense")
    retriever = build_retriever(cfg, MagicMock(), _pgvector_index(), _fake_chunks())
    assert isinstance(retriever, PgvectorDenseRetriever)


def test_elasticsearch_index_dispatches_to_elasticsearch_dense_retriever():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever

    cfg = _retrieval_config(retriever_type="dense")
    retriever = build_retriever(cfg, MagicMock(), _elasticsearch_index(), _fake_chunks())
    assert isinstance(retriever, ElasticsearchDenseRetriever)


def test_bm25_local_dispatches_to_bm25_retriever():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.bm25_retriever import BM25Retriever

    cfg = _retrieval_config(retriever_type="bm25", bm25=BM25Config(backend="local"))
    retriever = build_retriever(cfg, MagicMock(), _faiss_index(), _fake_chunks())
    assert isinstance(retriever, BM25Retriever)


def test_category_aware_dense_with_faiss_builds_category_aware_retriever():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever

    cfg = _retrieval_config(retriever_type="category_aware_dense")
    retriever = build_retriever(cfg, MagicMock(), _faiss_index(), _fake_chunks())
    assert isinstance(retriever, CategoryAwareDenseRetriever)
    assert retriever._pgvector_mode is False


def test_category_aware_dense_with_pgvector_builds_pgvector_mode():
    from src.pipeline1.retrieval.factory import build_retriever
    from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever

    cfg = _retrieval_config(retriever_type="category_aware_dense")
    retriever = build_retriever(cfg, MagicMock(), _pgvector_index(), _fake_chunks())
    assert isinstance(retriever, CategoryAwareDenseRetriever)
    assert retriever._pgvector_mode is True


def test_bm25_elasticsearch_host_env_overrides_host(monkeypatch):
    from src.pipeline1.retrieval.factory import _build_bm25_retriever
    from src.pipeline1.retrieval.bm25_retriever import BM25Retriever
    from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Error

    monkeypatch.setenv("MY_ES_HOST_TEST", "http://custom-host:9200")
    bm25_cfg = BM25Config(
        backend="elasticsearch",
        host_env="MY_ES_HOST_TEST",
        allow_fallback=True,
    )
    cfg = _retrieval_config(retriever_type="bm25", bm25=bm25_cfg)
    captured = []

    def fake_bm25(chunks, host, **kwargs):
        captured.append(host)
        raise ElasticsearchBM25Error("mocked unavailable")

    with patch("src.pipeline1.retrieval.factory.ElasticsearchBM25Retriever", side_effect=fake_bm25):
        result = _build_bm25_retriever(cfg, _fake_chunks())

    assert isinstance(result, BM25Retriever)
    assert captured == ["http://custom-host:9200"]
