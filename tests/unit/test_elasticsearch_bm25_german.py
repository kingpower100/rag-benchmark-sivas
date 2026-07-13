"""Tests for ElasticsearchBM25Retriever German analyzer and diagnostics."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Retriever
from src.pipeline1.schemas.chunk import ChunkRecord


def _chunk(chunk_id: str, text: str = "Text") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"ctx-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={},
    )


def _mock_es_client(hits=None):
    client = MagicMock()
    client.info.return_value = {}
    client.indices.exists.return_value = True
    client.indices.create.return_value = {}
    client.search.return_value = {
        "hits": {
            "hits": hits or [
                {"_id": "c1", "_score": 1.5, "_source": {"chunk_id": "c1", "cleaned_context": "text", "metadata": {}}}
            ]
        }
    }
    return client


def _retriever(analyzer="german", hits=None):
    chunks = [_chunk("c1")]
    client = _mock_es_client(hits=hits)
    return ElasticsearchBM25Retriever(
        chunks=chunks,
        host="http://localhost:9200",
        index_name="test_index",
        client=client,
        analyzer=analyzer,
    ), client


def test_analyzer_stored_on_init():
    retriever, _ = _retriever(analyzer="german")
    assert retriever.analyzer == "german"


def test_analyzer_custom_value():
    retriever, _ = _retriever(analyzer="standard")
    assert retriever.analyzer == "standard"


def test_index_body_uses_analyzer():
    retriever, _ = _retriever(analyzer="german")
    body = retriever._index_body()
    cleaned_ctx = body["mappings"]["properties"]["cleaned_context"]
    assert cleaned_ctx.get("analyzer") == "german"


def test_last_retrieval_diagnostics_populated_after_retrieve():
    retriever, _ = _retriever()
    retriever.retrieve("Was ist das?", top_k=1)
    diag = retriever.last_retrieval_diagnostics
    assert diag.get("backend") == "elasticsearch_bm25"
    assert diag.get("analyzer") == "german"
    assert "hits_count" in diag


def test_retrieve_populates_last_bm25_candidates():
    retriever, _ = _retriever()
    results = retriever.retrieve("query", top_k=1)
    assert len(retriever.last_bm25_candidates) == 1
    assert retriever.last_bm25_candidates[0].chunk_id == "c1"
