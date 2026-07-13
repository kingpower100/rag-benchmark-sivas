"""Unit tests for PgvectorDenseRetriever with mocked index."""
from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pipeline1.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import MetadataBoostingConfig, MetadataFilteringConfig


def _chunk(chunk_id: str, text: str = "text", category: str = "") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"ctx-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"kategorie": category},
    )


def _embedder(vec=None):
    emb = MagicMock()
    emb.encode_query.return_value = vec if vec is not None else np.zeros(4, dtype="float32")
    return emb


def _index(chunk_ids=None, scores=None):
    idx = MagicMock()
    idx.search.return_value = (chunk_ids or [], scores or [])
    idx.search_category.return_value = (chunk_ids or [], scores or [])
    return idx


def _no_boosting():
    m = MetadataBoostingConfig()
    m = MetadataBoostingConfig(enabled=False)
    return m


def _no_filtering():
    return MetadataFilteringConfig(enabled=False)


def _retriever(chunks=None, chunk_ids=None, scores=None):
    if chunks is None:
        chunks = [_chunk("c1", "alpha", "Finanzen"), _chunk("c2", "beta", "Einkauf")]
    idx = _index(chunk_ids or ["c1"], scores or [0.9])
    return PgvectorDenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=10,
        metadata_boosting=_no_boosting(),
        metadata_filtering=_no_filtering(),
    ), idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_returns_retrieval_items():
    retriever, idx = _retriever()
    results = retriever.retrieve("Wie buche ich?", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == pytest.approx(0.9)


def test_retrieve_calls_index_search():
    retriever, idx = _retriever()
    retriever.retrieve("query", top_k=1)
    idx.search.assert_called_once()


def test_set_active_category_uses_search_category():
    chunks = [_chunk("c1", "text", "Finanzen")]
    idx = _index(["c1"], [0.88])
    retriever = PgvectorDenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=5,
        metadata_boosting=_no_boosting(),
        metadata_filtering=_no_filtering(),
    )
    retriever.set_active_category("Finanzen")
    retriever.retrieve("query", top_k=1)
    idx.search_category.assert_called_once()
    idx.search.assert_not_called()


def test_set_active_category_none_uses_global_search():
    retriever, idx = _retriever()
    retriever.set_active_category(None)
    retriever.retrieve("query", top_k=1)
    idx.search.assert_called_once()
    idx.search_category.assert_not_called()


def test_last_dense_candidates_populated():
    retriever, _ = _retriever()
    retriever.retrieve("query", top_k=1)
    assert len(retriever.last_dense_candidates) == 1
    assert retriever.last_dense_candidates[0].chunk_id == "c1"


def test_last_retrieval_diagnostics_populated():
    retriever, _ = _retriever()
    retriever.retrieve("query", top_k=1)
    diag = retriever.last_retrieval_diagnostics
    assert "candidates_before_filter" in diag
    assert "retrieval_backend" in diag


def test_missing_chunk_id_in_index_result_is_skipped():
    chunks = [_chunk("c1")]
    idx = _index(["c99", "c1"], [0.9, 0.8])
    retriever = PgvectorDenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=5,
        metadata_boosting=_no_boosting(),
        metadata_filtering=_no_filtering(),
    )
    results = retriever.retrieve("query", top_k=2)
    assert all(r.chunk_id != "c99" for r in results)
    assert any(r.chunk_id == "c1" for r in results)


def test_retrieval_backend_label_is_pgvector():
    retriever, _ = _retriever()
    results = retriever.retrieve("query", top_k=1)
    assert results[0].retrieval_source == "pgvector"
