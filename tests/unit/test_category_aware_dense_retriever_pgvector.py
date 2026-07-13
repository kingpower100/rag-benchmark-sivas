"""Tests for CategoryAwareDenseRetriever in pgvector mode."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem


def _item(chunk_id: str, score: float = 0.9, category: str = "") -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"ctx-{chunk_id}",
        text="text",
        score=score,
        retrieval_source="pgvector",
        metadata={"kategorie": category},
    )


def _pgvector_retriever(items=None, category_items=None):
    from src.pipeline1.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever

    ret = MagicMock(spec=PgvectorDenseRetriever)
    ret.retrieve.return_value = items or [_item("c1", 0.9, "Finanzen")]
    ret.last_dense_candidates = items or [_item("c1", 0.9, "Finanzen")]
    ret.last_retrieval_diagnostics = {"retrieval_backend": "pgvector"}
    ret.chunks = []
    return ret


def test_pgvector_mode_detected():
    inner = _pgvector_retriever()
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    assert cadr._pgvector_mode is True


def test_non_pgvector_mode_not_detected():
    inner = MagicMock()
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    assert cadr._pgvector_mode is False


def test_set_active_category_propagates_to_inner_retriever():
    inner = _pgvector_retriever()
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    cadr.set_active_category("Einkauf")
    assert cadr.active_category == "Einkauf"
    inner.set_active_category.assert_called_once_with("Einkauf")


def test_set_active_category_none_clears():
    inner = _pgvector_retriever()
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    cadr.set_active_category("Finanzen")
    cadr.set_active_category(None)
    assert cadr.active_category is None


def test_retrieve_delegates_to_inner_in_pgvector_mode():
    items = [_item("c1", 0.95, "Finanzen")]
    inner = _pgvector_retriever(items=items)
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    cadr.set_active_category("Finanzen")
    result = cadr.retrieve("Was ist Finanzen?", top_k=1)
    assert len(result) == 1
    assert result[0].chunk_id == "c1"
    inner.retrieve.assert_called_once()


def test_pgvector_mode_does_not_build_faiss_sub_indexes():
    inner = _pgvector_retriever()
    import numpy as np
    embeddings = np.zeros((1, 4), dtype="float32")
    cadr = CategoryAwareDenseRetriever(
        dense_retriever=inner,
        embeddings=embeddings,
    )
    assert cadr._category_retrievers == {}


def test_diagnostics_include_category_index_used_true_when_category_set():
    items = [_item("c1", 0.9, "Finanzen")]
    inner = _pgvector_retriever(items=items)
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    cadr.set_active_category("Finanzen")
    cadr.retrieve("query", top_k=1)
    assert cadr.last_retrieval_diagnostics.get("category_index_used") is True


def test_diagnostics_category_index_used_false_when_no_category():
    items = [_item("c1", 0.9)]
    inner = _pgvector_retriever(items=items)
    cadr = CategoryAwareDenseRetriever(dense_retriever=inner)
    cadr.set_active_category(None)
    cadr.retrieve("query", top_k=1)
    assert cadr.last_retrieval_diagnostics.get("category_index_used") is False
