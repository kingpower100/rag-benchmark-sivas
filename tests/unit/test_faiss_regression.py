"""Regression tests to verify FAISS backend behavior is unchanged after multi-backend additions."""
from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever
from src.pipeline1.retrieval.dense_retriever import DenseRetriever
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


def _faiss_index(scores=None, idxs=None):
    idx = MagicMock()
    idx.metric = "cosine"
    idx.uses_external_storage = False
    idx.search.return_value = (
        np.array(scores or [0.9, 0.8]),
        np.array(idxs or [0, 1]),
    )
    return idx


def _embedder():
    emb = MagicMock()
    emb.encode_query.return_value = np.zeros(4, dtype="float32")
    return emb


def test_dense_retriever_with_faiss_still_works():
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]
    idx = _faiss_index(scores=[0.9, 0.8], idxs=[0, 1])
    retriever = DenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=10,
        metadata_boosting=MetadataBoostingConfig(enabled=False),
        metadata_filtering=MetadataFilteringConfig(enabled=False),
    )
    results = retriever.retrieve("test query", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[1].chunk_id == "c2"


def test_category_aware_retriever_faiss_mode_not_pgvector():
    chunks = [_chunk("c1", "alpha", "Finanzen"), _chunk("c2", "beta", "Einkauf")]
    idx = _faiss_index(scores=[0.9, 0.8], idxs=[0, 1])
    dense = DenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=10,
        metadata_boosting=MetadataBoostingConfig(enabled=False),
        metadata_filtering=MetadataFilteringConfig(enabled=False),
    )
    cadr = CategoryAwareDenseRetriever(dense_retriever=dense)
    assert cadr._pgvector_mode is False


def test_category_aware_retriever_builds_sub_indexes_from_embeddings():
    chunks = [_chunk("c1", "alpha", "Finanzen"), _chunk("c2", "beta", "Einkauf")]
    idx = _faiss_index(scores=[0.9, 0.8], idxs=[0, 1])
    dense = DenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=10,
        metadata_boosting=MetadataBoostingConfig(enabled=False),
        metadata_filtering=MetadataFilteringConfig(enabled=False),
    )
    embeddings = np.random.rand(2, 4).astype("float32")
    cadr = CategoryAwareDenseRetriever(dense_retriever=dense, embeddings=embeddings)
    assert "Finanzen" in cadr._category_retrievers
    assert "Einkauf" in cadr._category_retrievers


def test_category_aware_retriever_global_fallback_faiss():
    chunks = [_chunk("c1", "alpha", "Finanzen"), _chunk("c2", "beta", "Einkauf")]
    idx = _faiss_index(scores=[0.9, 0.8], idxs=[0, 1])
    dense = DenseRetriever(
        embedder=_embedder(),
        index=idx,
        chunks=chunks,
        fetch_k=10,
        metadata_boosting=MetadataBoostingConfig(enabled=False),
        metadata_filtering=MetadataFilteringConfig(enabled=False),
    )
    cadr = CategoryAwareDenseRetriever(dense_retriever=dense)
    cadr.set_active_category(None)
    results = cadr.retrieve("query", top_k=2)
    assert len(results) == 2


def test_indexing_factory_faiss_still_default():
    from src.pipeline1.indexing.factory import build_index
    from src.pipeline1.indexing.faiss_index import FaissIndex
    from src.pipeline1.schemas.config_schema import IndexConfig

    cfg = IndexConfig(type="faiss", metric="cosine")
    index = build_index(cfg)
    assert isinstance(index, FaissIndex)
