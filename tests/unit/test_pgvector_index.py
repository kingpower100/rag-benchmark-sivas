"""Unit tests for PgvectorIndex using mocked psycopg2."""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

from src.pipeline1.indexing.pgvector_index import PgvectorIndex, _vec_to_pgvector


# ---------------------------------------------------------------------------
# _vec_to_pgvector helper
# ---------------------------------------------------------------------------

def test_vec_to_pgvector_produces_bracket_format():
    vec = np.array([0.1, 0.2, 0.3], dtype="float32")
    result = _vec_to_pgvector(vec)
    assert result.startswith("[")
    assert result.endswith("]")
    assert "0.1" in result or "0.10" in result


def test_vec_to_pgvector_single_element():
    vec = np.array([1.0], dtype="float32")
    assert _vec_to_pgvector(vec) == "[1.0]"


# ---------------------------------------------------------------------------
# PgvectorIndex construction
# ---------------------------------------------------------------------------

def _mock_pool():
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
    return pool, conn, cur


def test_pgvector_index_construction_defaults():
    pool, _, _ = _mock_pool()
    idx = PgvectorIndex(_pool=pool)
    assert idx.schema_name == "rag"
    assert idx.table_name == "chunk_embeddings"
    assert idx.dense_dim == 1024
    assert idx.metric == "cosine"
    assert idx.index_type == "hnsw"
    assert idx.uses_external_storage is True


def test_pgvector_index_set_chunks():
    pool, _, _ = _mock_pool()
    idx = PgvectorIndex(_pool=pool)
    chunk = MagicMock()
    chunk.chunk_id = "c1"
    idx.set_chunks([chunk])
    assert "c1" in idx._chunk_by_id
    assert idx._chunks == [chunk]


def test_pgvector_index_dim_property():
    pool, _, _ = _mock_pool()
    idx = PgvectorIndex(dense_dim=768, _pool=pool)
    assert idx.dim == 768


def test_pgvector_index_ntotal_queries_db():
    pool, conn, cur = _mock_pool()
    cur.fetchone.return_value = (42,)
    idx = PgvectorIndex(_pool=pool)
    total = idx.ntotal
    assert total == 42


def test_pgvector_index_save_writes_pointer_file():
    import tempfile, pathlib
    pool, _, _ = _mock_pool()
    idx = PgvectorIndex(schema_name="myschema", table_name="mytable", _pool=pool)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "index.pgvector")
        idx.save(path)
        content = pathlib.Path(path).read_text(encoding="utf-8")
    assert "myschema" in content
    assert "mytable" in content


def test_pgvector_index_build_skips_upsert_when_count_matches():
    pool, conn, cur = _mock_pool()
    chunk = MagicMock()
    chunk.chunk_id = "c1"
    chunk.metadata = {}
    cur.fetchone.return_value = (1,)
    idx = PgvectorIndex(_pool=pool)
    idx.set_chunks([chunk])
    embeddings = np.zeros((1, 1024), dtype="float32")
    idx.build(embeddings)
    # No upsert should have been called since count already matches
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    upsert_calls = [c for c in execute_calls if "INSERT" in c or "ON CONFLICT" in c]
    assert len(upsert_calls) == 0


def test_pgvector_index_search_returns_chunk_ids_and_scores():
    pool, conn, cur = _mock_pool()
    cur.fetchall.return_value = [("chunk-1", 0.95), ("chunk-2", 0.87)]
    idx = PgvectorIndex(_pool=pool)
    qvec = np.zeros(1024, dtype="float32")
    chunk_ids, scores = idx.search(qvec, top_k=2)
    assert chunk_ids == ["chunk-1", "chunk-2"]
    assert scores == pytest.approx([0.95, 0.87])


def test_pgvector_index_search_category_passes_category_filter():
    pool, conn, cur = _mock_pool()
    cur.fetchall.return_value = [("chunk-3", 0.80)]
    idx = PgvectorIndex(_pool=pool)
    qvec = np.zeros(1024, dtype="float32")
    chunk_ids, scores = idx.search_category(qvec, top_k=5, category="Finanzen")
    assert chunk_ids == ["chunk-3"]
    sql_call = str(cur.execute.call_args_list[-1])
    assert "Finanzen" in sql_call or "category" in sql_call.lower()
