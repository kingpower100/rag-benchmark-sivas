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


def _chunk(chunk_id: str = "c1"):
    chunk = MagicMock()
    chunk.chunk_id = chunk_id
    chunk.document_id = "doc1"
    chunk.original_context_id = "ctx1"
    chunk.text = f"text {chunk_id}"
    chunk.metadata = {}
    return chunk


def _identity() -> dict:
    return {
        "dataset_fingerprint": "dataset-a",
        "source_document_fingerprint": "docs-a",
        "chunk_store_fingerprint": "chunks-a",
        "chunking_configuration_fingerprint": "chunk-cfg-a",
        "embedding_model_name": "model-a",
        "embedding_normalization": True,
        "framework_config_hash": "cfg-a",
        "framework_code_version": "commit-a",
    }


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


def test_pgvector_index_build_reuses_only_when_manifest_and_chunk_ids_match():
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    embeddings = np.zeros((1, 1024), dtype="float32")
    expected = idx._expected_identity(embeddings)
    cur.fetchone.side_effect = [(1,), ({"identity": expected},)]
    cur.fetchall.return_value = [("c1",)]
    idx.build(embeddings)
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    upsert_calls = [c for c in execute_calls if "INSERT" in c or "ON CONFLICT" in c]
    assert len(upsert_calls) == 0
    assert idx.last_health["reuse_allowed"] is True
    assert idx.last_health["reuse_rejected"] is False


def test_pgvector_index_missing_manifest_forces_rebuild_and_updates_manifest():
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    cur.fetchone.side_effect = [(1,), None, (1,)]
    embeddings = np.zeros((1, 1024), dtype="float32")
    idx.build(embeddings)
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("TRUNCATE TABLE" in c for c in execute_calls)
    assert any("INSERT INTO" in c and "ON CONFLICT" in c for c in execute_calls)
    assert idx.last_health["reuse_allowed"] is False
    assert idx.last_health["rejection_reason"] == "manifest_missing"
    assert idx.last_health["rebuilt_row_count"] == 1


def test_pgvector_index_matching_count_different_chunk_fingerprint_forces_rebuild():
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    embeddings = np.zeros((1, 1024), dtype="float32")
    stale = dict(idx._expected_identity(embeddings))
    stale["chunk_store_fingerprint"] = "stale-chunks"
    cur.fetchone.side_effect = [(1,), ({"identity": stale},), (1,)]
    idx.build(embeddings)
    assert idx.last_health["rejection_reason"] == "chunk_fingerprint_mismatch"


def test_pgvector_index_matching_count_different_chunk_ids_forces_rebuild():
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    embeddings = np.zeros((1, 1024), dtype="float32")
    expected = idx._expected_identity(embeddings)
    cur.fetchone.side_effect = [(1,), ({"identity": expected},), (1,)]
    cur.fetchall.return_value = [("stale-c1",)]
    idx.build(embeddings)
    assert idx.last_health["rejection_reason"] == "chunk_id_fingerprint_mismatch"


def test_pgvector_index_same_chunk_id_different_content_changes_fingerprint():
    pool, _, _ = _mock_pool()
    old_chunk = _chunk("stable-id")
    old_chunk.text = "old text"
    old_chunk.chunk_start = 0
    old_chunk.chunk_end = 8
    new_chunk = _chunk("stable-id")
    new_chunk.text = "new text"
    new_chunk.chunk_start = 0
    new_chunk.chunk_end = 9
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)

    idx.set_chunks([old_chunk])
    old_fingerprint = idx._chunk_content_fingerprint()
    idx.set_chunks([new_chunk])
    new_fingerprint = idx._chunk_content_fingerprint()

    assert old_fingerprint != new_fingerprint


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("embedding_model_name", "other-model", "embedding_model_mismatch"),
        ("embedding_dimension", 768, "embedding_dimension_mismatch"),
        ("embedding_normalization", False, "normalization_mismatch"),
        ("distance_metric", "l2", "distance_metric_mismatch"),
        ("manifest_format_version", "0.9", "manifest_version_mismatch"),
        ("table_name", "other_table", "table_identity_mismatch"),
    ],
)
def test_pgvector_index_manifest_mismatches_force_rebuild(field, value, reason):
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    embeddings = np.zeros((1, 1024), dtype="float32")
    stale = dict(idx._expected_identity(embeddings))
    stale[field] = value
    cur.fetchone.side_effect = [(1,), ({"identity": stale},), (1,)]
    idx.build(embeddings)
    assert idx.last_health["rejection_reason"] == reason


def test_pgvector_index_rebuild_requested_replaces_rows():
    pool, conn, cur = _mock_pool()
    chunk = _chunk("c1")
    idx = PgvectorIndex(logical_index_name="logical", rebuild_index=True, _pool=pool)
    idx.set_chunks([chunk])
    idx.set_artifact_identity(_identity())
    cur.fetchone.side_effect = [(1,), None, (1,)]
    embeddings = np.zeros((1, 1024), dtype="float32")
    idx.build(embeddings)
    execute_calls = [str(c) for c in cur.execute.call_args_list]
    assert any("TRUNCATE TABLE" in c for c in execute_calls)
    assert idx.last_health["rejection_reason"] == "rebuild_index_requested"


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
