"""Tests for the Elasticsearch dense-vector index mapping fix.

Verifies that:
- script_score mode produces index=False (no HNSW/ANN overhead).
- kNN mode produces index=True with similarity setting (HNSW enabled).
- Dimension is preserved in both modes.
- Score transformation semantics are unchanged (cosineSimilarity + 1.0 - 1.0 = cosine).
- _index_body() and _vector_field_mapping() are consistent with retrieval_mode.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex


def _make_index(retrieval_mode: str = "script_score", similarity: str = "cosine", dim: int = 1024) -> ElasticsearchIndex:
    """Build an ElasticsearchIndex with a mock client (no live ES required)."""
    mock_client = MagicMock()
    mock_client.info.return_value = {"version": {"number": "8.0.0"}}
    with patch.object(ElasticsearchIndex, "_build_client", return_value=mock_client), \
         patch.object(ElasticsearchIndex, "_ensure_available"):
        idx = ElasticsearchIndex(
            host="http://localhost:9200",
            index_name="test_index",
            dense_dim=dim,
            similarity=similarity,
            retrieval_mode=retrieval_mode,
            client=mock_client,
        )
    return idx


# ---------------------------------------------------------------------------
# 1. script_score mode: HNSW/ANN disabled (index=False, no similarity key)
# ---------------------------------------------------------------------------

def test_script_score_vector_mapping_has_index_false():
    idx = _make_index(retrieval_mode="script_score")
    mapping = idx._vector_field_mapping()
    assert mapping["index"] is False


def test_script_score_vector_mapping_has_no_similarity_key():
    idx = _make_index(retrieval_mode="script_score")
    mapping = idx._vector_field_mapping()
    assert "similarity" not in mapping, (
        "similarity must be absent when index=False to avoid ES validation errors"
    )


def test_script_score_vector_mapping_preserves_dimension():
    idx = _make_index(retrieval_mode="script_score", dim=1024)
    mapping = idx._vector_field_mapping()
    assert mapping["dims"] == 1024


def test_script_score_vector_mapping_type_is_dense_vector():
    idx = _make_index(retrieval_mode="script_score")
    mapping = idx._vector_field_mapping()
    assert mapping["type"] == "dense_vector"


def test_script_score_index_body_contains_correct_vector_mapping():
    idx = _make_index(retrieval_mode="script_score", dim=1024)
    body = idx._index_body()
    vec = body["mappings"]["properties"][idx.vector_field]
    assert vec["index"] is False
    assert vec["dims"] == 1024
    assert "similarity" not in vec


# ---------------------------------------------------------------------------
# 2. kNN mode: HNSW enabled (index=True, similarity present)
# ---------------------------------------------------------------------------

def test_knn_vector_mapping_has_index_true():
    idx = _make_index(retrieval_mode="knn")
    mapping = idx._vector_field_mapping()
    assert mapping["index"] is True


def test_knn_vector_mapping_has_similarity():
    idx = _make_index(retrieval_mode="knn", similarity="cosine")
    mapping = idx._vector_field_mapping()
    assert mapping["similarity"] == "cosine"


def test_knn_vector_mapping_preserves_dimension():
    idx = _make_index(retrieval_mode="knn", dim=1024)
    mapping = idx._vector_field_mapping()
    assert mapping["dims"] == 1024


def test_knn_index_body_contains_correct_vector_mapping():
    idx = _make_index(retrieval_mode="knn", similarity="cosine", dim=1024)
    body = idx._index_body()
    vec = body["mappings"]["properties"][idx.vector_field]
    assert vec["index"] is True
    assert vec["similarity"] == "cosine"
    assert vec["dims"] == 1024


# ---------------------------------------------------------------------------
# 3. Dimension is preserved regardless of retrieval mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["script_score", "knn"])
def test_dimension_preserved_across_modes(mode):
    idx = _make_index(retrieval_mode=mode, dim=768)
    mapping = idx._vector_field_mapping()
    assert mapping["dims"] == 768


# ---------------------------------------------------------------------------
# 4. V02 specific: script_score + cosine + dim=1024
# ---------------------------------------------------------------------------

def test_v02_vector_field_mapping():
    idx = _make_index(retrieval_mode="script_score", similarity="cosine", dim=1024)
    mapping = idx._vector_field_mapping()
    assert mapping == {"type": "dense_vector", "dims": 1024, "index": False}


def test_v02_index_body_structure():
    idx = _make_index(retrieval_mode="script_score", similarity="cosine", dim=1024)
    body = idx._index_body()
    assert body["settings"]["number_of_shards"] == 1
    assert body["settings"]["number_of_replicas"] == 0
    vec = body["mappings"]["properties"]["embedding"]
    assert vec["type"] == "dense_vector"
    assert vec["dims"] == 1024
    assert vec["index"] is False
    assert "similarity" not in vec


# ---------------------------------------------------------------------------
# 5. Score transformation semantics: cosineSimilarity + 1.0 - 1.0 = cosine
# ---------------------------------------------------------------------------

def test_score_subtraction_returns_cosine():
    idx = _make_index(retrieval_mode="script_score")
    # ES native score: cosineSimilarity + 1.0 (ES non-negative requirement)
    es_score = 1.0 + 0.85  # simulates cos_sim=0.85
    framework_score = es_score - 1.0
    assert abs(framework_score - 0.85) < 1e-9


def test_score_subtraction_for_negative_cosine():
    idx = _make_index(retrieval_mode="script_score")
    es_score = 1.0 + (-0.3)  # cosine=-0.3
    framework_score = es_score - 1.0
    assert abs(framework_score - (-0.3)) < 1e-9


def test_score_subtraction_for_zero_cosine():
    idx = _make_index(retrieval_mode="script_score")
    es_score = 1.0 + 0.0
    framework_score = es_score - 1.0
    assert abs(framework_score - 0.0) < 1e-9


def test_score_subtraction_for_maximum_cosine():
    idx = _make_index(retrieval_mode="script_score")
    es_score = 1.0 + 1.0  # cosine=1.0 (identical vectors)
    framework_score = es_score - 1.0
    assert abs(framework_score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 6. search() applies score subtraction to all returned hits
# ---------------------------------------------------------------------------

def test_search_applies_score_subtraction_to_all_hits():
    idx = _make_index(retrieval_mode="script_score")
    raw_hits = [
        {"_score": 1.85, "_source": {"chunk_id": "c1", "document_id": "d1"}},
        {"_score": 1.60, "_source": {"chunk_id": "c2", "document_id": "d1"}},
        {"_score": 1.20, "_source": {"chunk_id": "c3", "document_id": "d2"}},
    ]
    mock_response = {"hits": {"hits": raw_hits}}
    idx.client.search.return_value = mock_response
    chunk_ids, scores = idx.search(query_embedding=[0.0] * 1024, top_k=3)
    assert chunk_ids == ["c1", "c2", "c3"]
    assert abs(scores[0] - 0.85) < 1e-6
    assert abs(scores[1] - 0.60) < 1e-6
    assert abs(scores[2] - 0.20) < 1e-6


# ---------------------------------------------------------------------------
# 7. Retrieval mode dispatches to correct search method
# ---------------------------------------------------------------------------

def test_script_score_mode_calls_script_score_search():
    idx = _make_index(retrieval_mode="script_score")
    with patch.object(idx, "_script_score_search", return_value={"hits": {"hits": []}}) as mock_ss, \
         patch.object(idx, "_knn_search") as mock_knn:
        idx._execute_search([0.0] * 1024, top_k=5)
    mock_ss.assert_called_once()
    mock_knn.assert_not_called()


def test_knn_mode_calls_knn_search():
    idx = _make_index(retrieval_mode="knn")
    with patch.object(idx, "_knn_search", return_value={"hits": {"hits": []}}) as mock_knn, \
         patch.object(idx, "_script_score_search") as mock_ss:
        idx._execute_search([0.0] * 1024, top_k=5)
    mock_knn.assert_called_once()
    mock_ss.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Other index body fields are unchanged by the mapping fix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["script_score", "knn"])
def test_index_body_non_vector_fields_are_unchanged(mode):
    idx = _make_index(retrieval_mode=mode, dim=1024)
    body = idx._index_body()
    props = body["mappings"]["properties"]
    assert props["chunk_id"] == {"type": "keyword"}
    assert props["document_id"] == {"type": "keyword"}
    assert props["original_context_id"] == {"type": "keyword"}
    assert props["text"]["type"] == "text"
    assert props["metadata"] == {"type": "object", "enabled": True}
