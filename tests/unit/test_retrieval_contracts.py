import pytest

from src.pipeline1.retrieval.adapters import (
    retrieval_item_to_search_result,
    retrieval_items_to_search_results,
    search_result_to_retrieval_item,
    search_results_to_retrieval_items,
)
from src.pipeline1.retrieval.contracts import DedupePolicy, SearchResult
from src.pipeline1.retrieval.dedupe import dedupe_search_results
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline2.metrics.retrieval_metrics import compute_retrieval_metrics_for_ks


def test_search_result_converts_to_retrieval_item_without_field_loss():
    result = SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        original_context_id="ctx-1",
        text="Revenue was 10.",
        score=0.87,
        rank=2,
        retrieval_backend="dense",
        metadata={"file_name": "source.txt", "chunk_unit": "token"},
        diagnostics={
            "dense_score": 0.82,
            "metadata_boost": 0.05,
            "metadata_boost_components": {"year": 0.05},
            "ranking_score_type": "dense_score_plus_metadata",
            "score_before_metadata": 0.82,
        },
    )

    item = search_result_to_retrieval_item(result)
    roundtrip = retrieval_item_to_search_result(item)

    assert item.chunk_id == "chunk-1"
    assert item.original_context_id == "ctx-1"
    assert item.text == "Revenue was 10."
    assert item.score == pytest.approx(0.87)
    assert item.dense_score == pytest.approx(0.82)
    assert item.metadata["document_id"] == "doc-1"
    assert item.metadata["file_name"] == "source.txt"
    assert item.chunk_unit == "token"
    assert roundtrip.document_id == "doc-1"
    assert roundtrip.rank == 2
    assert roundtrip.diagnostics["metadata_boost_components"] == {"year": 0.05}


def test_retrieval_item_converts_to_search_result_without_field_loss():
    item = RetrievalItem(
        chunk_id="chunk-1",
        original_context_id="ctx-1",
        text="Revenue was 10.",
        score=0.7,
        dense_score=0.6,
        bm25_score=None,
        rerank_score=0.7,
        ranking_score_type="rerank_score",
        retrieval_source="dense",
        chunk_unit="token",
        metadata={"document_id": "doc-1", "file_name": "source.txt", "chunk_unit": "token"},
        metadata_boost=0.1,
        metadata_boost_components={"company": 0.1},
        score_before_metadata=0.6,
    )

    result = retrieval_item_to_search_result(item, rank=1)
    converted = search_result_to_retrieval_item(result)

    assert result.chunk_id == item.chunk_id
    assert result.document_id == "doc-1"
    assert result.original_context_id == item.original_context_id
    assert result.text == item.text
    assert result.score == pytest.approx(item.score)
    assert result.rank == 1
    assert result.metadata["file_name"] == "source.txt"
    assert converted.rerank_score == pytest.approx(0.7)
    assert converted.metadata_boost_components == {"company": 0.1}


def test_bulk_adapter_helpers_preserve_order():
    items = [
        RetrievalItem(chunk_id="c1", original_context_id="ctx1", text="one", score=1.0),
        RetrievalItem(chunk_id="c2", original_context_id="ctx2", text="two", score=0.5),
    ]

    results = retrieval_items_to_search_results(items)
    converted = search_results_to_retrieval_items(results)

    assert [result.rank for result in results] == [1, 2]
    assert [item.chunk_id for item in converted] == ["c1", "c2"]


def test_dedupe_by_chunk_id_works_and_matches_default():
    results = [
        _result("c1", "doc1", "ctx1", 1.0),
        _result("c1", "doc2", "ctx2", 0.9),
        _result("c2", "doc2", "ctx2", 0.8),
    ]

    explicit, explicit_diag = dedupe_search_results(results, top_k=10, policy=DedupePolicy.CHUNK_ID)
    default, default_diag = dedupe_search_results(results, top_k=10)

    assert [item.chunk_id for item in explicit] == ["c1", "c2"]
    assert [item.chunk_id for item in default] == ["c1", "c2"]
    assert explicit_diag["dedupe_policy"] == "chunk_id"
    assert default_diag["duplicate_collapse_count"] == 1


def test_dedupe_by_document_id_works():
    results = [
        _result("c1", "doc1", "ctx1", 1.0),
        _result("c2", "doc1", "ctx2", 0.9),
        _result("c3", "doc2", "ctx3", 0.8),
    ]

    deduped, diagnostics = dedupe_search_results(results, top_k=10, policy=DedupePolicy.DOCUMENT_ID)

    assert [item.chunk_id for item in deduped] == ["c1", "c3"]
    assert diagnostics["dedupe_policy"] == "document_id"
    assert diagnostics["duplicate_collapse_count"] == 1


def test_dedupe_by_original_context_id_works():
    results = [
        _result("c1", "doc1", "ctx1", 1.0),
        _result("c2", "doc2", "ctx1", 0.9),
        _result("c3", "doc3", "ctx3", 0.8),
    ]

    deduped, diagnostics = dedupe_search_results(results, top_k=10, policy=DedupePolicy.ORIGINAL_CONTEXT_ID)

    assert [item.chunk_id for item in deduped] == ["c1", "c3"]
    assert diagnostics["dedupe_policy"] == "original_context_id"


def test_dedupe_none_preserves_duplicates():
    results = [_result("c1", "doc1", "ctx1", 1.0), _result("c1", "doc1", "ctx1", 0.9)]

    deduped, diagnostics = dedupe_search_results(results, top_k=10, policy=DedupePolicy.NONE)

    assert [item.score for item in deduped] == [1.0, 0.9]
    assert diagnostics["duplicate_collapse_count"] == 0


def test_pipeline2_retrieval_metrics_accept_adapter_outputs():
    results = [
        _result("chunk-1", "doc-key-1", "doc-key-1", 1.0),
        _result("chunk-2", "doc-key-2", "doc-key-2", 0.5),
    ]
    items = search_results_to_retrieval_items(results)

    metrics = compute_retrieval_metrics_for_ks(
        [item.original_context_id for item in items],
        ["doc-key-1"],
        [1, 2],
    )

    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 1.0
    assert metrics["hit_at_2"] == 1.0


def _result(chunk_id: str, document_id: str, original_context_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        original_context_id=original_context_id,
        text=chunk_id,
        score=score,
        retrieval_backend="dense",
        metadata={"document_id": document_id, "file_name": f"{document_id}.txt"},
    )
