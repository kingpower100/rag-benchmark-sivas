from __future__ import annotations

from src.pipeline1.retrieval.contracts import SearchResult
from src.pipeline1.schemas.retrieval import RetrievalItem


_ADAPTER_METADATA_KEY = "_search_result_adapter"


def retrieval_item_to_search_result(item: RetrievalItem, rank: int | None = None) -> SearchResult:
    metadata = dict(item.metadata or {})
    adapter_payload = metadata.pop(_ADAPTER_METADATA_KEY, {}) if isinstance(metadata.get(_ADAPTER_METADATA_KEY), dict) else {}
    document_id = (
        metadata.get("doc_key")
        or metadata.get("document_id")
        or metadata.get("doc_id")
        or adapter_payload.get("document_id")
        or item.original_context_id
    )
    resolved_rank = rank if rank is not None else adapter_payload.get("rank")
    diagnostics = dict(adapter_payload.get("diagnostics") or {})
    diagnostics.update(_retrieval_item_diagnostics(item))
    return SearchResult(
        chunk_id=item.chunk_id,
        document_id=str(document_id),
        original_context_id=item.original_context_id,
        text=item.text,
        score=float(item.score),
        rank=int(resolved_rank) if resolved_rank is not None else None,
        retrieval_backend=item.retrieval_source,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def search_result_to_retrieval_item(result: SearchResult) -> RetrievalItem:
    metadata = dict(result.metadata or {})
    metadata.setdefault("document_id", result.document_id)
    if result.rank is not None or result.diagnostics:
        metadata[_ADAPTER_METADATA_KEY] = {
            "rank": result.rank,
            "document_id": result.document_id,
            "diagnostics": dict(result.diagnostics or {}),
        }
    dense_score = result.diagnostics.get("dense_score")
    bm25_score = result.diagnostics.get("bm25_score")
    rrf_score = result.diagnostics.get("rrf_score")
    rerank_score = result.diagnostics.get("rerank_score")
    if dense_score is None and result.retrieval_backend in {"dense", "elasticsearch_dense"}:
        dense_score = result.score
    if bm25_score is None and result.retrieval_backend in {"bm25", "elasticsearch_bm25"}:
        bm25_score = result.score
    if rrf_score is None and result.retrieval_backend == "hybrid_rrf":
        rrf_score = result.score
    metadata_boost = float(result.diagnostics.get("metadata_boost") or 0.0)
    return RetrievalItem(
        chunk_id=result.chunk_id,
        original_context_id=result.original_context_id,
        text=result.text,
        score=float(result.score),
        dense_score=_optional_float(dense_score),
        bm25_score=_optional_float(bm25_score),
        rrf_score=_optional_float(rrf_score),
        rerank_score=_optional_float(rerank_score),
        ranking_score_type=str(result.diagnostics.get("ranking_score_type") or _default_ranking_score_type(result)),
        retrieval_source=result.retrieval_backend,
        chunk_unit=metadata.get("chunk_unit"),
        metadata=metadata,
        metadata_boost=metadata_boost,
        metadata_boost_components=dict(result.diagnostics.get("metadata_boost_components") or {}),
        score_before_metadata=_optional_float(result.diagnostics.get("score_before_metadata")),
        metadata_filter_matched=result.diagnostics.get("metadata_filter_matched"),
    )


def search_results_to_retrieval_items(results: list[SearchResult]) -> list[RetrievalItem]:
    return [search_result_to_retrieval_item(result) for result in results]


def retrieval_items_to_search_results(items: list[RetrievalItem]) -> list[SearchResult]:
    return [retrieval_item_to_search_result(item, rank=index) for index, item in enumerate(items, start=1)]


def strip_adapter_metadata(items: list[RetrievalItem]) -> list[RetrievalItem]:
    cleaned = []
    for item in items:
        metadata = dict(item.metadata or {})
        metadata.pop(_ADAPTER_METADATA_KEY, None)
        cleaned.append(item.model_copy(update={"metadata": metadata}))
    return cleaned


def _retrieval_item_diagnostics(item: RetrievalItem) -> dict:
    return {
        "dense_score": item.dense_score,
        "bm25_score": item.bm25_score,
        "rrf_score": item.rrf_score,
        "rerank_score": item.rerank_score,
        "ranking_score_type": item.ranking_score_type,
        "metadata_boost": item.metadata_boost,
        "metadata_boost_components": dict(item.metadata_boost_components or {}),
        "score_before_metadata": item.score_before_metadata,
        "metadata_filter_matched": item.metadata_filter_matched,
    }


def _default_ranking_score_type(result: SearchResult) -> str:
    if result.retrieval_backend in {"bm25", "elasticsearch_bm25"}:
        return "bm25_score"
    if result.retrieval_backend == "hybrid_rrf":
        return "rrf_score"
    return "dense_score"


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)
