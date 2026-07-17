from __future__ import annotations

import time

from src.pipeline1.retrieval.adapters import (
    retrieval_items_to_search_results,
    search_results_to_retrieval_items,
    strip_adapter_metadata,
)
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery
from src.pipeline1.schemas.retrieval import RetrievalItem


class ElasticsearchHybridRRFRetriever(BaseRetriever):
    """Fuses Elasticsearch dense vector search and BM25 search via Reciprocal Rank Fusion.

    Both retrieval legs run against Elasticsearch (or any retriever implementing
    the BaseRetriever.retrieve interface). Each candidate is identified by its
    chunk_id. Items returned by both legs receive contributions from both RRF terms;
    items returned by only one leg receive a single contribution.

    Policy for malformed results: items with an empty chunk_id string are silently
    skipped during fusion so that a single bad Elasticsearch hit does not abort the
    entire retrieval call.  All remaining valid items are fused normally.

    Category filtering is intentionally not implemented.  This retriever always
    performs global (full-index) retrieval on both legs.
    """

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        fetch_k: int,
        dense_fetch_k: int | None = None,
        bm25_fetch_k: int | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fetch_k = fetch_k
        self.dense_fetch_k = dense_fetch_k if dense_fetch_k is not None else fetch_k
        self.bm25_fetch_k = bm25_fetch_k if bm25_fetch_k is not None else fetch_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_bm25_candidates: list[RetrievalItem] = []
        self.last_fused_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        results, trace = self.search(
            SearchQuery(
                question_id="",
                query_text=question,
                top_k=top_k,
                fetch_k=max(top_k, self.fetch_k),
            )
        )
        fused = strip_adapter_metadata(search_results_to_retrieval_items(results))
        self.last_fused_candidates = list(fused)
        self.last_retrieval_diagnostics = dict(trace.diagnostics)
        return fused[:top_k]

    def search(self, query: SearchQuery):
        start = time.perf_counter()
        dense_k = max(query.top_k, self.dense_fetch_k)
        bm25_k = max(query.top_k, self.bm25_fetch_k)

        dense = self.dense_retriever.retrieve(query.query_text, dense_k)
        bm25 = self.bm25_retriever.retrieve(query.query_text, bm25_k)

        fused = self._fuse(dense, bm25)
        self.last_dense_candidates = list(dense)
        self.last_bm25_candidates = list(bm25)
        self.last_fused_candidates = list(fused)

        diagnostics = {
            **getattr(self.dense_retriever, "last_retrieval_diagnostics", {}),
            "es_hybrid_dense_candidates": len(dense),
            "es_hybrid_bm25_candidates": len(bm25),
            "es_hybrid_fused_candidates": len(fused),
            "dedupe_policy": DedupePolicy.CHUNK_ID.value,
        }
        top_fused = fused[: query.top_k]
        results = retrieval_items_to_search_results(top_fused)
        return results, RetrievalTrace(
            question_id=query.question_id,
            backend="elasticsearch_hybrid_rrf",
            query_latency_ms=(time.perf_counter() - start) * 1000,
            raw_results_count=len(dense) + len(bm25),
            final_results_count=len(results),
            dedupe_policy=DedupePolicy.CHUNK_ID.value,
            filters_applied=query.filters,
            diagnostics=diagnostics,
        )

    def extract_query_metadata(self, question: str):
        if hasattr(self.dense_retriever, "extract_query_metadata"):
            return self.dense_retriever.extract_query_metadata(question)
        return None

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _fuse(
        self,
        dense: list[RetrievalItem],
        bm25: list[RetrievalItem],
    ) -> list[RetrievalItem]:
        # Accumulate per-chunk data in a single pass over each leg.
        by_chunk: dict[str, RetrievalItem] = {}
        rrf_scores: dict[str, float] = {}
        dense_scores: dict[str, float | None] = {}
        bm25_scores: dict[str, float | None] = {}
        dense_ranks: dict[str, int] = {}
        bm25_ranks: dict[str, int] = {}
        retrieval_sources: dict[str, list[str]] = {}
        # first_seen tracks insertion order for deterministic tie-breaking.
        first_seen: dict[str, int] = {}
        seen_index = 0

        for rank, item in enumerate(dense, start=1):
            key = str(item.chunk_id)
            if not key:
                continue  # skip items with empty chunk_id (documented policy)
            by_chunk.setdefault(key, item)
            first_seen.setdefault(key, seen_index)
            seen_index += 1
            dense_scores[key] = (
                item.dense_score if item.dense_score is not None else item.score
            )
            dense_ranks[key] = rank
            retrieval_sources.setdefault(key, []).append("dense")
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.dense_weight * (
                1.0 / (self.rrf_k + rank)
            )

        for rank, item in enumerate(bm25, start=1):
            key = str(item.chunk_id)
            if not key:
                continue  # skip items with empty chunk_id (documented policy)
            by_chunk.setdefault(key, item)
            first_seen.setdefault(key, seen_index)
            seen_index += 1
            bm25_scores[key] = (
                item.bm25_score if item.bm25_score is not None else item.score
            )
            bm25_ranks[key] = rank
            retrieval_sources.setdefault(key, []).append("bm25")
            rrf_scores[key] = rrf_scores.get(key, 0.0) + self.bm25_weight * (
                1.0 / (self.rrf_k + rank)
            )

        fused: list[RetrievalItem] = []
        for key, item in by_chunk.items():
            score = rrf_scores.get(key, 0.0)
            item_metadata = dict(item.metadata)
            item_metadata["_es_hybrid_diagnostics"] = {
                "rrf_score": score,
                "dense_rank": dense_ranks.get(key),
                "bm25_rank": bm25_ranks.get(key),
                "retrieval_sources": list(retrieval_sources.get(key, [])),
            }
            fused.append(
                item.model_copy(
                    update={
                        "score": score,
                        "dense_score": dense_scores.get(key),
                        "bm25_score": bm25_scores.get(key),
                        "rrf_score": score,
                        "rerank_score": None,
                        "ranking_score_type": "rrf_score",
                        "retrieval_source": "elasticsearch_hybrid_rrf",
                        "metadata": item_metadata,
                    }
                )
            )

        # Primary sort: descending RRF score.
        # Tie-break: ascending first_seen index (earlier appearance = higher priority).
        return sorted(
            fused,
            key=lambda i: (-i.score, first_seen.get(str(i.chunk_id), 0)),
        )
