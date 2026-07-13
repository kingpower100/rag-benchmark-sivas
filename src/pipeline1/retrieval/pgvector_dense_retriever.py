from __future__ import annotations

import time

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery, SearchResult
from src.pipeline1.retrieval.adapters import search_results_to_retrieval_items, strip_adapter_metadata
from src.pipeline1.retrieval.dedupe import dedupe_search_results
from src.pipeline1.retrieval.metadata import (
    extract_query_metadata,
    filter_candidates_by_metadata_with_diagnostics,
    metadata_boost_components,
)
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


class PgvectorDenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder,
        index,
        chunks: list[ChunkRecord],
        fetch_k: int,
        metadata_boosting,
        metadata_filtering,
        category_field: str = "kategorie",
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.fetch_k = fetch_k
        self.metadata_boosting = metadata_boosting
        self.metadata_filtering = metadata_filtering
        self.category_field = category_field
        self.chunk_by_id: dict[str, ChunkRecord] = {chunk.chunk_id: chunk for chunk in chunks}
        self._active_category: str | None = None
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    def set_active_category(self, category: str | None) -> None:
        self._active_category = str(category).strip() if category else None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        query = SearchQuery(
            question_id="",
            query_text=question,
            query_embedding=None,
            top_k=top_k,
            fetch_k=max(top_k, self.fetch_k),
        )
        results, trace = self.search(query)
        ranked = strip_adapter_metadata(search_results_to_retrieval_items(results))
        self.last_dense_candidates = ranked
        self.last_retrieval_diagnostics = dict(trace.diagnostics)
        return ranked

    def search(self, query: SearchQuery) -> tuple[list[SearchResult], RetrievalTrace]:
        start = time.perf_counter()
        query_vec = (
            query.query_embedding
            if query.query_embedding is not None
            else self.embedder.encode_query(query.query_text)
        )
        k = max(query.top_k, query.fetch_k)

        if self._active_category:
            chunk_ids, scores = self.index.search_category(
                query_vec, k, self._active_category, self.category_field
            )
            backend_label = "pgvector_category"
        else:
            chunk_ids, scores = self.index.search(query_vec, k)
            backend_label = "pgvector"

        query_metadata = extract_query_metadata(
            query.query_text, (chunk.metadata for chunk in self.chunks)
        )
        results: list[SearchResult] = []
        for chunk_id, score in zip(chunk_ids, scores):
            ch = self.chunk_by_id.get(chunk_id)
            if ch is None:
                continue
            boost_components = (
                metadata_boost_components(
                    ch.metadata,
                    query_metadata,
                    company_weight=self.metadata_boosting.company_weight,
                    year_weight=self.metadata_boosting.year_weight,
                    month_weight=self.metadata_boosting.month_weight,
                    year_month_weight=self.metadata_boosting.year_month_weight,
                    wrong_year_penalty=self.metadata_boosting.wrong_year_penalty,
                    symbol_weight=self.metadata_boosting.symbol_weight,
                    file_name_weight=self.metadata_boosting.file_name_weight,
                )
                if self.metadata_boosting.enabled
                else {}
            )
            boost = sum(boost_components.values())
            results.append(
                SearchResult(
                    chunk_id=ch.chunk_id,
                    document_id=getattr(ch, "document_id", chunk_id),
                    original_context_id=getattr(ch, "original_context_id", None) or getattr(ch, "document_id", chunk_id),
                    text=ch.text,
                    score=float(score) + boost,
                    retrieval_backend=backend_label,
                    metadata=dict(ch.metadata or {}),
                    diagnostics={
                        "dense_score": float(score),
                        "score_before_metadata": float(score),
                        "ranking_score_type": "dense_score_plus_metadata" if boost else "dense_score",
                        "metadata_boost": boost,
                        "metadata_boost_components": boost_components,
                    },
                )
            )

        before_filter_count = len(results)
        after_filter_count = len(results)
        filter_fallback = False
        if self.metadata_filtering.enabled:
            items = search_results_to_retrieval_items(results)
            filtered, filter_diagnostics = filter_candidates_by_metadata_with_diagnostics(
                items,
                query_metadata,
                self.metadata_filtering.strict,
                self.metadata_filtering.strict_year_match,
                self.metadata_filtering.strict_year_month_match,
            )
            after_filter_count = int(filter_diagnostics["candidates_after_filter"])
            filter_fallback = bool(filter_diagnostics["filter_fallback"])
            allowed = {item.chunk_id for item in filtered}
            results = [r for r in results if r.chunk_id in allowed]

        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        ranked, dedupe_diagnostics = dedupe_search_results(sorted_results, query.top_k, DedupePolicy.NONE)
        ranked = [
            SearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                original_context_id=r.original_context_id,
                text=r.text,
                score=r.score,
                rank=i,
                retrieval_backend=r.retrieval_backend,
                metadata=r.metadata,
                diagnostics=r.diagnostics,
            )
            for i, r in enumerate(ranked, start=1)
        ]
        diagnostics = {
            "query_metadata": query_metadata.__dict__,
            "candidates_before_filter": before_filter_count,
            "candidates_after_filter": after_filter_count,
            "filter_fallback": filter_fallback,
            "active_category": self._active_category,
            "retrieval_backend": backend_label,
        }
        diagnostics.update(dedupe_diagnostics)
        return ranked, RetrievalTrace(
            question_id=query.question_id,
            backend=backend_label,
            query_latency_ms=(time.perf_counter() - start) * 1000,
            raw_results_count=before_filter_count,
            final_results_count=len(ranked),
            dedupe_policy=DedupePolicy.NONE.value,
            filters_applied=query.filters,
            diagnostics=diagnostics,
        )

    def extract_query_metadata(self, question: str):
        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
