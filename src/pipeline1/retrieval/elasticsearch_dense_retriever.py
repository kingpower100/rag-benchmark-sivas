from __future__ import annotations

import logging
import time
from typing import Any

from src.pipeline1.embedding.base import BaseEmbedder
from src.pipeline1.retrieval.adapters import search_results_to_retrieval_items, strip_adapter_metadata
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery, SearchResult
from src.pipeline1.retrieval.dedupe import dedupe_search_results
from src.pipeline1.retrieval.metadata import (
    extract_query_metadata,
    filter_candidates_by_metadata_with_diagnostics,
    metadata_boost_components,
)
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


class ElasticsearchDenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        index,
        chunks: list[ChunkRecord],
        top_k: int,
        fetch_k: int,
        metadata_boosting,
        metadata_filtering,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.metadata_boosting = metadata_boosting
        self.metadata_filtering = metadata_filtering
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}
        self.logger = logging.getLogger("pipeline1")
        self.logger.info("Elasticsearch dense retriever configured top_k=%s fetch_k=%s", top_k, fetch_k)

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        results, trace = self.search(SearchQuery(question_id="", query_text=question, top_k=top_k, fetch_k=max(top_k, self.fetch_k)))
        ranked = strip_adapter_metadata(search_results_to_retrieval_items(results))
        self.last_dense_candidates = ranked
        self.last_retrieval_diagnostics = dict(trace.diagnostics)
        return ranked

    def search(self, query: SearchQuery) -> tuple[list[SearchResult], RetrievalTrace]:
        candidate_k = max(query.top_k, query.fetch_k, self.fetch_k)
        start = time.perf_counter()
        query_vec = query.query_embedding if query.query_embedding is not None else self.embedder.encode_query(query.query_text)
        hits = self._search_hits(query_vec, candidate_k)
        query_latency_ms = (time.perf_counter() - start) * 1000
        self.logger.info(
            "Elasticsearch dense retrieval top_k=%s fetch_k=%s latency_ms=%.2f",
            query.top_k,
            candidate_k,
            query_latency_ms,
        )

        query_metadata = extract_query_metadata(query.query_text, (chunk.metadata for chunk in self.chunks))
        results = [self._hit_to_result(hit, query_metadata) for hit in hits]
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
            results = [result for result in results if result.chunk_id in allowed]

        ranked, dedupe_diagnostics = dedupe_search_results(
            sorted(results, key=lambda item: item.score, reverse=True),
            query.top_k,
            DedupePolicy.NONE,
        )
        ranked = [
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                original_context_id=result.original_context_id,
                text=result.text,
                score=result.score,
                rank=index,
                retrieval_backend=result.retrieval_backend,
                metadata=result.metadata,
                diagnostics=result.diagnostics,
            )
            for index, result in enumerate(ranked, start=1)
        ]
        diagnostics = {
            "query_metadata": query_metadata.__dict__,
            "candidates_before_filter": before_filter_count,
            "candidates_after_filter": after_filter_count,
            "filter_fallback": filter_fallback,
            "boosted_candidates": [
                {
                    "chunk_id": item.chunk_id,
                    "dense_score": item.diagnostics.get("dense_score"),
                    "score": item.score,
                    "metadata_boost": item.diagnostics.get("metadata_boost", 0.0),
                    "metadata_boost_components": item.diagnostics.get("metadata_boost_components", {}),
                    "file_name": item.metadata.get("file_name"),
                    "doc_key": item.metadata.get("doc_key"),
                    "doc_name": item.metadata.get("doc_name"),
                    "kategorie": item.metadata.get("kategorie"),
                    "wissensart": item.metadata.get("wissensart"),
                }
                for item in ranked[:10]
            ],
        }
        diagnostics.update(dedupe_diagnostics)
        return ranked, RetrievalTrace(
            question_id=query.question_id,
            backend="elasticsearch_dense",
            query_latency_ms=query_latency_ms,
            raw_results_count=before_filter_count,
            final_results_count=len(ranked),
            dedupe_policy=DedupePolicy.NONE.value,
            filters_applied=query.filters,
            diagnostics=diagnostics,
        )

    def extract_query_metadata(self, question: str):
        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))

    def _search_hits(self, query_vec, candidate_k: int) -> list[dict[str, Any]]:
        if hasattr(self.index, "search_hits"):
            return self.index.search_hits(query_vec, candidate_k)
        chunk_ids, scores = self.index.search(query_vec, candidate_k)
        hits = []
        for score, chunk_id in zip(scores, chunk_ids):
            chunk = self.chunk_by_id.get(str(chunk_id))
            if not chunk:
                continue
            hits.append(
                {
                    "_id": chunk.chunk_id,
                    "_score": float(score) + 1.0,
                    "_source": {
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "original_context_id": chunk.original_context_id or chunk.document_id,
                        "text": chunk.text,
                        "metadata": dict(chunk.metadata),
                    },
                }
            )
        return hits

    def _hit_to_item(self, hit: dict[str, Any], query_metadata) -> RetrievalItem:
        return search_results_to_retrieval_items([self._hit_to_result(hit, query_metadata)])[0]

    def _hit_to_result(self, hit: dict[str, Any], query_metadata) -> SearchResult:
        source = hit.get("_source") or {}
        chunk_id = str(source.get("chunk_id") or hit.get("_id"))
        chunk = self.chunk_by_id.get(chunk_id)
        metadata = dict(source.get("metadata") or (chunk.metadata if chunk else {}))
        document_id = str(source.get("document_id") or (chunk.document_id if chunk else chunk_id))
        original_context_id = str(
            source.get("original_context_id")
            or metadata.get("original_context_id")
            or (chunk.original_context_id if chunk else None)
            or document_id
        )
        text = source.get(getattr(self.index, "text_field", "text")) or source.get("text") or (chunk.text if chunk else "")
        dense_score = float(hit.get("_score") or 0.0) - 1.0
        boost_components = (
            metadata_boost_components(
                metadata,
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
        return SearchResult(
            chunk_id=chunk_id,
            document_id=document_id,
            original_context_id=original_context_id,
            text=str(text),
            score=dense_score + boost,
            retrieval_backend="elasticsearch_dense",
            metadata=metadata,
            diagnostics={
                "dense_score": dense_score,
                "score_before_metadata": dense_score,
                "ranking_score_type": "dense_score_plus_metadata" if boost else "dense_score",
                "metadata_boost": boost,
                "metadata_boost_components": boost_components,
            },
        )
