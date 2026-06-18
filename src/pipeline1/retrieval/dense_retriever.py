from src.pipeline1.embedding.base import BaseEmbedder
from src.pipeline1.indexing.base import BaseVectorIndex
from src.pipeline1.retrieval.adapters import search_results_to_retrieval_items, strip_adapter_metadata
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.retrieval.contracts import DedupePolicy, RetrievalTrace, SearchQuery, SearchResult
from src.pipeline1.retrieval.dedupe import dedupe_search_results
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.retrieval.metadata import (
    extract_query_metadata,
    filter_candidates_by_metadata_with_diagnostics,
    metadata_boost_components,
)


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        index: BaseVectorIndex,
        chunks: list[ChunkRecord],
        fetch_k: int,
        metadata_boosting,
        metadata_filtering,
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.fetch_k = fetch_k
        self.metadata_boosting = metadata_boosting
        self.metadata_filtering = metadata_filtering
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

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
        import time

        start = time.perf_counter()
        query_vec = query.query_embedding if query.query_embedding is not None else self.embedder.encode_query(query.query_text)
        scores, idxs = self.index.search(query_vec, max(query.top_k, query.fetch_k))
        results: list[SearchResult] = []
        query_metadata = extract_query_metadata(query.query_text, (chunk.metadata for chunk in self.chunks))
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.chunks):
                continue
            ch = self.chunks[idx]
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
            diagnostics = {
                "dense_score": float(score),
                "score_before_metadata": float(score),
                "ranking_score_type": "dense_score_plus_metadata" if boost else "dense_score",
                "metadata_boost": boost,
                "metadata_boost_components": boost_components,
            }
            results.append(SearchResult(
                chunk_id=ch.chunk_id,
                document_id=ch.document_id,
                original_context_id=ch.original_context_id or ch.document_id,
                text=ch.text,
                score=float(score) + boost,
                retrieval_backend="dense",
                metadata=dict(ch.metadata),
                diagnostics=diagnostics,
            ))
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
        sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
        ranked, dedupe_diagnostics = dedupe_search_results(sorted_results, query.top_k, DedupePolicy.NONE)
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
            backend="dense",
            query_latency_ms=(time.perf_counter() - start) * 1000,
            raw_results_count=before_filter_count,
            final_results_count=len(ranked),
            dedupe_policy=DedupePolicy.NONE.value,
            filters_applied=query.filters,
            diagnostics=diagnostics,
        )

    def extract_query_metadata(self, question: str):
        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
