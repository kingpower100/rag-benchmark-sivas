from src.pipeline1.embedding.base import BaseEmbedder
from src.pipeline1.indexing.base import BaseVectorIndex
from src.pipeline1.retrieval.base import BaseRetriever
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
        query_vec = self.embedder.encode_query(question)
        scores, idxs = self.index.search(query_vec, max(top_k, self.fetch_k))
        items: list[RetrievalItem] = []
        query_metadata = extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
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
                    symbol_weight=self.metadata_boosting.symbol_weight,
                    file_name_weight=self.metadata_boosting.file_name_weight,
                )
                if self.metadata_boosting.enabled
                else {}
            )
            boost = sum(boost_components.values())
            items.append(RetrievalItem(
                chunk_id=ch.chunk_id,
                original_context_id=ch.original_context_id or ch.document_id,
                text=ch.text,
                score=float(score) + boost,
                dense_score=float(score),
                score_before_metadata=float(score),
                rerank_score=None,
                ranking_score_type="dense_score_plus_metadata" if boost else "dense_score",
                chunk_unit=ch.metadata.get("chunk_unit"),
                metadata=dict(ch.metadata),
                metadata_boost=boost,
                metadata_boost_components=boost_components,
            ))
        before_filter_count = len(items)
        after_filter_count = len(items)
        filter_fallback = False
        if self.metadata_filtering.enabled:
            filtered, filter_diagnostics = filter_candidates_by_metadata_with_diagnostics(
                items,
                query_metadata,
                self.metadata_filtering.strict,
                self.metadata_filtering.strict_year_match,
                self.metadata_filtering.strict_year_month_match,
            )
            after_filter_count = int(filter_diagnostics["candidates_after_filter"])
            filter_fallback = bool(filter_diagnostics["filter_fallback"])
            items = filtered
        ranked = sorted(items, key=lambda item: item.score, reverse=True)[:top_k]
        self.last_dense_candidates = ranked
        self.last_retrieval_diagnostics = {
            "query_metadata": query_metadata.__dict__,
            "candidates_before_filter": before_filter_count,
            "candidates_after_filter": after_filter_count,
            "filter_fallback": filter_fallback,
            "boosted_candidates": [
                {
                    "chunk_id": item.chunk_id,
                    "dense_score": item.dense_score,
                    "score": item.score,
                    "metadata_boost": item.metadata_boost,
                    "metadata_boost_components": item.metadata_boost_components,
                    "file_name": item.metadata.get("file_name"),
                    "treasury_year": item.metadata.get("treasury_year"),
                    "treasury_month": item.metadata.get("treasury_month"),
                    "treasury_year_month": item.metadata.get("treasury_year_month"),
                }
                for item in ranked[:10]
            ],
        }
        return ranked

    def extract_query_metadata(self, question: str):
        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
