from src.pipeline1.embedding.base import BaseEmbedder
from src.pipeline1.indexing.base import BaseVectorIndex
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.retrieval.metadata import extract_query_metadata, filter_candidates_by_metadata, metadata_boost


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

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        query_vec = self.embedder.encode_query(question)
        scores, idxs = self.index.search(query_vec, max(top_k, self.fetch_k))
        items: list[RetrievalItem] = []
        query_metadata = extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.chunks):
                continue
            ch = self.chunks[idx]
            boost = (
                metadata_boost(
                    ch.metadata,
                    query_metadata,
                    company_weight=self.metadata_boosting.company_weight,
                    year_weight=self.metadata_boosting.year_weight,
                    symbol_weight=self.metadata_boosting.symbol_weight,
                    file_name_weight=self.metadata_boosting.file_name_weight,
                )
                if self.metadata_boosting.enabled
                else 0.0
            )
            items.append(RetrievalItem(
                chunk_id=ch.chunk_id,
                original_context_id=ch.original_context_id or ch.document_id,
                text=ch.text,
                score=float(score) + boost,
                dense_score=float(score),
                rerank_score=None,
                ranking_score_type="dense_score_plus_metadata" if boost else "dense_score",
                chunk_unit=ch.metadata.get("chunk_unit"),
                metadata=dict(ch.metadata),
                metadata_boost=boost,
            ))
        if self.metadata_filtering.enabled:
            items = filter_candidates_by_metadata(items, query_metadata, self.metadata_filtering.strict)
        return sorted(items, key=lambda item: item.score, reverse=True)[:top_k]

    def extract_query_metadata(self, question: str):
        return extract_query_metadata(question, (chunk.metadata for chunk in self.chunks))
