from src.pipeline1.embedding.base import BaseEmbedder
from src.pipeline1.indexing.base import BaseVectorIndex
from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.retrieval import RetrievalItem


class DenseRetriever(BaseRetriever):
    def __init__(self, embedder: BaseEmbedder, index: BaseVectorIndex, chunks: list[ChunkRecord], fetch_k: int) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.fetch_k = fetch_k

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        query_vec = self.embedder.encode_query(question)
        scores, idxs = self.index.search(query_vec, max(top_k, self.fetch_k))
        items: list[RetrievalItem] = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self.chunks):
                continue
            ch = self.chunks[idx]
            items.append(RetrievalItem(
                chunk_id=ch.chunk_id,
                original_context_id=ch.original_context_id or ch.document_id,
                text=ch.text,
                score=float(score),
                dense_score=float(score),
                rerank_score=None,
                ranking_score_type="dense_score",
                chunk_unit=ch.metadata.get("chunk_unit"),
            ))
        return items[:top_k]
