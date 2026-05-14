from pydantic import BaseModel


class RetrievalItem(BaseModel):
    chunk_id: str
    original_context_id: str
    text: str
    score: float
    dense_score: float
    rerank_score: float | None = None
    ranking_score_type: str = "dense_score"
    chunk_unit: str | None = None
