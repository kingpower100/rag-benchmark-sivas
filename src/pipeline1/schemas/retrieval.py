from pydantic import BaseModel, Field


class RetrievalItem(BaseModel):
    chunk_id: str
    original_context_id: str
    text: str
    score: float
    dense_score: float
    rerank_score: float | None = None
    ranking_score_type: str = "dense_score"
    chunk_unit: str | None = None
    metadata: dict = Field(default_factory=dict)
    metadata_boost: float = 0.0
