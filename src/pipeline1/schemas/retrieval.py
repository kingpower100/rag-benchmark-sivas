from pydantic import BaseModel, Field


class RetrievalItem(BaseModel):
    chunk_id: str
    original_context_id: str
    text: str
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    ranking_score_type: str = "dense_score"
    retrieval_source: str = "dense"
    chunk_unit: str | None = None
    metadata: dict = Field(default_factory=dict)
    metadata_boost: float = 0.0
    metadata_boost_components: dict = Field(default_factory=dict)
    score_before_metadata: float | None = None
    metadata_filter_matched: bool | None = None
