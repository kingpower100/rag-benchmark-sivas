from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class OutputRecord(BaseModel):
    experiment_id: str
    question_id: str
    question: str
    generated_answer: str
    retrieved_chunk_ids: list[str]
    retrieved_original_context_ids: list[str]
    raw_retrieved_original_context_ids: list[str] = Field(default_factory=list)
    retrieved_context_ids: list[str] = Field(default_factory=list)
    retrieved_chunk_units: list[str | None] = Field(default_factory=list)
    retrieved_chunk_texts: list[str] = Field(default_factory=list)
    retrieved_context_texts: list[str]
    retrieval_scores: list[float]
    dense_scores: list[float] = Field(default_factory=list)
    rerank_scores: list[float | None] = Field(default_factory=list)
    ranking_score_type: str = "dense_score"
    retrieved_unique_count: int = 0
    raw_retrieved_unique_count: int = 0
    raw_duplicate_rate: float | None = None
    retrieval_warnings: list[str] = Field(default_factory=list)
    top_k: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    retriever_type: str
    reranker_used: bool
    llm_model: str
    retrieval_time_ms: float
    generation_time_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float = 0.0
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = "0.1.0"
    prompt_template_version: str = "v1"
    error: Optional[str] = None

    @model_validator(mode="after")
    def validate_retrieval_arrays(self) -> "OutputRecord":
        if not self.retrieved_chunk_texts:
            self.retrieved_chunk_texts = list(self.retrieved_context_texts)
        if not self.retrieved_chunk_units:
            self.retrieved_chunk_units = [None] * len(self.retrieved_chunk_ids)
        if not self.dense_scores:
            self.dense_scores = list(self.retrieval_scores)
        if not self.rerank_scores:
            self.rerank_scores = [None] * len(self.retrieved_chunk_ids)
        if self.retrieved_unique_count == 0:
            self.retrieved_unique_count = len(set(self.retrieved_original_context_ids))
        if self.raw_retrieved_unique_count == 0 and self.raw_retrieved_original_context_ids:
            self.raw_retrieved_unique_count = len(set(self.raw_retrieved_original_context_ids))
        if self.raw_duplicate_rate is None and self.raw_retrieved_original_context_ids:
            self.raw_duplicate_rate = (
                len(self.raw_retrieved_original_context_ids) - len(set(self.raw_retrieved_original_context_ids))
            ) / len(self.raw_retrieved_original_context_ids)
        if not (
            len(self.retrieved_chunk_ids)
            == len(self.retrieved_original_context_ids)
            == len(self.retrieved_context_ids)
            == len(self.retrieved_chunk_units)
            == len(self.retrieved_chunk_texts)
            == len(self.retrieved_context_texts)
            == len(self.retrieval_scores)
            == len(self.dense_scores)
            == len(self.rerank_scores)
        ):
            raise ValueError("retrieval arrays must align")
        if len(self.retrieved_chunk_ids) > self.top_k:
            raise ValueError(f"len(retrieved_chunk_ids)={len(self.retrieved_chunk_ids)} cannot exceed top_k={self.top_k}")
        if self.retrieved_original_context_ids is None:
            raise ValueError("retrieved_original_context_ids must be present")
        return self
