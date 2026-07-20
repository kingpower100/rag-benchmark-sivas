from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class OutputRecord(BaseModel):
    experiment_id: str
    question_id: str
    uid: str | None = None
    question: str
    cleaned_question: str | None = None
    detected_category: str | None = None
    category_validated: bool = False
    category_validation_reason: str | None = None
    orchestration_error: str | None = None
    generated_answer: str
    retrieved_chunks: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str]
    retrieved_original_context_ids: list[str]
    raw_retrieved_context_ids: list[str] = Field(default_factory=list)
    raw_retrieved_original_context_ids: list[str] = Field(default_factory=list)
    raw_dense_retrieved_context_ids: list[str] = Field(default_factory=list)
    raw_bm25_retrieved_context_ids: list[str] = Field(default_factory=list)
    retrieved_context_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str | None] = Field(default_factory=list)
    retrieved_documents: list[str | None] = Field(default_factory=list)
    raw_retrieved_document_ids: list[str | None] = Field(default_factory=list)
    retrieved_categories: list[str | None] = Field(default_factory=list)
    category_filter_applied: bool = False
    category_fallback_used: bool = False
    retrieved_files: list[str | None] = Field(default_factory=list)
    retrieved_file_names: list[str | None] = Field(default_factory=list)
    raw_retrieved_file_names: list[str | None] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
    retrieved_chunk_units: list[str | None] = Field(default_factory=list)
    retrieved_chunk_texts: list[str] = Field(default_factory=list)
    retrieved_chunk_metadata: list[dict] = Field(default_factory=list)
    retrieved_context_texts: list[str]
    retrieval_scores: list[float]
    dense_scores: list[float | None] = Field(default_factory=list)
    bm25_scores: list[float | None] = Field(default_factory=list)
    rrf_scores: list[float | None] = Field(default_factory=list)
    rerank_scores: list[float | None] = Field(default_factory=list)
    ranking_score_type: str = "dense_score"
    retrieval_mode: str = "dense"
    retrieved_unique_count: int = 0
    raw_retrieved_unique_count: int = 0
    raw_duplicate_rate: float | None = None
    retrieval_warnings: list[str] = Field(default_factory=list)
    query_metadata: dict = Field(default_factory=dict)
    retrieval_diagnostics: dict = Field(default_factory=dict)
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
    latency_ms: float | None = None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_usage: dict = Field(default_factory=dict)
    estimated_cost: float = 0.0
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = "0.1.0"
    prompt_template_version: str = "v1"
    prompt_stats: dict = Field(default_factory=dict)
    prompt_chars: int | None = None
    prompt_tokens: int | None = None
    context_chars_before: int | None = None
    context_chars_after: int | None = None
    context_tokens_before: int | None = None
    context_tokens_after: int | None = None
    chunks_before: int | None = None
    chunks_after: int | None = None
    chunks_truncated: int | None = None
    chunks_dropped: int | None = None
    generation_context_texts: list[str] = Field(default_factory=list)
    parent_context_diagnostics: dict = Field(default_factory=dict)
    parent_context_enabled: bool = False
    error: Optional[str] = None

    def to_export_record(self) -> dict:
        """Produce the target results.jsonl schema.

        Top-level keys match the benchmark target spec.  Pipeline 2 compatibility
        aliases (old names) are appended so evaluation continues to work without
        any changes to pipeline2/orchestrator.py.
        """
        chunk_ids = self.retrieved_chunk_ids
        chunk_meta = self.retrieved_chunk_metadata
        categories = self.retrieved_categories
        scores = self.retrieval_scores
        texts = self.retrieved_chunk_texts

        retrieved_chunks = [
            {
                "rank": i + 1,
                "chunk_id": chunk_ids[i],
                "doc_id": chunk_meta[i].get("doc_id") if i < len(chunk_meta) else None,
                "doc_name": chunk_meta[i].get("doc_name") if i < len(chunk_meta) else None,
                "category": categories[i] if i < len(categories) else None,
                "score": scores[i] if i < len(scores) else None,
                "chunk_text": texts[i] if i < len(texts) else None,
            }
            for i in range(len(chunk_ids))
        ]

        return {
            # ── Target schema fields ────────────────────────────────────────
            "question_id": self.question_id,
            "question": self.question,
            "clean_question": self.cleaned_question,
            "detected_category": self.detected_category,
            "category_validated": self.category_validated,
            "category_validation_reason": self.category_validation_reason,
            "retrieval_mode": self.retrieval_mode,
            "category_filter_applied": self.category_filter_applied,
            "category_fallback_used": self.category_fallback_used,
            "retrieved_chunks": retrieved_chunks,
            "answer": self.generated_answer,
            "config_id": self.experiment_id,
            "embedding_model": self.embedding_model,
            "generation_model": self.llm_model,
            "retrieval_k": self.top_k,
            # ── Pipeline 2 compatibility aliases ────────────────────────────
            # pipeline2/orchestrator.py reads these fields by their original
            # names; keeping them here avoids any changes to the eval layer.
            "experiment_id": self.experiment_id,
            "generated_answer": self.generated_answer,
            "llm_model": self.llm_model,
            "retrieved_original_context_ids": self.retrieved_original_context_ids,
            "raw_retrieved_original_context_ids": self.raw_retrieved_original_context_ids,
            "raw_retrieved_document_ids": self.raw_retrieved_document_ids,
            "retrieved_file_names": [
                chunk_meta[i].get("doc_name") if i < len(chunk_meta) else None
                for i in range(len(chunk_ids))
            ],
            "raw_retrieved_file_names": [
                chunk_meta[i].get("doc_name") if i < len(chunk_meta) else None
                for i in range(len(chunk_ids))
            ],
            "retrieved_chunk_metadata": self.retrieved_chunk_metadata,
            "query_metadata": self.query_metadata,
            "retrieval_diagnostics": self.retrieval_diagnostics,
            "retrieval_warnings": self.retrieval_warnings,
            "retriever_type": self.retriever_type,
            "reranker_used": self.reranker_used,
            "retrieval_time_ms": self.retrieval_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "total_latency_ms": self.total_latency_ms,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
            "error": self.error,
            "orchestration_error": self.orchestration_error,
            "generation_context_texts": self.generation_context_texts,
            "parent_context_diagnostics": self.parent_context_diagnostics,
            "parent_context_enabled": self.parent_context_enabled,
        }

    @model_validator(mode="after")
    def validate_retrieval_arrays(self) -> "OutputRecord":
        if self.uid is None:
            self.uid = self.question_id
        if not self.retrieved_chunk_texts:
            self.retrieved_chunk_texts = list(self.retrieved_context_texts)
        if not self.retrieved_chunks:
            self.retrieved_chunks = list(self.retrieved_chunk_ids)
        if not self.retrieved_chunk_ids:
            self.retrieved_chunk_ids = list(self.retrieved_chunks)
        if not self.retrieved_chunk_units:
            self.retrieved_chunk_units = [None] * len(self.retrieved_chunk_ids)
        if not self.dense_scores and self.retrieval_mode == "dense":
            self.dense_scores = list(self.retrieval_scores)
        elif not self.dense_scores:
            self.dense_scores = [None] * len(self.retrieved_chunk_ids)
        if not self.bm25_scores:
            self.bm25_scores = [None] * len(self.retrieved_chunk_ids)
        if not self.rrf_scores:
            self.rrf_scores = [None] * len(self.retrieved_chunk_ids)
        if not self.rerank_scores:
            self.rerank_scores = [None] * len(self.retrieved_chunk_ids)
        if not self.retrieved_chunk_metadata:
            self.retrieved_chunk_metadata = [{} for _ in self.retrieved_chunk_ids]
        if not self.retrieved_context_ids:
            self.retrieved_context_ids = list(self.retrieved_chunk_ids)
        if not self.raw_retrieved_context_ids:
            self.raw_retrieved_context_ids = list(self.raw_retrieved_original_context_ids)
        if not self.retrieved_document_ids:
            self.retrieved_document_ids = list(self.retrieved_original_context_ids)
        if not self.retrieved_documents:
            self.retrieved_documents = list(self.retrieved_document_ids)
        if not self.retrieved_categories:
            self.retrieved_categories = [
                metadata.get("kategorie")
                for metadata in self.retrieved_chunk_metadata
            ] if self.retrieved_chunk_metadata else [None] * len(self.retrieved_chunk_ids)
        if not self.retrieved_file_names:
            self.retrieved_file_names = [
                metadata.get("file_name") or metadata.get("source_file")
                for metadata in self.retrieved_chunk_metadata
            ]
        if not self.retrieved_files:
            self.retrieved_files = list(self.retrieved_file_names)
        if not self.citations:
            self.citations = [
                {
                    "source_file": metadata.get("source_file") or metadata.get("file_name") or file_name,
                    "source_id": metadata.get("source_id"),
                    "chunk_id": chunk_id,
                    "rank": rank,
                    "score": score,
                    "year": metadata.get("year") or metadata.get("report_year"),
                    "month": metadata.get("month"),
                }
                for rank, (chunk_id, file_name, metadata, score) in enumerate(
                    zip(
                        self.retrieved_chunk_ids,
                        self.retrieved_file_names,
                        self.retrieved_chunk_metadata,
                        self.retrieval_scores,
                    ),
                    start=1,
                )
            ]
        if self.latency_ms is None:
            self.latency_ms = self.total_latency_ms
        if not self.token_usage:
            self.token_usage = {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
            }
        if self.prompt_stats:
            for key in (
                "prompt_chars",
                "prompt_tokens",
                "context_chars_before",
                "context_chars_after",
                "context_tokens_before",
                "context_tokens_after",
                "chunks_before",
                "chunks_after",
                "chunks_truncated",
                "chunks_dropped",
            ):
                if getattr(self, key) is None and key in self.prompt_stats:
                    setattr(self, key, self.prompt_stats[key])
        if self.retrieved_unique_count == 0:
            self.retrieved_unique_count = len(set(self.retrieved_chunk_ids))
        if self.raw_retrieved_unique_count == 0 and self.raw_retrieved_original_context_ids:
            self.raw_retrieved_unique_count = len(set(self.raw_retrieved_original_context_ids))
        if self.raw_duplicate_rate is None and self.raw_retrieved_original_context_ids:
            self.raw_duplicate_rate = (
                len(self.raw_retrieved_original_context_ids) - len(set(self.raw_retrieved_original_context_ids))
            ) / len(self.raw_retrieved_original_context_ids)
        if not (
            len(self.retrieved_chunk_ids)
            == len(self.retrieved_chunks)
            == len(self.retrieved_original_context_ids)
            == len(self.retrieved_context_ids)
            == len(self.retrieved_document_ids)
            == len(self.retrieved_documents)
            == len(self.retrieved_categories)
            == len(self.retrieved_files)
            == len(self.retrieved_file_names)
            == len(self.retrieved_chunk_units)
            == len(self.retrieved_chunk_texts)
            == len(self.retrieved_chunk_metadata)
            == len(self.retrieved_context_texts)
            == len(self.retrieval_scores)
            == len(self.dense_scores)
            == len(self.bm25_scores)
            == len(self.rrf_scores)
            == len(self.rerank_scores)
        ):
            raise ValueError("retrieval arrays must align")
        if len(self.retrieved_chunk_ids) > self.top_k:
            raise ValueError(f"len(retrieved_chunk_ids)={len(self.retrieved_chunk_ids)} cannot exceed top_k={self.top_k}")
        if self.retrieved_original_context_ids is None:
            raise ValueError("retrieved_original_context_ids must be present")
        return self
