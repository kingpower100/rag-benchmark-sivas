from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from tqdm.auto import tqdm

from src.pipeline1.generation.cost_estimator import estimate_cost
from src.pipeline1.generation.factory import build_generator
from src.pipeline1.generation.prompt_builder import (
PROMPT_TEMPLATE_VERSION,
    PromptBudget,
    build_prompt_with_stats,
    dedupe_prompt_contexts,
)

MAX_GENERATION_RETRIES = 3
GENERATION_BACKOFF_BASE_SECONDS = 1.0
from src.pipeline1.observability.events import EventType
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.output_record import OutputRecord
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput
from src.pipeline1.stages.retrieval_stage import RetrievalRow
from src.pipeline1.utils.ids import stable_retrieved_document_id


@dataclass(frozen=True)
class GenerationRow:
    retrieval_row: RetrievalRow
    output_record: OutputRecord
    prompt_stats: dict
    generation_time_ms: float
    error: str | None = None


@dataclass(frozen=True)
class GenerationStageOutput(StageOutput):
    generator: object = None
    generation_rows: list[GenerationRow] = field(default_factory=list)


class GenerationStage(BaseStage):
    stage_name = "generation"

    def __init__(
        self,
        cfg: PipelineConfig,
        retriever,
        event_writer=None,
        logger=None,
        generator_factory: Callable = build_generator,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.event_writer = event_writer
        self.logger = logger
        self.generator_factory = generator_factory

    def run(self, stage_input: StageInput) -> GenerationStageOutput:
        retrieval_rows: list[RetrievalRow] = list(stage_input.payload["retrieval_rows"])
        final_top_k = int(stage_input.payload["final_top_k"])
        generator = self.generator_factory(self.cfg.generation)
        generation_rows: list[GenerationRow] = []
        for row_index, retrieval_row in enumerate(
            tqdm(retrieval_rows, desc="Generating answers", unit="question"),
            start=1,
        ):
            generation_rows.append(self._generate_row(generator, retrieval_row, final_top_k, row_index, len(retrieval_rows)))
        return GenerationStageOutput(
            stage_name=self.stage_name,
            artifacts={"generator": generator, "generation_rows": generation_rows},
            diagnostics={"generated_rows": len(generation_rows)},
            metadata={"llm_model": self.cfg.generation.model_name},
            generator=generator,
            generation_rows=generation_rows,
        )

    def _generate_row(
        self,
        generator,
        retrieval_row: RetrievalRow,
        final_top_k: int,
        row_index: int,
        total_rows: int,
    ) -> GenerationRow:
        query = retrieval_row.query
        raw_retrieved = retrieval_row.raw_retrieved
        raw_dense_retrieved = retrieval_row.raw_dense_retrieved
        raw_bm25_retrieved = retrieval_row.raw_bm25_retrieved
        retrieved = retrieval_row.retrieved
        retrieval_time_ms = retrieval_row.retrieval_time_ms
        reranker_used = retrieval_row.reranker_used
        retrieval_warnings = retrieval_row.retrieval_warnings
        retrieval_diagnostics = retrieval_row.retrieval_diagnostics

        # C03: use expanded parent sections; C00 and others: use retrieved children.
        if self.cfg.parent_context.enabled and retrieval_row.generation_contexts:
            prompt_contexts = list(retrieval_row.generation_contexts)
            generation_context_texts = [gc.text for gc in prompt_contexts]
            parent_context_enabled = True
        else:
            prompt_contexts = dedupe_prompt_contexts(retrieved)
            generation_context_texts = [item.text for item in prompt_contexts]
            parent_context_enabled = False
        if self.logger:
            self.logger.info(
                "row_start phase=generation question_id=%s row=%s/%s saved_contexts=%s prompt_contexts=%s parent_context=%s",
                query.question_id,
                row_index,
                total_rows,
                len(retrieved),
                len(prompt_contexts),
                parent_context_enabled,
            )
        retrieval_metadata = None
        if self.cfg.generation.include_retrieval_metadata:
            retrieval_metadata = {
                "detected_category": query.detected_category,
                "category_validated": query.category_validated,
                "retrieval_mode": retrieval_diagnostics.get("retrieval_mode"),
                "fallback_used": retrieval_diagnostics.get("category_fallback_used"),
            }
        prompt, prompt_stats = build_prompt_with_stats(
            self.cfg.generation.system_prompt,
            query.retrieval_question,
            prompt_contexts,
            include_metadata_headers=self.cfg.generation.include_metadata_headers,
            budget=PromptBudget(
                max_prompt_tokens=self.cfg.generation.max_prompt_tokens,
                max_context_tokens=self.cfg.generation.max_context_tokens,
                max_chunk_tokens=self.cfg.generation.max_chunk_tokens,
                max_context_chars=self.cfg.generation.max_context_chars,
                max_chunk_chars=self.cfg.generation.max_chunk_chars,
                tokenizer_name=self.cfg.chunking.tokenizer_name,
                context_truncation_strategy=self.cfg.generation.context_truncation_strategy,
            ),
            retrieval_metadata=retrieval_metadata,
        )
        if self.cfg.generation.log_prompt_stats and self.logger:
            self.logger.info("prompt_stats question_id=%s stats=%s", query.question_id, prompt_stats)
        query_metadata = (
            self.retriever.extract_query_metadata(query.question)
            if hasattr(self.retriever, "extract_query_metadata")
            else None
        )
        generation_start = time.perf_counter()
        self._write_event(
            stage="generation",
            event_type=EventType.GENERATION_START,
            message="Generation started.",
            question_id=query.question_id,
            metrics={
                "llm_model": self.cfg.generation.model_name,
                "prompt_tokens": prompt_stats.get("prompt_tokens"),
                "context_tokens_after": prompt_stats.get("context_tokens_after"),
            },
            diagnostics=prompt_stats,
        )
        generation, error, attempts = self._generate_with_retries(generator, prompt, query.question_id)
        if generation is None:
            answer = ""
            input_tokens = 0
            output_tokens = 0
        else:
            answer = generation.answer
            input_tokens = generation.input_tokens
            output_tokens = generation.output_tokens
        generation_time_ms = (time.perf_counter() - generation_start) * 1000
        self._write_event(
            stage="generation",
            event_type=EventType.GENERATION_END,
            message="Generation completed.",
            question_id=query.question_id,
            duration_ms=generation_time_ms,
            metrics={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "answer_chars": len(answer),
                "generation_failed": error is not None,
                "generation_attempts": attempts,
            },
            diagnostics={"error": error, "generation_attempts": attempts} if error else {"generation_attempts": attempts},
        )
        total_tokens = input_tokens + output_tokens
        cost = (
            estimate_cost(
                input_tokens,
                output_tokens,
                self.cfg.telemetry.pricing.input_per_1k_tokens_usd,
                self.cfg.telemetry.pricing.output_per_1k_tokens_usd,
            )
            if self.cfg.telemetry.estimate_cost
            else 0.0
        )
        record = OutputRecord(
            experiment_id=self.cfg.experiment.experiment_id,
            question_id=query.question_id,
            uid=query.question_id,
            question=query.question,
            cleaned_question=query.cleaned_question,
            detected_category=query.detected_category,
            category_validated=query.category_validated,
            category_validation_reason=query.category_validation_reason,
            orchestration_error=query.orchestration_error,
            generated_answer=answer,
            retrieved_chunks=[item.chunk_id for item in retrieved],
            retrieved_chunk_ids=[item.chunk_id for item in retrieved],
            retrieved_original_context_ids=[item.original_context_id for item in retrieved],
            raw_retrieved_context_ids=[item.chunk_id for item in raw_retrieved],
            raw_retrieved_original_context_ids=[item.original_context_id for item in raw_retrieved],
            raw_dense_retrieved_context_ids=[item.chunk_id for item in raw_dense_retrieved],
            raw_bm25_retrieved_context_ids=[item.chunk_id for item in raw_bm25_retrieved],
            retrieved_context_ids=[item.chunk_id for item in retrieved],
            retrieved_document_ids=[stable_retrieved_document_id(item.metadata, item.original_context_id) for item in retrieved],
            retrieved_documents=[
                stable_retrieved_document_id(item.metadata, item.original_context_id)
                for item in retrieved
            ],
            retrieved_categories=[item.metadata.get(self.cfg.retrieval.category_field) for item in retrieved],
            category_filter_applied=bool(retrieval_diagnostics.get("category_filter_applied", False)),
            category_fallback_used=bool(retrieval_diagnostics.get("category_fallback_used", False)),
            raw_retrieved_document_ids=[
                stable_retrieved_document_id(item.metadata, item.original_context_id)
                for item in raw_retrieved
            ],
            retrieved_file_names=[item.metadata.get("file_name") or item.metadata.get("source_file") for item in retrieved],
            retrieved_files=[item.metadata.get("source_file") or item.metadata.get("file_name") for item in retrieved],
            raw_retrieved_file_names=[
                item.metadata.get("file_name") or item.metadata.get("source_file") for item in raw_retrieved
            ],
            citations=build_citations(retrieved),
            retrieved_chunk_units=[item.chunk_unit for item in retrieved],
            retrieved_chunk_texts=[item.text for item in retrieved],
            retrieved_chunk_metadata=[dict(item.metadata) for item in retrieved],
            retrieved_context_texts=[item.text for item in retrieved],
            retrieval_scores=[item.score for item in retrieved],
            dense_scores=[item.dense_score for item in retrieved],
            bm25_scores=[item.bm25_score for item in retrieved],
            rrf_scores=[item.rrf_score for item in retrieved],
            rerank_scores=[item.rerank_score for item in retrieved],
            ranking_score_type="rerank_score" if reranker_used else (
                retrieved[0].ranking_score_type if retrieved else self.cfg.retrieval.retriever_type
            ),
            retrieval_mode=str(retrieval_diagnostics.get("retrieval_mode") or self.cfg.retrieval.retriever_type),
            retrieved_unique_count=len({item.chunk_id for item in retrieved}),
            raw_retrieved_unique_count=len({item.chunk_id for item in raw_retrieved}),
            raw_duplicate_rate=duplicate_rate([item.chunk_id for item in raw_retrieved]),
            retrieval_warnings=retrieval_warnings,
            query_metadata={} if query_metadata is None else {
                "company_names": sorted(query_metadata.company_names),
                "company_symbols": sorted(query_metadata.company_symbols),
                "years": sorted(query_metadata.years),
                "months": sorted(query_metadata.months),
                "year_months": sorted(query_metadata.year_months),
                "fiscal_years": sorted(query_metadata.fiscal_years),
                "report_periods": sorted(query_metadata.report_periods),
                "file_names": sorted(query_metadata.file_names),
                "source_datasets": sorted(query_metadata.source_datasets),
            },
            retrieval_diagnostics=retrieval_diagnostics,
            top_k=final_top_k,
            chunking_strategy=self.cfg.chunking.strategy,
            chunk_size=self.cfg.chunking.chunk_size,
            chunk_overlap=self.cfg.chunking.chunk_overlap,
            embedding_model=self.cfg.embedding.model_name,
            retriever_type=self.cfg.retrieval.retriever_type,
            reranker_used=reranker_used,
            llm_model=self.cfg.generation.model_name,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_latency_ms=retrieval_time_ms + generation_time_ms,
            latency_ms=retrieval_time_ms + generation_time_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            token_usage={"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens},
            estimated_cost=cost,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
            prompt_stats=prompt_stats,
            prompt_chars=prompt_stats.get("prompt_chars"),
            prompt_tokens=prompt_stats.get("prompt_tokens"),
            context_chars_before=prompt_stats.get("context_chars_before"),
            context_chars_after=prompt_stats.get("context_chars_after"),
            context_tokens_before=prompt_stats.get("context_tokens_before"),
            context_tokens_after=prompt_stats.get("context_tokens_after"),
            chunks_before=prompt_stats.get("chunks_before"),
            chunks_after=prompt_stats.get("chunks_after"),
            chunks_truncated=prompt_stats.get("chunks_truncated"),
            chunks_dropped=prompt_stats.get("chunks_dropped"),
            generation_context_texts=generation_context_texts,
            parent_context_diagnostics=retrieval_row.parent_context_diagnostics if parent_context_enabled else {},
            parent_context_enabled=parent_context_enabled,
            error=error,
        )
        return GenerationRow(
            retrieval_row=retrieval_row,
            output_record=record,
            prompt_stats=prompt_stats,
            generation_time_ms=generation_time_ms,
            error=error,
        )

    def _write_event(self, **kwargs) -> None:
        if self.event_writer is not None:
            self.event_writer.write(**kwargs)

    def _generate_with_retries(self, generator, prompt: str, question_id: str):
        last_error = None
        for attempt in range(1, MAX_GENERATION_RETRIES + 1):
            try:
                return generator.generate(prompt), None, attempt
            except Exception as ex:
                last_error = str(ex)
                if self.logger:
                    self.logger.warning(
                        "row_generation_attempt_failed question_id=%s attempt=%s/%s error=%s",
                        question_id,
                        attempt,
                        MAX_GENERATION_RETRIES,
                        last_error,
                        exc_info=True,
                    )
                self._write_event(
                    stage="generation",
                    event_type=EventType.GENERATION_ERROR,
                    message="Generation attempt failed.",
                    question_id=question_id,
                    diagnostics={
                        "error": last_error,
                        "attempt": attempt,
                        "max_retries": MAX_GENERATION_RETRIES,
                        "will_retry": attempt < MAX_GENERATION_RETRIES,
                    },
                )
                if attempt < MAX_GENERATION_RETRIES:
                    time.sleep(GENERATION_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        self._write_event(
            stage="pipeline",
            event_type=EventType.PIPELINE_ERROR,
            message="Generation failed after retries.",
            question_id=question_id,
            diagnostics={"error": last_error, "attempts": MAX_GENERATION_RETRIES},
        )
        return None, last_error, MAX_GENERATION_RETRIES


def build_citations(items: list) -> list[dict]:
    citations = []
    for rank, item in enumerate(items, start=1):
        metadata = item.metadata or {}
        citations.append(
            {
                "source_file": metadata.get("source_file") or metadata.get("file_name"),
                "source_id": metadata.get("source_id"),
                "chunk_id": item.chunk_id,
                "rank": rank,
                "score": item.score,
                "year": metadata.get("year") or metadata.get("report_year"),
                "month": metadata.get("month"),
            }
        )
    return citations


def duplicate_rate(ids: list[str]) -> float:
    if not ids:
        return 0.0
    return (len(ids) - len(set(ids))) / len(ids)
