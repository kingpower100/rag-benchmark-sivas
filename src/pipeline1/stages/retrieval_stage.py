from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from tqdm.auto import tqdm

from src.pipeline1.observability.events import EventType
from src.pipeline1.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput
from src.pipeline1.utils.ids import stable_retrieved_document_id


@dataclass(frozen=True)
class RetrievalRow:
    query: QueryRecord
    raw_retrieved: list
    raw_dense_retrieved: list
    raw_bm25_retrieved: list
    fused_retrieved: list
    retrieved: list
    retrieval_time_ms: float
    reranker_used: bool
    retrieval_warnings: list[str]
    retrieval_diagnostics: dict

    def as_generation_tuple(self) -> tuple:
        return (
            self.query,
            self.raw_retrieved,
            self.raw_dense_retrieved,
            self.raw_bm25_retrieved,
            self.fused_retrieved,
            self.retrieved,
            self.retrieval_time_ms,
            self.reranker_used,
            self.retrieval_warnings,
            self.retrieval_diagnostics,
        )


@dataclass(frozen=True)
class RetrievalStageOutput(StageOutput):
    retriever: object = None
    reranker: object = None
    final_top_k: int = 0
    retrieval_rows: list[RetrievalRow] = field(default_factory=list)
    attempted: int = 0


class RetrievalStage(BaseStage):
    stage_name = "retrieval"

    def __init__(
        self,
        cfg: PipelineConfig,
        embedder,
        index,
        chunks: list,
        event_writer=None,
        logger=None,
        retriever_factory: Callable = build_retriever,
        reranker_factory: Callable = CrossEncoderReranker,
        embeddings=None,
    ) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.event_writer = event_writer
        self.logger = logger
        self.retriever_factory = retriever_factory
        self.reranker_factory = reranker_factory
        # Pre-computed embeddings (numpy array, shape [N, D]) used by
        # CategoryAwareDenseRetriever to build per-category FAISS sub-indexes.
        self.embeddings = embeddings

    def run(self, stage_input: StageInput) -> RetrievalStageOutput:
        queries = list(stage_input.payload["queries"])
        retriever = self.retriever_factory(self.cfg.retrieval, self.embedder, self.index, self.chunks, embeddings=self.embeddings)
        reranker = (
            self.reranker_factory(self.cfg.reranker.model_name, self.cfg.reranker.device)
            if self.cfg.reranker.enabled and self.cfg.reranker.model_name
            else None
        )
        self._print_reranker_runtime_state(reranker)
        final_top_k = (
            self.cfg.reranker.final_top_k
            if self.cfg.reranker.enabled and self.cfg.reranker.final_top_k
            else self.cfg.retrieval.top_k
        )
        rerank_top_k = (
            self.cfg.reranker.rerank_top_k
            if self.cfg.reranker.enabled and self.cfg.reranker.rerank_top_k
            else final_top_k
        )
        rows: list[RetrievalRow] = []
        for row_index, query in enumerate(tqdm(queries, desc="Retrieving contexts", unit="question"), start=1):
            if self.logger:
                self.logger.info(
                    "row_start phase=retrieval question_id=%s row=%s/%s",
                    query.question_id,
                    row_index,
                    len(queries),
                )
            retrieval_start = time.perf_counter()
            self._write_event(
                stage="retrieval",
                event_type=EventType.RETRIEVAL_START,
                message="Retrieval started.",
                question_id=query.question_id,
                metrics={
                    "top_k": final_top_k,
                    "rerank_top_k": rerank_top_k,
                    "fetch_k": self.cfg.retrieval.fetch_k,
                    "retriever_type": self.cfg.retrieval.retriever_type,
                },
            )
            if reranker is not None:
                self._write_event(
                    stage="rerank",
                    event_type=EventType.RERANK_START,
                    message="Reranking started.",
                    question_id=query.question_id,
                    metrics={
                        "final_top_k": final_top_k,
                        "rerank_top_k": rerank_top_k,
                        "fetch_k": self.cfg.retrieval.fetch_k,
                    },
            )
            category_filter_applied = False
            category_fallback_used = False
            fallback_reason: str | None = None
            retrieval_mode = self.cfg.retrieval.retriever_type
            number_of_category_results = 0
            number_of_global_fallback_results = 0
            retrieval_warnings: list[str] = []
            if self.cfg.retrieval.retriever_type == "category_aware_dense" and hasattr(retriever, "set_active_category"):
                if query.category_validated:
                    category_filter_applied = True
                    retrieval_mode = "category_aware_dense"
                    retriever.set_active_category(query.detected_category)
                    raw_retrieved, retrieved, retrieval_warnings, reranker_used = retrieve_top_k_unique_contexts(
                        query.retrieval_question,
                        retriever,
                        reranker,
                        rerank_top_k,
                        self.cfg.retrieval.fetch_k,
                        max_candidates=len(self.chunks),
                    )
                    number_of_category_results = len(retrieved)
                    enough_retrieved_chunks = len(retrieved) >= final_top_k
                    if self.logger:
                        self.logger.info(
                            "retrieval_decision question_id=%s decision='Enough Retrieved Chunks?' retrieved=%s top_k=%s result=%s",
                            query.question_id,
                            len(retrieved),
                            final_top_k,
                            enough_retrieved_chunks,
                        )
                    if not enough_retrieved_chunks:
                        category_fallback_used = True
                        fallback_reason = "insufficient_category_results"
                        retrieval_mode = "global_fallback"
                        retriever.set_active_category(None)
                        raw_retrieved, retrieved, retrieval_warnings, reranker_used = retrieve_top_k_unique_contexts(
                            query.retrieval_question,
                            retriever,
                            reranker,
                            rerank_top_k,
                            self.cfg.retrieval.fetch_k,
                            max_candidates=len(self.chunks),
                        )
                        number_of_global_fallback_results = len(retrieved)
                else:
                    category_fallback_used = True
                    fallback_reason = query.category_validation_reason or "category_not_validated"
                    retrieval_mode = "global_fallback"
                    retriever.set_active_category(None)
                    raw_retrieved, retrieved, retrieval_warnings, reranker_used = retrieve_top_k_unique_contexts(
                        query.retrieval_question,
                        retriever,
                        reranker,
                        rerank_top_k,
                        self.cfg.retrieval.fetch_k,
                        max_candidates=len(self.chunks),
                    )
                    number_of_global_fallback_results = len(retrieved)
                    if self.logger:
                        self.logger.info(
                            "retrieval_decision question_id=%s decision='Category Validation' category_validated=false reason=%s",
                            query.question_id,
                            query.category_validation_reason,
                        )
            # If category validation fails, category-restricted retrieval is skipped
            # because there is no trusted category scope. The pipeline directly
            # performs global retrieval as controlled fallback.
            else:
                raw_retrieved, retrieved, retrieval_warnings, reranker_used = retrieve_top_k_unique_contexts(
                    query.retrieval_question,
                    retriever,
                    reranker,
                    rerank_top_k,
                    self.cfg.retrieval.fetch_k,
                    max_candidates=len(self.chunks),
                )
            reranked_candidates = list(retrieved)
            if reranker_used and len(retrieved) > final_top_k:
                retrieved = retrieved[:final_top_k]
            raw_dense_retrieved = last_candidates(retriever, "last_dense_candidates")
            raw_bm25_retrieved = last_candidates(retriever, "last_bm25_candidates")
            fused_retrieved = last_candidates(retriever, "last_fused_candidates")
            retrieval_diagnostics = retrieval_diagnostics_from(retriever)
            retrieval_diagnostics.update(
                {
                    "final_top_k": final_top_k,
                    "rerank_top_k": rerank_top_k,
                    "cleaned_question": query.cleaned_question,
                    "detected_category": query.detected_category,
                    "category_validated": query.category_validated,
                    "category_validation_reason": query.category_validation_reason,
                    "retrieval_mode": retrieval_mode,
                    "category_filter_applied": category_filter_applied,
                    "category_fallback_used": category_fallback_used,
                    "number_of_category_results": number_of_category_results,
                    "number_of_global_fallback_results": number_of_global_fallback_results,
                    "top_k": final_top_k,
                    "fetch_k": self.cfg.retrieval.fetch_k,
                    "decision": "Enough Retrieved Chunks?",
                    "retrieved_chunks": [item.chunk_id for item in retrieved],
                    "retrieved_documents": [
                        stable_retrieved_document_id(item.metadata, item.original_context_id)
                        for item in retrieved
                    ],
                    "retrieval_scores": [item.score for item in retrieved],
                    "retrieved_categories": [
                        item.metadata.get(self.cfg.retrieval.category_field)
                        for item in retrieved
                    ],
                    "reranked_candidate_ids": [item.chunk_id for item in reranked_candidates],
                    "final_candidate_ids": [item.chunk_id for item in retrieved],
                    # Fields required for per-question output records.
                    "retriever_type": self.cfg.retrieval.retriever_type,
                    "retrieval_scope": "category" if (category_filter_applied and not category_fallback_used) else "global",
                    "category_index_used": bool(retrieval_diagnostics.get("category_index_used", False)),
                    "fallback_used": category_fallback_used,
                    "fallback_reason": fallback_reason,
                }
            )
            retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            self._write_event(
                stage="retrieval",
                event_type=EventType.RETRIEVAL_END,
                message="Retrieval completed.",
                question_id=query.question_id,
                duration_ms=retrieval_time_ms,
                metrics={
                    "raw_candidates": len(raw_retrieved),
                    "final_contexts": len(retrieved),
                    "rerank_candidates": len(reranked_candidates),
                    "retriever_type": self.cfg.retrieval.retriever_type,
                },
                diagnostics={"warnings": retrieval_warnings, **retrieval_diagnostics},
            )
            if reranker_used:
                self._write_event(
                    stage="rerank",
                    event_type=EventType.RERANK_END,
                    message="Reranking completed.",
                    question_id=query.question_id,
                    duration_ms=retrieval_time_ms,
                    metrics={
                        "raw_candidates": len(raw_retrieved),
                        "final_contexts": len(retrieved),
                        "rerank_candidates": len(reranked_candidates),
                    },
                    diagnostics={"duration_includes_retrieval": True},
                )
            for warning in retrieval_warnings:
                if self.logger:
                    self.logger.warning("row_retrieval_warning question_id=%s warning=%s", query.question_id, warning)
            if self.logger:
                self.logger.info(
                    "row_retrieved question_id=%s raw_candidates=%s unique_final_contexts=%s scores=%s retrieval_time_ms=%.2f",
                    query.question_id,
                    len(raw_retrieved),
                    len(retrieved),
                    len([item.score for item in retrieved]),
                    retrieval_time_ms,
                )
            rows.append(
                RetrievalRow(
                    query=query,
                    raw_retrieved=raw_retrieved,
                    raw_dense_retrieved=raw_dense_retrieved,
                    raw_bm25_retrieved=raw_bm25_retrieved,
                    fused_retrieved=fused_retrieved,
                    retrieved=retrieved,
                    retrieval_time_ms=retrieval_time_ms,
                    reranker_used=reranker_used,
                    retrieval_warnings=retrieval_warnings,
                    retrieval_diagnostics=retrieval_diagnostics,
                )
            )
        return RetrievalStageOutput(
            stage_name=self.stage_name,
            artifacts={"retriever": retriever, "reranker": reranker, "retrieval_rows": rows},
            diagnostics={"attempted": len(queries), "retrieved_rows": len(rows)},
            metadata={
                "final_top_k": final_top_k,
                "rerank_top_k": rerank_top_k,
                "retriever_type": self.cfg.retrieval.retriever_type,
            },
            retriever=retriever,
            reranker=reranker,
            final_top_k=final_top_k,
            retrieval_rows=rows,
            attempted=len(queries),
        )

    def _write_event(self, **kwargs) -> None:
        if self.event_writer is not None:
            self.event_writer.write(**kwargs)

    def _print_reranker_runtime_state(self, reranker) -> None:
        if reranker is None:
            print("[startup] reranker=disabled")
            return
        runtime_device = getattr(reranker, "runtime_device", "<unknown>")
        requested_device = getattr(reranker, "requested_device", self.cfg.reranker.device)
        print(
            "[startup] "
            f"reranker_device={requested_device} "
            f"reranker_runtime_device={runtime_device}"
        )


def retrieve_top_k_unique_contexts(
    question: str,
    retriever,
    reranker,
    top_k: int,
    fetch_k: int,
    max_candidates: int,
) -> tuple[list, list, list[str], bool]:
    candidate_k = max(fetch_k, top_k)
    raw_retrieved = []
    retrieved = []
    reranker_used = reranker is not None
    while True:
        raw_retrieved = retriever.retrieve(question, candidate_k)
        ranked = reranker.rerank(question, raw_retrieved, top_k) if reranker is not None else raw_retrieved
        retrieved = dedupe_retrieval_by_chunk_id(ranked, top_k)
        if len(retrieved) >= top_k or candidate_k >= max_candidates:
            break
        candidate_k = min(max_candidates, max(candidate_k + 1, candidate_k * 2))
    warnings = []
    if len(retrieved) < top_k:
        warnings.append(
            f"Only {len(retrieved)} unique chunks were available after deduplication; requested top_k={top_k}."
        )
    return raw_retrieved, retrieved, warnings, reranker_used


def dedupe_retrieval_by_chunk_id(items: list, top_k: int) -> list:
    seen: set[str] = set()
    unique = []
    for item in items:
        key = str(item.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def last_candidates(retriever, attribute: str) -> list:
    value = getattr(retriever, attribute, None)
    return list(value) if isinstance(value, list) else []


def retrieval_diagnostics_from(retriever) -> dict:
    value = getattr(retriever, "last_retrieval_diagnostics", None)
    if not isinstance(value, dict):
        return {}
    return json_safe(value)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(item) for item in value)
    return value
