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
    retriever_time_ms: float
    rerank_time_ms: float
    retrieval_pipeline_time_ms: float
    reranker_used: bool
    retrieval_warnings: list[str]
    retrieval_diagnostics: dict
    generation_contexts: list = field(default_factory=list)
    parent_context_diagnostics: dict = field(default_factory=dict)

    def as_generation_tuple(self) -> tuple:
        return (
            self.query,
            self.raw_retrieved,
            self.raw_dense_retrieved,
            self.raw_bm25_retrieved,
            self.fused_retrieved,
            self.retrieved,
            self.retrieval_time_ms,
            self.retriever_time_ms,
            self.rerank_time_ms,
            self.retrieval_pipeline_time_ms,
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
            adaptive_diagnostics: dict = {}
            if self.cfg.retrieval.retriever_type == "adaptive_category_aware_dense" and hasattr(retriever, "set_active_category"):
                (
                    raw_retrieved,
                    retrieved,
                    retrieval_warnings,
                    reranker_used,
                    selection_diagnostics,
                    adaptive_diagnostics,
                ) = run_adaptive_category_aware_retrieval(
                    query=query,
                    retriever=retriever,
                    reranker=reranker,
                    cfg=self.cfg,
                    final_top_k=final_top_k,
                    rerank_top_k=rerank_top_k,
                    max_candidates=len(self.chunks),
                )
                retrieval_mode = str(adaptive_diagnostics["retrieval_mode"])
                category_filter_applied = bool(adaptive_diagnostics["category_filter_applied"])
                category_fallback_used = bool(adaptive_diagnostics["category_fallback_used"])
                fallback_reason = adaptive_diagnostics.get("fallback_reason")
                number_of_category_results = int(adaptive_diagnostics["number_of_category_results"])
                number_of_global_fallback_results = int(adaptive_diagnostics["number_of_global_fallback_results"])
            elif self.cfg.retrieval.retriever_type == "category_aware_dense" and hasattr(retriever, "set_active_category"):
                if query.category_validated:
                    category_filter_applied = True
                    retrieval_mode = "category_aware_dense"
                    retriever.set_active_category(query.detected_category)
                    raw_retrieved, retrieved, retrieval_warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
                        query.retrieval_question,
                        retriever,
                        reranker,
                        final_top_k,
                        self.cfg.retrieval.fetch_k,
                        max_candidates=len(self.chunks),
                        rerank_top_k=rerank_top_k,
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
                        if self.cfg.retrieval.fallback_to_global:
                            category_fallback_used = True
                            fallback_reason = "insufficient_category_results_global_fallback"
                            retrieval_mode = "global_fallback"
                            retriever.set_active_category(None)
                            raw_retrieved, retrieved, retrieval_warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
                                query.retrieval_question,
                                retriever,
                                reranker,
                                final_top_k,
                                self.cfg.retrieval.fetch_k,
                                max_candidates=len(self.chunks),
                                rerank_top_k=rerank_top_k,
                            )
                            number_of_global_fallback_results = len(retrieved)
                        else:
                            fallback_reason = "fallback_disabled_insufficient_results"
                            retrieval_mode = "category_aware_dense_no_fallback"
                else:
                    if self.cfg.retrieval.fallback_to_global:
                        category_fallback_used = True
                        fallback_reason = "invalid_category_global_fallback"
                        retrieval_mode = "global_fallback"
                        retriever.set_active_category(None)
                        raw_retrieved, retrieved, retrieval_warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
                            query.retrieval_question,
                            retriever,
                            reranker,
                            final_top_k,
                            self.cfg.retrieval.fetch_k,
                            max_candidates=len(self.chunks),
                            rerank_top_k=rerank_top_k,
                        )
                        number_of_global_fallback_results = len(retrieved)
                    else:
                        fallback_reason = "fallback_disabled_invalid_category"
                        retrieval_mode = "category_unavailable_no_fallback"
                        raw_retrieved = []
                        retrieved = []
                        reranker_used = reranker is not None
                        selection_diagnostics = _empty_selection_diagnostics(reranker, final_top_k, rerank_top_k)
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
                raw_retrieved, retrieved, retrieval_warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
                    query.retrieval_question,
                    retriever,
                    reranker,
                    final_top_k,
                    self.cfg.retrieval.fetch_k,
                    max_candidates=len(self.chunks),
                    rerank_top_k=rerank_top_k,
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
                    **selection_diagnostics,
                    **(adaptive_diagnostics if self.cfg.retrieval.retriever_type == "adaptive_category_aware_dense" else {}),
                    "final_top_k": final_top_k,
                    "rerank_top_k": rerank_top_k,
                    "cleaned_question": query.cleaned_question,
                    "detected_category": query.detected_category,
                    "category_validated": query.category_validated,
                    "category_validation_reason": query.category_validation_reason,
                    "orchestration_status": (
                        "disabled"
                        if query.category_validation_reason == "orchestration_disabled"
                        else "enabled"
                    ),
                    "retrieval_mode": retrieval_mode,
                    "category_filter_applied": category_filter_applied,
                    "category_fallback_used": category_fallback_used,
                    "number_of_category_results": number_of_category_results,
                    "number_of_global_fallback_results": number_of_global_fallback_results,
                    "top_k": final_top_k,
                    "fetch_k": self.cfg.retrieval.fetch_k,
                    "configured_fetch_k": self.cfg.retrieval.fetch_k,
                    "raw_candidate_request_k": self.cfg.retrieval.fetch_k,
                    "actual_raw_candidates_returned": len(raw_retrieved),
                    "unique_final_contexts": len(retrieved),
                    "candidate_expansion_enabled": False,
                    "candidate_expansion_occurred": False,
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
                    "final_chunk_ids": [item.chunk_id for item in retrieved],
                    # Fields required for per-question output records.
                    "retriever_type": self.cfg.retrieval.retriever_type,
                    "retrieval_scope": "category" if (category_filter_applied and not category_fallback_used) else "global",
                    "category_index_used": bool(retrieval_diagnostics.get("category_index_used", False)),
                    "fallback_used": category_fallback_used,
                    "fallback_reason": fallback_reason,
                }
            )
            retrieval_pipeline_time_ms = (time.perf_counter() - retrieval_start) * 1000
            retrieval_time_ms = retrieval_pipeline_time_ms
            retrieval_diagnostics["retrieval_pipeline_time_ms"] = retrieval_pipeline_time_ms
            self._write_event(
                stage="retrieval",
                event_type=EventType.RETRIEVAL_END,
                message="Retrieval completed.",
                question_id=query.question_id,
                duration_ms=retrieval_pipeline_time_ms,
                metrics={
                    "raw_candidates": len(raw_retrieved),
                    "final_contexts": len(retrieved),
                    "rerank_candidates": len(reranked_candidates),
                    "retriever_type": self.cfg.retrieval.retriever_type,
                    "retriever_time_ms": selection_diagnostics["retriever_time_ms"],
                    "rerank_time_ms": selection_diagnostics["rerank_time_ms"],
                },
                diagnostics={"warnings": retrieval_warnings, **retrieval_diagnostics},
            )
            if reranker_used:
                self._write_event(
                    stage="rerank",
                    event_type=EventType.RERANK_END,
                    message="Reranking completed.",
                    question_id=query.question_id,
                    duration_ms=selection_diagnostics["rerank_time_ms"],
                    metrics={
                        "raw_candidates": len(raw_retrieved),
                        "final_contexts": len(retrieved),
                        "rerank_candidates": len(reranked_candidates),
                    },
                    diagnostics={"duration_includes_retrieval": False},
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
                    retrieval_pipeline_time_ms,
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
                    retriever_time_ms=float(selection_diagnostics["retriever_time_ms"]),
                    rerank_time_ms=float(selection_diagnostics["rerank_time_ms"]),
                    retrieval_pipeline_time_ms=retrieval_pipeline_time_ms,
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


def run_adaptive_category_aware_retrieval(
    query: QueryRecord,
    retriever,
    reranker,
    cfg: PipelineConfig,
    final_top_k: int,
    rerank_top_k: int | None,
    max_candidates: int,
) -> tuple[list, list, list[str], bool, dict, dict]:
    validation_cfg = cfg.retrieval.category_routing_validation
    thresholds = {
        "minimum_category_share": validation_cfg.minimum_category_share,
        "minimum_category_count": validation_cfg.minimum_category_count,
        "minimum_margin": validation_cfg.minimum_margin,
    }
    predicted_category = query.detected_category
    base_diagnostics = {
        "predicted_category": predicted_category,
        "category_validated": query.category_validated,
        "probe_fetch_k": validation_cfg.probe_fetch_k,
        "routing_thresholds": thresholds,
        "retriever_type": cfg.retrieval.retriever_type,
    }

    if not query.category_validated or not predicted_category:
        retriever.set_active_category(None)
        raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
            query.retrieval_question,
            retriever,
            reranker,
            final_top_k,
            cfg.retrieval.fetch_k,
            max_candidates=max_candidates,
            rerank_top_k=rerank_top_k,
        )
        diagnostics = {
            **base_diagnostics,
            **empty_probe_diagnostics(),
            "routing_decision": "rejected",
            "routing_accepted": False,
            "decision_reason": "invalid_or_missing_category",
            "final_retrieval_mode": "global",
            "retrieval_mode": "global_fallback",
            "category_filter_applied": False,
            "category_fallback_used": True,
            "fallback_used": True,
            "fallback_reason": "invalid_category_global_fallback",
            "number_of_category_results": 0,
            "number_of_global_fallback_results": len(retrieved),
        }
        return raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics, diagnostics

    if not validation_cfg.enabled:
        retriever.set_active_category(predicted_category)
        raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
            query.retrieval_question,
            retriever,
            reranker,
            final_top_k,
            cfg.retrieval.fetch_k,
            max_candidates=max_candidates,
            rerank_top_k=rerank_top_k,
        )
        diagnostics = {
            **base_diagnostics,
            **empty_probe_diagnostics(),
            "routing_decision": "accepted",
            "routing_accepted": True,
            "decision_reason": "routing_validation_disabled",
            "final_retrieval_mode": "category",
            "retrieval_mode": "adaptive_category_aware_dense",
            "category_filter_applied": True,
            "category_fallback_used": False,
            "fallback_used": False,
            "fallback_reason": None,
            "number_of_category_results": len(retrieved),
            "number_of_global_fallback_results": 0,
        }
        return raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics, diagnostics

    retriever.set_active_category(None)
    if hasattr(retriever, "retrieve_global_probe"):
        probe_candidates = retriever.retrieve_global_probe(query.retrieval_question, validation_cfg.probe_fetch_k)
    else:
        probe_candidates = retriever.retrieve(query.retrieval_question, validation_cfg.probe_fetch_k)
    probe_stats = category_probe_support_stats(
        probe_candidates,
        predicted_category,
        cfg.retrieval.category_field,
    )
    accepted, decision_reason = adaptive_routing_decision(probe_stats, thresholds)
    if accepted:
        retriever.set_active_category(predicted_category)
        final_mode = "category"
        retrieval_mode = "adaptive_category_aware_dense"
        category_filter_applied = True
        category_fallback_used = False
        fallback_used = False
        fallback_reason = None
    else:
        retriever.set_active_category(None)
        final_mode = "global"
        retrieval_mode = "global_fallback"
        category_filter_applied = False
        category_fallback_used = True
        fallback_used = True
        fallback_reason = decision_reason

    raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics = retrieve_top_k_unique_contexts(
        query.retrieval_question,
        retriever,
        reranker,
        final_top_k,
        cfg.retrieval.fetch_k,
        max_candidates=max_candidates,
        rerank_top_k=rerank_top_k,
    )
    diagnostics = {
        **base_diagnostics,
        **probe_stats,
        "routing_decision": "accepted" if accepted else "rejected",
        "routing_accepted": accepted,
        "decision_reason": decision_reason,
        "final_retrieval_mode": final_mode,
        "retrieval_mode": retrieval_mode,
        "category_filter_applied": category_filter_applied,
        "category_fallback_used": category_fallback_used,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "number_of_category_results": len(retrieved) if accepted else 0,
        "number_of_global_fallback_results": 0 if accepted else len(retrieved),
    }
    return raw_retrieved, retrieved, warnings, reranker_used, selection_diagnostics, diagnostics


def category_probe_support_stats(candidates: list, predicted_category: str, category_field: str) -> dict:
    total = len(candidates)
    counts: dict[str, int] = {}
    score_sums: dict[str, float] = {}
    for item in candidates:
        category = str((item.metadata or {}).get(category_field) or "").strip()
        if not category:
            category = "<missing>"
        counts[category] = counts.get(category, 0) + 1
        score_sums[category] = score_sums.get(category, 0.0) + float(item.score)

    predicted_count = counts.get(predicted_category, 0)
    competitors = {category: count for category, count in counts.items() if category != predicted_category}
    strongest_competing_category = None
    competing_count = 0
    if competitors:
        strongest_competing_category, competing_count = sorted(
            competitors.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
    share = (predicted_count / total) if total else 0.0
    average_scores = {
        category: score_sums[category] / counts[category]
        for category in sorted(counts)
    }
    return {
        "probe_candidate_ids": [item.chunk_id for item in candidates],
        "probe_candidate_categories": [
            item.metadata.get(category_field)
            for item in candidates
        ],
        "probe_candidate_scores": [item.score for item in candidates],
        "probe_score_semantics": "higher_is_better",
        "total_probe_candidates": total,
        "predicted_category_count": predicted_count,
        "predicted_category_share": share,
        "strongest_competing_category": strongest_competing_category,
        "competing_category": strongest_competing_category,
        "competing_category_count": competing_count,
        "support_margin": predicted_count - competing_count,
        "average_similarity_by_category": average_scores,
    }


def adaptive_routing_decision(probe_stats: dict, thresholds: dict) -> tuple[bool, str]:
    if int(probe_stats["total_probe_candidates"]) == 0:
        return False, "empty_global_probe"
    failures = []
    if float(probe_stats["predicted_category_share"]) < float(thresholds["minimum_category_share"]):
        failures.append("category_share_below_threshold")
    if int(probe_stats["predicted_category_count"]) < int(thresholds["minimum_category_count"]):
        failures.append("category_count_below_threshold")
    if int(probe_stats["support_margin"]) < int(thresholds["minimum_margin"]):
        failures.append("support_margin_below_threshold")
    if failures:
        return False, ",".join(failures)
    return True, "thresholds_satisfied"


def empty_probe_diagnostics() -> dict:
    return {
        "probe_candidate_ids": [],
        "probe_candidate_categories": [],
        "total_probe_candidates": 0,
        "predicted_category_count": 0,
        "predicted_category_share": 0.0,
        "strongest_competing_category": None,
        "competing_category": None,
        "competing_category_count": 0,
        "support_margin": 0,
        "average_similarity_by_category": {},
    }


def retrieve_top_k_unique_contexts(
    question: str,
    retriever,
    reranker,
    top_k: int,
    fetch_k: int,
    max_candidates: int,
    rerank_top_k: int | None = None,
) -> tuple[list, list, list[str], bool, dict]:
    candidate_k = fetch_k
    reranker_used = reranker is not None
    retriever_start = time.perf_counter()
    raw_retrieved = retriever.retrieve(question, candidate_k)
    retriever_time_ms = (time.perf_counter() - retriever_start) * 1000
    rerank_time_ms = 0.0
    rerank_candidate_limit = max(0, min(rerank_top_k or max_candidates, max_candidates))
    ranked = raw_retrieved
    if reranker is not None:
        rerank_start = time.perf_counter()
        ranked = reranker.rerank(question, raw_retrieved, len(raw_retrieved))
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
    ranked_for_selection = ranked[:rerank_candidate_limit]
    retrieved = dedupe_retrieval_by_chunk_id(ranked_for_selection, top_k)
    if reranker is not None and len(retrieved) < top_k and len(ranked_for_selection) < len(ranked):
        retrieved = dedupe_retrieval_by_chunk_id(ranked, top_k)
    warnings = []
    if len(retrieved) < top_k:
        warnings.append(
            f"Only {len(retrieved)} unique chunks were available after deduplication within fetch_k={fetch_k}; requested top_k={top_k}."
        )
    duplicate_count = len(raw_retrieved) - len({item.chunk_id for item in raw_retrieved})
    diagnostics = {
        "retriever_time_ms": retriever_time_ms,
        "rerank_time_ms": rerank_time_ms,
        "reranker_enabled": reranker is not None,
        "reranker_applied": reranker is not None,
        "reranker_model_name": getattr(reranker, "model_name", None),
        "reranker_device_requested": getattr(reranker, "requested_device", None),
        "reranker_device_actual": getattr(reranker, "runtime_device", None),
        "reranker_candidate_count": len(raw_retrieved),
        "reranker_scored_count": len(raw_retrieved) if reranker is not None else 0,
        "reranker_output_count": len(ranked_for_selection) if reranker is not None else 0,
        "reranker_failure": False,
        "reranker_failure_reason": None,
        "raw_candidate_count": len(raw_retrieved),
        "duplicate_count": duplicate_count,
        "unique_candidate_count": len({item.chunk_id for item in raw_retrieved}),
        "final_result_count": len(retrieved),
    }
    return raw_retrieved, retrieved, warnings, reranker_used, diagnostics


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


def _empty_selection_diagnostics(reranker, final_top_k: int, rerank_top_k: int | None) -> dict:
    return {
        "retriever_time_ms": 0.0,
        "rerank_time_ms": 0.0,
        "reranker_enabled": reranker is not None,
        "reranker_applied": False,
        "reranker_model_name": getattr(reranker, "model_name", None),
        "reranker_device_requested": getattr(reranker, "requested_device", None),
        "reranker_device_actual": getattr(reranker, "runtime_device", None),
        "reranker_candidate_count": 0,
        "reranker_scored_count": 0,
        "reranker_output_count": 0,
        "reranker_failure": False,
        "reranker_failure_reason": None,
        "raw_candidate_count": 0,
        "duplicate_count": 0,
        "unique_candidate_count": 0,
        "final_result_count": 0,
    }


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
