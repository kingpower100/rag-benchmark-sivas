from __future__ import annotations

import logging

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.utils.ids import stable_retrieved_document_id

_log = logging.getLogger(__name__)


class AdaptiveCategoryAwareHybridRRFRetriever(BaseRetriever):
    """Hybrid RRF retriever with adaptive category-aware routing.

    Wraps ElasticsearchHybridRRFRetriever and exposes set_active_category()
    and retrieve_global_probe() so run_adaptive_category_aware_retrieval()
    can drive it unchanged.  When a category is active, both the dense and
    BM25 legs are restricted to that category via retrieve_with_category().
    When no category is active (global mode or probe), the existing
    retrieve() path is used with no change to algorithm or scoring.
    """

    def __init__(
        self,
        hybrid_retriever,
        category_field: str = "kategorie",
        category_filter_field: str | None = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.category_field = category_field
        self.category_filter_field = category_filter_field or f"metadata.{category_field}.keyword"
        self.active_category: str | None = None
        self.category_retrieval_empty: bool = False
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_bm25_candidates: list[RetrievalItem] = []
        self.last_fused_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    def set_active_category(self, category: str | None) -> None:
        self.active_category = str(category).strip() if category else None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        self.category_retrieval_empty = False
        _category_filter_diag: dict = {}

        if self.active_category and hasattr(self.hybrid_retriever, "retrieve_with_category"):
            _log.info(
                "AdaptiveCategoryAwareHybridRRF retrieve: category_filter=True category=%r top_k=%s",
                self.active_category, top_k,
            )
            results = self.hybrid_retriever.retrieve_with_category(
                question, top_k, self.active_category, self.category_filter_field
            )
            _log.info(
                "AdaptiveCategoryAwareHybridRRF category retrieve done: results=%s",
                len(results),
            )
            if not results:
                self.category_retrieval_empty = True
                _category_filter_diag = {
                    k: dict(getattr(self.hybrid_retriever, "last_retrieval_diagnostics", {})).get(k)
                    for k in ("category_filter_applied_dense", "category_filter_applied_bm25")
                }
                _log.warning(
                    "AdaptiveCategoryAwareHybridRRF: category=%r returned 0 candidates; falling back to global",
                    self.active_category,
                )
                results = self.hybrid_retriever.retrieve(question, top_k)
                _log.info(
                    "AdaptiveCategoryAwareHybridRRF empty-category global fallback done: results=%s",
                    len(results),
                )
        else:
            _log.info(
                "AdaptiveCategoryAwareHybridRRF retrieve: category_filter=False (global) top_k=%s",
                top_k,
            )
            results = self.hybrid_retriever.retrieve(question, top_k)
            _log.info(
                "AdaptiveCategoryAwareHybridRRF global retrieve done: results=%s",
                len(results),
            )
        self._sync_candidates()
        self.last_retrieval_diagnostics = {
            **dict(getattr(self.hybrid_retriever, "last_retrieval_diagnostics", {})),
            "category_filter_field": self.category_filter_field,
            "detected_category": self.active_category,
            "category_filter_applied": bool(self.active_category),
            "category_fallback_used": self.category_retrieval_empty,
            "category_filter_fallback": False,
            "category_retrieval_empty": self.category_retrieval_empty,
            "retrieval_backend": (
                "adaptive_category_aware_hybrid_rrf_category"
                if (self.active_category and not self.category_retrieval_empty)
                else "adaptive_category_aware_hybrid_rrf_global"
            ),
            "category_index_used": False,
            **_result_payload(results, self.category_field),
            **_category_filter_diag,
        }
        return results

    def retrieve_global_probe(self, question: str, probe_fetch_k: int) -> list[RetrievalItem]:
        """Global Hybrid RRF probe used by the adaptive routing validation logic."""
        _log.info(
            "AdaptiveCategoryAwareHybridRRF global probe: probe_fetch_k=%s", probe_fetch_k
        )
        self.set_active_category(None)
        results = self.hybrid_retriever.retrieve(question, probe_fetch_k)
        _log.info(
            "AdaptiveCategoryAwareHybridRRF global probe done: results=%s", len(results)
        )
        self._sync_candidates()
        self.last_retrieval_diagnostics = {
            **dict(getattr(self.hybrid_retriever, "last_retrieval_diagnostics", {})),
            "category_filter_field": self.category_filter_field,
            "detected_category": None,
            "category_filter_applied": False,
            "category_fallback_used": False,
            "category_filter_fallback": False,
            "retrieval_backend": "adaptive_category_aware_hybrid_rrf_probe",
            "category_index_used": False,
            "probe_score_semantics": "higher_is_better",
            **_result_payload(results, self.category_field),
        }
        return results

    def extract_query_metadata(self, question: str):
        if hasattr(self.hybrid_retriever, "extract_query_metadata"):
            return self.hybrid_retriever.extract_query_metadata(question)
        return None

    def _sync_candidates(self) -> None:
        self.last_dense_candidates = list(getattr(self.hybrid_retriever, "last_dense_candidates", []))
        self.last_bm25_candidates = list(getattr(self.hybrid_retriever, "last_bm25_candidates", []))
        self.last_fused_candidates = list(getattr(self.hybrid_retriever, "last_fused_candidates", []))


def _result_payload(items: list[RetrievalItem], category_field: str) -> dict:
    return {
        "retrieved_chunks": [item.chunk_id for item in items],
        "retrieved_documents": [
            stable_retrieved_document_id(item.metadata, item.original_context_id)
            for item in items
        ],
        "retrieval_scores": [item.score for item in items],
        "retrieved_categories": [item.metadata.get(category_field) for item in items],
    }
