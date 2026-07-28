from __future__ import annotations

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.utils.ids import stable_retrieved_document_id


class AdaptiveCategoryAwareHybridRRFRetriever(BaseRetriever):
    """Hybrid RRF retriever with adaptive category-aware routing.

    Wraps ElasticsearchHybridRRFRetriever and exposes set_active_category()
    and retrieve_global_probe() so run_adaptive_category_aware_retrieval()
    can drive it unchanged.  When a category is active, both the dense and
    BM25 legs are restricted to that category via retrieve_with_category().
    When no category is active (global mode or probe), the existing
    retrieve() path is used with no change to algorithm or scoring.
    """

    def __init__(self, hybrid_retriever, category_field: str = "kategorie") -> None:
        self.hybrid_retriever = hybrid_retriever
        self.category_field = category_field
        self.active_category: str | None = None
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_bm25_candidates: list[RetrievalItem] = []
        self.last_fused_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    def set_active_category(self, category: str | None) -> None:
        self.active_category = str(category).strip() if category else None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        if self.active_category and hasattr(self.hybrid_retriever, "retrieve_with_category"):
            results = self.hybrid_retriever.retrieve_with_category(
                question, top_k, self.active_category, self.category_field
            )
        else:
            results = self.hybrid_retriever.retrieve(question, top_k)
        self._sync_candidates()
        self.last_retrieval_diagnostics = {
            **dict(getattr(self.hybrid_retriever, "last_retrieval_diagnostics", {})),
            "category_filter_field": self.category_field,
            "detected_category": self.active_category,
            "category_filter_applied": bool(self.active_category),
            "category_fallback_used": False,
            "category_filter_fallback": False,
            "retrieval_backend": (
                "adaptive_category_aware_hybrid_rrf_category"
                if self.active_category
                else "adaptive_category_aware_hybrid_rrf_global"
            ),
            "category_index_used": False,
            **_result_payload(results, self.category_field),
        }
        return results

    def retrieve_global_probe(self, question: str, probe_fetch_k: int) -> list[RetrievalItem]:
        """Global Hybrid RRF probe used by the adaptive routing validation logic."""
        self.set_active_category(None)
        results = self.hybrid_retriever.retrieve(question, probe_fetch_k)
        self._sync_candidates()
        self.last_retrieval_diagnostics = {
            **dict(getattr(self.hybrid_retriever, "last_retrieval_diagnostics", {})),
            "category_filter_field": self.category_field,
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
