from __future__ import annotations

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.utils.ids import stable_retrieved_document_id


class CategoryAwareDenseRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever,
        category_field: str = "kategorie",
        fallback_to_global: bool = True,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.category_field = category_field
        self.fallback_to_global = fallback_to_global
        self.active_category: str | None = None
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}

    def set_active_category(self, category: str | None) -> None:
        self.active_category = str(category).strip() if category else None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        candidates = self.dense_retriever.retrieve(question, top_k)
        self.last_dense_candidates = list(getattr(self.dense_retriever, "last_dense_candidates", candidates))
        diagnostics = dict(getattr(self.dense_retriever, "last_retrieval_diagnostics", {}) or {})
        category_filter_applied = bool(self.active_category)
        diagnostics.update(
            {
                "category_filter_field": self.category_field,
                "detected_category": self.active_category,
                "category_filter_applied": category_filter_applied,
                "category_fallback_used": False,
                "category_filter_fallback": False,
            }
        )
        if not self.active_category:
            selected = candidates[:top_k]
            diagnostics.update(_result_payload(selected, self.category_field))
            self.last_retrieval_diagnostics = diagnostics
            return selected
        category_matches = [
            item
            for item in candidates
            if str((item.metadata or {}).get(self.category_field) or "").strip() == self.active_category
        ]
        selected = list(category_matches[:top_k])
        diagnostics.update(
            {
                "category_matches": len(category_matches),
                "global_candidates": len(candidates),
                **_result_payload(selected, self.category_field),
            }
        )
        self.last_retrieval_diagnostics = diagnostics
        return selected

    def extract_query_metadata(self, question: str):
        if hasattr(self.dense_retriever, "extract_query_metadata"):
            return self.dense_retriever.extract_query_metadata(question)
        return None


def _result_payload(items: list[RetrievalItem], category_field: str) -> dict:
    return {
        "retrieved_chunks": [item.chunk_id for item in items],
        "retrieved_documents": [
            stable_retrieved_document_id(item.metadata, item.original_context_id)
            for item in items
        ],
        "retrieval_scores": [item.score for item in items],
        "retrieved_categories": [
            item.metadata.get(category_field)
            for item in items
        ],
    }
