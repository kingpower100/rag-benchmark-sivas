from __future__ import annotations

from src.pipeline1.retrieval.base import BaseRetriever
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.utils.ids import stable_retrieved_document_id


class CategoryAwareDenseRetriever(BaseRetriever):
    def __init__(
        self,
        dense_retriever,
        category_field: str = "kategorie",
        embeddings=None,
        index_metric: str = "cosine",
    ) -> None:
        self.dense_retriever = dense_retriever
        self.category_field = category_field
        self.active_category: str | None = None
        self.last_dense_candidates: list[RetrievalItem] = []
        self.last_retrieval_diagnostics: dict = {}
        # Per-category FAISS retrievers built from pre-computed embeddings.
        # When embeddings is None the retriever falls back to global search + post-hoc
        # category filter, which preserves full backwards-compatibility with tests and
        # any caller that does not supply embeddings.
        self._category_retrievers: dict[str, object] = {}
        if embeddings is not None:
            self._build_category_retrievers(embeddings, index_metric)

    def _build_category_retrievers(self, embeddings, index_metric: str) -> None:
        import numpy as np
        from src.pipeline1.indexing.faiss_index import FaissIndex
        from src.pipeline1.retrieval.dense_retriever import DenseRetriever

        chunks = self.dense_retriever.chunks
        groups: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            cat = str((chunk.metadata or {}).get(self.category_field) or "").strip()
            if cat:
                groups.setdefault(cat, []).append(i)

        for cat, indices in groups.items():
            cat_chunks = [chunks[i] for i in indices]
            cat_embeddings = np.array(embeddings[indices], dtype="float32")
            cat_index = FaissIndex(metric=index_metric)
            cat_index.build(cat_embeddings)
            self._category_retrievers[cat] = DenseRetriever(
                embedder=self.dense_retriever.embedder,
                index=cat_index,
                chunks=cat_chunks,
                fetch_k=self.dense_retriever.fetch_k,
                metadata_boosting=self.dense_retriever.metadata_boosting,
                metadata_filtering=self.dense_retriever.metadata_filtering,
            )

    def set_active_category(self, category: str | None) -> None:
        self.active_category = str(category).strip() if category else None

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        if self.active_category and self.active_category in self._category_retrievers:
            # TRUE CATEGORY-SCOPED: search only within this category's dedicated FAISS index.
            # No global index is consulted; only chunks belonging to the predicted category
            # are ranked, so no globally-high-scoring off-category chunks can displace them.
            cat_retriever = self._category_retrievers[self.active_category]
            candidates = cat_retriever.retrieve(question, top_k)
            self.last_dense_candidates = list(getattr(cat_retriever, "last_dense_candidates", candidates))
            diagnostics = dict(getattr(cat_retriever, "last_retrieval_diagnostics", {}) or {})
            diagnostics.update(
                {
                    "category_filter_field": self.category_field,
                    "detected_category": self.active_category,
                    "category_filter_applied": True,
                    "category_fallback_used": False,
                    "category_filter_fallback": False,
                    "retrieval_backend": "category_faiss",
                }
            )
            selected = candidates[:top_k]

        elif self.active_category:
            # Category was requested but no per-category retriever exists (embeddings were
            # not provided at construction time). Fall back to global search + post-hoc filter.
            candidates = self.dense_retriever.retrieve(question, top_k)
            self.last_dense_candidates = list(getattr(self.dense_retriever, "last_dense_candidates", candidates))
            diagnostics = dict(getattr(self.dense_retriever, "last_retrieval_diagnostics", {}) or {})
            category_matches = [
                item
                for item in candidates
                if str((item.metadata or {}).get(self.category_field) or "").strip() == self.active_category
            ]
            selected = list(category_matches[:top_k])
            diagnostics.update(
                {
                    "category_filter_field": self.category_field,
                    "detected_category": self.active_category,
                    "category_filter_applied": True,
                    "category_fallback_used": False,
                    "category_filter_fallback": False,
                    "retrieval_backend": "dense_post_filter",
                    "category_matches": len(category_matches),
                    "global_candidates": len(candidates),
                }
            )

        else:
            # No active category: full global retrieval.
            candidates = self.dense_retriever.retrieve(question, top_k)
            self.last_dense_candidates = list(getattr(self.dense_retriever, "last_dense_candidates", candidates))
            diagnostics = dict(getattr(self.dense_retriever, "last_retrieval_diagnostics", {}) or {})
            selected = candidates[:top_k]
            diagnostics.update(
                {
                    "category_filter_field": self.category_field,
                    "detected_category": self.active_category,
                    "category_filter_applied": False,
                    "category_fallback_used": False,
                    "category_filter_fallback": False,
                }
            )

        diagnostics.update(_result_payload(selected, self.category_field))
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
