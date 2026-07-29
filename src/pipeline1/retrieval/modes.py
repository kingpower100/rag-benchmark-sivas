"""Shared retrieval-mode groups used by runtime, preflight, and schema checks."""

CATEGORY_PREDICTION_RETRIEVER_TYPES = frozenset(
    {
        "category_aware_dense",
        "adaptive_category_aware_dense",
        # Adaptive hybrid must run orchestration first: the predicted category
        # is validated, probed globally, then used to choose category or global Hybrid RRF.
        "adaptive_category_aware_hybrid_rrf",
    }
)

ADAPTIVE_CATEGORY_AWARE_RETRIEVER_TYPES = frozenset(
    {
        "adaptive_category_aware_dense",
        "adaptive_category_aware_hybrid_rrf",
    }
)
