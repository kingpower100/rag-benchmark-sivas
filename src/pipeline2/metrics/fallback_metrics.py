from __future__ import annotations

from typing import Any


def compute_fallback_flag(row: dict[str, Any]) -> tuple[bool, str | None]:
    """Return (fallback_used, fallback_reason) for a single Pipeline 1 result row.

    Reads ``category_fallback_used`` written by the Pipeline 1 generation stage
    and derives a reason string from orchestration/retrieval diagnostic fields
    that are already present in every result row.
    """
    fallback_used = bool(row.get("category_fallback_used", False))
    if not fallback_used:
        return False, None
    return True, _fallback_reason(row)


def _fallback_reason(row: dict[str, Any]) -> str:
    if row.get("orchestration_error"):
        return "no_category_detected"
    validated = bool(row.get("category_validated", False))
    reason_text = str(row.get("category_validation_reason") or "").lower()
    if not validated:
        if "empty" in reason_text or "missing" in reason_text:
            return "no_category_detected"
        if "not found" in reason_text:
            return "invalid_category"
        if "failed" in reason_text:
            return "no_category_detected"
        return "other"
    # Category was validated but category-scoped retrieval returned too few chunks.
    diagnostics = row.get("retrieval_diagnostics") or {}
    n_cat = int(diagnostics.get("number_of_category_results", 0))
    top_k = int(diagnostics.get("top_k", 1))
    if n_cat < top_k:
        return "insufficient_category_results"
    return "other"


def compute_fallback_summary(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute experiment-level fallback rate from a list of evaluated rows.

    Returns:
        total_queries           — number of completed queries
        queries_using_fallback  — queries where fallback_used=True
        fallback_rate           — proportion on 0–1 scale, matching all other rate metrics
    """
    total = len(per_question)
    using_fallback = sum(1 for row in per_question if row.get("fallback_used", False))
    fallback_rate = round(using_fallback / total, 6) if total > 0 else 0.0
    return {
        "total_queries": total,
        "queries_using_fallback": using_fallback,
        "fallback_rate": fallback_rate,
    }
