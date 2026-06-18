from __future__ import annotations

from typing import Any


def compute_efficiency_metrics(row: dict[str, Any]) -> dict[str, float | int | None]:
    retrieval_time = _number(row.get("retrieval_time_ms"))
    rerank_time = _number(row.get("rerank_time_ms"))
    generation_time = _number(row.get("generation_time_ms"))
    return {
        "retrieval_time_ms": retrieval_time,
        "rerank_time_ms": rerank_time,
        "generation_time_ms": generation_time,
        "total_latency_ms": _total_latency(row, retrieval_time, rerank_time, generation_time),
        "input_tokens": _number(row.get("input_tokens")),
        "output_tokens": _number(row.get("output_tokens")),
        "total_tokens": _number(row.get("total_tokens")),
        "estimated_cost": _number(row.get("estimated_cost")),
    }


def _number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _total_latency(
    row: dict[str, Any],
    retrieval_time: float | int | None,
    rerank_time: float | int | None,
    generation_time: float | int | None,
) -> float | int | None:
    components = [retrieval_time, rerank_time, generation_time]
    if any(value is not None for value in components):
        total = sum(float(value or 0.0) for value in components)
        return int(total) if total.is_integer() else total
    return _number(row.get("total_latency_ms", row.get("latency_ms")))
