from __future__ import annotations

from typing import Any


def compute_efficiency_metrics(row: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "retrieval_time_ms": _number(row.get("retrieval_time_ms")),
        "generation_time_ms": _number(row.get("generation_time_ms")),
        "total_latency_ms": _number(row.get("total_latency_ms")),
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
