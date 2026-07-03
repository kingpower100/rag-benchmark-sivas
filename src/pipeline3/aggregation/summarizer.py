from __future__ import annotations

from statistics import mean as _stat_mean
from typing import Any

_JUDGE_COLS = frozenset({
    "judge_correctness",
    "judge_faithfulness",
    "judge_relevancy",
    "judge_completeness",
    "judge_hallucination",
    "judge_context_relevance",
    "judge_overall_score",
})

_RAGAS_PREFIXES = (
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
)


def summarize_semantic_metrics(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean values across all per-question semantic metric rows."""
    if not per_question:
        return {"n_questions": 0}

    numeric_columns = _collect_numeric_columns(per_question)
    summary: dict[str, Any] = {"n_questions": len(per_question)}

    for col in sorted(numeric_columns):
        values = [
            float(row[col])
            for row in per_question
            if row.get(col) is not None and _is_numeric(row[col])
        ]
        summary[f"mean_{col}"] = _stat_mean(values) if values else None

    successes = sum(1 for row in per_question if row.get("judge_success"))
    summary["judge_success_count"] = successes
    summary["judge_failure_count"] = len(per_question) - successes
    summary["judge_success_rate"] = (
        successes / len(per_question) if per_question else 0.0
    )

    return summary


def _collect_numeric_columns(rows: list[dict[str, Any]]) -> set[str]:
    cols: set[str] = set()
    for row in rows:
        for key, val in row.items():
            if key in _JUDGE_COLS or any(key.startswith(p) for p in _RAGAS_PREFIXES):
                if _is_numeric(val):
                    cols.add(key)
    return cols


def _is_numeric(val: Any) -> bool:
    if val is None:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False
