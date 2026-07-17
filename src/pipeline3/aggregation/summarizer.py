from __future__ import annotations

import math
from statistics import mean as _stat_mean
from typing import Any

_JUDGE_COLS = frozenset({
    "judge_correctness",
    "judge_faithfulness",
    "judge_completeness",
    "judge_hallucination",
    "judge_context_relevance",
    "judge_overall_score",
})

_RAGAS_PREFIXES = (
    "ragas_faithfulness",
    "ragas_answer_relevancy",
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
            if row.get(col) is not None and _is_valid_number(row[col])
        ]
        summary[f"mean_{col}"] = _stat_mean(values) if values else None
        if _is_ragas_metric(col):
            total_rows = len(per_question)
            valid_rows = len(values)
            nan_rows = total_rows - valid_rows
            coverage = valid_rows / total_rows if total_rows else 0.0
            summary[f"{col}_valid_rows"] = valid_rows
            summary[f"{col}_nan_rows"] = nan_rows
            summary[f"{col}_total_rows"] = total_rows
            summary[f"{col}_coverage"] = coverage
            summary[f"{col}_coverage_percentage"] = coverage * 100

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
            if key in _JUDGE_COLS or _is_ragas_metric(key):
                if _is_numeric(val) or _is_ragas_metric(key):
                    cols.add(key)
    return cols


def _is_ragas_metric(key: str) -> bool:
    return key in _RAGAS_PREFIXES


def _is_valid_number(val: Any) -> bool:
    if not _is_numeric(val):
        return False
    return not math.isnan(float(val))


def _is_numeric(val: Any) -> bool:
    if val is None:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False
