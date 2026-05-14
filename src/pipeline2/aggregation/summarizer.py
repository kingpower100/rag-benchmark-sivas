from __future__ import annotations

from statistics import mean
from typing import Any


def summarize_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row.get("experiment_id", "")), []).append(row)

    summaries = []
    for experiment_id, group in sorted(by_experiment.items()):
        metric_cols = _dynamic_metric_columns(group)
        successful_rows = [row for row in group if not row.get("pipeline1_error")]
        summary = {
            "experiment_id": experiment_id,
            "n_questions": len(group),
            "pipeline_success_rate": _mean([1.0 if not row.get("pipeline1_error") else 0.0 for row in group]),
            "eval_success_rate": _mean([1.0 if not row.get("evaluation_errors") else 0.0 for row in group]),
        }
        for col in metric_cols:
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        for col in ("numeric_accuracy", "answer_coverage_rate"):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        for col in (
            "duplicate_context_rate",
            "retrieval_time_ms",
            "generation_time_ms",
            "total_latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost",
        ):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        summaries.append(summary)
    return summaries


def build_leaderboard(summary_rows: list[dict[str, Any]], sort_metric: str, sort_ascending: bool = False) -> list[dict[str, Any]]:
    present = [row for row in summary_rows if row.get(sort_metric) is not None]
    missing = [row for row in summary_rows if row.get(sort_metric) is None]
    sorted_rows = sorted(
        present,
        key=lambda row: (float(row[sort_metric]), str(row.get("experiment_id", ""))),
        reverse=not sort_ascending,
    )
    if not sort_ascending:
        sorted_rows = sorted(
            present,
            key=lambda row: (-float(row[sort_metric]), str(row.get("experiment_id", ""))),
        )
    sorted_rows.extend(sorted(missing, key=lambda row: str(row.get("experiment_id", ""))))
    return [{"rank": index, "sort_metric": sort_metric, **row} for index, row in enumerate(sorted_rows, start=1)]


def _dynamic_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    prefixes = ("hit_at_", "recall_at_", "mrr_at_", "context_precision_at_")
    cols = []
    for prefix in prefixes:
        names = sorted({key for row in rows for key in row if key.startswith(prefix)})
        cols.extend(names)
    return cols


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)
