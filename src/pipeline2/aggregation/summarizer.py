from __future__ import annotations

from statistics import mean
from typing import Any

from src.pipeline2.metrics.fallback_metrics import compute_fallback_summary

SIVAS_CATEGORIES = ["Technik", "Vertrieb", "Materialwirtschaft", "Einkauf", "Service"]


def summarize_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row.get("experiment_id", "")), []).append(row)

    summaries = []
    for experiment_id, group in sorted(by_experiment.items()):
        metric_cols = _dynamic_metric_columns(group)
        summary = {
            "experiment_id": experiment_id,
            "n_questions": len(group),
            "pipeline_success_rate": _mean([1.0 if not row.get("generation_failed", bool(row.get("pipeline1_error"))) else 0.0 for row in group]),
            "eval_success_rate": _mean([1.0 if not row.get("evaluation_errors") else 0.0 for row in group]),
        }
        for col in metric_cols:
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])
        for col in (
            "category_accuracy",
            "non_empty_answer_rate",
            # answer_coverage_rate is a deprecated alias for non_empty_answer_rate; not averaged separately
            "abstention_rate",
            "embedding_similarity",
            "hashed_embedding_cosine_similarity",
            "official_bertscore_precision",
            "official_bertscore_recall",
            "official_bertscore_f1",
        ):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])
        # question_answer_lexical_f1 is a lexical diagnostic (token-overlap F1), not a quality metric.
        # Retained for diagnostic inspection but not promoted to a headline number.
        summary["diagnostic_mean_question_answer_lexical_f1"] = _mean(
            [row.get("question_answer_lexical_f1") for row in group if row.get("question_answer_lexical_f1") is not None]
        )
        summary["mean_category_accuracy"] = _mean(
            [row.get("category_accuracy") for row in group if row.get("category_accuracy") is not None]
        )
        # UNKNOWN-specific tracking (separate from general abstention)
        unknown_count = sum(1 for row in group if row.get("is_unknown") == 1.0)
        summary["unknown_count"] = unknown_count
        summary["unknown_rate"] = unknown_count / len(group) if group else 0.0
        summary.update(compute_fallback_summary(group))
        for col in (
            "duplicate_context_rate",
            "raw_duplicate_rate",
            "raw_retrieved_count",
            "unique_retrieved_document_count",
            "duplicate_document_count",
            "duplicate_document_rate",
            "retrieval_time_ms",
            "rerank_time_ms",
            "generation_time_ms",
            "total_latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost",
        ):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])
        summaries.append(summary)
    return summaries


def _dynamic_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    prefixes = (
        "hit_at_",
        "recall_at_",
        "mrr_at_",
        "context_precision_at_",
        "ndcg_at_",
        "duplicate_count_at_",
        "duplicate_rate_at_",
        "deduped_hit_at_",
        "deduped_recall_at_",
        "deduped_mrr_at_",
        "deduped_ndcg_at_",
    )
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


def summarize_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-question metrics grouped by SIVAS document category.

    Groups are determined by the ``category_gold`` field written by
    ``compute_category_metrics()``.  An ``"all"`` group covering every row is
    always first.  Known SIVAS categories that are absent from the evaluated
    rows are still included (with ``n_questions=0``) so downstream consumers
    see a stable schema regardless of which categories a particular P1 run
    happened to cover.
    """
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        cat = str(row.get("category_gold") or "unknown")
        groups.setdefault(cat, []).append(row)

    # Guarantee all canonical SIVAS categories appear in output
    for cat in SIVAS_CATEGORIES:
        groups.setdefault(cat, [])

    output = []
    category_order = ["all", *sorted(key for key in groups if key != "all")]
    for cat in category_order:
        group = groups[cat]
        summary: dict[str, Any] = {"category": cat, "n_questions": len(group)}
        if not group:
            output.append(summary)
            continue

        metric_cols = _dynamic_metric_columns(group)
        for col in metric_cols:
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])

        for col in (
            "category_accuracy",
            "non_empty_answer_rate",
            "abstention_rate",
            "embedding_similarity",
            "hashed_embedding_cosine_similarity",
            "official_bertscore_precision",
            "official_bertscore_recall",
            "official_bertscore_f1",
            "total_latency_ms",
            "total_tokens",
        ):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in group if row.get(col) is not None])

        summary["pipeline_success_rate"] = _mean(
            [1.0 if not row.get("generation_failed") else 0.0 for row in group]
        )
        unknown_count = sum(1 for row in group if row.get("is_unknown") == 1.0)
        summary["unknown_count"] = unknown_count
        summary["unknown_rate"] = unknown_count / len(group) if group else 0.0
        output.append(summary)
    return output
