"""Unit tests for Phase 1 fixes: category accuracy, referenzantwort, German tokenization,
and per-category aggregation."""
from __future__ import annotations

import pytest

from src.pipeline2.metrics.category_metrics import compute_category_metrics, compute_category_routing_report
from src.pipeline2.metrics.answer_metrics import (
    resolve_ground_truth_answer,
)
from src.pipeline2.aggregation.summarizer import summarize_by_category, SIVAS_CATEGORIES


# ---------------------------------------------------------------------------
# compute_category_metrics
# ---------------------------------------------------------------------------

def test_category_accuracy_exact_match():
    result = compute_category_metrics("Einkauf", "Einkauf")
    assert result["category_accuracy"] == 1.0
    assert result["category_predicted"] == "Einkauf"
    assert result["category_gold"] == "Einkauf"


def test_category_accuracy_case_insensitive():
    result = compute_category_metrics("einkauf", "Einkauf")
    assert result["category_accuracy"] == 1.0


def test_category_incorrect_mismatch():
    result = compute_category_metrics("Technik", "Einkauf")
    assert result["category_accuracy"] == 0.0
    assert result["category_predicted"] == "Technik"
    assert result["category_gold"] == "Einkauf"


def test_category_none_predicted_returns_null():
    result = compute_category_metrics(None, "Einkauf")
    assert result["category_accuracy"] is None
    assert result["category_predicted"] is None
    assert result["category_gold"] == "Einkauf"


def test_category_none_gold_returns_null():
    result = compute_category_metrics("Technik", None)
    assert result["category_accuracy"] is None


def test_category_both_none_returns_null():
    result = compute_category_metrics(None, None)
    assert result["category_accuracy"] is None


def test_category_strips_whitespace():
    result = compute_category_metrics("  Vertrieb  ", "Vertrieb")
    assert result["category_accuracy"] == 1.0
    assert result["category_predicted"] == "Vertrieb"


def test_category_empty_string_treated_as_missing():
    result = compute_category_metrics("", "Technik")
    assert result["category_accuracy"] is None


def test_category_routing_report_inactive_for_global_rows_with_predictions():
    rows = [
        {
            "retriever_type": "dense",
            "category_predicted": "Einkauf",
            "category_gold": "Einkauf",
            "category_accuracy": 1.0,
        }
    ]

    report = compute_category_routing_report(rows, SIVAS_CATEGORIES)

    assert report["category_routing_active"] is False
    assert report["category_accuracy"] is None
    assert report["category_coverage"] is None


def test_category_routing_report_uses_runtime_routing_rows():
    rows = [
        {
            "retriever_type": "category_aware_dense",
            "category_predicted": "Einkauf",
            "category_gold": "Einkauf",
            "category_accuracy": 1.0,
            "category_validated": True,
            "fallback_used": False,
            "category_index_used": True,
        },
        {
            "retriever_type": "category_aware_dense",
            "category_predicted": "Unknown",
            "category_gold": "Technik",
            "category_accuracy": 0.0,
            "category_validated": False,
            "fallback_used": True,
            "category_index_used": False,
        },
    ]

    report = compute_category_routing_report(rows, SIVAS_CATEGORIES)

    assert report["category_routing_active"] is True
    assert report["category_coverage"] == 1.0
    assert report["validated_category_coverage"] == 0.5
    assert report["category_accuracy"] == 0.5
    assert report["effective_category_accuracy"] == 0.5
    assert report["fallback_rate"] == 0.5
    assert report["unknown_rate"] == 0.5
    assert report["category_index_usage_rate"] == 0.5


def test_category_all_five_sivas_categories_work():
    for cat in SIVAS_CATEGORIES:
        r = compute_category_metrics(cat, cat)
        assert r["category_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# referenzantwort field resolution
# ---------------------------------------------------------------------------

def test_resolve_referenzantwort_picks_up_german_field():
    row = {"question_id": "Q001"}
    qa_by_id = {"Q001": {"referenzantwort": "Die Antwort ist 42."}}
    result = resolve_ground_truth_answer(row, qa_by_id)
    assert result == "Die Antwort ist 42."


def test_resolve_prefers_english_ground_truth_answer_over_referenzantwort():
    row = {"question_id": "Q001"}
    qa_by_id = {
        "Q001": {
            "ground_truth_answer": "English answer",
            "referenzantwort": "German answer",
        }
    }
    result = resolve_ground_truth_answer(row, qa_by_id)
    assert result == "English answer"


def test_resolve_returns_empty_string_when_no_answer_field():
    row = {"question_id": "Q001"}
    qa_by_id = {"Q001": {"frage": "Was ist das?"}}
    result = resolve_ground_truth_answer(row, qa_by_id)
    assert result == ""


def test_resolve_returns_empty_string_for_missing_question_id():
    row = {"question_id": "Q999"}
    qa_by_id = {"Q001": {"referenzantwort": "Antwort"}}
    result = resolve_ground_truth_answer(row, qa_by_id)
    assert result == ""


def test_resolve_handles_none_value_in_referenzantwort():
    row = {"question_id": "Q001"}
    qa_by_id = {"Q001": {"referenzantwort": None}}
    result = resolve_ground_truth_answer(row, qa_by_id)
    assert result == ""


# ---------------------------------------------------------------------------
# summarize_by_category
# ---------------------------------------------------------------------------

def _make_rows(category_gold: str, n: int, category_accuracy: float | None = 1.0) -> list[dict]:
    return [
        {
            "category_gold": category_gold,
            "category_accuracy": category_accuracy,
            "embedding_similarity": 0.9,
            "total_latency_ms": 200.0,
            "total_tokens": 500,
            "generation_failed": False,
        }
        for _ in range(n)
    ]


def test_summarize_by_category_includes_all_sivas_categories():
    rows = _make_rows("Technik", 3)
    result = summarize_by_category(rows)
    categories = [r["category"] for r in result]
    for cat in SIVAS_CATEGORIES:
        assert cat in categories, f"Missing category: {cat}"


def test_summarize_by_category_first_row_is_all():
    rows = _make_rows("Einkauf", 5)
    result = summarize_by_category(rows)
    assert result[0]["category"] == "all"
    assert result[0]["n_questions"] == 5


def test_summarize_by_category_aggregates_category_accuracy():
    rows = _make_rows("Einkauf", 4, category_accuracy=1.0) + _make_rows("Technik", 4, category_accuracy=0.0)
    result = summarize_by_category(rows)
    all_row = next(r for r in result if r["category"] == "all")
    einkauf_row = next(r for r in result if r["category"] == "Einkauf")
    technik_row = next(r for r in result if r["category"] == "Technik")
    assert all_row["mean_category_accuracy"] == pytest.approx(0.5)
    assert einkauf_row["mean_category_accuracy"] == pytest.approx(1.0)
    assert technik_row["mean_category_accuracy"] == pytest.approx(0.0)


def test_summarize_by_category_empty_category_has_zero_questions():
    rows = _make_rows("Technik", 2)
    result = summarize_by_category(rows)
    service_row = next(r for r in result if r["category"] == "Service")
    assert service_row["n_questions"] == 0


def test_summarize_by_category_pipeline_success_rate():
    rows = [
        {"category_gold": "Einkauf", "generation_failed": False},
        {"category_gold": "Einkauf", "generation_failed": True},
    ]
    result = summarize_by_category(rows)
    einkauf_row = next(r for r in result if r["category"] == "Einkauf")
    assert einkauf_row["pipeline_success_rate"] == pytest.approx(0.5)


def test_summarize_by_category_null_category_accuracy_excluded_from_mean():
    rows = [
        {"category_gold": "Vertrieb", "category_accuracy": 1.0, "generation_failed": False},
        {"category_gold": "Vertrieb", "category_accuracy": None, "generation_failed": False},
    ]
    result = summarize_by_category(rows)
    vertrieb_row = next(r for r in result if r["category"] == "Vertrieb")
    assert vertrieb_row["mean_category_accuracy"] == pytest.approx(1.0)
