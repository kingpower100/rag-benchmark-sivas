"""Unit tests for Phase 1 fixes: category accuracy, referenzantwort, German tokenization,
and per-category aggregation."""
from __future__ import annotations

import pytest

from src.pipeline2.metrics.category_metrics import compute_category_metrics
from src.pipeline2.metrics.answer_metrics import (
    compute_rouge_l,
    resolve_ground_truth_answer,
    _rouge_tokens,
)
from src.pipeline2.aggregation.summarizer import summarize_by_category, SIVAS_CATEGORIES


# ---------------------------------------------------------------------------
# compute_category_metrics
# ---------------------------------------------------------------------------

def test_category_correct_exact_match():
    result = compute_category_metrics("Einkauf", "Einkauf")
    assert result["category_correct"] == 1.0
    assert result["category_predicted"] == "Einkauf"
    assert result["category_gold"] == "Einkauf"


def test_category_correct_case_insensitive():
    result = compute_category_metrics("einkauf", "Einkauf")
    assert result["category_correct"] == 1.0


def test_category_incorrect_mismatch():
    result = compute_category_metrics("Technik", "Einkauf")
    assert result["category_correct"] == 0.0
    assert result["category_predicted"] == "Technik"
    assert result["category_gold"] == "Einkauf"


def test_category_none_predicted_returns_null():
    result = compute_category_metrics(None, "Einkauf")
    assert result["category_correct"] is None
    assert result["category_predicted"] is None
    assert result["category_gold"] == "Einkauf"


def test_category_none_gold_returns_null():
    result = compute_category_metrics("Technik", None)
    assert result["category_correct"] is None


def test_category_both_none_returns_null():
    result = compute_category_metrics(None, None)
    assert result["category_correct"] is None


def test_category_strips_whitespace():
    result = compute_category_metrics("  Vertrieb  ", "Vertrieb")
    assert result["category_correct"] == 1.0
    assert result["category_predicted"] == "Vertrieb"


def test_category_empty_string_treated_as_missing():
    result = compute_category_metrics("", "Technik")
    assert result["category_correct"] is None


def test_category_all_five_sivas_categories_work():
    for cat in SIVAS_CATEGORIES:
        r = compute_category_metrics(cat, cat)
        assert r["category_correct"] == 1.0


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
# German ROUGE-L tokenization
# ---------------------------------------------------------------------------

def test_rouge_tokens_includes_german_umlauts():
    tokens = _rouge_tokens("Für Änderungen müssen Überarbeitungen erfolgen.")
    assert "für" in tokens
    assert "änderungen" in tokens
    assert "müssen" in tokens
    assert "überarbeitungen" in tokens


def test_rouge_tokens_includes_eszett():
    tokens = _rouge_tokens("Das ist straßenweise möglich.")
    assert "straßenweise" in tokens


def test_rouge_l_identical_german_sentence_is_one():
    sentence = "Arbeitsplan definiert die einzelnen Arbeitsgänge und Qualitätsprüfungen."
    score = compute_rouge_l(sentence, sentence)
    assert score == pytest.approx(1.0)


def test_rouge_l_german_partial_overlap_is_nonzero():
    generated = "Der Arbeitsplan enthält Vorgänge und Qualitätsprüfungen."
    reference = "Der Arbeitsplan enthält Arbeitsgänge und Qualitätsprüfungen."
    score = compute_rouge_l(generated, reference)
    assert score > 0.0


def test_rouge_l_no_overlap_is_zero():
    score = compute_rouge_l("völlig anderer Text", "komplett verschiedene Wörter")
    assert score == pytest.approx(0.0)


def test_rouge_l_german_better_than_pure_ascii_would_score():
    generated = "Überarbeitungen der Qualitätsprüfungen sind notwendig."
    reference = "Überarbeitungen der Qualitätsprüfungen sind erforderlich."
    score = compute_rouge_l(generated, reference)
    assert score > 0.7


# ---------------------------------------------------------------------------
# summarize_by_category
# ---------------------------------------------------------------------------

def _make_rows(category_gold: str, n: int, category_correct: float | None = 1.0) -> list[dict]:
    return [
        {
            "category_gold": category_gold,
            "category_correct": category_correct,
            "rouge_l": 0.8,
            "exact_match": 1.0,
            "literal_exact_match": 1.0,
            "canonical_exact_match": 1.0,
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
    rows = _make_rows("Einkauf", 4, category_correct=1.0) + _make_rows("Technik", 4, category_correct=0.0)
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


def test_summarize_by_category_null_category_correct_excluded_from_mean():
    rows = [
        {"category_gold": "Vertrieb", "category_correct": 1.0, "generation_failed": False},
        {"category_gold": "Vertrieb", "category_correct": None, "generation_failed": False},
    ]
    result = summarize_by_category(rows)
    vertrieb_row = next(r for r in result if r["category"] == "Vertrieb")
    assert vertrieb_row["mean_category_accuracy"] == pytest.approx(1.0)
