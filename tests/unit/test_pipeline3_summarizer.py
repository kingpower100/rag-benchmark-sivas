from __future__ import annotations

import math

import pytest

from src.pipeline3.aggregation.summarizer import summarize_semantic_metrics


def _make_row(qid="q1", **kwargs):
    defaults = {
        "question_id": qid,
        "judge_success": True,
        "judge_correctness": 4,
        "judge_faithfulness": 5,
        "judge_completeness": 3,
        "judge_hallucination": 1,
        "judge_context_relevance": 4,
        "judge_overall_score": 4.0,
        "ragas_faithfulness": 0.8,
        "ragas_answer_relevancy": 0.9,
    }
    defaults.update(kwargs)
    return defaults


def test_empty_rows_returns_zero_count():
    summary = summarize_semantic_metrics([])
    assert summary["n_questions"] == 0


def test_single_row_means_equal_values():
    row = _make_row()
    summary = summarize_semantic_metrics([row])
    assert summary["n_questions"] == 1
    assert summary["mean_judge_correctness"] == pytest.approx(4.0)
    assert summary["mean_ragas_faithfulness"] == pytest.approx(0.8)


def test_mean_computed_over_multiple_rows():
    rows = [
        _make_row("q1", judge_correctness=4),
        _make_row("q2", judge_correctness=2),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["mean_judge_correctness"] == pytest.approx(3.0)


def test_none_values_excluded_from_mean():
    rows = [
        _make_row("q1", ragas_faithfulness=0.8),
        _make_row("q2", ragas_faithfulness=None),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["mean_ragas_faithfulness"] == pytest.approx(0.8)


def test_nan_values_excluded_from_ragas_faithfulness_mean_and_not_zeroed():
    rows = [
        _make_row("q1", ragas_faithfulness=0.8),
        _make_row("q2", ragas_faithfulness=math.nan),
        _make_row("q3", ragas_faithfulness=1.0),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["mean_ragas_faithfulness"] == pytest.approx(0.9)
    assert summary["mean_ragas_faithfulness"] != pytest.approx(0.6)


def test_ragas_faithfulness_coverage_counts_are_reported():
    rows = [
        _make_row("q1", ragas_faithfulness=0.8),
        _make_row(
            "q2",
            ragas_faithfulness=None,
            ragas_faithfulness_status="no_statements_generated",
        ),
        _make_row("q3", ragas_faithfulness=1.0),
        _make_row("q4", ragas_faithfulness=math.nan),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["mean_ragas_faithfulness"] == pytest.approx(0.9)
    assert summary["ragas_faithfulness_valid_rows"] == 2
    assert summary["ragas_faithfulness_nan_rows"] == 2
    assert summary["ragas_faithfulness_total_rows"] == 4
    assert summary["ragas_faithfulness_coverage"] == pytest.approx(0.5)
    assert summary["ragas_faithfulness_coverage_percentage"] == pytest.approx(50.0)


def test_all_none_values_give_none_mean():
    rows = [
        _make_row("q1", ragas_faithfulness=None),
        _make_row("q2", ragas_faithfulness=None),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary.get("mean_ragas_faithfulness") is None


def test_judge_success_rate_computed():
    rows = [
        _make_row("q1", judge_success=True),
        _make_row("q2", judge_success=False),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["judge_success_rate"] == pytest.approx(0.5)
    assert summary["judge_success_count"] == 1
    assert summary["judge_failure_count"] == 1


def test_all_judge_failures():
    rows = [
        _make_row("q1", judge_success=False),
        _make_row("q2", judge_success=False),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["judge_success_rate"] == pytest.approx(0.0)
    assert summary["judge_failure_count"] == 2


def test_all_judge_successes():
    rows = [_make_row("q1"), _make_row("q2")]
    summary = summarize_semantic_metrics(rows)
    assert summary["judge_success_rate"] == pytest.approx(1.0)
    assert summary["judge_failure_count"] == 0


def test_mean_judge_overall_score():
    rows = [
        _make_row("q1", judge_overall_score=3.0),
        _make_row("q2", judge_overall_score=5.0),
    ]
    summary = summarize_semantic_metrics(rows)
    assert summary["mean_judge_overall_score"] == pytest.approx(4.0)


def test_n_questions_matches_row_count():
    rows = [_make_row(f"q{i}") for i in range(7)]
    summary = summarize_semantic_metrics(rows)
    assert summary["n_questions"] == 7


def test_removed_metrics_not_in_summary():
    rows = [_make_row()]
    summary = summarize_semantic_metrics(rows)
    assert "mean_judge_relevancy" not in summary
    assert "mean_ragas_context_precision" not in summary
    assert "mean_ragas_context_recall" not in summary
