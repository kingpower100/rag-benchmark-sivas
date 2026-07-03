from __future__ import annotations

import pytest

from src.pipeline3.judge.response_parser import JudgeResponse
from src.pipeline3.metrics.judge_metrics import compute_weighted_overall_score, enabled_judge_metric_names
from src.pipeline3.schemas.pipeline3_config_schema import (
    P3JudgeMetricsConfig,
    P3ScoringConfig,
    P3WeightsConfig,
)


def _make_response(**kwargs) -> JudgeResponse:
    defaults = dict(
        correctness=5,
        faithfulness=5,
        completeness=5,
        hallucination=0,
        context_relevance=5,
        overall_score=5.0,
        reasoning="test",
    )
    defaults.update(kwargs)
    return JudgeResponse(**defaults)


_DEFAULT_WEIGHTS = P3WeightsConfig()
_DEFAULT_SCORING = P3ScoringConfig()


def test_perfect_scores_give_max_overall():
    response = _make_response()
    score = compute_weighted_overall_score(response, _DEFAULT_WEIGHTS, _DEFAULT_SCORING)
    assert score == pytest.approx(5.0, abs=0.01)


def test_zero_scores_give_only_hallucination_contribution():
    response = _make_response(
        correctness=0,
        faithfulness=0,
        completeness=0,
        hallucination=0,
        context_relevance=0,
    )
    score = compute_weighted_overall_score(response, _DEFAULT_WEIGHTS, _DEFAULT_SCORING)
    # Only hallucination contributes (inverted, 0=best): weight=0.15, scale=5
    # contribution = (5-0)/5 * 0.15 * 5 = 1.0 * 0.15 * 5 = 0.75
    assert score == pytest.approx(0.15 * 5.0, abs=0.01)


def test_max_hallucination_reduces_score():
    perfect_no_hallucination = _make_response(hallucination=0)
    perfect_with_hallucination = _make_response(hallucination=5)
    score_good = compute_weighted_overall_score(
        perfect_no_hallucination, _DEFAULT_WEIGHTS, _DEFAULT_SCORING
    )
    score_bad = compute_weighted_overall_score(
        perfect_with_hallucination, _DEFAULT_WEIGHTS, _DEFAULT_SCORING
    )
    assert score_good > score_bad


def test_weighted_sum_correctness_only():
    weights = P3WeightsConfig(
        correctness=1.0,
        faithfulness=0.0,
        completeness=0.0,
        hallucination=0.0,
        context_relevance=0.0,
    )
    response = _make_response(
        correctness=3,
        faithfulness=0,
        completeness=0,
        hallucination=5,
        context_relevance=0,
    )
    score = compute_weighted_overall_score(response, weights, _DEFAULT_SCORING)
    # correctness=3/5=0.6, weight=1.0, scale=5 → 0.6 * 5.0 = 3.0
    assert score == pytest.approx(3.0, abs=0.01)


def test_enabled_judge_metric_names_all_enabled():
    cfg = P3JudgeMetricsConfig()
    names = enabled_judge_metric_names(cfg)
    assert set(names) == {
        "correctness",
        "faithfulness",
        "completeness",
        "hallucination",
        "context_relevance",
    }


def test_enabled_judge_metric_names_does_not_include_relevancy():
    cfg = P3JudgeMetricsConfig()
    names = enabled_judge_metric_names(cfg)
    assert "relevancy" not in names


def test_enabled_judge_metric_names_some_disabled():
    cfg = P3JudgeMetricsConfig(hallucination=False, context_relevance=False)
    names = enabled_judge_metric_names(cfg)
    assert "hallucination" not in names
    assert "context_relevance" not in names
    assert "correctness" in names


def test_score_result_is_within_scale_range():
    response = _make_response(correctness=3, faithfulness=3, completeness=3, hallucination=2, context_relevance=3)
    score = compute_weighted_overall_score(response, _DEFAULT_WEIGHTS, _DEFAULT_SCORING)
    assert _DEFAULT_SCORING.scale_min <= score <= _DEFAULT_SCORING.scale_max
