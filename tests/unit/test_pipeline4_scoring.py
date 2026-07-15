from __future__ import annotations

import pytest

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import RetrievalScoreWeights, RQIWeights
from src.pipeline4.scoring import compute_retrieval_score, compute_rqi, retrieval_score_contributions, rqi_contributions


def _p2(
    recall=0.5, mrr=0.8, ndcg=0.6, cp=0.4, unknown_rate=0.1
) -> P2Summary:
    return P2Summary(
        experiment_id="test_exp",
        n_questions=96,
        run_valid=True,
        generation_failure_rate=0.0,
        mean_recall_at_5=recall,
        mean_mrr_at_5=mrr,
        mean_ndcg_at_5=ndcg,
        mean_context_precision_at_5=cp,
        unknown_rate=unknown_rate,
        mean_embedding_similarity=0.88,
        mean_official_bertscore_f1=0.66,
        qa_hash="abc123",
        gold_contexts_hash="abc123",
        p2_run_dir="/fake/p2",
    )


def _p3(correctness=4.0, faithfulness=3.5, context_relevance=4.5) -> P3Summary:
    return P3Summary(
        run_id="p3_test_exp",
        experiment_id="test_exp",
        n_questions=96,
        judge_model="qwen2.5:14b",
        prompt_version="v2",
        qa_sha256="abc",
        judge_success_rate=1.0,
        judge_failure_count=0,
        mean_judge_correctness=correctness,
        mean_judge_faithfulness=faithfulness,
        mean_judge_completeness=2.0,
        mean_judge_hallucination=1.0,
        mean_judge_context_relevance=context_relevance,
        mean_judge_overall_score=3.5,
        mean_ragas_faithfulness=0.65,
        mean_ragas_answer_relevancy=0.70,
        ragas_faithfulness_nan_rate=0.01,
        ragas_answer_relevancy_nan_rate=0.0,
        p3_run_dir="/fake/p3",
    )


def test_retrieval_score_formula():
    p2 = _p2(recall=0.5, mrr=0.8, ndcg=0.6, cp=0.4)
    weights = RetrievalScoreWeights()
    score = compute_retrieval_score(p2, weights)
    expected = 0.35 * 0.5 + 0.25 * 0.8 + 0.20 * 0.6 + 0.20 * 0.4
    assert score == pytest.approx(expected, abs=1e-9)


def test_retrieval_score_perfect():
    p2 = _p2(recall=1.0, mrr=1.0, ndcg=1.0, cp=1.0)
    weights = RetrievalScoreWeights()
    assert compute_retrieval_score(p2, weights) == pytest.approx(1.0, abs=1e-9)


def test_retrieval_score_zero():
    p2 = _p2(recall=0.0, mrr=0.0, ndcg=0.0, cp=0.0)
    weights = RetrievalScoreWeights()
    assert compute_retrieval_score(p2, weights) == pytest.approx(0.0, abs=1e-9)


def test_retrieval_score_custom_weights():
    p2 = _p2(recall=1.0, mrr=0.0, ndcg=0.0, cp=0.0)
    weights = RetrievalScoreWeights(recall_at_5=1.0, mrr_at_5=0.0, ndcg_at_5=0.0, context_precision_at_5=0.0)
    assert compute_retrieval_score(p2, weights) == pytest.approx(1.0, abs=1e-9)


def test_rqi_formula():
    p2 = _p2(recall=0.5, unknown_rate=0.2)
    p3 = _p3(correctness=4.0, faithfulness=3.5, context_relevance=4.5)
    weights = RQIWeights()
    rqi = compute_rqi(p2, p3, weights)
    expected = (
        0.25 * (4.0 / 5.0)
        + 0.25 * (3.5 / 5.0)
        + 0.20 * (4.5 / 5.0)
        + 0.15 * 0.5
        + 0.15 * (1.0 - 0.2)
    )
    assert rqi == pytest.approx(expected, abs=1e-9)


def test_rqi_perfect():
    p2 = _p2(recall=1.0, unknown_rate=0.0)
    p3 = _p3(correctness=5.0, faithfulness=5.0, context_relevance=5.0)
    weights = RQIWeights()
    assert compute_rqi(p2, p3, weights) == pytest.approx(1.0, abs=1e-9)


def test_rqi_zero():
    p2 = _p2(recall=0.0, unknown_rate=1.0)
    p3 = _p3(correctness=0.0, faithfulness=0.0, context_relevance=0.0)
    weights = RQIWeights()
    assert compute_rqi(p2, p3, weights) == pytest.approx(0.0, abs=1e-9)


def test_retrieval_score_weights_must_sum_to_one():
    with pytest.raises(Exception):
        RetrievalScoreWeights(recall_at_5=0.5, mrr_at_5=0.5, ndcg_at_5=0.5, context_precision_at_5=0.5)


def test_rqi_weights_must_sum_to_one():
    with pytest.raises(Exception):
        RQIWeights(correctness=0.5, faithfulness=0.5, context_relevance=0.5, recall_at_5=0.5, no_unknown=0.5)


def test_retrieval_score_contributions_sum_to_score():
    p2 = _p2(recall=0.4, mrr=0.9, ndcg=0.53, cp=0.32)
    weights = RetrievalScoreWeights()
    score = compute_retrieval_score(p2, weights)
    contribs = retrieval_score_contributions(p2, weights)
    total = sum(contribs.values())
    assert total == pytest.approx(score, abs=1e-9)


def test_rqi_contributions_sum_to_rqi():
    p2 = _p2(recall=0.41, unknown_rate=0.23)
    p3 = _p3(correctness=2.39, faithfulness=2.83, context_relevance=3.76)
    weights = RQIWeights()
    rqi = compute_rqi(p2, p3, weights)
    contribs = rqi_contributions(p2, p3, weights)
    total = sum(contribs.values())
    assert total == pytest.approx(rqi, abs=1e-9)


def test_retrieval_score_uses_real_data():
    p2 = _p2(
        recall=0.40982142857142856,
        mrr=0.9045138888888888,
        ndcg=0.5281562132605336,
        cp=0.31875000000000003,
    )
    weights = RetrievalScoreWeights()
    score = compute_retrieval_score(p2, weights)
    expected = (
        0.35 * 0.40982142857142856
        + 0.25 * 0.9045138888888888
        + 0.20 * 0.5281562132605336
        + 0.20 * 0.31875000000000003
    )
    assert score == pytest.approx(expected, abs=1e-9)
    assert 0.0 < score < 1.0
