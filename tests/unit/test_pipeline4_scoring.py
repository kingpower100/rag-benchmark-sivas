from __future__ import annotations

import pytest

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import RetrievalScoreWeights, RQIWeights
from src.pipeline4.scoring import (
    compute_answer_quality,
    compute_faithfulness,
    compute_generation,
    compute_retrieval_score,
    compute_rqi,
    retrieval_score_contributions,
    rqi_contributions,
)


def _p2(
    recall=0.5,
    mrr=0.8,
    ndcg=0.6,
    bert=0.66,
    embedding=0.88,
) -> P2Summary:
    return P2Summary(
        experiment_id="test_exp",
        n_questions=96,
        run_valid=True,
        generation_failure_rate=0.0,
        mean_recall_at_5=recall,
        mean_mrr_at_5=mrr,
        mean_ndcg_at_5=ndcg,
        mean_context_precision_at_5=0.4,
        unknown_rate=0.1,
        mean_embedding_similarity=embedding,
        mean_official_bertscore_f1=bert,
        qa_hash="abc123",
        gold_contexts_hash="abc123",
        p2_run_dir="/fake/p2",
    )


def _p3(
    correctness=4.0,
    faithfulness=3.5,
    context_relevance=4.5,
    completeness=2.0,
    ragas_faithfulness=0.65,
) -> P3Summary:
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
        mean_judge_completeness=completeness,
        mean_judge_hallucination=1.0,
        mean_judge_context_relevance=context_relevance,
        mean_judge_overall_score=3.5,
        mean_ragas_faithfulness=ragas_faithfulness,
        mean_ragas_answer_relevancy=0.70,
        ragas_faithfulness_nan_rate=0.01,
        ragas_answer_relevancy_nan_rate=0.0,
        p3_run_dir="/fake/p3",
    )


def test_retrieval_score_formula():
    p2 = _p2(recall=0.5, mrr=0.8, ndcg=0.6)
    expected = 0.60 * 0.6 + 0.25 * 0.5 + 0.15 * 0.8
    assert compute_retrieval_score(p2, RetrievalScoreWeights()) == pytest.approx(expected, abs=1e-9)


def test_retrieval_score_perfect_and_zero_bounds():
    assert compute_retrieval_score(_p2(recall=1.0, mrr=1.0, ndcg=1.0), RetrievalScoreWeights()) == pytest.approx(1.0)
    assert compute_retrieval_score(_p2(recall=0.0, mrr=0.0, ndcg=0.0), RetrievalScoreWeights()) == pytest.approx(0.0)


def test_retrieval_score_custom_weights():
    weights = RetrievalScoreWeights(ndcg_at_5=1.0, recall_at_5=0.0, mrr_at_5=0.0)
    assert compute_retrieval_score(_p2(recall=0.0, mrr=0.0, ndcg=0.9), weights) == pytest.approx(0.9)


def test_retrieval_score_requires_at5_metrics():
    p2 = _p2()
    p2.mean_ndcg_at_5 = None
    with pytest.raises(ValueError, match="mean_ndcg_at_5"):
        compute_retrieval_score(p2, RetrievalScoreWeights())


def test_answer_quality_formula():
    assert compute_answer_quality(_p2(bert=0.7, embedding=0.9)) == pytest.approx(0.8, abs=1e-9)


def test_faithfulness_formula():
    p3 = _p3(faithfulness=4.0, ragas_faithfulness=0.6)
    assert compute_faithfulness(p3) == pytest.approx(0.5 * 0.6 + 0.5 * 0.8, abs=1e-9)


def test_generation_formula():
    p3 = _p3(correctness=5.0, context_relevance=4.0, completeness=3.0)
    expected = 0.40 * 1.0 + 0.30 * 0.8 + 0.30 * 0.6
    assert compute_generation(p3) == pytest.approx(expected, abs=1e-9)


def test_rqi_formula():
    p2 = _p2(recall=0.5, mrr=0.8, ndcg=0.6, bert=0.7, embedding=0.9)
    p3 = _p3(correctness=5.0, faithfulness=4.0, context_relevance=4.0, completeness=3.0, ragas_faithfulness=0.6)
    rsi = 0.60 * 0.6 + 0.25 * 0.5 + 0.15 * 0.8
    answer_quality = 0.50 * 0.7 + 0.50 * 0.9
    faithfulness = 0.50 * 0.6 + 0.50 * (4.0 / 5.0)
    generation = 0.40 * (5.0 / 5.0) + 0.30 * (4.0 / 5.0) + 0.30 * (3.0 / 5.0)
    expected = 0.30 * rsi + 0.20 * answer_quality + 0.25 * faithfulness + 0.25 * generation
    assert compute_rqi(p2, p3, RQIWeights()) == pytest.approx(expected, abs=1e-9)


def test_rqi_perfect_and_zero_bounds():
    assert compute_rqi(
        _p2(recall=1.0, mrr=1.0, ndcg=1.0, bert=1.0, embedding=1.0),
        _p3(correctness=5.0, faithfulness=5.0, context_relevance=5.0, completeness=5.0, ragas_faithfulness=1.0),
        RQIWeights(),
    ) == pytest.approx(1.0)
    assert compute_rqi(
        _p2(recall=0.0, mrr=0.0, ndcg=0.0, bert=0.0, embedding=0.0),
        _p3(correctness=0.0, faithfulness=0.0, context_relevance=0.0, completeness=0.0, ragas_faithfulness=0.0),
        RQIWeights(),
    ) == pytest.approx(0.0)


def test_weights_must_sum_to_one():
    with pytest.raises(Exception):
        RetrievalScoreWeights(ndcg_at_5=0.5, recall_at_5=0.5, mrr_at_5=0.5)
    with pytest.raises(Exception):
        RQIWeights(retrieval_score=0.5, answer_quality=0.5, faithfulness=0.5, generation=0.5)


def test_contributions_sum_to_scores():
    p2 = _p2(recall=0.41, mrr=0.73, ndcg=0.52, bert=0.7, embedding=0.81)
    p3 = _p3(correctness=2.39, faithfulness=2.83, context_relevance=3.76, completeness=4.1, ragas_faithfulness=0.62)

    ret_contribs = retrieval_score_contributions(p2, RetrievalScoreWeights())
    assert sum(ret_contribs.values()) == pytest.approx(compute_retrieval_score(p2, RetrievalScoreWeights()), abs=1e-9)

    rqi_contribs = rqi_contributions(p2, p3, RQIWeights())
    weighted_total = sum(value for key, value in rqi_contribs.items() if key.endswith("_contribution"))
    assert weighted_total == pytest.approx(compute_rqi(p2, p3, RQIWeights()), abs=1e-9)
