from __future__ import annotations

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import RetrievalScoreWeights, RQIWeights


def compute_retrieval_score(p2: P2Summary, weights: RetrievalScoreWeights) -> float:
    return (
        weights.recall_at_5 * p2.mean_recall_at_5
        + weights.mrr_at_5 * p2.mean_mrr_at_5
        + weights.ndcg_at_5 * p2.mean_ndcg_at_5
        + weights.context_precision_at_5 * p2.mean_context_precision_at_5
    )


def compute_rqi(p2: P2Summary, p3: P3Summary, weights: RQIWeights) -> float:
    correctness_norm = p3.mean_judge_correctness / 5.0
    faithfulness_norm = p3.mean_judge_faithfulness / 5.0
    context_relevance_norm = p3.mean_judge_context_relevance / 5.0
    recall_at_5 = p2.mean_recall_at_5
    no_unknown = 1.0 - p2.unknown_rate

    return (
        weights.correctness * correctness_norm
        + weights.faithfulness * faithfulness_norm
        + weights.context_relevance * context_relevance_norm
        + weights.recall_at_5 * recall_at_5
        + weights.no_unknown * no_unknown
    )


def retrieval_score_contributions(
    p2: P2Summary, weights: RetrievalScoreWeights
) -> dict[str, float]:
    return {
        "recall_at_5_contribution": weights.recall_at_5 * p2.mean_recall_at_5,
        "mrr_at_5_contribution": weights.mrr_at_5 * p2.mean_mrr_at_5,
        "ndcg_at_5_contribution": weights.ndcg_at_5 * p2.mean_ndcg_at_5,
        "context_precision_at_5_contribution": weights.context_precision_at_5
        * p2.mean_context_precision_at_5,
    }


def rqi_contributions(
    p2: P2Summary, p3: P3Summary, weights: RQIWeights
) -> dict[str, float]:
    return {
        "correctness_contribution": weights.correctness * (p3.mean_judge_correctness / 5.0),
        "faithfulness_contribution": weights.faithfulness * (p3.mean_judge_faithfulness / 5.0),
        "context_relevance_contribution": weights.context_relevance
        * (p3.mean_judge_context_relevance / 5.0),
        "recall_at_5_contribution": weights.recall_at_5 * p2.mean_recall_at_5,
        "no_unknown_contribution": weights.no_unknown * (1.0 - p2.unknown_rate),
    }
