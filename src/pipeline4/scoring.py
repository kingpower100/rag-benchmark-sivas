from __future__ import annotations

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import RetrievalScoreWeights, RQIWeights


def compute_retrieval_score(p2: P2Summary, weights: RetrievalScoreWeights) -> float:
    weighted = [
        (weights.recall_at_5, p2.primary_recall),
        (weights.mrr_at_5, p2.primary_mrr),
        (weights.ndcg_at_5, p2.primary_ndcg),
        (weights.context_precision_at_5, p2.primary_context_precision),
    ]
    available = [(weight, value) for weight, value in weighted if value is not None]
    weight_sum = sum(weight for weight, _value in available)
    if weight_sum <= 0.0:
        raise ValueError(
            f"No primary retrieval metrics available for experiment {p2.experiment_id!r}."
        )
    return sum(weight * value for weight, value in available) / weight_sum


def compute_rqi(p2: P2Summary, p3: P3Summary, weights: RQIWeights) -> float:
    correctness_norm = p3.mean_judge_correctness / 5.0
    faithfulness_norm = p3.mean_judge_faithfulness / 5.0
    context_relevance_norm = p3.mean_judge_context_relevance / 5.0
    recall_at_primary = p2.primary_recall
    no_unknown = 1.0 - p2.unknown_rate

    return (
        weights.correctness * correctness_norm
        + weights.faithfulness * faithfulness_norm
        + weights.context_relevance * context_relevance_norm
        + weights.recall_at_5 * recall_at_primary
        + weights.no_unknown * no_unknown
    )


def retrieval_score_contributions(
    p2: P2Summary, weights: RetrievalScoreWeights
) -> dict[str, float | None]:
    weighted = {
        f"recall_at_{p2.primary_k}_contribution": (weights.recall_at_5, p2.primary_recall),
        f"mrr_at_{p2.primary_k}_contribution": (weights.mrr_at_5, p2.primary_mrr),
        f"ndcg_at_{p2.primary_k}_contribution": (weights.ndcg_at_5, p2.primary_ndcg),
        f"context_precision_at_{p2.primary_k}_contribution": (
            weights.context_precision_at_5,
            p2.primary_context_precision,
        ),
    }
    available_weight = sum(weight for weight, value in weighted.values() if value is not None)
    if available_weight <= 0.0:
        raise ValueError(
            f"No primary retrieval metrics available for experiment {p2.experiment_id!r}."
        )
    return {
        name: None if value is None else (weight / available_weight) * value
        for name, (weight, value) in weighted.items()
    }


def rqi_contributions(
    p2: P2Summary, p3: P3Summary, weights: RQIWeights
) -> dict[str, float]:
    return {
        "correctness_contribution": weights.correctness * (p3.mean_judge_correctness / 5.0),
        "faithfulness_contribution": weights.faithfulness * (p3.mean_judge_faithfulness / 5.0),
        "context_relevance_contribution": weights.context_relevance
        * (p3.mean_judge_context_relevance / 5.0),
        f"recall_at_{p2.primary_k}_contribution": weights.recall_at_5 * p2.primary_recall,
        "no_unknown_contribution": weights.no_unknown * (1.0 - p2.unknown_rate),
    }
