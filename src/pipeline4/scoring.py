from __future__ import annotations

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.schemas import RetrievalScoreWeights, RQIWeights


def compute_retrieval_score(p2: P2Summary, weights: RetrievalScoreWeights) -> float:
    return (
        weights.ndcg_at_5 * _required(p2.mean_ndcg_at_5, p2.experiment_id, "mean_ndcg_at_5")
        + weights.recall_at_5 * _required(p2.mean_recall_at_5, p2.experiment_id, "mean_recall_at_5")
        + weights.mrr_at_5 * _required(p2.mean_mrr_at_5, p2.experiment_id, "mean_mrr_at_5")
    )


def compute_rqi(p2: P2Summary, p3: P3Summary, weights: RQIWeights) -> float:
    return (
        weights.retrieval_score * compute_retrieval_score(p2, RetrievalScoreWeights())
        + weights.answer_quality * compute_answer_quality(p2)
        + weights.faithfulness * compute_faithfulness(p3)
        + weights.generation * compute_generation(p3)
    )


def compute_answer_quality(p2: P2Summary) -> float:
    bert = _required(p2.mean_official_bertscore_f1, p2.experiment_id, "mean_official_bertscore_f1")
    embedding = _required(p2.mean_embedding_similarity, p2.experiment_id, "mean_embedding_similarity")
    return 0.50 * bert + 0.50 * embedding


def compute_faithfulness(p3: P3Summary) -> float:
    ragas = _required(p3.mean_ragas_faithfulness, p3.experiment_id, "mean_ragas_faithfulness")
    judge = p3.mean_judge_faithfulness / 5.0
    return 0.50 * ragas + 0.50 * judge


def compute_generation(p3: P3Summary) -> float:
    correctness = p3.mean_judge_correctness / 5.0
    context_relevance = p3.mean_judge_context_relevance / 5.0
    completeness = p3.mean_judge_completeness / 5.0
    return 0.40 * correctness + 0.30 * context_relevance + 0.30 * completeness


def retrieval_score_contributions(
    p2: P2Summary, weights: RetrievalScoreWeights
) -> dict[str, float | None]:
    return {
        "ndcg_at_5_contribution": weights.ndcg_at_5
        * _required(p2.mean_ndcg_at_5, p2.experiment_id, "mean_ndcg_at_5"),
        "recall_at_5_contribution": weights.recall_at_5
        * _required(p2.mean_recall_at_5, p2.experiment_id, "mean_recall_at_5"),
        "mrr_at_5_contribution": weights.mrr_at_5
        * _required(p2.mean_mrr_at_5, p2.experiment_id, "mean_mrr_at_5"),
    }


def rqi_contributions(
    p2: P2Summary, p3: P3Summary, weights: RQIWeights
) -> dict[str, float]:
    retrieval_score = compute_retrieval_score(p2, RetrievalScoreWeights())
    answer_quality = compute_answer_quality(p2)
    faithfulness = compute_faithfulness(p3)
    generation = compute_generation(p3)
    return {
        "retrieval_score_component": retrieval_score,
        "answer_quality_component": answer_quality,
        "faithfulness_component": faithfulness,
        "generation_component": generation,
        "retrieval_score_contribution": weights.retrieval_score * retrieval_score,
        "answer_quality_contribution": weights.answer_quality * answer_quality,
        "faithfulness_contribution": weights.faithfulness * faithfulness,
        "generation_contribution": weights.generation * generation,
    }


def _required(value: float | None, experiment_id: str, metric_name: str) -> float:
    if value is None:
        raise ValueError(f"Required metric {metric_name!r} is missing for experiment {experiment_id!r}.")
    return float(value)
