from __future__ import annotations

from typing import Any

from src.pipeline3.judge.response_parser import JudgeResponse
from src.pipeline3.schemas.pipeline3_config_schema import P3ScoringConfig, P3WeightsConfig


def compute_weighted_overall_score(
    response: JudgeResponse,
    weights: P3WeightsConfig,
    scoring: P3ScoringConfig,
) -> float:
    scale_max = float(scoring.scale_max)
    # Hallucination is inverted: 0 = no hallucination (best), 5 = worst.
    # (scale_max - score) ensures 0 hallucination contributes full weight.
    hallucination_contribution = (scale_max - response.hallucination) / scale_max
    weighted_sum = (
        response.correctness / scale_max * weights.correctness
        + response.faithfulness / scale_max * weights.faithfulness
        + response.completeness / scale_max * weights.completeness
        + hallucination_contribution * weights.hallucination
        + response.context_relevance / scale_max * weights.context_relevance
    )
    return round(weighted_sum * scale_max, 4)


def enabled_judge_metric_names(metrics_cfg: Any) -> list[str]:
    mapping = {
        "correctness": metrics_cfg.correctness,
        "faithfulness": metrics_cfg.faithfulness,
        "completeness": metrics_cfg.completeness,
        "hallucination": metrics_cfg.hallucination,
        "context_relevance": metrics_cfg.context_relevance,
    }
    return [name for name, enabled in mapping.items() if enabled]
