from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalScoreWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recall_at_5: float = Field(default=0.35, ge=0.0, le=1.0)
    mrr_at_5: float = Field(default=0.25, ge=0.0, le=1.0)
    ndcg_at_5: float = Field(default=0.20, ge=0.0, le=1.0)
    context_precision_at_5: float = Field(default=0.20, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RetrievalScoreWeights":
        total = (
            self.recall_at_5 + self.mrr_at_5 + self.ndcg_at_5 + self.context_precision_at_5
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"retrieval_score_weights must sum to 1.0, got {total:.6f}"
            )
        return self


class RQIWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(default=0.25, ge=0.0, le=1.0)
    faithfulness: float = Field(default=0.25, ge=0.0, le=1.0)
    context_relevance: float = Field(default=0.20, ge=0.0, le=1.0)
    recall_at_5: float = Field(default=0.15, ge=0.0, le=1.0)
    no_unknown: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RQIWeights":
        total = (
            self.correctness
            + self.faithfulness
            + self.context_relevance
            + self.recall_at_5
            + self.no_unknown
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rqi_weights must sum to 1.0, got {total:.6f}")
        return self


class ValidationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_generation_failure_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    min_judge_success_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    max_ragas_nan_rate: float = Field(default=0.10, ge=0.0, le=1.0)


class Pipeline4Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pipeline2_runs_dir: str = Field(default="data/eval/runs/pipeline2")
    pipeline3_runs_dir: str = Field(default="data/eval/runs/pipeline3")
    output_dir: str = Field(default="data/eval/runs/pipeline4")
    run_id: str = Field(default="pipeline4_run")
    ranking_mode: str = Field(default="retrieval_only")
    retrieval_score_weights: RetrievalScoreWeights = Field(
        default_factory=RetrievalScoreWeights
    )
    rqi_weights: RQIWeights = Field(default_factory=RQIWeights)
    validation: ValidationThresholds = Field(default_factory=ValidationThresholds)

    @model_validator(mode="after")
    def validate_ranking_mode(self) -> "Pipeline4Config":
        valid_modes = ("retrieval_only", "overall_rag")
        if self.ranking_mode not in valid_modes:
            raise ValueError(
                f"ranking_mode must be one of {valid_modes}, got '{self.ranking_mode}'"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline4Config":
        data = _load_with_extends(path)
        return cls(**data)


def _load_with_extends(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "extends" in data:
        base_path = Path(path).parent / data.pop("extends")
        base = _load_with_extends(str(base_path))
        data = _deep_merge(base, data)
    return data


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
