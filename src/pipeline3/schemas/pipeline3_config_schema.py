from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.pipeline3.config_loader import load_pipeline3_config_payload


class StrictP3ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class P3RunConfig(StrictP3ConfigModel):
    run_id: str
    output_dir: str = "data/eval/runs/pipeline3"
    overwrite: bool = True
    save_csv: bool = True
    version: str = "1.0.0"
    prompt_version: str = "v2"


class P3InputsConfig(StrictP3ConfigModel):
    pipeline1_results_path: str
    questions_path: str = "data/raw/questions_fixed.jsonl"
    qa_path: str = "data/raw/qa_ground_truth_fixed.jsonl"
    gold_contexts_path: str = "data/raw/qa_ground_truth_fixed.jsonl"


class P3JudgeConfig(StrictP3ConfigModel):
    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=120, gt=0)
    prompt_version: str = "v2"


class P3RagasMetricsConfig(StrictP3ConfigModel):
    faithfulness: bool = True
    answer_relevancy: bool = True


class P3RagasConfig(StrictP3ConfigModel):
    enabled: bool = True
    fail_on_ragas_error: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:14b"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    embeddings_model: str = "intfloat/multilingual-e5-large"
    embeddings_device: str = "cuda"
    require_cuda: bool = True
    timeout_seconds: int = Field(default=300, gt=0)
    metrics: P3RagasMetricsConfig = Field(default_factory=P3RagasMetricsConfig)


class P3JudgeMetricsConfig(StrictP3ConfigModel):
    correctness: bool = True
    faithfulness: bool = True
    completeness: bool = True
    hallucination: bool = True
    context_relevance: bool = True


class P3ScoringConfig(StrictP3ConfigModel):
    scale_min: int = 0
    scale_max: int = 5


class P3WeightsConfig(StrictP3ConfigModel):
    correctness: float = 0.30
    faithfulness: float = 0.25
    completeness: float = 0.20
    hallucination: float = 0.15
    context_relevance: float = 0.10

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "P3WeightsConfig":
        total = (
            self.correctness
            + self.faithfulness
            + self.completeness
            + self.hallucination
            + self.context_relevance
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"llm_judge.weights must sum to 1.0, got {total:.4f}")
        return self


class P3LLMJudgeConfig(StrictP3ConfigModel):
    enabled: bool = True
    metrics: P3JudgeMetricsConfig = Field(default_factory=P3JudgeMetricsConfig)
    scoring: P3ScoringConfig = Field(default_factory=P3ScoringConfig)
    weights: P3WeightsConfig = Field(default_factory=P3WeightsConfig)


class P3RuntimeConfig(StrictP3ConfigModel):
    parallel_workers: int = Field(default=1, ge=1)
    batch_size: int = Field(default=10, ge=1)
    save_raw_outputs: bool = True


class Pipeline3Config(StrictP3ConfigModel):
    pipeline3: P3RunConfig
    inputs: P3InputsConfig
    judge: P3JudgeConfig = Field(default_factory=P3JudgeConfig)
    ragas: P3RagasConfig = Field(default_factory=P3RagasConfig)
    llm_judge: P3LLMJudgeConfig = Field(default_factory=P3LLMJudgeConfig)
    runtime: P3RuntimeConfig = Field(default_factory=P3RuntimeConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline3Config":
        return cls.model_validate(load_pipeline3_config_payload(path))
