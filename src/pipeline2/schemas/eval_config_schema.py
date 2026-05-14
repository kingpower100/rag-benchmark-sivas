from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.pipeline2.config_loader import load_eval_config_payload


class StrictEvalConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationConfig(StrictEvalConfigModel):
    eval_run_id: str
    output_dir: str = "data/eval/runs/pipeline2"


class InputsConfig(StrictEvalConfigModel):
    rag_outputs: list[str]
    qa_path: str = "data/raw/qa_test.jsonl"
    gold_contexts_path: str = "data/raw/gold_contexts.jsonl"


class RetrievalEvalConfig(StrictEvalConfigModel):
    k: int = Field(default=5, gt=0)
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])


class AnswerQualityConfig(StrictEvalConfigModel):
    enable_numeric_accuracy: bool = True


class RuntimeConfig(StrictEvalConfigModel):
    overwrite: bool = True
    save_csv: bool = True


class LeaderboardConfig(StrictEvalConfigModel):
    sort_metric: str = "mean_recall_at_5"
    sort_ascending: bool = False


class EvalConfig(StrictEvalConfigModel):
    evaluation: EvaluationConfig
    inputs: InputsConfig
    retrieval: RetrievalEvalConfig = RetrievalEvalConfig()
    answer_quality: AnswerQualityConfig = AnswerQualityConfig()
    leaderboard: LeaderboardConfig = LeaderboardConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        return cls.model_validate(load_eval_config_payload(path))
