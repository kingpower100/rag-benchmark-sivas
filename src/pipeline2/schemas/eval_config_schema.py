from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.pipeline2.config_loader import load_eval_config_payload


class StrictEvalConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationConfig(StrictEvalConfigModel):
    eval_run_id: str
    output_dir: str = "data/eval/runs/pipeline2"
    retrieval_only: bool = False
    retrieval_eval_field: Literal[
        "retrieved_file_names",
        "retrieved_files",
        "retrieved_document_ids",
        "retrieved_original_context_ids",
    ] = "retrieved_file_names"
    max_generation_failure_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    strict_failure_threshold: bool = False


class InputsConfig(StrictEvalConfigModel):
    pipeline1_results_path: str = "data/runs/pipeline1/11_sivas_fixed512_faiss_dense_mistralsmall_baseline/results.jsonl"
    rag_outputs: list[str] = Field(default_factory=list)
    questions_path: str = "data/raw/questions_fixed.jsonl"
    qa_path: str = "data/raw/qa_ground_truth_fixed.jsonl"
    gold_contexts_path: str = "data/raw/qa_ground_truth_fixed.jsonl"

    @model_validator(mode="after")
    def ensure_pipeline1_output_paths(self) -> "InputsConfig":
        if not self.rag_outputs:
            self.rag_outputs = [self.pipeline1_results_path]
        return self


class RetrievalEvalConfig(StrictEvalConfigModel):
    k: int = Field(default=5, gt=0)
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])


class AnswerQualityConfig(StrictEvalConfigModel):
    abstention_patterns: list[str] = Field(
        default_factory=lambda: [
            # English
            "UNKNOWN", "NOT FOUND", "N/A", "CANNOT DETERMINE",
            # German
            "UNBEKANNT", "NICHT GEFUNDEN", "NICHT VERFÜGBAR", "NICHT BEKANNT",
            "KEINE INFORMATION", "KANN NICHT BESTIMMT WERDEN", "NICHT BESTIMMBAR",
            "KEINE ANGABE", "K.A.",
        ]
    )


class EmbeddingSimilarityConfig(StrictEvalConfigModel):
    # Default is deterministic_hash so configs that omit this section remain offline-safe.
    # Production SIVAS runs must explicitly set provider=sentence_transformers in the yaml.
    provider: Literal["deterministic_hash", "sentence_transformers"] = "deterministic_hash"
    model_name: str = "hashing-bow-v1"
    dimensions: int = Field(default=256, gt=0)
    enabled: bool = True
    device: str = "cuda"
    require_cuda: bool = True
    # Set offline_mode=true to explicitly allow deterministic_hash in non-production runs.
    # EvaluationOrchestrator.run() raises if enabled=True, provider=deterministic_hash, offline_mode=False.
    offline_mode: bool = False


class BertScoreConfig(StrictEvalConfigModel):
    enabled: bool = False
    model_name: str = "bert-base-multilingual-cased"
    device: str = "auto"
    # max_length is kept for YAML backward-compatibility; the official bert-score library
    # selects tokenisation limits internally per model and ignores this field.
    max_length: int = Field(default=512, gt=0)
    idf: bool = False
    rescale_with_baseline: bool = False


class RuntimeConfig(StrictEvalConfigModel):
    overwrite: bool = True
    save_csv: bool = True


class EvalConfig(StrictEvalConfigModel):
    evaluation: EvaluationConfig
    inputs: InputsConfig
    retrieval: RetrievalEvalConfig = RetrievalEvalConfig()
    answer_quality: AnswerQualityConfig = AnswerQualityConfig()
    embedding_similarity: EmbeddingSimilarityConfig = EmbeddingSimilarityConfig()
    bert_score: BertScoreConfig = BertScoreConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        return cls.model_validate(load_eval_config_payload(path))
