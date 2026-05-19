from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from src.pipeline1.config_loader import load_pipeline_config_payload


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(StrictConfigModel):
    experiment_id: str
    random_seed: int = 42
    output_dir: str


class DataConfig(StrictConfigModel):
    documents_path: str
    documents_source_type: Literal["jsonl", "txt_folder"] = "jsonl"
    documents_file_glob: str = "*.txt"
    questions_path: str = Field(validation_alias=AliasChoices("questions_path", "qa_test_path"))
    question_field: str = "question"
    question_id_field: str = "question_id"
    document_text_field: str = "cleaned_context"
    allow_document_text_fallback: bool = False
    allow_unsafe_query_fields: bool = False
    use_ground_truth_contexts: bool = False
    use_gold_answers: bool = False

    @field_validator("use_ground_truth_contexts", "use_gold_answers")
    @classmethod
    def ensure_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("Pipeline 1 forbids gold answers and ground-truth contexts.")
        return value


class ChunkingConfig(StrictConfigModel):
    strategy: Literal["fixed_token", "fixed_word", "sentence", "table_aware"]
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    tokenizer_name: str = "cl100k_base"
    allow_word_fallback: bool = False


class EmbeddingConfig(StrictConfigModel):
    provider: Literal["sentence_transformers"]
    model_name: str
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"


class IndexConfig(StrictConfigModel):
    type: Literal["faiss"]
    metric: Literal["cosine", "l2"] = "cosine"


class MetadataBoostingConfig(StrictConfigModel):
    enabled: bool = False
    company_weight: float = 0.3
    year_weight: float = 0.15
    symbol_weight: float = 0.2
    file_name_weight: float = 0.0


class MetadataFilteringConfig(StrictConfigModel):
    enabled: bool = False
    strict: bool = False


class RetrievalConfig(StrictConfigModel):
    retriever_type: Literal["dense"] = "dense"
    top_k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)
    metadata_boosting: MetadataBoostingConfig = Field(default_factory=MetadataBoostingConfig)
    metadata_filtering: MetadataFilteringConfig = Field(default_factory=MetadataFilteringConfig)


class RerankerConfig(StrictConfigModel):
    enabled: bool = False
    model_name: Optional[str] = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("reranker.model_name cannot be blank.")
        return value


class GenerationConfig(StrictConfigModel):
    provider: Literal["ollama"]
    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = Field(default=512, gt=0)
    timeout_s: int = Field(default=90, gt=0)
    system_prompt: str
    include_metadata_headers: bool = False


class PricingConfig(StrictConfigModel):
    input_per_1k_tokens_usd: float = 0.0
    output_per_1k_tokens_usd: float = 0.0


class TelemetryConfig(StrictConfigModel):
    estimate_cost: bool = True
    pricing: PricingConfig = PricingConfig()


class RuntimeConfig(StrictConfigModel):
    save_csv: bool = True
    log_level: str = "INFO"
    resume: bool = True
    overwrite: bool = False


class PipelineConfig(StrictConfigModel):
    experiment: ExperimentConfig
    data: DataConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    generation: GenerationConfig
    telemetry: TelemetryConfig
    runtime: RuntimeConfig

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        payload = load_pipeline_config_payload(path, validate_unique_experiment_id=True)
        return cls.model_validate(payload)
