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
    documents_recursive: bool = True
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
    max_chunk_chars: int = Field(default=8000, gt=0)
    max_chunk_tokens: int = Field(default=1800, gt=0)
    oversized_chunk_policy: Literal["split", "warn", "raise"] = "split"
    oversized_chunk_warning: bool = True


class EmbeddingConfig(StrictConfigModel):
    provider: Literal["sentence_transformers"]
    model_name: str
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str = "cpu"
    require_cuda: bool = False


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


class BM25Config(StrictConfigModel):
    enabled: bool = True
    backend: Literal["local", "elasticsearch"] = "local"
    host: str = "http://localhost:9200"
    index_name: str = "rag_benchmark_chunks"
    rebuild_index: bool = False
    allow_fallback: bool = False
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


class HybridConfig(StrictConfigModel):
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=1.0, ge=0)


class RetrievalConfig(StrictConfigModel):
    retriever_type: Literal["dense", "bm25", "hybrid_rrf"] = "dense"
    top_k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)
    metadata_boosting: MetadataBoostingConfig = Field(default_factory=MetadataBoostingConfig)
    metadata_filtering: MetadataFilteringConfig = Field(default_factory=MetadataFilteringConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class RerankerConfig(StrictConfigModel):
    enabled: bool = False
    model_name: Optional[str] = None
    device: str = "cpu"
    final_top_k: Optional[int] = Field(default=None, gt=0)

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
    max_prompt_tokens: int = Field(default=8192, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)
    max_chunk_tokens: int = Field(default=1800, gt=0)
    max_context_chars: int = Field(default=24000, gt=0)
    max_chunk_chars: int = Field(default=8000, gt=0)
    context_truncation_strategy: Literal["ranked_budget"] = "ranked_budget"
    log_prompt_stats: bool = True


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
    cache_mismatch_policy: Literal["raise", "rebuild"] = "raise"


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
