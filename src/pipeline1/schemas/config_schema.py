from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.pipeline1.config_loader import load_pipeline_config_payload

FIXED_ORCHESTRATION_MODEL = "mistral-small"


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_by_alias=True, validate_by_name=True)


class ExperimentConfig(StrictConfigModel):
    experiment_id: str
    random_seed: int = 42
    output_dir: str


class DataConfig(StrictConfigModel):
    documents_path: str
    dataset_schema: Literal["sivas"] = "sivas"
    documents_source_type: Literal["jsonl", "txt_folder"] = "jsonl"
    documents_file_glob: str = "*.txt"
    documents_recursive: bool = True
    questions_path: str
    question_field: str = "frage"
    question_id_field: str = "question_id"
    document_text_field: str = "text"
    allow_document_text_fallback: bool = False
    allow_unsafe_query_fields: bool = False


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
    cache_dir: Optional[str] = None


class IndexConfig(StrictConfigModel):
    type: Literal["faiss", "elasticsearch"]
    metric: Literal["cosine", "l2"] = "cosine"
    host: str = "http://localhost:9200"
    index_name: str = "sivas_fixed512_bge_small"
    index_alias: Optional[str] = None
    index_version: Optional[str] = None
    dense_dim: int = Field(default=384, gt=0)
    vector_field: str = "embedding"
    text_field: str = "text"
    similarity: Literal["cosine"] = "cosine"
    recreate: bool = False
    retrieval_mode: Literal["script_score", "knn"] = "script_score"
    num_candidates: int = Field(default=100, gt=0)
    shards: int = Field(default=1, gt=0)
    replicas: int = Field(default=0, ge=0)
    refresh_after_index: bool = True
    request_timeout: int = Field(default=60, gt=0)
    verify_certs: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None


class MetadataBoostingConfig(StrictConfigModel):
    enabled: bool = False
    company_weight: float = 0.3
    year_weight: float = 0.15
    month_weight: float = 0.0
    year_month_weight: float = 0.0
    wrong_year_penalty: float = 0.0
    symbol_weight: float = 0.2
    file_name_weight: float = 0.0


class MetadataFilteringConfig(StrictConfigModel):
    enabled: bool = False
    strict: bool = False
    strict_year_match: bool = False
    strict_year_month_match: bool = False


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
    retriever_type: Literal["dense", "bm25", "hybrid_rrf", "elasticsearch_dense", "category_aware_dense"] = Field(
        default="dense",
        validation_alias=AliasChoices("retriever_type", "type"),
    )
    top_k: int = Field(gt=0)
    fetch_k: int = Field(gt=0)
    category_field: str = "kategorie"
    fallback_to_global: bool = True
    metadata_boosting: MetadataBoostingConfig = Field(default_factory=MetadataBoostingConfig)
    metadata_filtering: MetadataFilteringConfig = Field(default_factory=MetadataFilteringConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)


class RerankerConfig(StrictConfigModel):
    enabled: bool = False
    model_name: Optional[str] = None
    device: str = "cpu"
    rerank_top_k: Optional[int] = Field(default=None, gt=0)
    final_top_k: Optional[int] = Field(default=None, gt=0)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("reranker.model_name cannot be blank.")
        return value


class OrchestrationConfig(StrictConfigModel):
    provider: Literal["ollama"] = "ollama"
    model_name: str = Field(default=FIXED_ORCHESTRATION_MODEL, validation_alias=AliasChoices("model_name", "model"))
    base_url: str = "http://localhost:11434"
    fixed: bool = True
    tasks: list[Literal["clean_question", "detect_category"]] = Field(
        default_factory=lambda: ["clean_question", "detect_category"]
    )
    temperature: float = 0.0
    max_tokens: int = Field(default=256, gt=0)
    timeout_s: int = Field(default=60, gt=0)
    prompt_path: Optional[str] = None
    prompt_version: Optional[str] = None

    @field_validator("fixed")
    @classmethod
    def ensure_fixed(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Pipeline 1 requires orchestration.fixed=true for benchmark comparability.")
        return value

    @field_validator("model_name")
    @classmethod
    def ensure_fixed_model(cls, value: str) -> str:
        if value != FIXED_ORCHESTRATION_MODEL:
            raise ValueError(f"Orchestration model is fixed across experiments: {FIXED_ORCHESTRATION_MODEL}")
        return value

    @field_validator("tasks")
    @classmethod
    def ensure_limited_tasks(cls, value: list[str]) -> list[str]:
        expected = {"clean_question", "detect_category"}
        if set(value) != expected:
            raise ValueError("Orchestration LLM may only perform clean_question and detect_category.")
        return value


class GenerationConfig(StrictConfigModel):
    configurable: bool = False
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
    include_retrieval_metadata: bool = False


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
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    generation: GenerationConfig
    telemetry: TelemetryConfig
    runtime: RuntimeConfig

    @model_validator(mode="after")
    def validate_index_retriever_compatibility(self) -> "PipelineConfig":
        retriever_type = self.retrieval.retriever_type
        index_type = self.index.type
        if index_type == "faiss" and retriever_type == "elasticsearch_dense":
            raise ValueError(
                "Unsupported index/retriever combination: index.type='faiss' cannot be used with "
                "retrieval.retriever_type='elasticsearch_dense'. Use retrieval.retriever_type='dense' "
                "or set index.type='elasticsearch'."
            )
        if index_type == "elasticsearch" and retriever_type == "dense":
            raise ValueError(
                "Unsupported index/retriever combination: index.type='elasticsearch' cannot be used with "
                "retrieval.retriever_type='dense'. Use retrieval.retriever_type='elasticsearch_dense' "
                "or 'hybrid_rrf'."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        payload = load_pipeline_config_payload(path, validate_unique_experiment_id=True)
        return cls.model_validate(payload)
