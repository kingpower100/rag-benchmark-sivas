from pathlib import Path
from typing import Any, Literal, Optional

import warnings

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from src.pipeline1.config_loader import load_pipeline_config_payload

DEFAULT_ORCHESTRATION_MODEL = "mistral-small"
ALLOWED_ORCHESTRATION_MODELS = frozenset(
    {
        "mistral-small",
        "qwen2.5:7b",
        "llama3.1:8b",
    }
)


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
    strategy: Literal["fixed_token", "fixed_word", "sentence", "table_aware", "sivas_character"]
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    chunk_size_unit: Optional[Literal["tokens", "words", "sentences", "characters"]] = None
    chunk_overlap_unit: Optional[Literal["tokens", "words", "sentences", "characters"]] = None
    tokenizer_name: str = "cl100k_base"
    allow_word_fallback: bool = False
    max_chunk_chars: int = Field(default=8000, gt=0)
    max_chunk_tokens: int = Field(default=1800, gt=0)
    oversized_chunk_policy: Literal["split", "warn", "raise"] = "split"
    oversized_chunk_warning: bool = True

    @model_validator(mode="after")
    def validate_chunk_units(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunking.chunk_overlap must be < chunking.chunk_size.")
        if self.strategy == "sentence":
            if self.chunk_size_unit is None or self.chunk_overlap_unit is None:
                warnings.warn(
                    "Sentence chunking configs without explicit chunk_size_unit/chunk_overlap_unit "
                    "use legacy units: chunk_size_unit='words', chunk_overlap_unit='sentences'. "
                    "Official benchmark configs should set both units explicitly.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if self.chunk_size_unit is None:
                    self.chunk_size_unit = "words"
                if self.chunk_overlap_unit is None:
                    self.chunk_overlap_unit = "sentences"
            if "tokens" in (self.chunk_size_unit, self.chunk_overlap_unit):
                _validate_tiktoken_encoding(self.tokenizer_name)
        if self.strategy == "sivas_character" and self.oversized_chunk_policy != "warn":
            raise ValueError(
                "strategy='sivas_character' only supports chunking.oversized_chunk_policy='warn'. "
                "Oversized indivisible source spans are kept as one chunk and marked in diagnostics; "
                "'split' and 'raise' are not implemented for this strategy."
            )
        return self


class EmbeddingConfig(StrictConfigModel):
    provider: Literal["sentence_transformers", "mistral"]
    model_name: str
    normalize_embeddings: bool = Field(default=True, validation_alias=AliasChoices("normalize_embeddings", "normalize"))
    batch_size: int = 32
    device: str = "cpu"
    require_cuda: bool = False
    cache_dir: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_normalize_alias(cls, data: Any) -> Any:
        if (
            isinstance(data, dict)
            and "normalize" in data
            and "normalize_embeddings" in data
            and data["normalize"] != data["normalize_embeddings"]
        ):
            raise ValueError(
                "embedding.normalize is a legacy alias for embedding.normalize_embeddings; "
                "do not set conflicting values."
            )
        return data


class PgvectorConfig(StrictConfigModel):
    dsn_env: str = "PGVECTOR_DSN"
    schema_name: str = "rag"
    table_name: str = "chunk_embeddings"
    index_type: Literal["exact", "hnsw", "ivfflat"] = "hnsw"
    rebuild_index: bool = False
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construction: int = Field(default=64, gt=0)
    hnsw_ef_search: int = Field(default=40, gt=0)
    ivfflat_lists: int = Field(default=100, gt=0)
    pool_min: int = Field(default=1, gt=0)
    pool_max: int = Field(default=5, gt=0)


class IndexConfig(StrictConfigModel):
    type: Literal["faiss", "elasticsearch", "pgvector"]
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
    pgvector: Optional[PgvectorConfig] = None


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
    backend: Literal["local", "elasticsearch"] = "local"
    host: str = "http://localhost:9200"
    host_env: Optional[str] = None
    index_name: str = "rag_benchmark_chunks"
    rebuild_index: bool = False
    allow_fallback: bool = False
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    analyzer: str = "german"

    @model_validator(mode="before")
    @classmethod
    def reject_removed_enabled(cls, data: Any) -> Any:
        if isinstance(data, dict) and "enabled" in data:
            raise ValueError(
                "Deprecated field: retrieval.bm25.enabled. BM25 activation is controlled only by "
                "retrieval.retriever_type. Remove retrieval.bm25.enabled from YAML."
            )
        return data


class HybridConfig(StrictConfigModel):
    rrf_k: int = Field(default=60, gt=0)
    dense_weight: float = Field(default=1.0, ge=0)
    bm25_weight: float = Field(default=1.0, ge=0)
    dense_backend: Literal["faiss", "pgvector"] = "faiss"
    dense_fetch_k: Optional[int] = Field(default=None, gt=0)
    bm25_fetch_k: Optional[int] = Field(default=None, gt=0)


class RetrievalConfig(StrictConfigModel):
    retriever_type: Literal["dense", "bm25", "hybrid_rrf", "elasticsearch_dense", "category_aware_dense", "elasticsearch_hybrid_rrf"] = Field(
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

    @model_validator(mode="after")
    def validate_fetch_k_hard_cap(self) -> "RetrievalConfig":
        if self.fetch_k < self.top_k:
            raise ValueError(
                f"retrieval.fetch_k ({self.fetch_k}) must be >= retrieval.top_k ({self.top_k}). "
                "fetch_k is a strict raw-candidate cap."
            )
        return self


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

    @model_validator(mode="after")
    def require_model_when_enabled(self) -> "RerankerConfig":
        if self.enabled and not self.model_name:
            raise ValueError("reranker.model_name is required when reranker.enabled=true")
        return self


class OrchestrationConfig(StrictConfigModel):
    enabled: bool = True
    provider: Literal["ollama", "mistral"] = "ollama"
    model_name: str = Field(default=DEFAULT_ORCHESTRATION_MODEL, validation_alias=AliasChoices("model_name", "model"))
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
    def ensure_allowed_model(cls, value: str) -> str:
        if value not in ALLOWED_ORCHESTRATION_MODELS:
            allowed = ", ".join(sorted(ALLOWED_ORCHESTRATION_MODELS))
            raise ValueError(f"Unsupported orchestration.model_name '{value}'. Allowed values: {allowed}")
        return value

    @field_validator("tasks")
    @classmethod
    def ensure_limited_tasks(cls, value: list[str]) -> list[str]:
        expected = {"clean_question", "detect_category"}
        if set(value) != expected:
            raise ValueError("Orchestration LLM may only perform clean_question and detect_category.")
        return value

    @model_validator(mode="after")
    def validate_prompt_version_matches_prompt_path(self) -> "OrchestrationConfig":
        if "tasks" in self.model_fields_set:
            warnings.warn(
                "Deprecated field: orchestration.tasks. The current benchmark orchestration "
                "workflow is fixed to clean_question and detect_category; remove this field.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.prompt_version and self.prompt_path:
            prompt_stem = Path(self.prompt_path).stem.lower()
            version = self.prompt_version.lower()
            normalized_stem = prompt_stem.replace("orchestration_prompt", "")
            normalized_version = version.replace("orchestration_prompt", "")
            if normalized_version == "v0" and prompt_stem == "orchestration_prompt":
                return self
            if prompt_stem != version and normalized_stem != normalized_version:
                raise ValueError(
                    "orchestration.prompt_version must match orchestration.prompt_path. "
                    f"Got prompt_version={self.prompt_version!r}, prompt_path={self.prompt_path!r}."
                )
        return self


class GenerationConfig(StrictConfigModel):
    provider: Literal["ollama", "mistral"]
    model_name: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = Field(default=512, gt=0)
    timeout_s: int = Field(default=90, gt=0)
    system_prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    include_metadata_headers: bool = False
    max_prompt_tokens: int = Field(default=8192, gt=0)
    max_context_tokens: int = Field(default=6000, gt=0)
    max_chunk_tokens: int = Field(default=1800, gt=0)
    max_context_chars: int = Field(default=24000, gt=0)
    max_chunk_chars: int = Field(default=8000, gt=0)
    context_truncation_strategy: Literal["ranked_budget"] = "ranked_budget"
    log_prompt_stats: bool = True
    include_retrieval_metadata: bool = False

    @model_validator(mode="after")
    def require_prompt_source(self) -> "GenerationConfig":
        if self.prompt_path is not None and self.system_prompt is None:
            resolved = _resolve_project_path(self.prompt_path)
            if not resolved.is_file():
                raise ValueError(f"generation.prompt_path is missing or not a file: {resolved}")
            prompt = resolved.read_text(encoding="utf-8")
            if not prompt.strip():
                raise ValueError(f"generation.prompt_path is empty: {resolved}")
            self.system_prompt = prompt
        if self.prompt_path is None and not (self.system_prompt or "").strip():
            raise ValueError("generation requires either system_prompt or prompt_path.")
        return self

    @model_validator(mode="before")
    @classmethod
    def reject_removed_configurable(cls, data: Any) -> Any:
        if isinstance(data, dict) and "configurable" in data:
            raise ValueError(
                "Deprecated field: generation.configurable. It had no runtime effect; remove it "
                "and configure generation.temperature, max_tokens, timeout_s, and prompt_path explicitly."
            )
        return data


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / candidate).resolve()


def _validate_tiktoken_encoding(tokenizer_name: str) -> None:
    try:
        import tiktoken

        tiktoken.get_encoding(tokenizer_name)
    except Exception as ex:
        raise ValueError(
            "chunking.tokenizer_name must be a valid tiktoken encoding when sentence "
            f"chunking uses token units; got {tokenizer_name!r}."
        ) from ex


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


class ParentContextConfig(StrictConfigModel):
    enabled: bool = False
    parent_unit: Literal["markdown_section"] = "markdown_section"
    deduplicate: bool = True
    missing_parent_policy: Literal["use_child", "error"] = "use_child"
    unique_parent_top_k: int = Field(default=5, gt=0)
    max_parent_tokens: int = Field(default=1800, gt=0)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_policy_fields(cls, data: Any) -> Any:
        removed = {
            "mapping_policy",
            "score_policy",
            "preserve_child_provenance",
            "oversized_parent_policy",
        }
        if isinstance(data, dict):
            present = sorted(removed.intersection(data))
            if present:
                fields = ", ".join(f"parent_context.{field}" for field in present)
                raise ValueError(
                    f"Deprecated parent-context field(s): {fields}. These fields were removed because "
                    "the current parent-context implementation uses one fixed policy for parent mapping, "
                    "score propagation, child provenance preservation, and oversized-parent selection. "
                    "Remove the field(s) from the YAML."
                )
        return data


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
    parent_context: ParentContextConfig = Field(default_factory=ParentContextConfig)

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
                "or 'elasticsearch_hybrid_rrf'."
            )
        if index_type == "pgvector" and retriever_type == "elasticsearch_dense":
            raise ValueError(
                "Unsupported index/retriever combination: index.type='pgvector' cannot be used with "
                "retrieval.retriever_type='elasticsearch_dense'."
            )
        if index_type == "pgvector" and self.index.pgvector is None:
            raise ValueError(
                "index.type='pgvector' requires index.pgvector configuration block."
            )
        if retriever_type == "hybrid_rrf":
            dense_backend = self.retrieval.hybrid.dense_backend
            if dense_backend == "pgvector" and index_type != "pgvector":
                raise ValueError(
                    f"retrieval.hybrid.dense_backend='pgvector' requires index.type='pgvector', got '{index_type}'."
                )
            if dense_backend == "faiss" and index_type not in ("faiss",):
                raise ValueError(
                    f"retrieval.hybrid.dense_backend='faiss' requires index.type='faiss', got '{index_type}'."
                )
        if retriever_type == "elasticsearch_hybrid_rrf" and index_type != "elasticsearch":
            raise ValueError(
                f"retrieval.retriever_type='elasticsearch_hybrid_rrf' requires index.type='elasticsearch', "
                f"got '{index_type}'. Set index.type='elasticsearch' to use the Elasticsearch hybrid retriever."
            )
        if retriever_type == "category_aware_dense" and not self.orchestration.enabled:
            raise ValueError(
                "retrieval.retriever_type='category_aware_dense' requires orchestration.enabled=true "
                "because category-aware retrieval needs a validated category prediction."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        payload = load_pipeline_config_payload(path, validate_unique_experiment_id=True)
        return cls.model_validate(payload)
