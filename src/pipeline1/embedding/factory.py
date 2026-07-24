import logging

from src.pipeline1.embedding.bge_encoder import BGEEncoder
from src.pipeline1.schemas.config_schema import EmbeddingConfig

_log = logging.getLogger(__name__)


def build_embedder(config: EmbeddingConfig):
    if config.provider == "mistral":
        from src.pipeline1.embedding.mistral_embedder import MistralEmbedder
        _log.info(
            "Embedding provider 'mistral' uses the remote API; "
            "device, require_cuda and normalize_embeddings are ignored."
        )
        return MistralEmbedder(
            model_name=config.model_name,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            retry_backoff_base_s=config.retry_backoff_base_s,
            api_base_url=config.api_base_url,
            expected_dimension=config.expected_dimension,
        )
    return BGEEncoder(
        config.model_name,
        config.normalize_embeddings,
        config.batch_size,
        config.device,
        config.require_cuda,
        config.cache_dir,
        config.query_prefix,
        config.document_prefix,
        config.dense_output_only,
        config.pooling,
        config.max_seq_length,
        config.expected_dimension,
    )
