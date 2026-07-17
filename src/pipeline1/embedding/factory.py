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
        )
    return BGEEncoder(
        config.model_name,
        config.normalize_embeddings,
        config.batch_size,
        config.device,
        config.require_cuda,
        config.cache_dir,
    )
