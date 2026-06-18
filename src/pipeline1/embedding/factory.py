from src.pipeline1.embedding.bge_encoder import BGEEncoder
from src.pipeline1.schemas.config_schema import EmbeddingConfig


def build_embedder(config: EmbeddingConfig):
    return BGEEncoder(
        config.model_name,
        config.normalize_embeddings,
        config.batch_size,
        config.device,
        config.require_cuda,
        config.cache_dir,
    )
