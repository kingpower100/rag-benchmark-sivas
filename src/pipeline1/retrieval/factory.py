from src.pipeline1.retrieval.dense_retriever import DenseRetriever
from src.pipeline1.schemas.config_schema import RetrievalConfig


def build_retriever(config: RetrievalConfig, embedder, index, chunks):
    return DenseRetriever(
        embedder=embedder,
        index=index,
        chunks=chunks,
        fetch_k=config.fetch_k,
        metadata_boosting=config.metadata_boosting,
        metadata_filtering=config.metadata_filtering,
    )
