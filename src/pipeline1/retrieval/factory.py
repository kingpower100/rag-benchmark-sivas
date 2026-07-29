import os

from src.pipeline1.retrieval.adaptive_category_aware_dense_retriever import AdaptiveCategoryAwareDenseRetriever
from src.pipeline1.retrieval.bm25_retriever import BM25Retriever
from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever
from src.pipeline1.retrieval.dense_retriever import DenseRetriever
from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Error, ElasticsearchBM25Retriever
from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever
from src.pipeline1.retrieval.hybrid_rrf_retriever import HybridRRFRetriever
from src.pipeline1.schemas.config_schema import RetrievalConfig


def build_retriever(config: RetrievalConfig, embedder, index, chunks, embeddings=None):
    if config.retriever_type == "bm25":
        return _build_bm25_retriever(config, chunks)
    if config.retriever_type == "elasticsearch_dense":
        return ElasticsearchDenseRetriever(
            embedder=embedder,
            index=index,
            chunks=chunks,
            top_k=config.top_k,
            fetch_k=config.fetch_k,
            metadata_boosting=config.metadata_boosting,
            metadata_filtering=config.metadata_filtering,
        )
    if config.retriever_type == "elasticsearch_hybrid_rrf":
        return _build_elasticsearch_hybrid_rrf_retriever(config, embedder, index, chunks)
    if config.retriever_type == "adaptive_category_aware_hybrid_rrf":
        return _build_adaptive_category_aware_hybrid_rrf_retriever(config, embedder, index, chunks)

    dense_retriever = _build_dense_retriever(config, embedder, index, chunks)
    if config.retriever_type == "category_aware_dense":
        return CategoryAwareDenseRetriever(
            dense_retriever=dense_retriever,
            category_field=config.category_field,
            embeddings=embeddings,
            index_metric=getattr(index, "metric", "cosine"),
        )
    if config.retriever_type == "adaptive_category_aware_dense":
        return AdaptiveCategoryAwareDenseRetriever(
            dense_retriever=dense_retriever,
            category_field=config.category_field,
            embeddings=embeddings,
            index_metric=getattr(index, "metric", "cosine"),
        )
    if config.retriever_type == "hybrid_rrf":
        return HybridRRFRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=_build_bm25_retriever(config, chunks),
            fetch_k=config.fetch_k,
            rrf_k=config.hybrid.rrf_k,
            dense_weight=config.hybrid.dense_weight,
            bm25_weight=config.hybrid.bm25_weight,
        )
    return dense_retriever


def _build_dense_retriever(config: RetrievalConfig, embedder, index, chunks):
    from src.pipeline1.indexing.pgvector_index import PgvectorIndex

    if isinstance(index, PgvectorIndex):
        from src.pipeline1.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever

        return PgvectorDenseRetriever(
            embedder=embedder,
            index=index,
            chunks=chunks,
            fetch_k=config.fetch_k,
            metadata_boosting=config.metadata_boosting,
            metadata_filtering=config.metadata_filtering,
            category_field=config.category_field,
        )

    from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex

    if isinstance(index, ElasticsearchIndex):
        return ElasticsearchDenseRetriever(
            embedder=embedder,
            index=index,
            chunks=chunks,
            top_k=config.top_k,
            fetch_k=config.fetch_k,
            metadata_boosting=config.metadata_boosting,
            metadata_filtering=config.metadata_filtering,
        )

    return DenseRetriever(
        embedder=embedder,
        index=index,
        chunks=chunks,
        fetch_k=config.fetch_k,
        metadata_boosting=config.metadata_boosting,
        metadata_filtering=config.metadata_filtering,
    )


def _build_bm25_retriever(config: RetrievalConfig, chunks):
    if config.bm25.backend == "local":
        return BM25Retriever(chunks=chunks, k1=config.bm25.k1, b=config.bm25.b)
    host = config.bm25.host
    if config.bm25.host_env:
        host = os.environ.get(config.bm25.host_env, host)
    try:
        return ElasticsearchBM25Retriever(
            chunks=chunks,
            host=host,
            index_name=config.bm25.index_name,
            k1=config.bm25.k1,
            b=config.bm25.b,
            rebuild_index=config.bm25.rebuild_index,
            analyzer=config.bm25.analyzer,
        )
    except ElasticsearchBM25Error:
        if config.bm25.allow_fallback:
            return BM25Retriever(chunks=chunks, k1=config.bm25.k1, b=config.bm25.b)
        raise


def _build_adaptive_category_aware_hybrid_rrf_retriever(config: RetrievalConfig, embedder, index, chunks):
    """Build AdaptiveCategoryAwareHybridRRFRetriever wrapping an ElasticsearchHybridRRFRetriever.

    The schema guarantees index.type='elasticsearch' and orchestration.enabled=True when this
    retriever_type is selected.  The hybrid sub-retriever is built identically to the
    elasticsearch_hybrid_rrf path so both R00 and R01 share the same algorithm.
    """
    from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
        AdaptiveCategoryAwareHybridRRFRetriever,
    )

    hybrid = _build_elasticsearch_hybrid_rrf_retriever(config, embedder, index, chunks)
    return AdaptiveCategoryAwareHybridRRFRetriever(
        hybrid_retriever=hybrid,
        category_field=config.category_field,
    )


def _build_elasticsearch_hybrid_rrf_retriever(config: RetrievalConfig, embedder, index, chunks):
    """Build an ElasticsearchHybridRRFRetriever from config.

    The dense leg is always an ElasticsearchDenseRetriever (the schema validator
    guarantees index.type='elasticsearch' when this retriever_type is selected).
    Official R00/R01 configurations require Elasticsearch BM25 with
    allow_fallback=false. Local BM25 is only used by configurations that
    explicitly select bm25.backend='local' or permit fallback.
    """
    from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import (
        ElasticsearchHybridRRFRetriever,
    )

    hybrid_cfg = config.hybrid
    dense_fetch_k = hybrid_cfg.dense_fetch_k if hybrid_cfg.dense_fetch_k else config.fetch_k
    bm25_fetch_k = hybrid_cfg.bm25_fetch_k if hybrid_cfg.bm25_fetch_k else config.fetch_k

    dense = ElasticsearchDenseRetriever(
        embedder=embedder,
        index=index,
        chunks=chunks,
        top_k=config.top_k,
        fetch_k=dense_fetch_k,
        metadata_boosting=config.metadata_boosting,
        metadata_filtering=config.metadata_filtering,
    )
    bm25 = _build_bm25_retriever(config, chunks)
    return ElasticsearchHybridRRFRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fetch_k=config.fetch_k,
        dense_fetch_k=dense_fetch_k,
        bm25_fetch_k=bm25_fetch_k,
        rrf_k=hybrid_cfg.rrf_k,
        dense_weight=hybrid_cfg.dense_weight,
        bm25_weight=hybrid_cfg.bm25_weight,
    )
