from src.pipeline1.schemas.config_schema import IndexConfig


def build_index(config: IndexConfig):
    if config.type == "elasticsearch":
        from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex

        return ElasticsearchIndex(
            host=config.host,
            index_name=config.index_name,
            index_alias=config.index_alias,
            index_version=config.index_version,
            dense_dim=config.dense_dim,
            vector_field=config.vector_field,
            text_field=config.text_field,
            similarity=config.similarity,
            recreate=config.recreate,
            retrieval_mode=config.retrieval_mode,
            num_candidates=config.num_candidates,
            shards=config.shards,
            replicas=config.replicas,
            refresh_after_index=config.refresh_after_index,
            request_timeout=config.request_timeout,
            verify_certs=config.verify_certs,
            username=config.username,
            password=config.password,
            api_key=config.api_key,
        )
    from src.pipeline1.indexing.faiss_index import FaissIndex

    return FaissIndex(metric=config.metric)
