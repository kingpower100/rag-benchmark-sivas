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

    if config.type == "pgvector":
        from src.pipeline1.indexing.pgvector_index import PgvectorIndex

        pg = config.pgvector
        return PgvectorIndex(
            dsn_env=pg.dsn_env,
            schema_name=pg.schema_name,
            table_name=pg.table_name,
            logical_index_name=config.index_name,
            dense_dim=config.dense_dim,
            metric=config.metric,
            index_type=pg.index_type,
            rebuild_index=pg.rebuild_index,
            hnsw_m=pg.hnsw_m,
            hnsw_ef_construction=pg.hnsw_ef_construction,
            hnsw_ef_search=pg.hnsw_ef_search,
            ivfflat_lists=pg.ivfflat_lists,
            pool_min=pg.pool_min,
            pool_max=pg.pool_max,
        )

    from src.pipeline1.indexing.faiss_index import FaissIndex

    return FaissIndex(metric=config.metric)
