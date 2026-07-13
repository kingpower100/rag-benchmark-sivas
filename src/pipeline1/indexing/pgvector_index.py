from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import numpy as np

from src.pipeline1.indexing.base import BaseVectorIndex


def _vec_to_pgvector(vec: np.ndarray) -> str:
    return "[" + ",".join(str(float(v)) for v in vec) + "]"


class _PgConn:
    def __init__(self, pool):
        self._pool = pool
        self._conn = None

    def __enter__(self):
        self._conn = self._pool.getconn()
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pool.putconn(self._conn)
        self._conn = None


class PgvectorIndex(BaseVectorIndex):
    uses_external_storage = True

    def __init__(
        self,
        dsn_env: str = "PGVECTOR_DSN",
        schema_name: str = "rag",
        table_name: str = "chunk_embeddings",
        dense_dim: int = 1024,
        metric: str = "cosine",
        index_type: str = "hnsw",
        rebuild_index: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 40,
        ivfflat_lists: int = 100,
        pool_min: int = 1,
        pool_max: int = 5,
        _pool=None,
    ) -> None:
        self.dsn_env = dsn_env
        self.schema_name = schema_name
        self.table_name = table_name
        self.dense_dim = dense_dim
        self.metric = metric
        self.index_type = index_type
        self.rebuild_index = rebuild_index
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.ivfflat_lists = ivfflat_lists
        self.pool_min = pool_min
        self.pool_max = pool_max
        self._pool = _pool
        self._chunks: list = []
        self._chunk_by_id: dict[str, Any] = {}

    def set_chunks(self, chunks: list) -> None:
        self._chunks = chunks
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def _get_dsn(self) -> str:
        dsn = os.environ.get(self.dsn_env)
        if not dsn:
            raise RuntimeError(
                f"Environment variable '{self.dsn_env}' is not set. "
                "Set it to a valid PostgreSQL DSN before running with pgvector backend."
            )
        return dsn

    def _get_pool(self):
        if self._pool is None:
            self._pool = self._build_pool()
        return self._pool

    def _build_pool(self):
        try:
            from psycopg2.pool import ThreadedConnectionPool
        except ImportError as ex:
            raise RuntimeError(
                "index.type='pgvector' requires psycopg2. Install it with: pip install psycopg2-binary"
            ) from ex
        dsn = self._get_dsn()
        return ThreadedConnectionPool(self.pool_min, self.pool_max, dsn)

    @contextmanager
    def _conn(self):
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            pool.putconn(conn)

    def _ensure_schema(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema_name}")
        conn.commit()

    def _ensure_table(self, conn) -> None:
        qualified = f"{self.schema_name}.{self.table_name}"
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    original_context_id TEXT,
                    text TEXT,
                    category TEXT,
                    metadata JSONB,
                    embedding vector({self.dense_dim})
                )
            """)
        conn.commit()

    def _pg_op(self) -> str:
        return "<=>" if self.metric == "cosine" else "<->"

    def _ensure_vector_index(self, conn) -> None:
        qualified = f"{self.schema_name}.{self.table_name}"
        idx_name = f"{self.table_name}_embedding_idx"
        op_class = "vector_cosine_ops" if self.metric == "cosine" else "vector_l2_ops"
        with conn.cursor() as cur:
            if self.index_type == "hnsw":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {qualified} USING hnsw (embedding {op_class})
                    WITH (m = {self.hnsw_m}, ef_construction = {self.hnsw_ef_construction})
                """)
            elif self.index_type == "ivfflat":
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {qualified} USING ivfflat (embedding {op_class})
                    WITH (lists = {self.ivfflat_lists})
                """)
        conn.commit()

    def _bulk_upsert(self, embeddings: np.ndarray, batch_size: int = 500) -> None:
        import json as _json
        qualified = f"{self.schema_name}.{self.table_name}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                for start in range(0, len(self._chunks), batch_size):
                    batch_chunks = self._chunks[start:start + batch_size]
                    batch_embs = embeddings[start:start + batch_size]
                    for chunk, emb in zip(batch_chunks, batch_embs):
                        cat = str((chunk.metadata or {}).get("kategorie") or "")
                        cur.execute(
                            f"""
                            INSERT INTO {qualified}
                                (chunk_id, document_id, original_context_id, text, category, metadata, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                document_id = EXCLUDED.document_id,
                                original_context_id = EXCLUDED.original_context_id,
                                text = EXCLUDED.text,
                                category = EXCLUDED.category,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding
                            """,
                            (
                                chunk.chunk_id,
                                getattr(chunk, "document_id", None),
                                getattr(chunk, "original_context_id", None),
                                chunk.text,
                                cat,
                                _json.dumps(dict(chunk.metadata or {})),
                                _vec_to_pgvector(emb),
                            ),
                        )
            conn.commit()

    def build(self, embeddings: np.ndarray) -> None:
        with self._conn() as conn:
            self._ensure_schema(conn)
            self._ensure_table(conn)
            if not self.rebuild_index:
                with conn.cursor() as cur:
                    qualified = f"{self.schema_name}.{self.table_name}"
                    cur.execute(f"SELECT COUNT(*) FROM {qualified}")
                    row = cur.fetchone()
                    existing = row[0] if row else 0
                if existing == len(self._chunks):
                    self._set_hnsw_ef_search(conn)
                    return
        self._bulk_upsert(embeddings)
        with self._conn() as conn:
            if self.index_type != "exact":
                self._ensure_vector_index(conn)
            self._set_hnsw_ef_search(conn)

    def _set_hnsw_ef_search(self, conn) -> None:
        if self.index_type == "hnsw":
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self.hnsw_ef_search}")

    def save(self, path: str) -> None:
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"pgvector:{self.schema_name}.{self.table_name}@{self.dsn_env}",
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        with self._conn() as conn:
            self._ensure_schema(conn)
            self._ensure_table(conn)
            self._set_hnsw_ef_search(conn)

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        return self._search_sql(query_embedding, top_k, category=None, category_field=None)

    def search_category(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        category: str,
        category_field: str = "kategorie",
    ) -> tuple[list[str], list[float]]:
        return self._search_sql(query_embedding, top_k, category=category, category_field=category_field)

    def _search_sql(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        category: str | None,
        category_field: str | None,
    ) -> tuple[list[str], list[float]]:
        qualified = f"{self.schema_name}.{self.table_name}"
        op = self._pg_op()
        vec_str = _vec_to_pgvector(query_embedding)
        if category:
            sql = f"""
                SELECT chunk_id, 1 - (embedding {op} %s::vector) AS score
                FROM {qualified}
                WHERE category = %s
                ORDER BY embedding {op} %s::vector
                LIMIT %s
            """
            params = (vec_str, category, vec_str, top_k)
        else:
            sql = f"""
                SELECT chunk_id, 1 - (embedding {op} %s::vector) AS score
                FROM {qualified}
                ORDER BY embedding {op} %s::vector
                LIMIT %s
            """
            params = (vec_str, vec_str, top_k)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        chunk_ids = [str(r[0]) for r in rows]
        scores = [float(r[1]) for r in rows]
        return chunk_ids, scores

    @property
    def ntotal(self) -> int:
        qualified = f"{self.schema_name}.{self.table_name}"
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {qualified}")
                    row = cur.fetchone()
                    return row[0] if row else 0
        except Exception:
            return 0

    @property
    def dim(self) -> int:
        return self.dense_dim
