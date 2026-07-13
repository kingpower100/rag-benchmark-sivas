#!/usr/bin/env python3
"""Idempotent pgvector schema/table/index initializer.

Usage:
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag python scripts/init_pgvector.py

Optional env vars:
    PGVECTOR_DSN       - required, full PostgreSQL DSN
    PG_SCHEMA          - schema name (default: rag)
    PG_TABLE           - table name (default: chunk_embeddings)
    DENSE_DIM          - embedding dimension (default: 1024)
    INDEX_TYPE         - hnsw | ivfflat | exact (default: hnsw)
    HNSW_M             - HNSW m parameter (default: 16)
    HNSW_EF_CONSTRUCTION - HNSW ef_construction (default: 64)
    IVFFLAT_LISTS      - IVFFlat lists (default: 100)
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    dsn = os.environ.get("PGVECTOR_DSN")
    if not dsn:
        print("ERROR: PGVECTOR_DSN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    schema = os.environ.get("PG_SCHEMA", "rag")
    table = os.environ.get("PG_TABLE", "chunk_embeddings")
    dense_dim = int(os.environ.get("DENSE_DIM", "1024"))
    index_type = os.environ.get("INDEX_TYPE", "hnsw")
    hnsw_m = int(os.environ.get("HNSW_M", "16"))
    hnsw_ef_construction = int(os.environ.get("HNSW_EF_CONSTRUCTION", "64"))
    ivfflat_lists = int(os.environ.get("IVFFLAT_LISTS", "100"))

    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 is required. Install it with: pip install psycopg2-binary", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            print(f"Creating extension vector...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            print(f"Creating schema {schema}...")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

            qualified = f"{schema}.{table}"
            print(f"Creating table {qualified} (dim={dense_dim})...")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {qualified} (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    original_context_id TEXT,
                    text TEXT,
                    category TEXT,
                    metadata JSONB,
                    embedding vector({dense_dim})
                )
            """)

            idx_name = f"{table}_embedding_idx"
            if index_type == "hnsw":
                print(f"Creating HNSW index {idx_name} (m={hnsw_m}, ef_construction={hnsw_ef_construction})...")
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {qualified} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})
                """)
            elif index_type == "ivfflat":
                print(f"Creating IVFFlat index {idx_name} (lists={ivfflat_lists})...")
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON {qualified} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {ivfflat_lists})
                """)
            else:
                print(f"index_type='{index_type}' — no ANN index created (exact search).")

        conn.commit()
        print("pgvector initialization complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
