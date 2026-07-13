#!/usr/bin/env python3
"""Lightweight connectivity check for PostgreSQL+pgvector and Elasticsearch.

Does NOT modify any data. Exits with code 0 if all checks pass, 1 if any fail.

Usage:
    export PGVECTOR_DSN="postgresql://rag:rag@localhost:5432/rag"
    export ELASTICSEARCH_URL="http://localhost:9200"   # optional; default used if unset
    python scripts/check_backend_services.py
"""
from __future__ import annotations

import os
import sys

PGVECTOR_DSN_ENV = "PGVECTOR_DSN"
ELASTICSEARCH_URL_ENV = "ELASTICSEARCH_URL"
DEFAULT_ES_URL = "http://localhost:9200"

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

_failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  {_GREEN}[OK]{_RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {_RED}[FAIL]{_RESET} {msg}")
    _failures.append(msg)


def warn(msg: str) -> None:
    print(f"  {_YELLOW}[WARN]{_RESET} {msg}")


# ---------------------------------------------------------------------------
# PostgreSQL + pgvector
# ---------------------------------------------------------------------------

def check_postgres() -> None:
    print("\nPostgreSQL + pgvector")
    print("-" * 40)

    dsn = os.environ.get(PGVECTOR_DSN_ENV)
    if not dsn:
        fail(f"{PGVECTOR_DSN_ENV} is not set — skipping PostgreSQL checks.")
        warn(f"  Set it with: export {PGVECTOR_DSN_ENV}=postgresql://rag:rag@localhost:5432/rag")
        return

    try:
        import psycopg2
    except ImportError:
        fail("psycopg2 not installed. Run: pip install psycopg2-binary")
        return

    # 1 — connectivity
    try:
        conn = psycopg2.connect(dsn, connect_timeout=5)
    except Exception as ex:
        fail(f"Cannot connect to PostgreSQL: {ex}")
        return
    ok(f"Connected to PostgreSQL ({dsn.split('@')[-1]})")

    with conn.cursor() as cur:
        # 2 — server version
        cur.execute("SELECT version()")
        version_row = cur.fetchone()
        version = str(version_row[0]) if version_row else "unknown"
        ok(f"Server: {version.split(',')[0]}")

        # 3 — pgvector extension
        cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        if row:
            ok(f"pgvector extension installed (version {row[1]})")
        else:
            fail(
                "pgvector extension is NOT installed. "
                "Run: docker exec rag-benchmark-postgres psql -U rag -d rag "
                "-c \"CREATE EXTENSION IF NOT EXISTS vector;\""
            )

        # 4 — schema/table existence (informational only)
        cur.execute("""
            SELECT schemaname, tablename
            FROM pg_tables
            WHERE schemaname = 'rag' AND tablename = 'chunk_embeddings'
        """)
        tbl = cur.fetchone()
        if tbl:
            cur.execute("SELECT COUNT(*) FROM rag.chunk_embeddings")
            count_row = cur.fetchone()
            count = int(count_row[0]) if count_row else 0
            ok(f"Table rag.chunk_embeddings exists ({count} rows)")
        else:
            warn("Table rag.chunk_embeddings does not exist yet — run scripts/init_pgvector.py first.")

    conn.close()


# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------

def check_elasticsearch() -> None:
    print("\nElasticsearch")
    print("-" * 40)

    es_url = os.environ.get(ELASTICSEARCH_URL_ENV, DEFAULT_ES_URL)

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        fail("elasticsearch package not installed. Run: pip install elasticsearch")
        return

    # 1 — connectivity + cluster health
    try:
        client = Elasticsearch(es_url, request_timeout=5)
        health = client.cluster.health()
    except Exception as ex:
        fail(f"Cannot reach Elasticsearch at {es_url}: {ex}")
        return

    status = health.get("status", "unknown")
    node_count = health.get("number_of_nodes", "?")
    if status in ("green", "yellow"):
        ok(f"Cluster healthy (status={status}, nodes={node_count}) at {es_url}")
    else:
        fail(f"Cluster status is '{status}' — expected 'green' or 'yellow'.")

    # 2 — version
    try:
        info = client.info()
        es_version = info.get("version", {}).get("number", "unknown")
        ok(f"Elasticsearch version: {es_version}")
    except Exception:
        warn("Could not retrieve Elasticsearch version.")

    # 3 — index existence (informational only)
    index_name = "rag_benchmark_chunks"
    try:
        exists = client.indices.exists(index=index_name)
        if exists:
            stats = client.count(index=index_name)
            doc_count = stats.get("count", "?")
            ok(f"Index '{index_name}' exists ({doc_count} documents)")

            # 4 — German analyzer check
            try:
                result = client.indices.analyze(
                    index=index_name,
                    body={"analyzer": "german", "text": "Lieferungen Aufträge"},
                )
                tokens = [t["token"] for t in result.get("tokens", [])]
                if tokens:
                    ok(f"German analyzer active — tokens: {tokens}")
                else:
                    warn("German analyzer returned no tokens.")
            except Exception as ex:
                warn(f"Could not verify German analyzer: {ex}")
        else:
            warn(
                f"Index '{index_name}' does not exist yet — "
                "run scripts/init_elasticsearch.py first."
            )
    except Exception as ex:
        warn(f"Could not query index '{index_name}': {ex}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 50)
    print("RAG Benchmark — Backend Service Check")
    print("=" * 50)

    check_postgres()
    check_elasticsearch()

    print("\n" + "=" * 50)
    if _failures:
        print(f"{_RED}RESULT: {len(_failures)} check(s) failed:{_RESET}")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"{_GREEN}RESULT: All checks passed.{_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
