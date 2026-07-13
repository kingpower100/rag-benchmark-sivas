#!/usr/bin/env python3
"""Index documents into Elasticsearch BM25 from a pipeline1 YAML config.

Usage:
    python scripts/index_elasticsearch.py --config configs/pipeline1/experiments/my_es_bm25.yaml

The script is idempotent: bulk-indexes chunks with _id=chunk_id so re-running
overwrites existing documents without duplicating them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into Elasticsearch BM25.")
    parser.add_argument("--config", required=True, help="Path to pipeline1 YAML config.")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate the ES index.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.pipeline1.schemas.config_schema import PipelineConfig
    from src.pipeline1.stages.document_stage import DocumentStage
    from src.pipeline1.stages.chunking_stage import ChunkingStage
    from src.pipeline1.stages.base import StageInput

    cfg = PipelineConfig.from_yaml(args.config)

    docs_path = project_root / cfg.data.documents_path
    cache_dir = project_root / "data" / "processed"

    print("Loading documents...")
    document_output = DocumentStage(cfg, docs_path).run()
    docs = document_output.documents
    print(f"  {len(docs)} documents loaded.")

    print("Chunking...")
    chunking_output = ChunkingStage(cfg, project_root, cache_dir, docs_path).run(
        StageInput({"documents": docs})
    )
    chunks = chunking_output.chunks
    print(f"  {len(chunks)} chunks produced.")

    bm25_cfg = cfg.retrieval.bm25
    host_env = bm25_cfg.host_env
    if host_env:
        host = os.environ.get(host_env, "").strip()
        if not host:
            print(
                f"ERROR: bm25.host_env is set to '{host_env}' but the environment "
                f"variable is missing or empty. Export it before running:\n"
                f"  export {host_env}=http://<host>:<port>",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        host = bm25_cfg.host

    index_name = bm25_cfg.index_name
    k1 = bm25_cfg.k1
    b = bm25_cfg.b
    analyzer = bm25_cfg.analyzer

    print(f"Elasticsearch host: {host}")

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print("ERROR: elasticsearch package is required.", file=sys.stderr)
        sys.exit(1)

    client = Elasticsearch(host)
    exists = client.indices.exists(index=index_name)
    if exists and args.recreate:
        print(f"Deleting index '{index_name}'...")
        client.indices.delete(index=index_name)
        exists = False

    if not exists:
        print(f"Creating index '{index_name}'...")
        client.indices.create(index=index_name, body={
            "settings": {
                "index": {
                    "similarity": {
                        "default": {"type": "BM25", "k1": k1, "b": b}
                    }
                }
            },
            "mappings": {
                "properties": {
                    "context_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "cleaned_context": {"type": "text", "analyzer": analyzer},
                    "file_name": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        })

    print(f"Indexing {len(chunks)} chunks...")
    batch_size = 1000
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        operations = []
        for chunk in batch:
            operations.append({"index": {"_index": index_name, "_id": chunk.chunk_id}})
            operations.append({
                "context_id": chunk.metadata.get("context_id") or chunk.metadata.get("original_context_id"),
                "chunk_id": chunk.chunk_id,
                "cleaned_context": chunk.text,
                "file_name": chunk.metadata.get("file_name"),
                "document_id": getattr(chunk, "document_id", None) or chunk.metadata.get("document_id"),
                "metadata": chunk.metadata,
            })
        client.bulk(operations=operations, refresh=False)
        print(f"  Indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")

    client.indices.refresh(index=index_name)
    print("Elasticsearch BM25 indexing complete.")


if __name__ == "__main__":
    main()
