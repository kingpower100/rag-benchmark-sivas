#!/usr/bin/env python3
"""Idempotent Elasticsearch BM25 index initializer with German analyzer.

Usage:
    python scripts/init_elasticsearch.py \\
        --host http://localhost:9200 \\
        --index rag_benchmark_chunks

Optional:
    --k1         BM25 k1 parameter (default: 1.5)
    --b          BM25 b parameter (default: 0.75)
    --analyzer   Elasticsearch analyzer (default: german)
    --recreate   Drop and recreate the index if it exists
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize Elasticsearch BM25 index.")
    parser.add_argument("--host", default="http://localhost:9200")
    parser.add_argument("--index", default="rag_benchmark_chunks")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--analyzer", default="german")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print("ERROR: elasticsearch package is required. Install it with: pip install elasticsearch", file=sys.stderr)
        sys.exit(1)

    client = Elasticsearch(args.host)
    try:
        client.info()
    except Exception as ex:
        print(f"ERROR: Cannot reach Elasticsearch at {args.host}: {ex}", file=sys.stderr)
        sys.exit(1)

    exists = client.indices.exists(index=args.index)
    if exists and args.recreate:
        print(f"Deleting existing index '{args.index}'...")
        client.indices.delete(index=args.index)
        exists = False

    if exists:
        print(f"Index '{args.index}' already exists. Use --recreate to rebuild.")
        return

    body = {
        "settings": {
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                        "k1": args.k1,
                        "b": args.b,
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "context_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "cleaned_context": {"type": "text", "analyzer": args.analyzer},
                "file_name": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }
    print(f"Creating index '{args.index}' with analyzer='{args.analyzer}', k1={args.k1}, b={args.b}...")
    client.indices.create(index=args.index, body=body)
    print("Elasticsearch index initialized.")


if __name__ == "__main__":
    main()
