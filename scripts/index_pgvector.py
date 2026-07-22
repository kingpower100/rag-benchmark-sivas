#!/usr/bin/env python3
"""Index documents into pgvector from a pipeline1 YAML config.

Usage:
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag \\
        python scripts/index_pgvector.py --config configs/pipeline1/experiments/my_pgvector.yaml

The script is idempotent: chunks already present (matched by chunk_id) are upserted,
so re-running is safe without duplicating data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into pgvector.")
    parser.add_argument("--config", required=True, help="Path to pipeline1 YAML config.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.pipeline1.schemas.config_schema import PipelineConfig
    from src.pipeline1.io.jsonl_reader import JsonlReader
    from src.pipeline1.embedding.factory import build_embedder
    from src.pipeline1.indexing.factory import build_index
    from src.pipeline1.stages.document_stage import DocumentStage
    from src.pipeline1.stages.chunking_stage import ChunkingStage
    from src.pipeline1.stages.embedding_stage import EmbeddingStage
    from src.pipeline1.stages.base import StageInput
    from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict

    cfg = PipelineConfig.from_yaml(args.config)
    if cfg.index.type != "pgvector":
        print(f"ERROR: config index.type='{cfg.index.type}' — expected 'pgvector'.", file=sys.stderr)
        sys.exit(1)

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

    print("Embedding...")
    embedding_output = EmbeddingStage(cfg, cache_dir, embedder_factory=build_embedder).run(
        StageInput({"chunks": chunks, "chunks_key": chunking_output.chunks_key})
    )
    embeddings = embedding_output.embeddings
    print(f"  Embeddings shape: {embeddings.shape}")

    print("Building pgvector index (upsert)...")
    index = build_index(cfg.index)
    index.set_chunks(chunks)
    if hasattr(index, "set_artifact_identity"):
        index.set_artifact_identity(
            {
                "dataset_fingerprint": chunking_output.documents_fingerprint,
                "source_document_fingerprint": chunking_output.documents_fingerprint,
                "chunk_store_fingerprint": chunking_output.chunks_key,
                "chunking_configuration_fingerprint": stable_hash_dict(cfg.chunking.model_dump()),
                "embedding_model_name": cfg.embedding.model_name,
                "embedding_normalization": cfg.embedding.normalize_embeddings,
                "framework_config_hash": file_sha256(args.config),
                "framework_code_version": "index_pgvector_script",
            }
        )
    index.build(embeddings)
    print("pgvector indexing complete.")


if __name__ == "__main__":
    main()
