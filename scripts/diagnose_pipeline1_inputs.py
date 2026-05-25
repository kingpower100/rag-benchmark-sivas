from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline1.io.jsonl_reader import JsonlReader, list_txt_files
from src.pipeline1.orchestrator import _build_chunker
from src.pipeline1.schemas.config_schema import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Pipeline 1 document loading and chunking inputs.")
    parser.add_argument("--config", required=True, help="Path to a Pipeline 1 YAML config.")
    parser.add_argument("--sample-chunks", type=int, default=3, help="Number of sample chunks to print.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    project_root = Path(__file__).resolve().parents[1]
    docs_path = project_root / cfg.data.documents_path

    if cfg.data.documents_source_type == "jsonl":
        docs = JsonlReader.read_documents(
            str(docs_path),
            text_field=cfg.data.document_text_field,
            allow_text_fallback=cfg.data.allow_document_text_fallback,
        )
        files = []
    elif cfg.data.documents_source_type == "txt_folder":
        files = list_txt_files(docs_path, cfg.data.documents_file_glob, cfg.data.documents_recursive)
        docs = JsonlReader.read_txt_folder(str(docs_path), cfg.data.documents_file_glob, cfg.data.documents_recursive)
    else:
        raise ValueError(f"Unsupported documents_source_type={cfg.data.documents_source_type!r}.")
    chunks = _build_chunker(cfg).chunk_documents(docs, show_progress=False)

    doc_lengths = [len(doc.text) for doc in docs]
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    chunks_by_doc = Counter(chunk.document_id for chunk in chunks)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    print(f"documents_path: {docs_path}")
    print(f"documents_source_type: {cfg.data.documents_source_type}")
    print(f"document_text_field: {cfg.data.document_text_field}")
    if cfg.data.documents_source_type == "txt_folder":
        print(f"documents_file_glob: {cfg.data.documents_file_glob}")
        print(f"documents_recursive: {cfg.data.documents_recursive}")
        print(f"txt_files_found: {len(files)}")
        print("first_10_file_paths:")
        for path in files[:10]:
            print(f"  - {path.relative_to(docs_path).as_posix()}")
    print(f"documents_loaded: {len(docs)}")
    if cfg.data.documents_source_type == "txt_folder":
        print(f"txt_files_skipped_empty: {len(files) - len(docs)}")
    print(f"document_length_chars: {_stats(doc_lengths)}")
    print(f"chunks_generated: {len(chunks)}")
    print(f"empty_chunks: {sum(1 for chunk in chunks if not chunk.text.strip())}")
    print(f"duplicate_chunk_ids: {len(chunk_ids) - len(set(chunk_ids))}")
    print(f"chunk_length_chars: {_stats(chunk_lengths)}")
    print("top_10_documents_by_chunks:")
    for document_id, count in chunks_by_doc.most_common(10):
        print(f"  - {document_id}: {count}")
    print(f"sample_{args.sample_chunks}_chunks:")
    for chunk in chunks[: args.sample_chunks]:
        metadata = chunk.metadata or {}
        preview = " ".join(chunk.text.split())[:240]
        print(f"  chunk_id: {chunk.chunk_id}")
        print(f"    document_id: {chunk.document_id}")
        print(f"    original_context_id: {chunk.original_context_id}")
        print(f"    source_file: {metadata.get('source_file')}")
        print(f"    source_path: {metadata.get('source_path')}")
        print(f"    preview: {preview}")


def _stats(values: list[int]) -> str:
    if not values:
        return "min=0 max=0 avg=0.0"
    return f"min={min(values)} max={max(values)} avg={sum(values) / len(values):.1f}"


if __name__ == "__main__":
    main()
