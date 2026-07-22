from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline1.chunking.fixed_token_chunker import FixedTokenChunker
from src.pipeline1.chunking.sentence_chunker import SentenceChunker
from src.pipeline1.chunking.sivas_character_chunker import SivasCharacterChunker
from src.pipeline1.io.jsonl_reader import JsonlReader
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord


DEFAULT_SPANS = PROJECT_ROOT / "data/ground_truth/chunk_level/E00-G_sentence512_overlap200/gold_evidence_spans.jsonl"
DEFAULT_DOCS = PROJECT_ROOT / "data/raw/kb_documents_fixed.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "data/ground_truth/chunk_level"


CONFIGS = {
    "B00_sivas_character2048_overlap0": {
        "strategy": "sivas_character",
        "chunk_size": 2048,
        "chunk_overlap": 0,
        "max_chunk_chars": 2048,
        "tokenizer_name": None,
    },
    "E00-G_sentence512_overlap200": {
        "strategy": "sentence",
        "chunk_size": 512,
        "chunk_overlap": 200,
        "chunk_size_unit": "tokens",
        "chunk_overlap_unit": "tokens",
        "tokenizer_name": "cl100k_base",
    },
    "C01_sentence256_overlap100": {
        "strategy": "sentence",
        "chunk_size": 256,
        "chunk_overlap": 100,
        "chunk_size_unit": "tokens",
        "chunk_overlap_unit": "tokens",
        "tokenizer_name": "cl100k_base",
    },
    "C02_sentence1024_overlap400": {
        "strategy": "sentence",
        "chunk_size": 1024,
        "chunk_overlap": 400,
        "chunk_size_unit": "tokens",
        "chunk_overlap_unit": "tokens",
        "tokenizer_name": "cl100k_base",
    },
    "E91-E98_fixed512_overlap64": {
        "strategy": "fixed_token",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "tokenizer_name": "cl100k_base",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build derived chunk-level annotation packages from canonical spans.")
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--configs", nargs="*", choices=sorted(CONFIGS), default=sorted(CONFIGS))
    args = parser.parse_args()

    spans = _read_jsonl(args.spans)
    docs = JsonlReader.read_documents(str(args.documents), dataset_schema="sivas", text_field="text")
    docs_by_key = {str(doc.metadata.get("doc_key")): doc for doc in docs}

    for config_id in args.configs:
        build_package(config_id, CONFIGS[config_id], docs, docs_by_key, spans, args.spans, args.output_root)


def build_package(
    config_id: str,
    config: dict[str, Any],
    docs: list[DocumentRecord],
    docs_by_key: dict[str, DocumentRecord],
    spans: list[dict[str, Any]],
    spans_path: Path,
    output_root: Path,
) -> None:
    chunks = _chunk_documents(config, docs)
    chunks_by_doc: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[str(chunk.document_id)].append(chunk)

    chunk_ranges = _chunk_ranges_by_doc(config, docs_by_key, chunks_by_doc)

    mappings: list[dict[str, Any]] = []
    annotations: dict[str, set[str]] = defaultdict(set)
    unmapped = 0
    document_hash_mismatches: list[dict[str, Any]] = []

    for span in spans:
        qid = _required_str(span, "question_id")
        doc_key = _required_str(span, "source_document_key")
        start = _required_int(span, "start_char")
        end = _required_int(span, "end_char")
        expected_hash = str(span.get("source_document_hash") or "")
        doc = docs_by_key.get(doc_key)
        if doc is None:
            raise RuntimeError(f"Span references unknown document key {doc_key!r}.")
        actual_hash = str(doc.metadata.get("content_hash") or "")
        if expected_hash and actual_hash and expected_hash != actual_hash:
            document_hash_mismatches.append({"question_id": qid, "doc_key": doc_key, "expected": expected_hash, "actual": actual_hash})

        relevant = []
        for chunk, chunk_start, chunk_end in chunk_ranges.get(doc_key, []):
            overlap_start = max(start, chunk_start)
            overlap_end = min(end, chunk_end)
            overlap_chars = max(0, overlap_end - overlap_start)
            if overlap_chars <= 0:
                continue
            relevant.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": int(chunk.metadata.get("chunk_index", len(relevant))),
                    "chunk_start_char": chunk_start,
                    "chunk_end_char": chunk_end,
                    "overlap_start_char": overlap_start,
                    "overlap_end_char": overlap_end,
                    "overlap_chars": overlap_chars,
                    "evidence_coverage": round(overlap_chars / max(1, end - start), 6),
                    "chunk_coverage": round(overlap_chars / max(1, chunk_end - chunk_start), 6),
                }
            )
        if not relevant:
            unmapped += 1
        else:
            primary = max(relevant, key=lambda item: (item["overlap_chars"], item["evidence_coverage"]))
            for item in relevant:
                item["is_primary"] = item["chunk_id"] == primary["chunk_id"]
                annotations[qid].add(item["chunk_id"])

        mappings.append(
            {
                "question_id": qid,
                "evidence_rank": span.get("evidence_rank"),
                "source_document_id": span.get("source_document_id"),
                "source_document_key": doc_key,
                "source_document_path": span.get("source_document_path"),
                "source_document_name": span.get("source_document_name"),
                "source_document_hash": span.get("source_document_hash"),
                "evidence_start_char": start,
                "evidence_end_char": end,
                "evidence_length_chars": max(0, end - start),
                "evidence_text_raw": span.get("evidence_text_raw"),
                "chunk_config_id": config_id,
                "mapping_policy": "any_overlap",
                "primary_chunk_id": max(relevant, key=lambda item: item["overlap_chars"])["chunk_id"] if relevant else None,
                "relevant_chunk_ids": [item["chunk_id"] for item in relevant],
                "relevant_chunks": relevant,
                "mapping_status": "mapped" if relevant else "unmapped",
            }
        )

    if unmapped:
        raise RuntimeError(f"{config_id}: {unmapped} evidence spans could not be mapped to production chunks.")
    if document_hash_mismatches:
        raise RuntimeError(f"{config_id}: document hash mismatches detected: {document_hash_mismatches[:3]}")

    output_dir = output_root / config_id
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_file = output_dir / f"gold_chunk_annotations_{config_id}.jsonl"
    mapping_file = output_dir / f"gold_span_chunk_mappings_{config_id}.jsonl"
    summary_file = output_dir / f"chunk_mapping_summary_{config_id}.json"
    validation_file = output_dir / "final_annotation_validation.json"
    manifest_file = output_dir / "final_annotation_manifest.json"
    integration_file = output_dir / "integration_package.json"
    spans_copy = output_dir / "gold_evidence_spans.jsonl"
    if spans_path.resolve() != spans_copy.resolve():
        shutil.copyfile(spans_path, spans_copy)

    _write_jsonl(
        annotation_file,
        [
            {
                "question_id": qid,
                "chunk_config_id": config_id,
                "gold_relevant_chunk_ids": sorted(chunk_ids),
                "gold_relevant_chunk_count": len(chunk_ids),
                "mapping_policy": "any_overlap",
            }
            for qid, chunk_ids in sorted(annotations.items())
        ],
    )
    _write_jsonl(mapping_file, mappings)

    unique_gold_chunks = {chunk_id for chunk_ids in annotations.values() for chunk_id in chunk_ids}
    summary = {
        "chunk_config_id": config_id,
        "chunking": config,
        "mapping_policy": "any_overlap",
        "questions": len(annotations),
        "chunks_total": len(chunks),
        "gold_evidence_spans": len(spans),
        "mapped_evidence_spans": len(spans) - unmapped,
        "unmapped_evidence_spans": unmapped,
        "unique_gold_relevant_chunks": len(unique_gold_chunks),
        "total_question_gold_chunk_references": sum(len(v) for v in annotations.values()),
        "raw_files_modified": False,
    }
    _write_json(summary_file, summary)

    validation = {
        "dataset_status": "PASS",
        "chunk_config_id": config_id,
        "questions": len(annotations),
        "expected_questions": 96,
        "chunk_count": len(chunks),
        "documents": len(docs),
        "gold_evidence_spans": len(spans),
        "mapped_evidence_spans": len(spans),
        "unmapped_records": unmapped,
        "document_hash_mismatches": 0,
        "validation_checks_failed": 0,
        "raw_files_modified": False,
    }
    if len(annotations) != 96:
        raise RuntimeError(f"{config_id}: expected 96 annotated questions, got {len(annotations)}.")
    _write_json(validation_file, validation)

    files = []
    for path, purpose in [
        (annotation_file, "Primary runtime ground-truth file for Pipeline 2 chunk-level retrieval evaluation."),
        (mapping_file, "Traceability artifact linking canonical evidence spans to production chunks."),
        (spans_copy, "Copied canonical evidence-span provenance used to derive chunk-level labels."),
        (summary_file, "Chunk configuration and mapping summary."),
        (validation_file, "Final validation report for the derived chunk annotation dataset."),
    ]:
        files.append({"filename": path.name, "purpose": purpose, "sha256": _sha256(path)})
    _write_json(manifest_file, {"chunk_config_id": config_id, "created_utc": _now(), "files": files})
    files.append({"filename": manifest_file.name, "purpose": "Integrity manifest for finalized derived artifacts.", "sha256": _sha256(manifest_file)})
    _write_json(
        integration_file,
        {
            "package_name": "SIVAS chunk-level retrieval ground truth",
            "chunk_config_id": config_id,
            "primary_runtime_file": annotation_file.name,
            "mapping_policy": "any_overlap",
            "questions": len(annotations),
            "chunks": len(chunks),
            "gold_evidence_spans": len(spans),
            "mapped_evidence_spans": len(spans),
            "unmapped_evidence_spans": unmapped,
            "files": files,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config_id}",
                "",
                "Derived chunk-level retrieval annotation package.",
                "",
                "This package is generated from the canonical evidence spans in `gold_evidence_spans.jsonl` and the production chunking configuration listed in `chunk_mapping_summary_*.json`.",
                "The canonical evidence spans are copied for provenance and are not modified by the generator.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _chunk_documents(config: dict[str, Any], docs: list[DocumentRecord]) -> list[ChunkRecord]:
    strategy = config["strategy"]
    if strategy == "sentence":
        chunker = SentenceChunker(
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"]),
            chunk_size_unit=config.get("chunk_size_unit", "tokens"),
            chunk_overlap_unit=config.get("chunk_overlap_unit", "tokens"),
            tokenizer_name=config.get("tokenizer_name") or "cl100k_base",
        )
    elif strategy == "fixed_token":
        chunker = FixedTokenChunker(
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"]),
            tokenizer_name=config.get("tokenizer_name") or "cl100k_base",
            allow_word_fallback=False,
        )
    elif strategy == "sivas_character":
        chunker = SivasCharacterChunker(max_chars=int(config["max_chunk_chars"]))
    else:
        raise ValueError(f"Unsupported annotation package chunking strategy: {strategy}")
    return chunker.chunk_documents(docs, show_progress=False)


def _chunk_ranges_by_doc(
    config: dict[str, Any],
    docs_by_key: dict[str, DocumentRecord],
    chunks_by_doc: dict[str, list[ChunkRecord]],
) -> dict[str, list[tuple[ChunkRecord, int, int]]]:
    if config["strategy"] == "fixed_token":
        return {
            doc_key: _fixed_token_chunk_ranges(
                docs_by_key[doc_key],
                doc_chunks,
                int(config["chunk_size"]),
                int(config["chunk_overlap"]),
                config.get("tokenizer_name") or "cl100k_base",
            )
            for doc_key, doc_chunks in chunks_by_doc.items()
            if doc_key in docs_by_key
        }
    return {
        doc_key: _locate_chunk_ranges(docs_by_key[doc_key].text, doc_chunks)
        for doc_key, doc_chunks in chunks_by_doc.items()
        if doc_key in docs_by_key
    }


def _fixed_token_chunk_ranges(
    doc: DocumentRecord,
    chunks: list[ChunkRecord],
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_name: str,
) -> list[tuple[ChunkRecord, int, int]]:
    import tiktoken

    encoding = tiktoken.get_encoding(tokenizer_name)
    tokens = encoding.encode(doc.text or "")
    _, offsets = encoding.decode_with_offsets(tokens)
    step = max(1, chunk_size - chunk_overlap)
    ranges: list[tuple[ChunkRecord, int, int]] = []
    for chunk, start in zip(chunks, range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        if start >= len(offsets):
            break
        start_char = offsets[start]
        if end >= len(offsets):
            end_char = len(doc.text or "")
        else:
            end_char = offsets[end]
        ranges.append((chunk, start_char, end_char))
        if end == len(tokens):
            break
    if len(ranges) != len(chunks):
        raise RuntimeError(
            f"Fixed-token range derivation mismatch for {doc.document_id}: "
            f"{len(ranges)} ranges for {len(chunks)} chunks."
        )
    return ranges


def _locate_chunk_ranges(document_text: str, chunks: list[ChunkRecord]) -> list[tuple[ChunkRecord, int, int]]:
    normalized_doc, index_map = _normalize_with_index_map(document_text)
    cursor = 0
    ranges: list[tuple[ChunkRecord, int, int]] = []
    for chunk in chunks:
        normalized_chunk, _ = _normalize_with_index_map(chunk.text)
        found = normalized_doc.find(normalized_chunk, cursor)
        if found < 0:
            found = normalized_doc.find(normalized_chunk)
        if found < 0:
            raise RuntimeError(f"Unable to locate production chunk {chunk.chunk_id} in source document {chunk.document_id}.")
        end_norm = found + len(normalized_chunk)
        start_char = index_map[found] if found < len(index_map) else len(document_text)
        end_char = index_map[end_norm - 1] + 1 if end_norm - 1 < len(index_map) else len(document_text)
        ranges.append((chunk, start_char, end_char))
        cursor = max(cursor, found + 1)
    return ranges


def _normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    index_map: list[int] = []
    in_ws = False
    for idx, char in enumerate(text or ""):
        if char.isspace():
            if not in_ws and out:
                out.append(" ")
                index_map.append(idx)
            in_ws = True
            continue
        out.append(char)
        index_map.append(idx)
        in_ws = False
    if out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}.")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _required_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Canonical evidence span is missing string field {field!r}: {row}")
    return value.strip()


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int):
        raise RuntimeError(f"Canonical evidence span is missing integer field {field!r}: {row}")
    return value


if __name__ == "__main__":
    main()
