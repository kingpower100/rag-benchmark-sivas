from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.metadata import canonical_chunk_metadata
from src.pipeline1.utils.ids import make_configured_chunk_id


TABLE_AWARE_CHUNKER_VERSION = "table_aware_v1_markdown_blocks"


@dataclass
class _Block:
    text: str
    kind: str
    start_line: int
    end_line: int


class TableAwareChunker(BaseChunker):
    """Groups text blocks while keeping markdown tables intact when possible."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_config = {
            "strategy": "table_aware",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "block_parser": TABLE_AWARE_CHUNKER_VERSION,
        }

    def chunk_documents(self, docs: list[DocumentRecord], show_progress: bool = False) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        iterator = tqdm(docs, desc="Table-aware chunking documents", unit="doc") if show_progress else docs
        for doc in iterator:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: DocumentRecord) -> list[ChunkRecord]:
        blocks = _parse_blocks(doc.text)
        grouped = self._group_blocks(blocks)
        records: list[ChunkRecord] = []
        for chunk_index, group in enumerate(grouped):
            text = "\n\n".join(block.text for block in group).strip()
            if not text:
                continue
            contains_table = any(block.kind == "table" for block in group)
            oversized_table = any(block.kind == "table" and _word_count(block.text) > self.chunk_size for block in group)
            chunk_id = make_configured_chunk_id(doc.document_id, chunk_index, text, self.strategy_config)
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=text,
                    chunk_start=group[0].start_line,
                    chunk_end=group[-1].end_line,
                    metadata={
                        **dict(doc.metadata),
                        **canonical_chunk_metadata(doc.metadata, doc.original_context_id),
                        "doc_id": doc.document_id,
                        "original_context_id": doc.original_context_id,
                        "source_file": doc.metadata.get("source_file") or doc.metadata.get("file_name"),
                        "source_id": doc.metadata.get("source_id"),
                        "year": doc.metadata.get("year"),
                        "month": doc.metadata.get("month"),
                        "subset": doc.metadata.get("subset"),
                        "split": doc.metadata.get("split") or doc.metadata.get("source_split"),
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "chunk_strategy": "table_aware",
                        "chunk_unit": "table_or_text_block",
                        "contains_table": contains_table,
                        "oversized_table": oversized_table,
                    },
                )
            )
        return records

    def _group_blocks(self, blocks: list[_Block]) -> list[list[_Block]]:
        groups: list[list[_Block]] = []
        current: list[_Block] = []
        current_words = 0
        for block in blocks:
            block_words = _word_count(block.text)
            would_exceed = current and current_words + block_words > self.chunk_size
            if would_exceed:
                groups.append(current)
                overlap = min(self.chunk_overlap, len(current))
                current = current[-overlap:] if overlap else []
                current_words = sum(_word_count(item.text) for item in current)
            current.append(block)
            current_words += block_words
        if current:
            groups.append(current)
        return groups


def _parse_blocks(text: str) -> list[_Block]:
    lines = (text or "").splitlines()
    blocks: list[_Block] = []
    pending_text: list[str] = []
    pending_start = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if _is_table_line(line):
            table_start = idx
            table_lines = []
            while idx < len(lines) and (_is_table_line(lines[idx]) or _is_table_separator(lines[idx])):
                table_lines.append(lines[idx])
                idx += 1
            table_text = "\n".join(table_lines).strip()
            if pending_text and _should_attach_to_table(pending_text):
                table_text = "\n".join(pending_text).strip() + "\n" + table_text
                table_start = pending_start
                pending_text = []
            elif blocks and blocks[-1].kind == "text" and _should_attach_to_table(blocks[-1].text.splitlines()):
                previous = blocks.pop()
                table_text = previous.text + "\n" + table_text
                table_start = previous.start_line
            elif pending_text:
                blocks.append(_Block("\n".join(pending_text).strip(), "text", pending_start, table_start))
                pending_text = []
            blocks.append(_Block(table_text, "table", table_start, idx))
            continue
        if line.strip():
            if not pending_text:
                pending_start = idx
            pending_text.append(line)
        elif pending_text:
            blocks.append(_Block("\n".join(pending_text).strip(), "text", pending_start, idx))
            pending_text = []
        idx += 1
    if pending_text:
        blocks.append(_Block("\n".join(pending_text).strip(), "text", pending_start, len(lines)))
    return [block for block in blocks if block.text]


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_table_separator(line: str) -> bool:
    stripped = line.strip().replace("|", "").replace(":", "").replace("-", "").strip()
    return not stripped and "|" in line and "-" in line


def _should_attach_to_table(lines: list[str]) -> bool:
    text = " ".join(lines).lower()
    return len(text.split()) <= 80 or any(marker in text for marker in ("in millions", "in thousands", "amounts in", "$ in"))


def _word_count(text: str) -> int:
    return len(text.split())
