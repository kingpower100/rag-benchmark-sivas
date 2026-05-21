from __future__ import annotations

from dataclasses import dataclass

from tqdm.auto import tqdm

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.metadata import canonical_chunk_metadata
from src.pipeline1.utils.ids import make_configured_chunk_id


TABLE_AWARE_CHUNKER_VERSION = "table_aware_v3_oversized_guard"


@dataclass
class _Block:
    text: str
    kind: str
    start_line: int
    end_line: int


class TableAwareChunker(BaseChunker):
    """Groups text blocks while keeping markdown tables intact when possible."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        max_chunk_chars: int = 8000,
        max_chunk_tokens: int = 1800,
        oversized_chunk_policy: str = "split",
        oversized_chunk_warning: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_chars = max_chunk_chars
        self.max_chunk_tokens = max_chunk_tokens
        self.oversized_chunk_policy = oversized_chunk_policy
        self.oversized_chunk_warning = oversized_chunk_warning
        self.strategy_config = {
            "strategy": "table_aware",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "max_chunk_chars": max_chunk_chars,
            "max_chunk_tokens": max_chunk_tokens,
            "oversized_chunk_policy": oversized_chunk_policy,
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
            base_metadata = {
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
                "chunk_strategy": "table_aware",
                "chunk_unit": "table_or_text_block",
                "contains_table": contains_table,
                "oversized_table": oversized_table,
            }
            records.extend(
                self._records_for_text(doc, text, group[0].start_line, group[-1].end_line, chunk_index, base_metadata)
            )
        return records

    def _records_for_text(
        self,
        doc: DocumentRecord,
        text: str,
        start_line: int,
        end_line: int,
        chunk_index: int,
        base_metadata: dict,
    ) -> list[ChunkRecord]:
        pieces = [text]
        oversized = len(text) > self.max_chunk_chars or _word_count(text) > self.max_chunk_tokens
        if oversized and self.oversized_chunk_policy == "raise":
            raise ValueError(f"Oversized chunk in document_id={doc.document_id}: chars={len(text)} words={_word_count(text)}")
        if oversized and self.oversized_chunk_policy == "split":
            pieces = _split_oversized_text(text, self.max_chunk_chars, self.max_chunk_tokens)
        records = []
        for piece_index, piece in enumerate(pieces):
            if not piece.strip():
                continue
            local_index = chunk_index if len(pieces) == 1 else int(f"{chunk_index}{piece_index:03d}")
            chunk_id = make_configured_chunk_id(doc.document_id, local_index, piece, self.strategy_config)
            metadata = {
                **base_metadata,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "oversized_split": oversized and len(pieces) > 1,
                "oversized_split_index": piece_index if len(pieces) > 1 else None,
                "oversized_split_count": len(pieces) if len(pieces) > 1 else None,
            }
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=piece.strip(),
                    chunk_start=start_line,
                    chunk_end=end_line,
                    metadata=metadata,
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
                current = _overlap_blocks(current, self.chunk_overlap)
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


def _overlap_blocks(blocks: list[_Block], max_overlap_words: int) -> list[_Block]:
    if max_overlap_words <= 0 or not blocks:
        return []
    selected: list[_Block] = []
    selected_words = 0
    for block in reversed(blocks):
        block_words = _word_count(block.text)
        if block_words > max_overlap_words or selected_words + block_words > max_overlap_words:
            break
        selected.append(block)
        selected_words += block_words
    selected.reverse()
    if len(selected) == len(blocks):
        return selected[1:]
    return selected


def _split_oversized_text(text: str, max_chars: int, max_words: int) -> list[str]:
    lines = text.splitlines() or [text]
    pieces: list[str] = []
    current: list[str] = []
    current_chars = 0
    current_words = 0
    for line in lines:
        line_chars = len(line) + (1 if current else 0)
        line_words = _word_count(line)
        if current and (current_chars + line_chars > max_chars or current_words + line_words > max_words):
            pieces.append("\n".join(current).strip())
            current = []
            current_chars = 0
            current_words = 0
        if line_chars > max_chars or line_words > max_words:
            pieces.extend(_split_long_line(line, max_chars, max_words))
            continue
        current.append(line)
        current_chars += line_chars
        current_words += line_words
    if current:
        pieces.append("\n".join(current).strip())
    return [piece for piece in pieces if piece.strip()]


def _split_long_line(line: str, max_chars: int, max_words: int) -> list[str]:
    words = line.split()
    if not words:
        return [line[:max_chars]]
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(word[start:start + max_chars] for start in range(0, len(word), max_chars))
            continue
        candidate = " ".join([*current, word])
        if current and (len(candidate) > max_chars or len(current) + 1 > max_words):
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces
