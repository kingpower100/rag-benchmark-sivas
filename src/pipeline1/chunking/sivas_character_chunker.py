"""SIVAS-compatible character-based sentence chunker.

Implements the exact partner production algorithm:

1. Split document text with the documented boundary regex.
2. Iterate segments.
3. Accumulate segments into the current chunk while the combined text
   stays at or below max_chars (default 2048 characters).
4. When a segment would exceed the limit, emit the current chunk and
   start a new one with that segment.
5. Emit the final (possibly short) chunk unconditionally.
6. No overlap — each segment belongs to exactly one chunk.
7. All document metadata is forwarded to every chunk.

The boundary regex is the exact SIVAS partner regex::

    (?<=[.!?;:])\\s+|\\n\\n|\\n(?=#{1,6}\\s)|\\n(?=-\\s)

which fires at:

- sentence boundaries after . ! ? ; :  (lookbehind + whitespace)
- blank lines (paragraph breaks)       (double newline)
- lines immediately before a Markdown heading
- lines immediately before a Markdown bullet item (- )
"""
from __future__ import annotations

import re

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.metadata import chunk_metadata
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.ids import make_configured_chunk_id_for_document

SIVAS_BOUNDARY_RE = re.compile(
    r"(?<=[.!?;:])\s+|\n\n|\n(?=#{1,6}\s)|\n(?=-\s)"
)
SIVAS_CHARACTER_CHUNKER_VERSION = "sivas_character_v1"


class SivasCharacterChunker(BaseChunker):
    """Chunks documents using the exact SIVAS partner boundary regex and a
    character-count ceiling.  No overlap is applied.
    """

    def __init__(self, max_chars: int = 2048) -> None:
        self.max_chars = max_chars
        self.strategy_config = {
            "strategy": "sivas_character",
            "max_chars": max_chars,
            "splitter_regex": SIVAS_BOUNDARY_RE.pattern,
            "version": SIVAS_CHARACTER_CHUNKER_VERSION,
        }

    def chunk_documents(
        self,
        docs: list[DocumentRecord],
        show_progress: bool = False,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for doc in docs:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: DocumentRecord) -> list[ChunkRecord]:
        raw_segments = SIVAS_BOUNDARY_RE.split(doc.text or "")
        segments = [s for s in raw_segments if s.strip()]

        records: list[ChunkRecord] = []
        current_parts: list[str] = []
        current_chars: int = 0

        for segment in segments:
            if not current_parts:
                current_parts.append(segment)
                current_chars = len(segment)
            else:
                candidate_len = current_chars + 1 + len(segment)  # +1 for join sep
                if candidate_len <= self.max_chars:
                    current_parts.append(segment)
                    current_chars = candidate_len
                else:
                    records.append(self._make_chunk(doc, len(records), current_parts))
                    current_parts = [segment]
                    current_chars = len(segment)

        if current_parts:
            records.append(self._make_chunk(doc, len(records), current_parts))

        return records

    def _make_chunk(
        self,
        doc: DocumentRecord,
        chunk_index: int,
        parts: list[str],
    ) -> ChunkRecord:
        text = " ".join(parts).strip()
        chunk_id = make_configured_chunk_id_for_document(
            doc.document_id,
            chunk_index,
            text,
            self.strategy_config,
            doc.metadata,
        )
        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            original_context_id=doc.original_context_id,
            text=text,
            chunk_start=chunk_index,
            chunk_end=chunk_index + 1,
            metadata=chunk_metadata(
                doc.metadata,
                doc.document_id,
                doc.original_context_id,
                chunk_id,
                "sivas_character",
                "sivas_character",
                subset=doc.metadata.get("subset") if doc.metadata else None,
                split=(
                    (doc.metadata.get("split") or doc.metadata.get("source_split"))
                    if doc.metadata
                    else None
                ),
                chunk_index=chunk_index,
            ),
        )
