"""SIVAS-compatible character-based chunker.

Implements the documented SIVAS boundary accumulation behavior with
offset-preserving source slicing:

1. Locate boundaries with the documented regex using ``re.finditer``.
2. Treat each logical segment plus its following boundary separator as one
   source span.
3. Accumulate contiguous source spans while the exact source substring length
   stays at or below ``max_chars`` (default 2048 characters).
4. When the next full source span would exceed the limit, emit the current
   exact source substring and start the next chunk at the previous end offset.
5. Emit the final source span unconditionally.
6. No overlap; chunks are ordered, contiguous, and reconstruct the input text.
7. All document metadata is forwarded to every chunk.

Oversized indivisible source spans are kept as one chunk, are not truncated,
and are marked in chunk metadata. A ``RuntimeWarning`` is emitted so diagnostics
surface the policy while preserving source text exactly.

The boundary regex is the exact requested SIVAS regex::

    (?<=[.!?;:])\\s+|\\n\\n|\\n(?=#{1,6}\\s)|\\n(?=-\\s)
"""
from __future__ import annotations

import re
import warnings

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.metadata import chunk_metadata
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.ids import make_configured_chunk_id_for_document

SIVAS_BOUNDARY_RE = re.compile(
    r"(?<=[.!?;:])\s+|\n\n|\n(?=#{1,6}\s)|\n(?=-\s)"
)
SIVAS_CHARACTER_CHUNKER_VERSION = "sivas_character_v2"


class SivasCharacterChunker(BaseChunker):
    """Chunk documents with exact source-text preservation and no overlap."""

    def __init__(self, max_chars: int = 2048) -> None:
        self.max_chars = max_chars
        self.strategy_config = {
            "strategy": "sivas_character",
            "max_chars": max_chars,
            "splitter_regex": SIVAS_BOUNDARY_RE.pattern,
            "version": SIVAS_CHARACTER_CHUNKER_VERSION,
            "oversized_chunk_policy": "warn_keep_whole",
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
        text = doc.text or ""
        if text == "":
            return []

        records: list[ChunkRecord] = []
        current_start: int | None = None
        current_end: int | None = None

        for span_start, span_end in self._source_spans(text):
            if current_start is None:
                current_start = span_start
                current_end = span_end
                continue

            assert current_end is not None
            if span_end - current_start <= self.max_chars:
                current_end = span_end
                continue

            records.append(
                self._make_chunk(doc, len(records), current_start, current_end)
            )
            current_start = current_end
            current_end = span_end

        if current_start is not None and current_end is not None:
            records.append(self._make_chunk(doc, len(records), current_start, current_end))

        return records

    @staticmethod
    def _source_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        for match in SIVAS_BOUNDARY_RE.finditer(text):
            end = match.end()
            if end > start:
                spans.append((start, end))
            start = end
        if start < len(text):
            spans.append((start, len(text)))
        return spans

    def _make_chunk(
        self,
        doc: DocumentRecord,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> ChunkRecord:
        text = doc.text[start_char:end_char]
        is_oversized = len(text) > self.max_chars
        if is_oversized:
            warnings.warn(
                "sivas_character emitted an oversized indivisible source span "
                f"for document_id={doc.document_id!r}, chunk_index={chunk_index}, "
                f"length={len(text)}, max_chars={self.max_chars}.",
                RuntimeWarning,
                stacklevel=2,
            )

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
            chunk_start=start_char,
            chunk_end=end_char,
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
                start_char=start_char,
                end_char=end_char,
                max_chunk_chars=self.max_chars,
                chunker_version=SIVAS_CHARACTER_CHUNKER_VERSION,
                oversized_chunk=is_oversized,
                oversized_chunk_policy="warn_keep_whole",
            ),
        )
