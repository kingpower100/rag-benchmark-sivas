from __future__ import annotations

import re

from tqdm.auto import tqdm

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.ids import make_configured_chunk_id


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
SENTENCE_SPLITTER_VERSION = "regex_v1"
SENTENCE_CHUNKER_VERSION = "sentence_v2_regex_boundaries"
_COMMON_ABBREVIATIONS = {
    "co.",
    "corp.",
    "dr.",
    "e.g.",
    "fig.",
    "inc.",
    "jr.",
    "ltd.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "prof.",
    "sr.",
    "u.k.",
    "u.s.",
    "vs.",
}


class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy_config = {
            "strategy": "sentence",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "splitter": SENTENCE_SPLITTER_VERSION,
        }

    def chunk_documents(self, docs: list[DocumentRecord], show_progress: bool = False) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        iterator = tqdm(docs, desc="Sentence chunking documents", unit="doc") if show_progress else docs
        for doc in iterator:
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: DocumentRecord) -> list[ChunkRecord]:
        sentences = split_sentences(doc.text)
        unit = "sentence" if len(sentences) > 1 else "sentence_fallback"
        grouped = self._group_sentences(sentences)
        records: list[ChunkRecord] = []
        for chunk_index, group in enumerate(grouped):
            text = " ".join(group).strip()
            if not text:
                continue
            records.append(
                ChunkRecord(
                    chunk_id=make_configured_chunk_id(doc.document_id, chunk_index, text, self.strategy_config),
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=text,
                    chunk_start=chunk_index,
                    chunk_end=chunk_index + len(group),
                    metadata={
                        **dict(doc.metadata),
                        "doc_id": doc.document_id,
                        "original_context_id": doc.original_context_id,
                        "source_file": doc.metadata.get("source_file") or doc.metadata.get("file_name"),
                        "subset": doc.metadata.get("subset"),
                        "split": doc.metadata.get("split") or doc.metadata.get("source_split"),
                        "chunk_id": make_configured_chunk_id(doc.document_id, chunk_index, text, self.strategy_config),
                        "chunk_index": chunk_index,
                        "chunk_strategy": "sentence",
                        "chunk_unit": unit,
                    },
                )
            )
        return records

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        if not sentences:
            return []
        groups: list[list[str]] = []
        start = 0
        while start < len(sentences):
            current: list[str] = []
            current_words = 0
            idx = start
            while idx < len(sentences):
                sentence_words = _word_count(sentences[idx])
                if current and current_words + sentence_words > self.chunk_size:
                    break
                current.append(sentences[idx])
                current_words += sentence_words
                idx += 1
            groups.append(current)
            if idx >= len(sentences):
                break
            overlap = min(self.chunk_overlap, max(0, len(current) - 1))
            start = idx - overlap if overlap else idx
        return groups


def split_sentences(text: str) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    pieces = _SENTENCE_BOUNDARY_RE.split(normalized)
    sentences: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _is_common_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _is_common_abbreviation(text: str) -> bool:
    tail = text.strip().split()[-1].lower() if text.strip() else ""
    return tail in _COMMON_ABBREVIATIONS


def _word_count(text: str) -> int:
    return len(text.split())
