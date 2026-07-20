from __future__ import annotations

import re
from typing import Literal

from tqdm.auto import tqdm

from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.metadata import chunk_metadata
from src.pipeline1.utils.ids import make_configured_chunk_id_for_document


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
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        chunk_size_unit: Literal["tokens", "words", "sentences", "characters"] = "words",
        chunk_overlap_unit: Literal["tokens", "words", "sentences", "characters"] = "sentences",
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_size_unit = chunk_size_unit
        self.chunk_overlap_unit = chunk_overlap_unit
        self.tokenizer_name = tokenizer_name
        self._encoder = _load_token_encoder(tokenizer_name) if "tokens" in {chunk_size_unit, chunk_overlap_unit} else None
        self.strategy_config = {
            "strategy": "sentence",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_size_unit": chunk_size_unit,
            "chunk_overlap_unit": chunk_overlap_unit,
            "tokenizer_name": tokenizer_name if "tokens" in {chunk_size_unit, chunk_overlap_unit} else None,
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
            chunk_id = make_configured_chunk_id_for_document(
                doc.document_id,
                chunk_index,
                text,
                self.strategy_config,
                doc.metadata,
            )
            records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=text,
                    chunk_start=chunk_index,
                    chunk_end=chunk_index + len(group),
                    metadata=chunk_metadata(
                        doc.metadata,
                        doc.document_id,
                        doc.original_context_id,
                        chunk_id,
                        "sentence",
                        unit,
                        subset=doc.metadata.get("subset"),
                        split=doc.metadata.get("split") or doc.metadata.get("source_split"),
                        chunk_index=chunk_index,
                        chunk_size_unit=self.chunk_size_unit,
                        chunk_overlap_unit=self.chunk_overlap_unit,
                    ),
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
            current_size = 0
            idx = start
            while idx < len(sentences):
                sentence_size = _measure(sentences[idx], self.chunk_size_unit, self._encoder)
                if current and current_size + sentence_size > self.chunk_size:
                    break
                current.append(sentences[idx])
                current_size += sentence_size
                idx += 1
                if len(current) == 1 and sentence_size > self.chunk_size:
                    break
            groups.append(current)
            if idx >= len(sentences):
                break
            overlap_sentences = self._overlap_sentence_count(current)
            start = idx - overlap_sentences if overlap_sentences else idx
        return groups

    def _overlap_sentence_count(self, current: list[str]) -> int:
        if self.chunk_overlap == 0 or len(current) <= 1:
            return 0
        if self.chunk_overlap_unit == "sentences":
            return min(self.chunk_overlap, len(current) - 1)

        total = 0
        selected = 0
        for sentence in reversed(current):
            sentence_size = _measure(sentence, self.chunk_overlap_unit, self._encoder)
            if selected and total + sentence_size > self.chunk_overlap:
                break
            if not selected and sentence_size > self.chunk_overlap:
                selected = 1
                break
            total += sentence_size
            selected += 1
            if selected >= len(current) - 1:
                break
        return min(selected, len(current) - 1)


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


def _measure(text: str, unit: str, encoder) -> int:
    if unit == "words":
        return _word_count(text)
    if unit == "characters":
        return len(text)
    if unit == "sentences":
        return 1 if text.strip() else 0
    if unit == "tokens":
        if encoder is None:
            raise ValueError("Token-based sentence chunking requires a valid tokenizer.")
        return len(encoder.encode(text or ""))
    raise ValueError(f"Unsupported sentence chunk unit: {unit!r}")


def _load_token_encoder(tokenizer_name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(tokenizer_name)
    except Exception as ex:
        raise ValueError(
            f"Sentence chunking token units require a valid tiktoken encoding; got {tokenizer_name!r}."
        ) from ex
