from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.metadata import chunk_metadata
from src.pipeline1.utils.ids import make_chunk_id_for_document
from tqdm.auto import tqdm


FIXED_TOKEN_CHUNKER_VERSION = "fixed_token_v2_no_silent_fallback"


class FixedTokenChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        tokenizer_name: str = "cl100k_base",
        allow_word_fallback: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_name = tokenizer_name
        self.allow_word_fallback = allow_word_fallback
        self.encoding = self._load_encoding(tokenizer_name)
        if self.encoding is None and not allow_word_fallback:
            raise RuntimeError(
                f"Unable to load tokenizer '{tokenizer_name}' for fixed_token chunking. "
                "Install tiktoken or set chunking.allow_word_fallback=true to use explicit word fallback."
            )

    def chunk_documents(self, docs: list[DocumentRecord], show_progress: bool = False) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        iterator = tqdm(docs, desc="Chunking documents", unit="doc") if show_progress else docs
        for doc in iterator:
            units = self._encode_units(doc.text)
            chunk_index = 0
            for start in range(0, len(units), step):
                end = min(start + self.chunk_size, len(units))
                text = self._decode_units(units[start:end]).strip()
                if not text:
                    continue
                chunk_id = make_chunk_id_for_document(doc.document_id, start, end, text, doc.metadata, chunk_index)
                chunks.append(ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=text,
                    chunk_start=start,
                    chunk_end=end,
                    metadata=chunk_metadata(
                        doc.metadata,
                        doc.document_id,
                        doc.original_context_id,
                        chunk_id,
                        "fixed_token",
                        "token" if self.encoding is not None else "word_fallback",
                        tokenizer_name=self.tokenizer_name,
                    ),
                ))
                chunk_index += 1
                if end == len(units):
                    break
        return chunks

    @staticmethod
    def _load_encoding(tokenizer_name: str):
        try:
            import tiktoken

            return tiktoken.get_encoding(tokenizer_name)
        except Exception:
            return None

    def _encode_units(self, text: str) -> list[int] | list[str]:
        if self.encoding is None:
            return text.split()
        return self.encoding.encode(text)

    def _decode_units(self, units: list[int] | list[str]) -> str:
        if self.encoding is None:
            return " ".join(str(unit) for unit in units)
        return self.encoding.decode(units)
