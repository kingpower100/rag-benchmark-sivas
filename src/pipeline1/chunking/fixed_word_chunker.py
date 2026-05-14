from src.pipeline1.chunking.base import BaseChunker
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.utils.ids import make_chunk_id
from tqdm.auto import tqdm


FIXED_WORD_CHUNKER_VERSION = "fixed_word_v1"


class FixedWordChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: list[DocumentRecord], show_progress: bool = False) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        iterator = tqdm(docs, desc="Chunking documents", unit="doc") if show_progress else docs
        for doc in iterator:
            words = doc.text.split()
            for start in range(0, len(words), step):
                end = min(start + self.chunk_size, len(words))
                text = " ".join(words[start:end]).strip()
                if not text:
                    continue
                chunks.append(ChunkRecord(
                    chunk_id=make_chunk_id(doc.document_id, start, end, text),
                    document_id=doc.document_id,
                    original_context_id=doc.original_context_id,
                    text=text,
                    chunk_start=start,
                    chunk_end=end,
                    metadata={
                        **dict(doc.metadata),
                        "chunk_unit": "word",
                        "chunk_strategy": "fixed_word",
                    },
                ))
                if end == len(words):
                    break
        return chunks
