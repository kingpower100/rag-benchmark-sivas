from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline1.chunking.fixed_token_chunker import FIXED_TOKEN_CHUNKER_VERSION, FixedTokenChunker
from src.pipeline1.chunking.fixed_word_chunker import FIXED_WORD_CHUNKER_VERSION, FixedWordChunker
from src.pipeline1.chunking.sentence_chunker import SENTENCE_CHUNKER_VERSION, SENTENCE_SPLITTER_VERSION, SentenceChunker
from src.pipeline1.chunking.table_aware_chunker import TABLE_AWARE_CHUNKER_VERSION, TableAwareChunker
from src.pipeline1.io.jsonl_reader import list_txt_files
from src.pipeline1.metadata import METADATA_SCHEMA_VERSION
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.document import DocumentRecord
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput
from src.pipeline1.utils.hashing import file_sha256, stable_hash_dict


@dataclass(frozen=True)
class ChunkingStageOutput(StageOutput):
    chunks: list[ChunkRecord] = None
    chunks_key: str = ""
    chunks_path: Path | None = None
    chunker_versions: dict | None = None
    documents_fingerprint: str = ""
    chunk_diagnostics: dict | None = None
    cache_status: str = ""


class ChunkingStage(BaseStage):
    stage_name = "chunking"

    def __init__(
        self,
        cfg: PipelineConfig,
        project_root: Path,
        cache_dir: Path,
        docs_path: Path,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.cache_dir = cache_dir
        self.docs_path = docs_path
        self.logger = logger

    def run(self, stage_input: StageInput) -> ChunkingStageOutput:
        docs = stage_input.payload["documents"]
        chunker_versions = self.chunker_versions()
        documents_fingerprint = self.documents_fingerprint()
        chunks_key = stable_hash_dict(
            {
                "documents_fingerprint": documents_fingerprint,
                "documents_source_type": self.cfg.data.documents_source_type,
                "documents_file_glob": self.cfg.data.documents_file_glob,
                "documents_recursive": self.cfg.data.documents_recursive,
                "document_text_field": self.cfg.data.document_text_field,
                "allow_document_text_fallback": self.cfg.data.allow_document_text_fallback,
                "metadata_schema_version": METADATA_SCHEMA_VERSION,
                "chunking": self.cfg.chunking.model_dump(),
                "chunker_versions": chunker_versions,
            }
        )
        chunks_path = self.cache_dir / "chunks" / f"{chunks_key}.jsonl"
        chunks = self.load_chunks(chunks_path)
        if chunks is None:
            chunks = self.build_chunker().chunk_documents(docs, show_progress=True)
            self.save_chunks(chunks_path, chunks)
            cache_status = "built"
        else:
            if self.logger:
                self.logger.info("Loaded cached chunks: %s", chunks_path)
            cache_status = "loaded"
        chunk_diagnostics = self.chunk_diagnostics(chunks)
        return ChunkingStageOutput(
            stage_name=self.stage_name,
            artifacts={"chunks": chunks, "chunks_path": chunks_path},
            diagnostics=chunk_diagnostics,
            metadata={
                "chunks_key": chunks_key,
                "documents_fingerprint": documents_fingerprint,
                "chunker_versions": chunker_versions,
                "cache_status": cache_status,
            },
            chunks=chunks,
            chunks_key=chunks_key,
            chunks_path=chunks_path,
            chunker_versions=chunker_versions,
            documents_fingerprint=documents_fingerprint,
            chunk_diagnostics=chunk_diagnostics,
            cache_status=cache_status,
        )

    def documents_fingerprint(self) -> str:
        if self.cfg.data.documents_source_type == "jsonl":
            return file_sha256(self.docs_path)
        files = list_txt_files(
            self.docs_path,
            self.cfg.data.documents_file_glob,
            self.cfg.data.documents_recursive,
        )
        return stable_hash_dict(
            {
                "source_type": "txt_folder",
                "folder_path": str(self.docs_path),
                "file_glob": self.cfg.data.documents_file_glob,
                "recursive": self.cfg.data.documents_recursive,
                "files": [
                    {
                        "path": path.relative_to(self.docs_path).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in files
                ],
            }
        )

    def build_chunker(self):
        if self.cfg.chunking.strategy == "fixed_token":
            return FixedTokenChunker(
                self.cfg.chunking.chunk_size,
                self.cfg.chunking.chunk_overlap,
                self.cfg.chunking.tokenizer_name,
                self.cfg.chunking.allow_word_fallback,
            )
        if self.cfg.chunking.strategy == "fixed_word":
            return FixedWordChunker(self.cfg.chunking.chunk_size, self.cfg.chunking.chunk_overlap)
        if self.cfg.chunking.strategy == "sentence":
            print("Using sentence-aware chunking with regex sentence boundaries and full-sentence overlap.")
            return SentenceChunker(self.cfg.chunking.chunk_size, self.cfg.chunking.chunk_overlap)
        print("Using table-aware chunking that keeps markdown tables intact when possible.")
        return TableAwareChunker(
            self.cfg.chunking.chunk_size,
            self.cfg.chunking.chunk_overlap,
            self.cfg.chunking.max_chunk_chars,
            self.cfg.chunking.max_chunk_tokens,
            self.cfg.chunking.oversized_chunk_policy,
            self.cfg.chunking.oversized_chunk_warning,
        )

    def chunker_versions(self) -> dict[str, str]:
        versions = {"chunker_implementation": ""}
        if self.cfg.chunking.strategy == "fixed_token":
            versions["chunker_implementation"] = FIXED_TOKEN_CHUNKER_VERSION
        elif self.cfg.chunking.strategy == "fixed_word":
            versions["chunker_implementation"] = FIXED_WORD_CHUNKER_VERSION
        elif self.cfg.chunking.strategy == "sentence":
            versions["chunker_implementation"] = SENTENCE_CHUNKER_VERSION
            versions["sentence_splitter"] = SENTENCE_SPLITTER_VERSION
        else:
            versions["chunker_implementation"] = TABLE_AWARE_CHUNKER_VERSION
        return versions

    def chunk_diagnostics(self, chunks: list[ChunkRecord]) -> dict[str, int]:
        return {
            "total_chunks": len(chunks),
            "empty_chunks": sum(1 for chunk in chunks if not chunk.text.strip()),
            "over_max_chunk_chars": sum(1 for chunk in chunks if len(chunk.text) > self.cfg.chunking.max_chunk_chars),
            "over_max_chunk_tokens": sum(1 for chunk in chunks if len(chunk.text.split()) > self.cfg.chunking.max_chunk_tokens),
            "max_chunk_chars_observed": max((len(chunk.text) for chunk in chunks), default=0),
            "max_chunk_tokens_observed": max((len(chunk.text.split()) for chunk in chunks), default=0),
        }

    @staticmethod
    def load_chunks(path: Path) -> list[ChunkRecord] | None:
        if not path.exists():
            return None
        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(ChunkRecord.model_validate_json(line))
        return chunks

    @staticmethod
    def save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")
