from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.pipeline1.embedding.cache import EmbeddingCache
from src.pipeline1.embedding.factory import build_embedder
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput
from src.pipeline1.utils.hashing import stable_hash_dict


@dataclass(frozen=True)
class EmbeddingStageOutput(StageOutput):
    embedder: object = None
    embeddings: object = None
    embeddings_key: str = ""
    embeddings_path: Path | None = None
    cache_status: str = ""


class EmbeddingStage(BaseStage):
    stage_name = "embedding"

    def __init__(
        self,
        cfg: PipelineConfig,
        cache_dir: Path,
        embedder_factory: Callable = build_embedder,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.embedder_factory = embedder_factory
        self.logger = logger

    def run(self, stage_input: StageInput) -> EmbeddingStageOutput:
        chunks: list[ChunkRecord] = stage_input.payload["chunks"]
        chunks_key = str(stage_input.payload["chunks_key"])
        embedder = self.embedder_factory(self.cfg.embedding)
        embeddings_key = self._cache_key_for(chunks_key, self.cfg.embedding.model_dump())
        embeddings_path = self.cache_dir / "embeddings" / f"{embeddings_key}.npy"
        embeddings = EmbeddingCache.load(embeddings_path)
        if embeddings is None:
            embeddings = embedder.encode_texts([chunk.text for chunk in chunks], show_progress=True)
            EmbeddingCache.save(embeddings_path, embeddings, {"chunks_key": chunks_key, "embedding": self.cfg.embedding.model_dump()})
            cache_status = "built"
        else:
            try:
                self.validate_embedding_cache(embeddings, len(chunks), embeddings_path, chunks_key, self.cfg.embedding.model_dump())
                cache_status = "validated"
            except RuntimeError:
                if self.cfg.runtime.cache_mismatch_policy != "rebuild":
                    raise
                embeddings_path.unlink(missing_ok=True)
                embeddings_path.with_suffix(embeddings_path.suffix + ".meta.json").unlink(missing_ok=True)
                embeddings = embedder.encode_texts([chunk.text for chunk in chunks], show_progress=True)
                EmbeddingCache.save(embeddings_path, embeddings, {"chunks_key": chunks_key, "embedding": self.cfg.embedding.model_dump()})
                cache_status = "rebuilt_after_mismatch"
            if self.logger:
                self.logger.info("Loaded cached embeddings: %s", embeddings_path)
        diagnostics = {
            "embedding_rows": int(embeddings.shape[0]) if len(embeddings.shape) > 0 else 0,
            "embedding_dim": int(embeddings.shape[1]) if len(embeddings.shape) > 1 else None,
            "cache_status": cache_status,
        }
        return EmbeddingStageOutput(
            stage_name=self.stage_name,
            artifacts={"embeddings": embeddings, "embeddings_path": embeddings_path, "embedder": embedder},
            diagnostics=diagnostics,
            metadata={"embeddings_key": embeddings_key, "cache_status": cache_status},
            embedder=embedder,
            embeddings=embeddings,
            embeddings_key=embeddings_key,
            embeddings_path=embeddings_path,
            cache_status=cache_status,
        )

    @staticmethod
    def _cache_key_for(chunks_key: str, embedding_config: dict) -> str:
        return stable_hash_dict(
            {
                "chunks_key": chunks_key,
                "embedding": embedding_config,
            }
        )

    @staticmethod
    def validate_embedding_cache(embeddings, chunk_count: int, path: Path, chunks_key: str, embedding_config: dict) -> None:
        if len(embeddings) != chunk_count:
            raise RuntimeError(f"Cached embeddings row count mismatch for {path}: embeddings={len(embeddings)} chunks={chunk_count}")
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not meta_path.exists():
            raise RuntimeError(f"Cached embeddings metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"chunks_key": chunks_key, "embedding": embedding_config}
        if meta != expected:
            raise RuntimeError(f"Cached embeddings metadata mismatch for {path}")
