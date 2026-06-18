from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

from src.pipeline1.embedding.base import BaseEmbedder


class BGEEncoder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        device: str = "cpu",
        require_cuda: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.requested_device = device
        self.require_cuda = require_cuda
        self.cache_dir = cache_dir
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_path))
            os.environ.setdefault("HF_HOME", str(cache_path.parent / "huggingface"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path.parent / "huggingface"))
        if cache_dir:
            self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        else:
            self.model = SentenceTransformer(model_name, device=device)
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.runtime_device = self._resolve_runtime_device()
        self.embedding_tensor_device = self._probe_embedding_tensor_device()
        self._validate_device_selection()

    def encode_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        outputs = []
        total_batches = max(1, math.ceil(len(texts) / self.batch_size))
        start_time = time.perf_counter()
        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            batch_start = time.perf_counter()
            batch_embeddings = self.model.encode(
                batch,
                batch_size=min(self.batch_size, len(batch)),
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
            outputs.append(np.asarray(batch_embeddings))
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            processed = min(batch_index * self.batch_size, len(texts))
            batches_per_sec = batch_index / elapsed
            chunks_per_sec = processed / elapsed
            remaining_batches = total_batches - batch_index
            eta_s = remaining_batches / max(batches_per_sec, 1e-9)
            print(
                "[embedding] "
                f"device={self.embedding_tensor_device} "
                f"batch={batch_index}/{total_batches} "
                f"batch_ms={(time.perf_counter() - batch_start) * 1000:.1f} "
                f"batches/sec={batches_per_sec:.2f} "
                f"chunks/sec={chunks_per_sec:.2f} "
                f"eta_s={eta_s:.1f}"
            )
            if show_progress:
                pass
        return np.vstack(outputs)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def _resolve_runtime_device(self) -> str:
        device = getattr(self.model, "device", None)
        if device is not None:
            return str(device)
        if hasattr(self.model, "model"):
            try:
                parameter = next(self.model.model.parameters())
                return str(parameter.device)
            except Exception:
                pass
        target = getattr(self.model, "_target_device", None)
        if target is not None:
            return str(target)
        return str(self.requested_device)

    def _probe_embedding_tensor_device(self) -> str:
        try:
            tensor = self.model.encode(["cuda_probe"], convert_to_tensor=True, show_progress_bar=False)
            return str(tensor.device)
        except Exception:
            return self.runtime_device

    def _validate_device_selection(self) -> None:
        if self.require_cuda and not str(self.requested_device).startswith("cuda"):
            raise RuntimeError(
                f"embedding.require_cuda=true requires a CUDA embedding device, got requested_device={self.requested_device!r}"
            )
        requested_cuda = str(self.requested_device).startswith("cuda")
        runtime_cuda = str(self.runtime_device).startswith("cuda") or str(self.embedding_tensor_device).startswith("cuda")
        if requested_cuda and not runtime_cuda:
            message = (
                f"SentenceTransformer requested device={self.requested_device!r} but runtime device resolved to "
                f"{self.runtime_device!r} and embedding tensor device resolved to {self.embedding_tensor_device!r}."
            )
            if self.require_cuda:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
