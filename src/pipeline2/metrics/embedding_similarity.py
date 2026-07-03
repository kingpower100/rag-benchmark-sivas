from __future__ import annotations

import hashlib
import math
import re
from importlib import metadata
from functools import lru_cache
from typing import Protocol


_TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ0-9]+", re.UNICODE)


class AnswerEmbedder(Protocol):
    metric_name: str
    is_semantic: bool

    def encode(self, text: str) -> list[float]:
        ...


class DeterministicHashEmbedder:
    # Not a semantic model — produces a bag-of-words random projection via BLAKE2B hashing.
    # Results are reported under hashed_embedding_cosine_similarity, not embedding_similarity.
    # The metric approximates BOW cosine similarity but is NOT a lexical overlap metric.
    metric_name: str = "hashed_embedding_cosine_similarity"
    is_semantic: bool = False

    def __init__(self, model_name: str = "hashing-bow-v1", dimensions: int = 256) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.device = "cpu"

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.blake2b(f"{self.model_name}:{token}".encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if int.from_bytes(digest[4:], "big") % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector


class SentenceTransformerAnswerEmbedder:
    metric_name: str = "embedding_similarity"
    is_semantic: bool = True

    def __init__(self, model_name: str, device: str = "cuda", require_cuda: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.requested_device = device
        self.require_cuda = require_cuda
        self._validate_device_selection(device)
        self.model = SentenceTransformer(model_name, device=device)
        self.device = str(getattr(self.model, "device", "unknown"))

    def encode(self, text: str) -> list[float]:
        # normalize_embeddings=True is required for cosine similarity to be in [-1, 1]
        return self.model.encode([text or ""], normalize_embeddings=True)[0].tolist()


    def _validate_device_selection(self, device: str) -> None:
        requested_cuda = str(device).startswith("cuda")
        if self.require_cuda and not requested_cuda:
            raise RuntimeError(
                "embedding_similarity.require_cuda=true requires embedding_similarity.device to be cuda or cuda:N"
            )
        if requested_cuda or self.require_cuda:
            try:
                import torch
            except Exception as ex:
                raise RuntimeError(
                    f"embedding similarity requires CUDA but torch could not be imported: {ex}"
                ) from ex
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                raise RuntimeError(
                    "embedding similarity requested CUDA but torch reports no available CUDA device"
                )


@lru_cache(maxsize=8)
def build_answer_embedder(
    provider: str,
    model_name: str,
    dimensions: int,
    device: str = "cuda",
    require_cuda: bool = True,
) -> AnswerEmbedder:
    if provider == "sentence_transformers":
        return SentenceTransformerAnswerEmbedder(model_name, device=device, require_cuda=require_cuda)
    if provider == "deterministic_hash":
        return DeterministicHashEmbedder(model_name=model_name, dimensions=dimensions)
    raise ValueError(f"Unsupported embedding similarity provider: {provider!r}")


def compute_embedding_similarity(
    generated_answer: str,
    ground_truth_answer: str,
    embedder: AnswerEmbedder,
) -> float:
    if not (generated_answer or "").strip() or not (ground_truth_answer or "").strip():
        return 0.0
    return cosine_similarity(embedder.encode(generated_answer), embedder.encode(ground_truth_answer))


def embedding_model_metadata(
    provider: str,
    model_name: str,
    embedder: "AnswerEmbedder | None",
    offline_mode: bool = False,
) -> dict:
    return {
        "provider": provider,
        "model_name": model_name,
        "is_semantic": provider == "sentence_transformers",
        "offline_mode": offline_mode,
        "sentence_transformers_version": (
            _package_version("sentence-transformers") if provider == "sentence_transformers" else "n/a"
        ),
        "model_revision": "unknown",
        "requested_device": str(getattr(embedder, "requested_device", "unknown")) if embedder is not None else "unknown",
        "require_cuda": bool(getattr(embedder, "require_cuda", False)) if embedder is not None else False,
        "device_used": str(getattr(embedder, "device", "unknown")) if embedder is not None else "unknown",
    }


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
