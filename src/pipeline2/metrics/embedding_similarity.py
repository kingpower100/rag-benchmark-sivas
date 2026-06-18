from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Protocol


_TOKEN_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ0-9]+", re.UNICODE)


class AnswerEmbedder(Protocol):
    metric_name: str
    is_semantic: bool

    def encode(self, text: str) -> list[float]:
        ...


class DeterministicHashEmbedder:
    # Not a semantic model — produces a bag-of-words random projection.
    # Results are reported under bow_token_overlap_similarity, not embedding_similarity.
    metric_name: str = "bow_token_overlap_similarity"
    is_semantic: bool = False

    def __init__(self, model_name: str = "hashing-bow-v1", dimensions: int = 256) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

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

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> list[float]:
        # normalize_embeddings=True is required for cosine similarity to be in [-1, 1]
        return self.model.encode([text or ""], normalize_embeddings=True)[0].tolist()


@lru_cache(maxsize=8)
def build_answer_embedder(provider: str, model_name: str, dimensions: int) -> AnswerEmbedder:
    if provider == "sentence_transformers":
        return SentenceTransformerAnswerEmbedder(model_name)
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
