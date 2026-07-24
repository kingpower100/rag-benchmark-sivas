"""Mistral Embed API embedding provider.

Reads the API key from the MISTRAL_API_KEY environment variable.
The key is never logged or written to any output.

Model: mistral-embed (1024 dimensions, cosine similarity)
API:   https://api.mistral.ai/v1/embeddings
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np
import requests

from src.pipeline1.embedding.base import BaseEmbedder

MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0
_SUPPORTED_MODEL = "mistral-embed"
_EXPECTED_DIMENSION = 1024


class MistralEmbedder(BaseEmbedder):
    """Calls the Mistral Embed API to produce text embeddings.

    Authentication is read from the MISTRAL_API_KEY environment variable.
    The constructor raises EnvironmentError immediately if the key is absent
    so mis-configuration surfaces at startup rather than at the first call.
    """

    def __init__(
        self,
        model_name: str = "mistral-embed",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        timeout_s: int = 60,
        max_retries: int = _MAX_RETRIES,
        retry_backoff_base_s: float = _RETRY_BACKOFF_BASE,
        api_base_url: str | None = None,
        expected_dimension: int | None = _EXPECTED_DIMENSION,
    ) -> None:
        if model_name != _SUPPORTED_MODEL:
            raise ValueError(f"MistralEmbedder supports only model_name='{_SUPPORTED_MODEL}', got {model_name!r}.")
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Export it before starting the pipeline: "
                "export MISTRAL_API_KEY='<your-key>'"
            )
        self._api_key = api_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_base_s = retry_backoff_base_s
        self.api_base_url = (api_base_url or MISTRAL_EMBEDDINGS_URL).rstrip("/")
        if self.api_base_url != MISTRAL_EMBEDDINGS_URL:
            raise ValueError("MistralEmbedder only supports the official embeddings endpoint.")
        self.expected_dimension = expected_dimension
        self.client_library = f"requests/{requests.__version__}"

    def encode_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: list[list[float]] = []
        total_batches = math.ceil(len(texts) / self.batch_size)

        for batch_index, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(embeddings)
            if show_progress:
                print(
                    f"[mistral-embed] batch={batch_index}/{total_batches} "
                    f"texts={start + len(batch)}/{len(texts)}"
                )

        embeddings = np.array(all_embeddings, dtype=np.float32)
        self._validate_matrix(embeddings, len(texts))
        return embeddings

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._embed_batch(texts)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in _RETRY_STATUSES and attempt < self.max_retries:
                    time.sleep(self.retry_backoff_base_s * (2 ** (attempt - 1)))
                    last_error = exc
                    continue
                raise
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_base_s * (2 ** (attempt - 1)))
                    last_error = exc
                    continue
                raise
        raise RuntimeError(f"Mistral embedding failed after {self.max_retries} attempts") from last_error

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(
            self.api_base_url,
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        if response.status_code == 401:
            raise EnvironmentError(
                "Mistral API authentication failed. "
                "Verify that MISTRAL_API_KEY is correct and active."
            )
        response.raise_for_status()
        data = response.json()
        return self._extract_embeddings(data, len(texts))

    def _extract_embeddings(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError("Malformed Mistral embeddings response: missing data list.")
        if len(items) != expected_count:
            raise RuntimeError(
                f"Malformed Mistral embeddings response: expected {expected_count} embeddings, got {len(items)}."
            )
        seen_indices = set()
        embeddings_by_index: dict[int, list[float]] = {}
        for item in items:
            if not isinstance(item, dict) or "index" not in item or "embedding" not in item:
                raise RuntimeError("Malformed Mistral embeddings response item.")
            index = item["index"]
            if not isinstance(index, int) or index < 0 or index >= expected_count:
                raise RuntimeError(f"Malformed Mistral embeddings response index: {index!r}.")
            if index in seen_indices:
                raise RuntimeError(f"Malformed Mistral embeddings response: duplicate index {index}.")
            seen_indices.add(index)
            vector = item["embedding"]
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(f"Malformed Mistral embedding vector at index {index}.")
            try:
                arr = np.asarray(vector, dtype=np.float32)
            except (TypeError, ValueError) as ex:
                raise RuntimeError(f"Malformed Mistral embedding vector at index {index}.") from ex
            if arr.ndim != 1:
                raise RuntimeError(f"Malformed Mistral embedding vector at index {index}: expected 1D vector.")
            if self.expected_dimension is not None and int(arr.shape[0]) != self.expected_dimension:
                raise RuntimeError(
                    f"Mistral embedding dimension mismatch at index {index}: "
                    f"expected={self.expected_dimension} observed={int(arr.shape[0])}."
                )
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Malformed Mistral embedding vector at index {index}: non-finite values.")
            if self.normalize_embeddings:
                norm = float(np.linalg.norm(arr))
                if norm <= 0.0:
                    raise RuntimeError(f"Malformed Mistral embedding vector at index {index}: zero norm.")
                arr = arr / norm
            embeddings_by_index[index] = arr.astype("float32").tolist()
        missing = set(range(expected_count)).difference(seen_indices)
        if missing:
            raise RuntimeError(f"Malformed Mistral embeddings response: missing indices {sorted(missing)}.")
        return [embeddings_by_index[index] for index in range(expected_count)]

    def _validate_matrix(self, embeddings: np.ndarray, expected_count: int) -> None:
        if embeddings.ndim != 2:
            raise RuntimeError(f"Expected a 2D Mistral embedding matrix, got shape={embeddings.shape}.")
        if int(embeddings.shape[0]) != expected_count:
            raise RuntimeError(
                f"Mistral embedding row count mismatch: expected={expected_count} observed={int(embeddings.shape[0])}."
            )
        if self.expected_dimension is not None and int(embeddings.shape[1]) != self.expected_dimension:
            raise RuntimeError(
                f"Mistral embedding dimension mismatch: expected={self.expected_dimension} "
                f"observed={int(embeddings.shape[1])}."
            )
