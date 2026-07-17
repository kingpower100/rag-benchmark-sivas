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

import numpy as np
import requests

from src.pipeline1.embedding.base import BaseEmbedder

MISTRAL_EMBEDDINGS_URL = "https://api.mistral.ai/v1/embeddings"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0


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
        timeout_s: int = 60,
    ) -> None:
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
        self.timeout_s = timeout_s

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

        return np.array(all_embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._embed_batch(texts)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                    last_error = exc
                    continue
                raise
            except requests.RequestException as exc:
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                    last_error = exc
                    continue
                raise
        raise RuntimeError(f"Mistral embedding failed after {_MAX_RETRIES} attempts") from last_error

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(
            MISTRAL_EMBEDDINGS_URL,
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
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
