"""Mistral Chat Completions API generator.

Reads the API key from the MISTRAL_API_KEY environment variable.
The key is never logged or written to any output.

API: https://api.mistral.ai/v1/chat/completions
"""
from __future__ import annotations

import os
import time

import requests

from src.pipeline1.generation.base import BaseGenerator, GenerationResult
from src.pipeline1.generation.token_counter import count_tokens

MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0


class MistralGenerator(BaseGenerator):
    """Calls the Mistral chat completions endpoint.

    Authentication is read from the MISTRAL_API_KEY environment variable.
    The constructor raises EnvironmentError immediately if the key is absent.

    Model name examples:
        orchestration: "mistral-small-latest"  (or "mistral-small")
        generation:    "mistral-medium-latest" (or "mistral-medium")

    Verify the exact model IDs available on your account at:
        https://docs.mistral.ai/getting-started/models/
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 90,
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
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> GenerationResult:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._call_api(prompt)
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
        raise RuntimeError(f"Mistral generation failed after {_MAX_RETRIES} attempts") from last_error

    def _call_api(self, prompt: str) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            MISTRAL_CHAT_URL,
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
        answer = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage", {})
        return GenerationResult(
            answer=answer,
            input_tokens=int(usage.get("prompt_tokens") or count_tokens(prompt)),
            output_tokens=int(usage.get("completion_tokens") or count_tokens(answer)),
        )
