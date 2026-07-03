from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from requests import Timeout

logger = logging.getLogger("pipeline3.judge")


class OllamaClientError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float, timeout_seconds: int) -> None:
        self.model = model
        self.base_url = self._normalize_base_url(base_url)
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._generate_url = f"{self.base_url}/api/generate"
        self._show_url = f"{self.base_url}/api/show"

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        raw = base_url.strip().rstrip("/")
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Ollama base_url must start with http:// or https://, got: {base_url!r}"
            )
        return raw

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 2048,
            },
        }
        try:
            response = requests.post(
                self._generate_url,
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
        except Timeout as ex:
            raise OllamaClientError(
                f"Ollama request timed out after {self.timeout_seconds}s"
            ) from ex
        except requests.RequestException as ex:
            raise OllamaClientError(f"Ollama request failed: {ex}") from ex
        try:
            response.raise_for_status()
        except requests.HTTPError as ex:
            raise OllamaClientError(f"Ollama HTTP error: {ex}") from ex
        data = response.json()
        return data.get("response", "").strip()

    def get_model_info(self) -> dict:
        try:
            response = requests.post(
                self._show_url,
                json={"name": self.model},
                timeout=(10, 30),
            )
            response.raise_for_status()
            return response.json()
        except Exception as ex:
            logger.warning("Could not retrieve model info for %s: %s", self.model, ex)
            return {}
