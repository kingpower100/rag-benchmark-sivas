"""OpenAI Chat Completions API generator.

Reads the API key from the OPENAI_API_KEY environment variable.
The key is never logged or written to any output.

API: https://api.openai.com/v1/chat/completions
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any

import requests

from src.pipeline1.generation.base import BaseGenerator, GenerationResult
from src.pipeline1.generation.token_counter import count_tokens

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0
logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseGenerator):
    """Calls the OpenAI chat completions endpoint.

    Authentication is read from the OPENAI_API_KEY environment variable.
    The constructor raises EnvironmentError immediately if the key is absent.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: int = 90,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export it before starting the pipeline: "
                "export OPENAI_API_KEY='<your-key>'"
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
        raise RuntimeError(f"OpenAI generation failed after {_MAX_RETRIES} attempts") from last_error

    def _call_api(self, prompt: str) -> GenerationResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload, temperature_diagnostics = self._build_payload(prompt)
        logger.info(
            "openai_request_temperature model=%s configured_temperature=%s "
            "effective_api_temperature=%s temperature_omitted=%s",
            self.model_name,
            temperature_diagnostics["configured_temperature"],
            temperature_diagnostics["effective_api_temperature"],
            temperature_diagnostics["temperature_omitted"],
        )
        response = requests.post(
            OPENAI_CHAT_URL,
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        request_id = response.headers.get("x-request-id")
        if response.status_code >= 400:
            logger.error(
                "openai_request_failed status=%s request_id=%s model=%s body=%s",
                response.status_code,
                request_id,
                self.model_name,
                response.text[:4000],
            )
        if response.status_code == 401:
            raise EnvironmentError(
                "OpenAI API authentication failed. "
                "Verify that OPENAI_API_KEY is correct and active."
            )
        response.raise_for_status()
        data = self._parse_json_response(response, request_id)
        answer = self._extract_answer(data, request_id)
        usage = data.get("usage", {})
        return GenerationResult(
            answer=answer,
            input_tokens=int(usage.get("prompt_tokens") or count_tokens(prompt)),
            output_tokens=int(usage.get("completion_tokens") or count_tokens(answer)),
        )

    def _build_payload(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        configured_temperature = float(self.temperature)
        temperature_omitted = False
        effective_api_temperature = configured_temperature
        if _supports_custom_temperature(self.model_name):
            payload["temperature"] = configured_temperature
        else:
            temperature_omitted = True
            effective_api_temperature = 1.0
        return payload, {
            "configured_temperature": configured_temperature,
            "effective_api_temperature": effective_api_temperature,
            "temperature_omitted": temperature_omitted,
        }

    def _parse_json_response(self, response: requests.Response, request_id: str | None) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OpenAI response was not valid JSON for model={self.model_name} request_id={request_id}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"OpenAI response JSON must be an object for model={self.model_name} request_id={request_id}"
            )
        return data

    def _extract_answer(self, data: dict[str, Any], request_id: str | None) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"OpenAI response choices must be a non-empty list for model={self.model_name} request_id={request_id}"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(
                f"OpenAI response first choice must be an object for model={self.model_name} request_id={request_id}"
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(
                f"OpenAI response choice message must be an object for model={self.model_name} request_id={request_id}"
            )
        if "content" not in message:
            raise RuntimeError(
                f"OpenAI response message is missing content for model={self.model_name} request_id={request_id}"
            )
        content = message["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    raise RuntimeError(
                        "OpenAI response message content contains an unsupported part "
                        f"for model={self.model_name} request_id={request_id}"
                    )
            return "".join(parts).strip()
        raise RuntimeError(
            "OpenAI response message content has unsupported type "
            f"{type(content).__name__} for model={self.model_name} request_id={request_id}"
        )


def _supports_custom_temperature(model_name: str) -> bool:
    default_temperature_only_models = {"gpt-5.5"}
    return model_name not in default_temperature_only_models
