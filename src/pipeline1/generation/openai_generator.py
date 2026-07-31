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

# Models / model families whose temperature parameter the API rejects.
# Matching rule: exact name OR name that starts with "<prefix>-".
# Do NOT use split("-")[0] — it is too lossy (e.g. "gpt-4.1" would wrongly
# match a hypothetical "gpt" prefix).
_NO_CUSTOM_TEMPERATURE_PREFIXES: tuple[str, ...] = (
    "gpt-5.5",
    "o1",
    "o3",
    "o4-mini",
)


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
            except requests.RequestException as exc:
                # Retry transient network errors (connection refused, timeout, etc.)
                if attempt < _MAX_RETRIES:
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
        finish_reason, answer = self._extract_answer(data, request_id)

        # Empty-answer rejection: a successful API call must produce visible text.
        if not answer or not answer.strip():
            raise RuntimeError(
                f"OpenAI returned an empty generated answer "
                f"(finish_reason={finish_reason!r}) for model={self.model_name} "
                f"request_id={request_id}"
            )

        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens") or count_tokens(prompt))
        completion_tokens = int(usage.get("completion_tokens") or count_tokens(answer))
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        # reasoning_tokens lives under usage.completion_tokens_details.reasoning_tokens
        reasoning_tokens: int | None = None
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            rt = details.get("reasoning_tokens")
            if rt is not None:
                reasoning_tokens = int(rt)

        completion_diagnostics: dict[str, Any] = {
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        return GenerationResult(
            answer=answer,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            completion_diagnostics=completion_diagnostics,
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

    def _parse_json_response(
        self, response: requests.Response, request_id: str | None
    ) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OpenAI response was not valid JSON for model={self.model_name} "
                f"request_id={request_id}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"OpenAI response JSON must be an object for model={self.model_name} "
                f"request_id={request_id}"
            )
        return data

    def _extract_answer(
        self, data: dict[str, Any], request_id: str | None
    ) -> tuple[str | None, str]:
        """Return (finish_reason, answer_text).

        Raises RuntimeError for structurally invalid or policy-blocked responses.
        A finish_reason of "length" logs a warning but does not raise — the caller
        is responsible for rejecting the resulting empty answer if that is the case.
        """
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"OpenAI response choices must be a non-empty list for model={self.model_name} "
                f"request_id={request_id}"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError(
                f"OpenAI response first choice must be an object for model={self.model_name} "
                f"request_id={request_id}"
            )
        finish_reason: str | None = first_choice.get("finish_reason")
        if finish_reason == "content_filter":
            raise RuntimeError(
                f"OpenAI response blocked by content filter for model={self.model_name} "
                f"request_id={request_id}"
            )
        if finish_reason == "length":
            logger.warning(
                "openai_response_truncated model=%s request_id=%s max_completion_tokens=%s — "
                "answer may be cut short; consider increasing max_tokens",
                self.model_name,
                request_id,
                self.max_tokens,
            )
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError(
                f"OpenAI response choice message must be an object for model={self.model_name} "
                f"request_id={request_id}"
            )
        if "content" not in message:
            raise RuntimeError(
                f"OpenAI response message is missing content for model={self.model_name} "
                f"request_id={request_id}"
            )
        content = message["content"]
        if content is None:
            raise RuntimeError(
                f"OpenAI response message content is null "
                f"(finish_reason={finish_reason!r}) for model={self.model_name} "
                f"request_id={request_id}"
            )
        if isinstance(content, str):
            return finish_reason, content.strip()
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
            return finish_reason, "".join(parts).strip()
        raise RuntimeError(
            "OpenAI response message content has unsupported type "
            f"{type(content).__name__} for model={self.model_name} request_id={request_id}"
        )


def _supports_custom_temperature(model_name: str) -> bool:
    """Return True when the model accepts a custom temperature parameter."""
    return not any(
        model_name == prefix or model_name.startswith(prefix + "-")
        for prefix in _NO_CUSTOM_TEMPERATURE_PREFIXES
    )
