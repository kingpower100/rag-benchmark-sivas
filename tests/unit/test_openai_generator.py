import logging

import pytest
import requests

from src.pipeline1.generation.openai_generator import OpenAIGenerator


class _FakeResponse:
    def __init__(
        self,
        status_code=200,
        payload=None,
        text='{"choices":[{"message":{"content":"answer"}}]}',
        request_id="req_test",
        json_error=None,
    ):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }
        self.text = text
        self.headers = {"x-request-id": request_id} if request_id else {}
        self._json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_openai_payload_forwards_model_and_uses_max_completion_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    result = OpenAIGenerator("gpt-5.5", temperature=0.0, max_tokens=512, timeout_s=180).generate("Prompt")

    assert result.answer == "answer"
    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["max_completion_tokens"] == 512
    assert "max_tokens" not in captured["payload"]
    assert "temperature" not in captured["payload"]
    assert captured["headers"]["Authorization"] == "Bearer secret-test-key"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["timeout"] == 180


def test_openai_payload_sends_reasoning_effort_when_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    OpenAIGenerator("gpt-5.5", max_tokens=512, reasoning_effort="none").generate("Prompt")

    assert captured["payload"]["model"] == "gpt-5.5"
    assert captured["payload"]["max_completion_tokens"] == 512
    assert captured["payload"]["reasoning_effort"] == "none"


def test_openai_payload_omits_reasoning_effort_when_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert "reasoning_effort" not in captured["payload"]


def test_gpt55_temperature_zero_is_omitted_and_traced(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    with caplog.at_level(logging.INFO, logger="src.pipeline1.generation.openai_generator"):
        OpenAIGenerator("gpt-5.5", temperature=0.0).generate("Sensitive prompt")

    assert captured["payload"]["model"] == "gpt-5.5"
    assert "temperature" not in captured["payload"]
    assert captured["payload"]["max_completion_tokens"] == 512
    assert "configured_temperature=0.0" in caplog.text
    assert "effective_api_temperature=1.0" in caplog.text
    assert "temperature_omitted=True" in caplog.text
    assert "secret-test-key" not in caplog.text
    assert "Sensitive prompt" not in caplog.text


def test_gpt55_temperature_one_does_not_send_unsupported_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    OpenAIGenerator("gpt-5.5", temperature=1.0).generate("Prompt")

    assert captured["payload"]["model"] == "gpt-5.5"
    assert "temperature" not in captured["payload"] or captured["payload"]["temperature"] == 1.0
    assert captured["payload"]["max_completion_tokens"] == 512


def test_custom_temperature_supported_for_non_gpt55_models(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["payload"] = json
        return _FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    OpenAIGenerator("gpt-4.1", temperature=0.2).generate("Prompt")

    assert captured["payload"]["model"] == "gpt-4.1"
    assert captured["payload"]["temperature"] == 0.2
    assert captured["payload"]["max_completion_tokens"] == 512


def test_http_400_makes_exactly_one_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr("src.pipeline1.generation.openai_generator.time.sleep", lambda seconds: None)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(json)
        return _FakeResponse(status_code=400, text='{"error":{"message":"bad request"}}')

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises(requests.HTTPError):
        OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert len(calls) == 1


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_non_transient_client_errors_fail_immediately(monkeypatch, status_code):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr("src.pipeline1.generation.openai_generator.time.sleep", lambda seconds: None)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(json)
        return _FakeResponse(status_code=status_code, text='{"error":{"message":"client error"}}')

    monkeypatch.setattr("requests.post", fake_post)

    with pytest.raises((EnvironmentError, requests.HTTPError)):
        OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert len(calls) == 1


def test_http_429_is_retried_until_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr("src.pipeline1.generation.openai_generator.time.sleep", lambda seconds: None)
    responses = [
        _FakeResponse(status_code=429, text='{"error":{"message":"rate limited"}}'),
        _FakeResponse(status_code=429, text='{"error":{"message":"rate limited"}}'),
        _FakeResponse(payload={"choices": [{"message": {"content": "recovered"}}]}),
    ]

    def fake_post(url, json, headers, timeout):
        return responses.pop(0)

    monkeypatch.setattr("requests.post", fake_post)

    result = OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert result.answer == "recovered"
    assert responses == []


@pytest.mark.parametrize("status_code", [408, 409])
def test_408_and_409_are_transient(monkeypatch, status_code):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr("src.pipeline1.generation.openai_generator.time.sleep", lambda seconds: None)
    calls = []
    responses = [
        _FakeResponse(status_code=status_code, text='{"error":{"message":"transient"}}'),
        _FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]}),
    ]

    def fake_post(url, json, headers, timeout):
        calls.append(json)
        return responses.pop(0)

    monkeypatch.setattr("requests.post", fake_post)

    result = OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert result.answer == "ok"
    assert len(calls) == 2


def test_api_key_is_absent_from_error_logs(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")

    def fake_post(url, json, headers, timeout):
        return _FakeResponse(status_code=400, text='{"error":{"message":"bad request"}}')

    monkeypatch.setattr("requests.post", fake_post)

    with caplog.at_level(logging.ERROR, logger="src.pipeline1.generation.openai_generator"):
        with pytest.raises(requests.HTTPError):
            OpenAIGenerator("gpt-5.5").generate("Prompt")

    log_text = caplog.text
    assert "status=400" in log_text
    assert "request_id=req_test" in log_text
    assert "model=gpt-5.5" in log_text
    assert "bad request" in log_text
    assert "secret-test-key" not in log_text
    assert "Authorization" not in log_text
    assert "Prompt" not in log_text


def test_invalid_json_produces_descriptive_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(json_error=ValueError("not json")),
    )

    with pytest.raises(RuntimeError, match="not valid JSON.*model=gpt-5.5.*request_id=req_test"):
        OpenAIGenerator("gpt-5.5").generate("Prompt")


def test_empty_choices_produces_descriptive_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(payload={"choices": []}),
    )

    with pytest.raises(RuntimeError, match="choices must be a non-empty list.*model=gpt-5.5.*request_id=req_test"):
        OpenAIGenerator("gpt-5.5").generate("Prompt")


def test_valid_response_returns_answer_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"message": {"content": "  final answer  "}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
        ),
    )

    result = OpenAIGenerator("gpt-5.5").generate("Prompt")

    assert result.answer == "final answer"
    assert result.input_tokens == 10
    assert result.output_tokens == 4


# ── Empty-answer rejection ────────────────────────────────────────────────────

def test_empty_content_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={"choices": [{"message": {"content": ""}}]}
        ),
    )

    with pytest.raises(RuntimeError, match="empty generated answer"):
        OpenAIGenerator("gpt-4.1").generate("Prompt")


def test_empty_length_response_still_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        ),
    )

    with pytest.raises(RuntimeError, match="empty generated answer.*finish_reason='length'"):
        OpenAIGenerator("gpt-5.5", max_tokens=512, reasoning_effort="none").generate("Prompt")


def test_whitespace_only_content_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={"choices": [{"message": {"content": "   \n\t  "}}]}
        ),
    )

    with pytest.raises(RuntimeError, match="empty generated answer"):
        OpenAIGenerator("gpt-4.1").generate("Prompt")


def test_unknown_text_is_not_treated_as_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"message": {"content": "UNKNOWN"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        ),
    )

    result = OpenAIGenerator("gpt-4.1").generate("Prompt")
    assert result.answer == "UNKNOWN"


def test_content_filter_finish_reason_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={"choices": [{"finish_reason": "content_filter", "message": {"content": None}}]}
        ),
    )

    with pytest.raises(RuntimeError, match="content filter"):
        OpenAIGenerator("gpt-4.1").generate("Prompt")


def test_null_content_raises_descriptive_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={"choices": [{"message": {"content": None}}]}
        ),
    )

    with pytest.raises(RuntimeError, match="content is null"):
        OpenAIGenerator("gpt-4.1").generate("Prompt")


# ── Completion diagnostics ────────────────────────────────────────────────────

def test_finish_reason_is_recorded_in_completion_diagnostics(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ),
    )

    result = OpenAIGenerator("gpt-4.1").generate("Prompt")
    diag = result.completion_diagnostics
    assert diag is not None
    assert diag["finish_reason"] == "stop"
    assert diag["prompt_tokens"] == 5
    assert diag["completion_tokens"] == 1
    assert diag["total_tokens"] == 6
    assert diag["reasoning_tokens"] is None


def test_reasoning_tokens_recorded_when_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 8},
                },
            }
        ),
    )

    result = OpenAIGenerator("gpt-4.1").generate("Prompt")
    assert result.completion_diagnostics["reasoning_tokens"] == 8


def test_reasoning_tokens_none_when_absent(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        ),
    )

    result = OpenAIGenerator("gpt-4.1").generate("Prompt")
    assert result.completion_diagnostics["reasoning_tokens"] is None


def test_length_finish_reason_logs_warning_and_keeps_answer(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "requests.post",
        lambda url, json, headers, timeout: _FakeResponse(
            payload={
                "choices": [{"finish_reason": "length", "message": {"content": "partial answer"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 512},
            }
        ),
    )

    with caplog.at_level(logging.WARNING, logger="src.pipeline1.generation.openai_generator"):
        result = OpenAIGenerator("gpt-4.1", max_tokens=512).generate("Prompt")

    assert result.answer == "partial answer"
    assert result.completion_diagnostics["finish_reason"] == "length"
    assert "truncated" in caplog.text.lower() or "cut short" in caplog.text.lower()


def test_network_error_is_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    monkeypatch.setattr("src.pipeline1.generation.openai_generator.time.sleep", lambda s: None)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ConnectionError("refused")
        return _FakeResponse(
            payload={
                "choices": [{"finish_reason": "stop", "message": {"content": "recovered"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )

    monkeypatch.setattr("requests.post", fake_post)

    result = OpenAIGenerator("gpt-4.1").generate("Prompt")
    assert result.answer == "recovered"
    assert len(calls) == 3


# ── Temperature family hardening ──────────────────────────────────────────────

@pytest.mark.parametrize("model_name", [
    "gpt-5.5",
    "gpt-5.5-turbo",
    "gpt-5.5-preview-2025-01",
])
def test_gpt55_family_omits_temperature(monkeypatch, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}
    monkeypatch.setattr("requests.post", lambda url, json, headers, timeout: (captured.update(payload=json) or _FakeResponse()))
    OpenAIGenerator(model_name).generate("Prompt")
    assert "temperature" not in captured["payload"], f"temperature must not be sent for {model_name}"


@pytest.mark.parametrize("model_name", [
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-2024-12-17",
])
def test_o1_family_omits_temperature(monkeypatch, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}
    monkeypatch.setattr("requests.post", lambda url, json, headers, timeout: (captured.update(payload=json) or _FakeResponse()))
    OpenAIGenerator(model_name).generate("Prompt")
    assert "temperature" not in captured["payload"], f"temperature must not be sent for {model_name}"


@pytest.mark.parametrize("model_name", [
    "o3",
    "o3-mini",
    "o3-mini-2025-01-31",
])
def test_o3_family_omits_temperature(monkeypatch, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}
    monkeypatch.setattr("requests.post", lambda url, json, headers, timeout: (captured.update(payload=json) or _FakeResponse()))
    OpenAIGenerator(model_name).generate("Prompt")
    assert "temperature" not in captured["payload"], f"temperature must not be sent for {model_name}"


@pytest.mark.parametrize("model_name", [
    "o4-mini",
    "o4-mini-2025-04-16",
])
def test_o4_mini_family_omits_temperature(monkeypatch, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}
    monkeypatch.setattr("requests.post", lambda url, json, headers, timeout: (captured.update(payload=json) or _FakeResponse()))
    OpenAIGenerator(model_name).generate("Prompt")
    assert "temperature" not in captured["payload"], f"temperature must not be sent for {model_name}"


@pytest.mark.parametrize("model_name", [
    "gpt-4o",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
])
def test_standard_models_send_temperature(monkeypatch, model_name):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key")
    captured = {}
    monkeypatch.setattr("requests.post", lambda url, json, headers, timeout: (captured.update(payload=json) or _FakeResponse()))
    OpenAIGenerator(model_name, temperature=0.2).generate("Prompt")
    assert captured["payload"].get("temperature") == 0.2, f"{model_name} should send temperature"
