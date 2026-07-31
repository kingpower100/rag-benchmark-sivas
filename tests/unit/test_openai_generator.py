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
