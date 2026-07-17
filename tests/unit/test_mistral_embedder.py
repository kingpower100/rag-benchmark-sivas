"""Unit tests for MistralEmbedder.

All tests use mocked HTTP responses — the real Mistral API is never called.

Covers:
- Missing API key raises EnvironmentError at construction
- Invalid API key (401) raises EnvironmentError on encode
- Single text embedding call
- Batch encoding (multiple texts in one call)
- Batch splitting respects batch_size
- Correct response ordering (by index)
- Retry on transient HTTP errors (429, 503)
- No API key in request logs or error messages
- encode_query returns a 1-D array
- Empty input returns empty array
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import requests

from src.pipeline1.embedding.mistral_embedder import (
    MISTRAL_EMBEDDINGS_URL,
    MistralEmbedder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_response(embeddings: list[list[float]], status_code: int = 200):
    """Return a mock requests.Response with a Mistral embeddings payload."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_error_response(status_code: int):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    http_error = requests.HTTPError(response=resp)
    resp.raise_for_status.side_effect = http_error
    return resp


_DIM = 1024
_FAKE_EMB = [0.1] * _DIM


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_raises_if_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
            MistralEmbedder()

    def test_constructs_when_api_key_present(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        embedder = MistralEmbedder()
        assert embedder.model_name == "mistral-embed"

    def test_api_key_not_exposed_on_object(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "super-secret")
        embedder = MistralEmbedder()
        public_attrs = {k: v for k, v in vars(embedder).items() if not k.startswith("_")}
        for value in public_attrs.values():
            assert "super-secret" not in str(value)


# ---------------------------------------------------------------------------
# Single text encoding
# ---------------------------------------------------------------------------

class TestSingleTextEncoding:
    def test_encode_texts_single(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        mock_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = embedder.encode_texts(["Hello world"])
        assert result.shape == (1, _DIM)
        assert result.dtype == np.float32
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[0][0] == MISTRAL_EMBEDDINGS_URL

    def test_encode_query_returns_1d(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        mock_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", return_value=mock_resp):
            result = embedder.encode_query("test query")
        assert result.ndim == 1
        assert result.shape == (_DIM,)


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncoding:
    def test_three_texts_returned_in_order(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder(batch_size=10)
        embs = [[float(i)] * _DIM for i in range(3)]
        mock_resp = _make_response(embs)
        with patch("requests.post", return_value=mock_resp):
            result = embedder.encode_texts(["a", "b", "c"])
        assert result.shape == (3, _DIM)
        for i in range(3):
            assert result[i, 0] == pytest.approx(float(i))

    def test_batch_size_splits_calls(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder(batch_size=2)
        # 5 texts → 3 batches (2, 2, 1)
        embs_batch1 = [[1.0] * _DIM, [2.0] * _DIM]
        embs_batch2 = [[3.0] * _DIM, [4.0] * _DIM]
        embs_batch3 = [[5.0] * _DIM]
        responses = [
            _make_response(embs_batch1),
            _make_response(embs_batch2),
            _make_response(embs_batch3),
        ]
        with patch("requests.post", side_effect=responses) as mock_post:
            result = embedder.encode_texts(["a", "b", "c", "d", "e"])
        assert mock_post.call_count == 3
        assert result.shape == (5, _DIM)

    def test_response_ordering_by_index(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder(batch_size=10)
        # Simulate API returning items in reversed index order
        resp = MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.return_value = {
            "data": [
                {"index": 2, "embedding": [3.0] * _DIM},
                {"index": 0, "embedding": [1.0] * _DIM},
                {"index": 1, "embedding": [2.0] * _DIM},
            ]
        }
        resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=resp):
            result = embedder.encode_texts(["x", "y", "z"])
        assert result[0, 0] == pytest.approx(1.0)
        assert result[1, 0] == pytest.approx(2.0)
        assert result[2, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list_returns_empty_array(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        with patch("requests.post") as mock_post:
            result = embedder.encode_texts([])
        mock_post.assert_not_called()
        assert result.shape[0] == 0


# ---------------------------------------------------------------------------
# Authentication errors
# ---------------------------------------------------------------------------

class TestAuthenticationErrors:
    def test_401_raises_environment_error(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "bad-key")
        embedder = MistralEmbedder()
        resp = _make_error_response(401)
        resp.raise_for_status.side_effect = None  # reset so we check status manually
        resp.status_code = 401
        with patch("requests.post", return_value=resp):
            with pytest.raises(EnvironmentError, match="authentication"):
                embedder.encode_texts(["test"])

    def test_error_message_does_not_contain_api_key(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "leakable-secret")
        embedder = MistralEmbedder()
        resp = _make_error_response(401)
        resp.status_code = 401
        resp.raise_for_status.side_effect = None
        with patch("requests.post", return_value=resp):
            try:
                embedder.encode_texts(["test"])
            except EnvironmentError as exc:
                assert "leakable-secret" not in str(exc)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------

class TestRetryBehavior:
    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        rate_limit_resp = _make_error_response(429)
        success_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", side_effect=[rate_limit_resp, success_resp]):
            with patch("time.sleep"):  # suppress actual sleep in tests
                result = embedder.encode_texts(["hello"])
        assert result.shape == (1, _DIM)

    def test_retries_on_503_then_succeeds(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        server_error = _make_error_response(503)
        success_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", side_effect=[server_error, success_resp]):
            with patch("time.sleep"):
                result = embedder.encode_texts(["hello"])
        assert result.shape == (1, _DIM)

    def test_raises_after_max_retries_exhausted(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder()
        persistent_error = _make_error_response(503)
        with patch("requests.post", return_value=persistent_error):
            with patch("time.sleep"):
                with pytest.raises(requests.HTTPError):
                    embedder.encode_texts(["hello"])


# ---------------------------------------------------------------------------
# Authorization header is set correctly
# ---------------------------------------------------------------------------

class TestAuthorizationHeader:
    def test_bearer_token_in_headers(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "my-test-key")
        embedder = MistralEmbedder()
        mock_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder.encode_texts(["test"])
        _, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-test-key"

    def test_model_name_in_payload(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "key")
        embedder = MistralEmbedder(model_name="mistral-embed")
        mock_resp = _make_response([_FAKE_EMB])
        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder.encode_texts(["hello"])
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["model"] == "mistral-embed"
