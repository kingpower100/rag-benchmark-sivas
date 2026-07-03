"""Unit tests for judge_stage.py retry behavior and overall_score handling.

Ollama is mocked — no real HTTP calls are made.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.pipeline3.judge.ollama_client import OllamaClientError
from src.pipeline3.schemas.pipeline3_config_schema import P3JudgeConfig, P3LLMJudgeConfig
from src.pipeline3.stages.judge_stage import _evaluate_single

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_RESPONSE = json.dumps({
    "correctness": 4,
    "faithfulness": 4,
    "completeness": 3,
    "hallucination": 1,
    "context_relevance": 4,
    "overall_score": 4.0,
    "reasoning": "Good answer with minor omissions.",
})

_INVALID_JSON = "This is definitely not valid JSON."

_MISSING_KEY_RESPONSE = json.dumps({
    "correctness": 4,
    # faithfulness missing
    "completeness": 3,
    "hallucination": 1,
    "context_relevance": 4,
    "overall_score": 4.0,
    "reasoning": "Missing key.",
})


def _judge_cfg(max_retries: int = 3) -> P3JudgeConfig:
    return P3JudgeConfig(
        model="qwen2.5:14b",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_retries=max_retries,
        timeout_seconds=30,
        prompt_version="v2",
    )


def _llm_judge_cfg() -> P3LLMJudgeConfig:
    return P3LLMJudgeConfig()


def _rag_row() -> dict:
    return {
        "question_id": "q1",
        "question": "What is the standard item price?",
        "generated_answer": "The standard price is 100 EUR.",
        "retrieved_context_texts": ["Price list: 100 EUR for standard items."],
    }


def _qa_by_id() -> dict:
    return {"q1": {"answer": "100 EUR"}}


def _mock_client(*responses) -> MagicMock:
    """Return a mock OllamaClient whose generate() yields responses in order."""
    client = MagicMock()
    client.generate.side_effect = list(responses)
    return client


# ---------------------------------------------------------------------------
# Test 1: First call fails (network), second call succeeds
# ---------------------------------------------------------------------------

class TestRetrySucceedsOnSecondAttempt:
    def test_network_error_then_valid_response(self):
        """OllamaClientError on attempt 0; valid response on attempt 1."""
        client = _mock_client(OllamaClientError("connection refused"), _VALID_RESPONSE)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.success is True
        assert result.response is not None
        assert result.response.correctness == 4
        assert result.response.hallucination == 1
        assert result.retry_count == 1          # succeeded on attempt index 1
        assert result.error is None
        assert client.generate.call_count == 2

    def test_two_network_errors_then_valid(self):
        """Two failures followed by a valid response."""
        client = _mock_client(
            OllamaClientError("timeout"),
            OllamaClientError("timeout"),
            _VALID_RESPONSE,
        )

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=3), _llm_judge_cfg())

        assert result.success is True
        assert result.retry_count == 2          # succeeded on attempt index 2
        assert client.generate.call_count == 3


# ---------------------------------------------------------------------------
# Test 2: All judge calls fail
# ---------------------------------------------------------------------------

class TestAllRetriesExhausted:
    def test_all_network_errors(self):
        """All attempts raise OllamaClientError."""
        client = _mock_client(
            OllamaClientError("timeout"),
            OllamaClientError("timeout"),
            OllamaClientError("timeout"),
        )

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=3), _llm_judge_cfg())

        assert result.success is False
        assert result.response is None
        assert result.error is not None
        assert result.retry_count == 3         # equals max_retries on exhaustion
        assert client.generate.call_count == 3

    def test_no_fake_scores_on_failure(self):
        """A failed result must carry no numeric scores — none can be fabricated."""
        client = _mock_client(OllamaClientError("failed"))

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=1), _llm_judge_cfg())

        assert result.success is False
        assert result.response is None
        assert result.llm_overall_score is None
        assert result.error is not None

    def test_exhausted_retry_count_equals_max_retries_config(self):
        """retry_count on failure must equal the configured max_retries value."""
        cfg = _judge_cfg(max_retries=2)
        client = _mock_client(OllamaClientError("fail"), OllamaClientError("fail"))

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, cfg, _llm_judge_cfg())

        assert result.success is False
        assert result.retry_count == 2


# ---------------------------------------------------------------------------
# Test 3: Invalid JSON followed by valid JSON
# ---------------------------------------------------------------------------

class TestParseRetry:
    def test_invalid_json_then_valid(self):
        """First response is unparseable; second response is valid JSON."""
        client = _mock_client(_INVALID_JSON, _VALID_RESPONSE)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.success is True
        assert result.response is not None
        assert result.response.correctness == 4
        assert result.retry_count == 1
        assert client.generate.call_count == 2

    def test_missing_key_json_then_valid(self):
        """First response has a missing required key; second is valid."""
        client = _mock_client(_MISSING_KEY_RESPONSE, _VALID_RESPONSE)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.success is True
        assert result.response.faithfulness == 4
        assert result.retry_count == 1

    def test_all_invalid_json_exhausts_retries(self):
        """All responses are invalid JSON — failure is recorded honestly."""
        client = _mock_client(_INVALID_JSON, _INVALID_JSON, _INVALID_JSON)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=3), _llm_judge_cfg())

        assert result.success is False
        assert result.response is None
        assert result.error is not None
        assert client.generate.call_count == 3


# ---------------------------------------------------------------------------
# Overall score handling (Fix 7)
# ---------------------------------------------------------------------------

class TestOfficialOverallScore:
    def test_official_score_computed_from_weights_not_llm(self):
        """The official overall_score must differ from the LLM's raw value when math differs."""
        client = _mock_client(_VALID_RESPONSE)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.success is True
        # LLM said 4.0; weighted formula gives a different deterministic value.
        assert result.llm_overall_score == pytest.approx(4.0)
        # Deterministic: correctness=4, faithfulness=4, completeness=3, hallucination=1, ctx=4
        # Default weights: correctness=0.30, faithfulness=0.25, completeness=0.20, hallucination=0.15, ctx=0.10
        # hall_contrib=(5-1)/5=0.8
        # weighted=(4/5*0.30)+(4/5*0.25)+(3/5*0.20)+(0.8*0.15)+(4/5*0.10)
        # =0.240+0.200+0.120+0.120+0.080 = 0.760 * 5 = 3.80
        assert result.response.overall_score == pytest.approx(3.80, abs=0.01)
        assert result.response.overall_score != pytest.approx(4.0, abs=0.001)

    def test_llm_overall_score_is_none_on_failure(self):
        """No llm_overall_score must be set when the judge call fails."""
        client = _mock_client(OllamaClientError("fail"))

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=1), _llm_judge_cfg())

        assert result.success is False
        assert result.llm_overall_score is None

    def test_official_score_within_valid_range(self):
        """The recomputed overall_score must always be within [0, scale_max]."""
        client = _mock_client(_VALID_RESPONSE)
        llm_cfg = _llm_judge_cfg()

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), llm_cfg)

        assert result.success is True
        assert llm_cfg.scoring.scale_min <= result.response.overall_score <= llm_cfg.scoring.scale_max


# ---------------------------------------------------------------------------
# Context truncation flag (Fix 6)
# ---------------------------------------------------------------------------

class TestContextTruncation:
    def test_short_context_not_truncated(self):
        """Normal context should not trigger truncation."""
        client = _mock_client(_VALID_RESPONSE)

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.context_truncated is False

    def test_long_context_sets_truncated_flag(self):
        """Context exceeding max_chars must set context_truncated=True."""
        row = dict(_rag_row())
        row["retrieved_context_texts"] = ["x" * 7000]   # exceeds default 6000-char limit
        client = _mock_client(_VALID_RESPONSE)

        result = _evaluate_single(row, _qa_by_id(), client, _judge_cfg(), _llm_judge_cfg())

        assert result.context_truncated is True

    def test_truncated_flag_false_on_failure(self):
        """context_truncated must be False (no truncation) even when the judge call fails."""
        client = _mock_client(OllamaClientError("fail"))

        result = _evaluate_single(_rag_row(), _qa_by_id(), client, _judge_cfg(max_retries=1), _llm_judge_cfg())

        assert result.success is False
        assert result.context_truncated is False
