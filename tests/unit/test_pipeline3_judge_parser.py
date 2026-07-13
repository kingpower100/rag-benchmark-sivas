from __future__ import annotations

import json

import pytest

from src.pipeline3.judge.response_parser import (
    _extract_json_block,
    parse_judge_response,
)

_VALID_RESPONSE = json.dumps(
    {
        "correctness": 5,
        "faithfulness": 4,
        "completeness": 4,
        "hallucination": 0,
        "context_relevance": 5,
        "overall_score": 4.5,
        "reasoning": "The answer is accurate and well-supported.",
    }
)


def test_valid_response_parses_successfully():
    result = parse_judge_response(_VALID_RESPONSE)
    assert result.success is True
    assert result.response is not None
    assert result.response.correctness == 5
    assert result.response.hallucination == 0
    assert result.response.overall_score == pytest.approx(4.5)


def test_empty_response_fails():
    result = parse_judge_response("")
    assert result.success is False
    assert "Empty" in result.error


def test_missing_key_fails():
    data = json.loads(_VALID_RESPONSE)
    del data["correctness"]
    result = parse_judge_response(json.dumps(data))
    assert result.success is False
    assert "correctness" in result.error


def test_extra_key_is_ignored():
    # LLMs often include extra keys like "explanation" or "thoughts"; the parser
    # must silently ignore them as long as all required keys are present and valid.
    data = json.loads(_VALID_RESPONSE)
    data["unexpected"] = 1
    result = parse_judge_response(json.dumps(data))
    assert result.success is True
    assert result.response is not None
    assert result.response.correctness == 5


def test_wrong_key_name_fails_as_missing_and_extra():
    data = json.loads(_VALID_RESPONSE)
    data["context_relevancy"] = data.pop("context_relevance")
    result = parse_judge_response(json.dumps(data))
    assert result.success is False
    assert "context_relevance" in result.error


def test_out_of_range_score_fails():
    data = json.loads(_VALID_RESPONSE)
    data["correctness"] = 10
    result = parse_judge_response(json.dumps(data))
    assert result.success is False
    assert "out of range" in result.error.lower()


def test_markdown_fences_stripped():
    wrapped = f"```json\n{_VALID_RESPONSE}\n```"
    result = parse_judge_response(wrapped)
    assert result.success is True


def test_json_embedded_in_text_extracted():
    text = f"Here is the evaluation: {_VALID_RESPONSE} That's my assessment."
    result = parse_judge_response(text)
    assert result.success is True


def test_non_json_fails_gracefully():
    result = parse_judge_response("This is not JSON at all.")
    assert result.success is False


def test_response_to_dict_has_judge_prefix():
    result = parse_judge_response(_VALID_RESPONSE)
    assert result.success
    d = result.response.to_dict()
    assert "judge_correctness" in d
    assert "judge_overall_score" in d
    assert "judge_relevancy" not in d


def test_relevancy_key_as_extra_is_ignored():
    """The old 'relevancy' key is not a required field.  When it appears alongside
    all required keys the parser must succeed, extracting only the required fields."""
    data = json.loads(_VALID_RESPONSE)
    data["relevancy"] = 3
    result = parse_judge_response(json.dumps(data))
    assert result.success is True
    assert result.response is not None
    # The correct field is context_relevance; relevancy is an extra that was dropped.
    assert result.response.context_relevance == data["context_relevance"]


def test_extract_json_block_from_clean_json():
    raw = '{"key": "value"}'
    assert _extract_json_block(raw) == raw


def test_extract_json_block_strips_backtick_fences():
    raw = "```\n{\"key\": \"value\"}\n```"
    extracted = _extract_json_block(raw)
    assert extracted == '{"key": "value"}'


def test_non_numeric_score_field_fails():
    data = json.loads(_VALID_RESPONSE)
    data["correctness"] = "five"
    result = parse_judge_response(json.dumps(data))
    assert result.success is False
    assert "numeric" in result.error.lower()


def test_overall_score_non_numeric_fails():
    data = json.loads(_VALID_RESPONSE)
    data["overall_score"] = "high"
    result = parse_judge_response(json.dumps(data))
    assert result.success is False


def test_reasoning_coerced_to_string():
    data = json.loads(_VALID_RESPONSE)
    data["reasoning"] = 42
    result = parse_judge_response(json.dumps(data))
    assert result.success is True
    assert result.response.reasoning == "42"
