import pytest

from src.pipeline1.preflight import _ollama_model_available


def test_bare_name_matches_latest_tag():
    # "mistral-small" should match when Ollama reports "mistral-small:latest"
    assert _ollama_model_available("mistral-small", {"mistral-small:latest", "qwen2.5:7b"})


def test_exact_tagged_name_matches():
    # "qwen2.5:7b" should match exactly
    assert _ollama_model_available("qwen2.5:7b", {"mistral-small:latest", "qwen2.5:7b"})


def test_missing_model_fails():
    assert not _ollama_model_available("llama3", {"mistral-small:latest", "qwen2.5:7b"})


def test_explicit_tag_does_not_match_different_tag():
    # "mistral-small:v0.3" must not match "mistral-small:latest"
    assert not _ollama_model_available("mistral-small:v0.3", {"mistral-small:latest"})


def test_bare_name_matches_exact_if_present():
    # If Ollama somehow reports the bare name, it still matches
    assert _ollama_model_available("mistral-small", {"mistral-small"})


def test_bare_name_does_not_match_non_latest_tag_only():
    # "model" should not match "model:v2" (only :latest gets the implicit promotion)
    assert not _ollama_model_available("mistral-small", {"mistral-small:v2"})
