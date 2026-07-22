import pytest
import sys
import types

from src.pipeline1.preflight import _ollama_model_available, run_preflight_checks
from src.pipeline1.schemas.config_schema import PipelineConfig


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


def _cfg_for_reranker_device(device: str) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 2},
            "reranker": {"enabled": True, "model_name": "fake-reranker", "device": device},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {"estimate_cost": False},
            "runtime": {"resume": False, "overwrite": True},
        }
    )


def _write_minimal_inputs(tmp_path):
    (tmp_path / "documents.jsonl").write_text('{"context_id":"c1","cleaned_context":"text"}\n', encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text('{"question_id":"q1","question":"Q?"}\n', encoding="utf-8")


def _install_fake_torch(monkeypatch, cuda_available: bool, device_count: int = 0):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: device_count,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


def test_reranker_cuda_colon_zero_rejected_when_cuda_unavailable(tmp_path, monkeypatch):
    _write_minimal_inputs(tmp_path)
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    _install_fake_torch(monkeypatch, cuda_available=False)

    errors = run_preflight_checks(_cfg_for_reranker_device("cuda:0"), tmp_path)

    assert any("Reranker requested CUDA device cuda:0, but CUDA is unavailable." in error for error in errors)


def test_reranker_cuda_index_out_of_range_rejected(tmp_path, monkeypatch):
    _write_minimal_inputs(tmp_path)
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    _install_fake_torch(monkeypatch, cuda_available=True, device_count=1)

    errors = run_preflight_checks(_cfg_for_reranker_device("cuda:2"), tmp_path)

    assert any("cuda:2" in error and "only 1 CUDA device" in error for error in errors)


def test_reranker_cpu_device_does_not_require_cuda(tmp_path, monkeypatch):
    _write_minimal_inputs(tmp_path)
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

    errors = run_preflight_checks(_cfg_for_reranker_device("cpu"), tmp_path)

    assert not any("Reranker requested CUDA" in error for error in errors)
