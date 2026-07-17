from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.pipeline1.schemas.config_schema import PipelineConfig


def _config_text(generation_block: str) -> str:
    return f"""
experiment:
  experiment_id: "prompt_test"
  output_dir: "runs"
data:
  documents_path: "documents.jsonl"
  questions_path: "questions.jsonl"
chunking:
  strategy: "fixed_word"
  chunk_size: 10
  chunk_overlap: 0
embedding:
  provider: "sentence_transformers"
  model_name: "fake"
index:
  type: "faiss"
  metric: "cosine"
retrieval:
  retriever_type: "dense"
  top_k: 1
  fetch_k: 1
reranker:
  enabled: false
generation:
{generation_block}
telemetry:
  estimate_cost: false
runtime:
  resume: false
  overwrite: true
"""


def test_generation_prompt_path_loads_prompt_file(tmp_path: Path):
    prompt_path = tmp_path / "answer_prompt.txt"
    prompt_path.write_text("Shared answer prompt.\n", encoding="utf-8")
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        _config_text(
            f"""  provider: "ollama"
  model_name: "fake"
  prompt_path: "{prompt_path.as_posix()}"
"""
        ),
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(str(cfg_path))

    assert cfg.generation.system_prompt == "Shared answer prompt.\n"
    assert cfg.generation.prompt_path == prompt_path.as_posix()


def test_generation_prompt_path_loads_for_in_memory_config(tmp_path: Path):
    prompt_path = tmp_path / "answer_prompt.txt"
    prompt_path.write_text("In-memory shared prompt.\n", encoding="utf-8")
    payload = {
        "experiment": {"experiment_id": "prompt_test", "output_dir": "runs"},
        "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
        "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
        "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
        "index": {"type": "faiss", "metric": "cosine"},
        "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
        "reranker": {"enabled": False},
        "generation": {
            "provider": "ollama",
            "model_name": "fake",
            "prompt_path": prompt_path.as_posix(),
        },
        "telemetry": {"estimate_cost": False},
        "runtime": {"resume": False, "overwrite": True},
    }

    cfg = PipelineConfig.model_validate(payload)

    assert cfg.generation.system_prompt == "In-memory shared prompt.\n"


def test_generation_system_prompt_backward_compatibility(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        _config_text(
            """  provider: "ollama"
  model_name: "fake"
  system_prompt: "Inline prompt."
"""
        ),
        encoding="utf-8",
    )

    cfg = PipelineConfig.from_yaml(str(cfg_path))

    assert cfg.generation.system_prompt == "Inline prompt."
    assert cfg.generation.prompt_path is None


def test_generation_prompt_path_missing_file_fails(tmp_path: Path):
    missing_path = tmp_path / "missing_prompt.txt"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        _config_text(
            f"""  provider: "ollama"
  model_name: "fake"
  prompt_path: "{missing_path.as_posix()}"
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation.prompt_path is missing or not a file"):
        PipelineConfig.from_yaml(str(cfg_path))


def test_controlled_pipeline1_experiments_share_generation_prompt():
    config_dir = Path("configs/pipeline1/experiments Orchestration LLM")
    configs = sorted(config_dir.glob("9*.yaml"))

    prompts = {path.name: PipelineConfig.from_yaml(str(path)).generation.system_prompt for path in configs}
    prompt_hashes = {hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts.values()}

    assert len(configs) == 8
    assert len(prompt_hashes) == 1
    assert next(iter(prompt_hashes)).startswith("25e98edb664555df")
    assert {
        PipelineConfig.from_yaml(str(path)).generation.prompt_path for path in configs
    } == {"src/pipeline1/prompts/answer_generation_sivas_v1.txt"}
