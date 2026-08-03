"""Tests for F01 — Final Production System with Mistral-Small generation.

F01 is a controlled generation ablation of F00:
  F00 = Final system + GPT-5.5 (OpenAI API)
  F01 = Final system + Mistral-Small (local Ollama)

Only the generation provider and model-specific settings differ.
All retrieval, reranking, chunking, embedding, routing, and
evaluation settings must remain identical to F00.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.config_loader import load_pipeline3_config_payload
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------

F01_P1_PATH = "configs/pipeline1/final_experiments/F01_mistral_small_final_system.yaml"
F01_P2_PATH = "configs/pipeline2/final_experiments/F01_mistral_small_final_system_eval.yaml"
F01_P3_PATH = "configs/pipeline3/final_experiments/F01_mistral_small_final_system_eval.yaml"

F00_P1_PATH = "configs/pipeline1/final_experiments/F00_final_system.yaml"
F00_P2_PATH = "configs/pipeline2/final_experiments/F00_final_system_eval.yaml"
F00_P3_PATH = "configs/pipeline3/final_experiments/F00_final_system_eval.yaml"

MAPPING_PATH = "configs/official_experiment_mapping.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f01_cfg() -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(F01_P1_PATH))


def _f00_cfg() -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))


def _strip_approved_differences(cfg: PipelineConfig) -> dict[str, Any]:
    """Return the model dict with the approved-different fields removed.

    Approved differences between F00 and F01:
      experiment.experiment_id        — identity field, intentionally different
      generation.provider             — ollama vs openai
      generation.model_name           — mistral-small:latest vs gpt-5.5
      generation.base_url             — localhost:11434 vs api.openai.com/v1
      generation.system_prompt        — loaded from same prompt_path; identical content
      telemetry.estimate_cost         — true (Ollama, free) vs false (GPT-5.5, unpublished pricing)
    """
    d = cfg.model_dump()
    d.get("experiment", {}).pop("experiment_id", None)
    gen = d.get("generation", {})
    gen.pop("provider", None)
    gen.pop("model_name", None)
    gen.pop("base_url", None)
    gen.pop("system_prompt", None)   # loaded from same file; verified separately
    d.get("telemetry", {}).pop("estimate_cost", None)
    return d


# ---------------------------------------------------------------------------
# 1. F01 resolves successfully
# ---------------------------------------------------------------------------

def test_f01_resolves_successfully():
    cfg = _f01_cfg()
    assert cfg is not None
    assert cfg.experiment.experiment_id == "F01"


# ---------------------------------------------------------------------------
# 2. F01 and F00 have identical non-generation configurations
# ---------------------------------------------------------------------------

def test_f01_and_f00_non_generation_configs_are_identical():
    f01 = _f01_cfg()
    f00 = _f00_cfg()

    f01_stripped = _strip_approved_differences(f01)
    f00_stripped = _strip_approved_differences(f00)

    assert f01_stripped == f00_stripped, (
        "F01 and F00 differ outside approved generation fields. "
        "Inspect the diff carefully — no unintended change is allowed."
    )


# ---------------------------------------------------------------------------
# 3. F01 uses provider=ollama
# ---------------------------------------------------------------------------

def test_f01_generation_provider_is_ollama():
    cfg = _f01_cfg()
    assert cfg.generation.provider == "ollama"


# ---------------------------------------------------------------------------
# 4. F01 uses the exact G02 model name
# ---------------------------------------------------------------------------

def test_f01_model_name_matches_g02_mistral_small():
    cfg = _f01_cfg()
    assert cfg.generation.model_name == "mistral-small:latest"


# ---------------------------------------------------------------------------
# 5. F01 does not retain an OpenAI API-key requirement
# ---------------------------------------------------------------------------

def test_f01_factory_does_not_require_openai_api_key(monkeypatch):
    """Building the F01 generator must not raise even when OPENAI_API_KEY is absent."""
    from src.pipeline1.generation.factory import build_generator
    from src.pipeline1.generation.ollama_generator import OllamaGenerator

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _f01_cfg()

    generator = build_generator(cfg.generation)
    assert isinstance(generator, OllamaGenerator), (
        f"Expected OllamaGenerator for provider='ollama', got {type(generator).__name__}"
    )


# ---------------------------------------------------------------------------
# 6. F01 does not retain unsupported GPT-specific reasoning parameters
# ---------------------------------------------------------------------------

def test_f01_reasoning_effort_is_absent():
    """reasoning_effort must be None — it is only valid for provider='openai'.

    The schema (GenerationConfig._validate_reasoning_effort) would raise a
    ValueError if reasoning_effort were set for an ollama provider, so a
    successful load already proves this. This test makes it explicit.
    """
    cfg = _f01_cfg()
    assert cfg.generation.reasoning_effort is None


# ---------------------------------------------------------------------------
# 7. F01 keeps global Hybrid RRF retrieval
# ---------------------------------------------------------------------------

def test_f01_retriever_type_is_global_hybrid_rrf():
    cfg = _f01_cfg()
    assert cfg.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert cfg.retrieval.hybrid.rrf_k == 60
    assert cfg.retrieval.hybrid.dense_fetch_k == 20
    assert cfg.retrieval.hybrid.bm25_fetch_k == 20
    assert cfg.retrieval.fallback_to_global is False
    assert cfg.retrieval.category_routing_validation.enabled is False


# ---------------------------------------------------------------------------
# 8. F01 keeps the reranker enabled
# ---------------------------------------------------------------------------

def test_f01_reranker_is_enabled():
    cfg = _f01_cfg()
    assert cfg.reranker.enabled is True
    assert cfg.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.reranker.device == "cuda"
    assert cfg.reranker.rerank_top_k == 20


# ---------------------------------------------------------------------------
# 9. F01 keeps top_k=5
# ---------------------------------------------------------------------------

def test_f01_top_k_is_5():
    cfg = _f01_cfg()
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.fetch_k == 20


# ---------------------------------------------------------------------------
# 10. F01 keeps orchestration disabled
# ---------------------------------------------------------------------------

def test_f01_orchestration_is_disabled():
    cfg = _f01_cfg()
    assert cfg.orchestration.enabled is False


# ---------------------------------------------------------------------------
# 11. P2 and P3 point to F01 Pipeline 1 outputs
# ---------------------------------------------------------------------------

def test_f01_p2_points_to_f01_pipeline1_results():
    cfg = EvalConfig.model_validate(load_eval_config_payload(F01_P2_PATH))
    assert cfg.evaluation.eval_run_id == "F01"
    assert "F01" in cfg.inputs.pipeline1_results_path
    assert cfg.inputs.pipeline1_results_path == "data/runs/pipeline1/F01/results.jsonl"


def test_f01_p3_points_to_f01_pipeline1_results():
    cfg = Pipeline3Config.model_validate(load_pipeline3_config_payload(F01_P3_PATH))
    assert cfg.pipeline3.run_id == "F01"
    assert cfg.inputs.pipeline1_results_path == "data/runs/pipeline1/F01/results.jsonl"


# ---------------------------------------------------------------------------
# 12. F01 is registered in the official experiment mapping and Pipeline 4
# ---------------------------------------------------------------------------

def test_f01_is_in_official_experiment_mapping():
    from src.config_utils import load_yaml_mapping
    mapping = load_yaml_mapping(Path(MAPPING_PATH))
    experiments = mapping.get("official_experiment_mapping", {})
    assert "F01" in experiments, "F01 is missing from official_experiment_mapping.yaml"
    f01_entry = experiments["F01"]
    assert "pipeline1" in f01_entry
    assert "pipeline2" in f01_entry
    assert "pipeline3" in f01_entry
    assert "F01" in f01_entry["pipeline1"]
    assert "F01" in f01_entry["pipeline2"]
    assert "F01" in f01_entry["pipeline3"]


def test_f01_mapping_p1_path_exists():
    from src.config_utils import load_yaml_mapping
    mapping = load_yaml_mapping(Path(MAPPING_PATH))
    p1_path = mapping["official_experiment_mapping"]["F01"]["pipeline1"]
    assert Path(p1_path).exists(), f"F01 P1 YAML referenced in mapping does not exist: {p1_path}"


def test_f01_mapping_p2_path_exists():
    from src.config_utils import load_yaml_mapping
    mapping = load_yaml_mapping(Path(MAPPING_PATH))
    p2_path = mapping["official_experiment_mapping"]["F01"]["pipeline2"]
    assert Path(p2_path).exists(), f"F01 P2 YAML referenced in mapping does not exist: {p2_path}"


def test_f01_mapping_p3_path_exists():
    from src.config_utils import load_yaml_mapping
    mapping = load_yaml_mapping(Path(MAPPING_PATH))
    p3_path = mapping["official_experiment_mapping"]["F01"]["pipeline3"]
    assert Path(p3_path).exists(), f"F01 P3 YAML referenced in mapping does not exist: {p3_path}"


# ---------------------------------------------------------------------------
# 13. F00 remains unchanged
# ---------------------------------------------------------------------------

def test_f00_is_not_modified_by_f01():
    """Loading F01 must not alter F00's resolved configuration."""
    f00 = _f00_cfg()
    assert f00.experiment.experiment_id == "F00"
    assert f00.generation.provider == "openai"
    assert f00.generation.model_name == "gpt-5.5"
    assert f00.generation.base_url == "https://api.openai.com/v1"
    assert f00.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert f00.reranker.enabled is True
    assert f00.orchestration.enabled is False
    assert f00.retrieval.top_k == 5
    assert f00.telemetry.estimate_cost is False


# ---------------------------------------------------------------------------
# 14. Resolved-difference test: only approved fields differ
# ---------------------------------------------------------------------------

def test_f01_vs_f00_chunking_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.chunking == f00.chunking


def test_f01_vs_f00_embedding_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.embedding == f00.embedding


def test_f01_vs_f00_index_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.index == f00.index


def test_f01_vs_f00_retrieval_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.retrieval == f00.retrieval


def test_f01_vs_f00_reranker_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.reranker == f00.reranker


def test_f01_vs_f00_orchestration_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.orchestration == f00.orchestration


def test_f01_vs_f00_data_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.data == f00.data


def test_f01_vs_f00_runtime_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.runtime == f00.runtime


def test_f01_vs_f00_parent_context_is_identical():
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.parent_context == f00.parent_context


def test_f01_vs_f00_generation_prompt_is_identical():
    """Both F00 and F01 must use the same prompt text (loaded from same file)."""
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.generation.prompt_path == f00.generation.prompt_path
    assert f01.generation.system_prompt == f00.generation.system_prompt


def test_f01_vs_f00_generation_context_budget_is_identical():
    """All context-budget and decoding settings must match F00."""
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.generation.max_tokens == f00.generation.max_tokens
    assert f01.generation.temperature == f00.generation.temperature
    assert f01.generation.max_context_tokens == f00.generation.max_context_tokens
    assert f01.generation.max_chunk_tokens == f00.generation.max_chunk_tokens
    assert f01.generation.max_context_chars == f00.generation.max_context_chars
    assert f01.generation.max_chunk_chars == f00.generation.max_chunk_chars
    assert f01.generation.max_prompt_tokens == f00.generation.max_prompt_tokens
    assert f01.generation.context_truncation_strategy == f00.generation.context_truncation_strategy
    assert f01.generation.timeout_s == f00.generation.timeout_s


def test_f01_vs_f00_telemetry_pricing_is_identical():
    """Pricing values must remain 0.0 (identical to F00)."""
    f01, f00 = _f01_cfg(), _f00_cfg()
    assert f01.telemetry.pricing.input_per_1k_tokens_usd == f00.telemetry.pricing.input_per_1k_tokens_usd
    assert f01.telemetry.pricing.output_per_1k_tokens_usd == f00.telemetry.pricing.output_per_1k_tokens_usd


# ---------------------------------------------------------------------------
# P2 source_validation contract for F01
# ---------------------------------------------------------------------------

def test_f01_p2_source_validation_contract():
    cfg = EvalConfig.model_validate(load_eval_config_payload(F01_P2_PATH))
    sv = cfg.source_validation

    assert sv.expected_experiment_id == "F01"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "ollama"
    assert sv.expected_generator_model == "mistral-small:latest"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.require_hybrid_diagnostics is True
    assert sv.require_reranker_diagnostics is True
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f01_p2_inherits_f00_chunk_level_evaluation():
    """Chunk-level gold annotations path must be inherited from F00."""
    f01_cfg = EvalConfig.model_validate(load_eval_config_payload(F01_P2_PATH))
    f00_cfg = EvalConfig.model_validate(load_eval_config_payload(F00_P2_PATH))

    f01_chunk = f01_cfg.retrieval_evaluation.chunk_level
    f00_chunk = f00_cfg.retrieval_evaluation.chunk_level

    assert f01_chunk.enabled == f00_chunk.enabled
    assert f01_chunk.ground_truth_path == f00_chunk.ground_truth_path
    assert f01_chunk.missing_question_policy == f00_chunk.missing_question_policy


def test_f01_p2_inherits_f00_retrieval_settings():
    f01_cfg = EvalConfig.model_validate(load_eval_config_payload(F01_P2_PATH))
    f00_cfg = EvalConfig.model_validate(load_eval_config_payload(F00_P2_PATH))

    assert f01_cfg.retrieval.k == f00_cfg.retrieval.k
    assert f01_cfg.retrieval.ks == f00_cfg.retrieval.ks


# ---------------------------------------------------------------------------
# P3 source_validation contract for F01
# ---------------------------------------------------------------------------

def test_f01_p3_source_validation_contract():
    cfg = Pipeline3Config.model_validate(load_pipeline3_config_payload(F01_P3_PATH))
    sv = cfg.source_validation

    assert sv.expected_experiment_id == "F01"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "ollama"
    assert sv.expected_generator_model == "mistral-small:latest"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f01_p3_inherits_f00_ragas_and_judge_settings():
    """RAGAS and LLM-judge settings must be inherited unchanged from F00."""
    f01_cfg = Pipeline3Config.model_validate(load_pipeline3_config_payload(F01_P3_PATH))
    f00_cfg = Pipeline3Config.model_validate(load_pipeline3_config_payload(F00_P3_PATH))

    assert f01_cfg.ragas.llm_model == f00_cfg.ragas.llm_model
    assert f01_cfg.ragas.embeddings_model == f00_cfg.ragas.embeddings_model
    assert f01_cfg.judge.model == f00_cfg.judge.model
    assert f01_cfg.judge.prompt_version == f00_cfg.judge.prompt_version
