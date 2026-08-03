from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.evaluation.source_validation import validate_pipeline1_source
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.schemas.eval_config_schema import EvalConfig, SourceValidationConfig
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config, P3SourceValidationConfig


R01_P1 = "configs/pipeline1/final_experiments/R01_adaptive_category_aware.yaml"
R01_M_V1_P1 = "configs/pipeline1/final_experiments/R01_Mistral_V1_adaptive_category_aware.yaml"
R01_M_V1_P2 = "configs/pipeline2/final_experiments/R01_Mistral_V1_adaptive_category_aware_eval.yaml"
R01_M_V1_P3 = "configs/pipeline3/final_experiments/R01_Mistral_V1_adaptive_category_aware_eval.yaml"


def _p1(path: str) -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(path))


def test_r01_mistral_v1_mapping_and_configs_load():
    mapping = yaml.safe_load(Path("configs/official_experiment_mapping.yaml").read_text(encoding="utf-8"))[
        "official_experiment_mapping"
    ]

    assert mapping["R01-M-V1"] == {
        "pipeline1": R01_M_V1_P1,
        "pipeline2": R01_M_V1_P2,
        "pipeline3": R01_M_V1_P3,
    }

    p1 = _p1(mapping["R01-M-V1"]["pipeline1"])
    p2 = EvalConfig.from_yaml(mapping["R01-M-V1"]["pipeline2"])
    p3 = Pipeline3Config.from_yaml(mapping["R01-M-V1"]["pipeline3"])

    assert p1.experiment.experiment_id == "R01-M-V1"
    assert p2.evaluation.eval_run_id == "R01-M-V1_adaptive_category_aware_eval"
    assert p3.pipeline3.run_id == "R01-M-V1"
    assert p2.inputs.pipeline1_results_path == "data/runs/pipeline1/R01-M-V1/results.jsonl"
    assert p3.inputs.pipeline1_results_path == "data/runs/pipeline1/R01-M-V1/results.jsonl"
    assert "R01/results.jsonl" not in p2.inputs.pipeline1_results_path
    assert "R01/results.jsonl" not in p3.inputs.pipeline1_results_path


def test_r01_mistral_v1_changes_only_routing_model_and_prompt():
    r01 = _p1(R01_P1)
    variant = _p1(R01_M_V1_P1)

    assert r01.experiment.experiment_id == "R01"
    assert variant.experiment.experiment_id == "R01-M-V1"

    assert variant.orchestration.enabled is True
    assert variant.orchestration.provider == "ollama"
    assert variant.orchestration.model_name == "mistral-small"
    assert variant.orchestration.prompt_version == "v1"
    assert variant.orchestration.prompt_path == "src/pipeline1/prompts/orchestration_promptV1.txt"

    assert variant.chunking.model_dump() == r01.chunking.model_dump()
    assert variant.embedding.model_dump() == r01.embedding.model_dump()
    assert variant.index.model_dump() == r01.index.model_dump()
    assert variant.retrieval.model_dump() == r01.retrieval.model_dump()
    assert variant.reranker.model_dump() == r01.reranker.model_dump()
    assert variant.generation.model_dump() == r01.generation.model_dump()
    assert variant.runtime.model_dump() == r01.runtime.model_dump()

    assert variant.retrieval.retriever_type == "adaptive_category_aware_hybrid_rrf"
    assert variant.retrieval.fallback_to_global is True
    assert variant.retrieval.category_routing_validation.enabled is True
    assert variant.retrieval.category_field == "kategorie"
    assert variant.retrieval.top_k == r01.retrieval.top_k == 5
    assert variant.retrieval.fetch_k == r01.retrieval.fetch_k == 20


def test_r01_mistral_v1_pipeline2_and_pipeline3_source_validation_rules():
    p2 = EvalConfig.from_yaml(R01_M_V1_P2)
    p3 = Pipeline3Config.from_yaml(R01_M_V1_P3)

    for source_validation in (p2.source_validation, p3.source_validation):
        assert source_validation is not None
        assert source_validation.expected_experiment_id == "R01-M-V1"
        assert source_validation.expected_retriever_type == "adaptive_category_aware_hybrid_rrf"
        assert source_validation.expected_orchestration_enabled is True
        assert source_validation.expected_generator_provider == "ollama"
        assert source_validation.expected_generator_model == "qwen2.5:7b-instruct"
        assert source_validation.expected_top_k == 5
        assert source_validation.expected_fetch_k == 20
        assert source_validation.expected_reranker_enabled is True
        assert source_validation.expected_fallback_to_global is True
        assert source_validation.require_hybrid_diagnostics is True
        assert source_validation.require_reranker_diagnostics is True
        assert source_validation.require_routing_diagnostics is True
        assert source_validation.require_routing_reconciliation is True


@pytest.mark.parametrize("wrong_id", ["R01", "R00", "B00_sivas_pgvector_reference", "F00"])
def test_r01_mistral_v1_source_validation_rejects_other_experiments(wrong_id, tmp_path):
    manifest = _manifest(wrong_id)
    rows = [_row(wrong_id)]
    source_validation = SourceValidationConfig(
        expected_experiment_id="R01-M-V1",
        expected_retriever_type="adaptive_category_aware_hybrid_rrf",
        expected_orchestration_enabled=True,
        expected_generator_provider="ollama",
        expected_generator_model="qwen2.5:7b-instruct",
        expected_top_k=5,
        expected_fetch_k=20,
        expected_reranker_enabled=True,
        expected_fallback_to_global=True,
        require_hybrid_diagnostics=True,
        require_reranker_diagnostics=True,
        require_routing_diagnostics=True,
        require_routing_reconciliation=True,
    )

    with pytest.raises(ValueError, match="experiment_id"):
        validate_pipeline1_source(
            results_path=tmp_path / "results.jsonl",
            manifest=manifest,
            rows=rows,
            source_validation=source_validation,
            pipeline_name="Pipeline 2",
        )


def test_pipeline3_source_validation_config_type_has_same_rejection_policy():
    cfg = P3SourceValidationConfig(
        expected_experiment_id="R01-M-V1",
        expected_retriever_type="adaptive_category_aware_hybrid_rrf",
        expected_orchestration_enabled=True,
        expected_generator_provider="ollama",
        expected_generator_model="qwen2.5:7b-instruct",
        expected_top_k=5,
        expected_fetch_k=20,
        expected_reranker_enabled=True,
        expected_fallback_to_global=True,
        require_hybrid_diagnostics=True,
        require_reranker_diagnostics=True,
        require_routing_diagnostics=True,
        require_routing_reconciliation=True,
    )
    assert cfg.expected_experiment_id == "R01-M-V1"
    assert cfg.require_routing_reconciliation is True


def _manifest(experiment_id: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "run_status": "PASS",
        "failed_questions": 0,
        "expected_questions": 1,
        "successful_questions": 1,
        "config": {
            "experiment": {"experiment_id": experiment_id},
            "retrieval": {
                "retriever_type": "adaptive_category_aware_hybrid_rrf",
                "top_k": 5,
                "fetch_k": 20,
                "fallback_to_global": True,
            },
            "reranker": {"enabled": True},
            "orchestration": {"enabled": True},
        },
        "resolved_config": {
            "experiment": {"experiment_id": experiment_id},
            "retrieval": {
                "retriever_type": "adaptive_category_aware_hybrid_rrf",
                "top_k": 5,
                "fetch_k": 20,
                "fallback_to_global": True,
            },
            "reranker": {"enabled": True},
            "generation": {"provider": "ollama", "model_name": "qwen2.5:7b-instruct"},
        },
        "models": {
            "generator_provider": "ollama",
            "generator_model": "qwen2.5:7b-instruct",
            "orchestration_enabled": True,
        },
        "category_routing_validation": {
            "category_route_count": 1,
            "global_route_count": 0,
            "fallback_count": 0,
        },
    }


def _row(experiment_id: str) -> dict:
    return {
        "experiment_id": experiment_id,
        "question_id": "Q001",
        "retriever_type": "adaptive_category_aware_hybrid_rrf",
        "generated_answer": "answer",
        "reranker_applied": True,
        "retrieval_diagnostics": {
            "retriever_type": "adaptive_category_aware_hybrid_rrf",
            "routing_decision": "accepted",
            "final_retrieval_mode": "category",
            "retrieval_scope": "category",
            "fallback_used": False,
            "category_filter_applied_dense": True,
            "category_filter_applied_bm25": True,
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "fused_candidate_count": 20,
            "reranked_candidate_count": 20,
            "final_context_count": 5,
        },
    }
