"""Static contract tests for B04 and F04 controlled experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config_utils import load_yaml_mapping
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.config_loader import load_pipeline3_config_payload
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


B00_P1_PATH = "configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml"
B04_P1_PATH = "configs/pipeline1/final_experiments/B04_global_b00_dense.yaml"
F04_P1_PATH = "configs/pipeline1/final_experiments/F04_global_hybrid_b00_chunking.yaml"

B04_P2_PATH = "configs/pipeline2/final_experiments/B04_global_b00_dense_eval.yaml"
F04_P2_PATH = "configs/pipeline2/final_experiments/F04_global_hybrid_b00_chunking_eval.yaml"
B04_P3_PATH = "configs/pipeline3/final_experiments/B04_global_b00_dense_eval.yaml"
F04_P3_PATH = "configs/pipeline3/final_experiments/F04_global_hybrid_b00_chunking_eval.yaml"

MAPPING_PATH = "configs/official_experiment_mapping.yaml"
B00_CHUNK_GT = (
    "data/ground_truth/chunk_level/B00_sivas_character2048_overlap0/"
    "gold_chunk_annotations_B00_sivas_character2048_overlap0.jsonl"
)


def _p1(path: str) -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(path))


def _p2(path: str) -> EvalConfig:
    return EvalConfig.model_validate(load_eval_config_payload(path))


def _p3(path: str) -> Pipeline3Config:
    return Pipeline3Config.model_validate(load_pipeline3_config_payload(path))


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            flattened.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return flattened
    if isinstance(value, list):
        flattened: dict[str, Any] = {}
        for index, child in enumerate(value):
            flattened.update(_flatten(child, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _diff_paths(left: Any, right: Any) -> set[str]:
    left_flat = _flatten(left)
    right_flat = _flatten(right)
    return {key for key in set(left_flat) | set(right_flat) if left_flat.get(key) != right_flat.get(key)}


def test_b04_is_global_dense_b00_ablation():
    b00 = _p1(B00_P1_PATH)
    b04 = _p1(B04_P1_PATH)

    assert b04.experiment.experiment_id == "B04"
    assert b04.chunking == b00.chunking
    assert b04.embedding == b00.embedding
    assert b04.generation == b00.generation
    assert b04.reranker == b00.reranker
    assert b04.data == b00.data
    assert b04.runtime == b00.runtime
    assert b04.telemetry == b00.telemetry
    assert b04.parent_context == b00.parent_context

    assert b04.index.type == b00.index.type == "pgvector"
    assert b04.index.metric == b00.index.metric == "cosine"
    assert b04.index.dense_dim == b00.index.dense_dim == 1024
    assert b04.index.index_name != b00.index.index_name
    assert b04.index.pgvector is not None
    assert b00.index.pgvector is not None
    assert b04.index.pgvector.table_name != b00.index.pgvector.table_name
    assert b04.index.pgvector.dsn_env == b00.index.pgvector.dsn_env == "PGVECTOR_DSN"
    assert b04.index.pgvector.index_type == b00.index.pgvector.index_type == "hnsw"

    assert b00.retrieval.retriever_type == "adaptive_category_aware_dense"
    assert b04.retrieval.retriever_type == "dense"
    assert b04.retrieval.top_k == b00.retrieval.top_k == 5
    assert b04.retrieval.fetch_k == b00.retrieval.fetch_k == 20
    assert b04.retrieval.metadata_boosting == b00.retrieval.metadata_boosting
    assert b04.retrieval.metadata_filtering == b00.retrieval.metadata_filtering
    assert b04.retrieval.fallback_to_global is False
    assert b04.retrieval.category_routing_validation.enabled is False
    assert b04.orchestration.enabled is False

    allowed = {
        "experiment.experiment_id",
        "index.index_name",
        "index.pgvector.table_name",
        "retrieval.retriever_type",
        "retrieval.fallback_to_global",
        "retrieval.category_routing_validation.enabled",
        "orchestration.enabled",
    }
    assert _diff_paths(b00.model_dump(), b04.model_dump()) == allowed


def test_f04_keeps_b04_controls_and_changes_only_retrieval_stack():
    b04 = _p1(B04_P1_PATH)
    f04 = _p1(F04_P1_PATH)

    assert f04.experiment.experiment_id == "F04"
    assert f04.chunking == b04.chunking
    assert f04.embedding == b04.embedding
    assert f04.generation == b04.generation
    assert f04.reranker == b04.reranker
    assert f04.data == b04.data
    assert f04.runtime == b04.runtime
    assert f04.telemetry == b04.telemetry
    assert f04.parent_context == b04.parent_context
    assert f04.orchestration == b04.orchestration

    assert f04.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert f04.retrieval.top_k == b04.retrieval.top_k == 5
    assert f04.retrieval.fetch_k == b04.retrieval.fetch_k == 20
    assert f04.retrieval.fallback_to_global is False
    assert f04.retrieval.category_routing_validation.enabled is False
    assert f04.retrieval.hybrid.rrf_k == 60
    assert f04.retrieval.hybrid.dense_fetch_k == 20
    assert f04.retrieval.hybrid.bm25_fetch_k == 20
    assert f04.retrieval.bm25.backend == "elasticsearch"
    assert f04.retrieval.bm25.index_name == "sivas_phase4_f04_bm25_b00_chunks"
    assert "c02" not in f04.retrieval.bm25.index_name.lower()

    assert f04.index.type == "elasticsearch"
    assert f04.index.index_name == "sivas_phase4_f04_dense_b00_chunks"
    assert f04.index.pgvector is None
    assert "c02" not in f04.index.index_name.lower()


def test_b04_and_f04_pipeline2_use_b00_chunk_annotations_and_source_validation():
    b04 = _p2(B04_P2_PATH)
    f04 = _p2(F04_P2_PATH)

    for exp_id, cfg, retriever in [
        ("B04", b04, "dense"),
        ("F04", f04, "elasticsearch_hybrid_rrf"),
    ]:
        assert cfg.evaluation.eval_run_id == exp_id
        assert cfg.inputs.pipeline1_results_path == f"data/runs/pipeline1/{exp_id}/results.jsonl"
        assert cfg.inputs.rag_outputs == [f"data/runs/pipeline1/{exp_id}/results.jsonl"]
        assert cfg.inputs.questions_path == "data/raw/questions_fixed.jsonl"
        assert cfg.inputs.qa_path == "data/raw/qa_ground_truth_fixed.jsonl"
        assert cfg.retrieval.ks == [1, 3, 5]
        assert cfg.retrieval_evaluation is not None
        assert cfg.retrieval_evaluation.document_level.enabled is True
        assert cfg.retrieval_evaluation.chunk_level.enabled is True
        assert cfg.retrieval_evaluation.chunk_level.ground_truth_path == B00_CHUNK_GT
        assert Path(cfg.retrieval_evaluation.chunk_level.ground_truth_path).is_file()

        sv = cfg.source_validation
        assert sv is not None
        assert sv.expected_experiment_id == exp_id
        assert sv.expected_retriever_type == retriever
        assert sv.expected_orchestration_enabled is False
        assert sv.expected_generator_provider == "ollama"
        assert sv.expected_generator_model == "mistral-small"
        assert sv.expected_top_k == 5
        assert sv.expected_fetch_k == 20
        assert sv.expected_reranker_enabled is False
        assert sv.expected_fallback_to_global is False
        assert sv.require_reranker_diagnostics is False
        assert sv.require_routing_diagnostics is False
        assert sv.require_routing_reconciliation is False

    assert b04.source_validation.require_hybrid_diagnostics is False
    assert f04.source_validation.require_hybrid_diagnostics is True


def test_b04_and_f04_pipeline3_use_official_b00_f00_evaluator_defaults():
    b04 = _p3(B04_P3_PATH)
    f04 = _p3(F04_P3_PATH)

    for exp_id, cfg, retriever in [
        ("B04", b04, "dense"),
        ("F04", f04, "elasticsearch_hybrid_rrf"),
    ]:
        assert cfg.pipeline3.run_id == exp_id
        assert cfg.inputs.pipeline1_results_path == f"data/runs/pipeline1/{exp_id}/results.jsonl"
        assert cfg.inputs.questions_path == "data/raw/questions_fixed.jsonl"
        assert cfg.inputs.qa_path == "data/raw/qa_ground_truth_fixed.jsonl"
        assert cfg.judge.model == "qwen2.5:14b"
        assert cfg.ragas.enabled is True
        assert cfg.ragas.metrics.context_recall is True
        assert cfg.llm_judge.enabled is True
        assert cfg.llm_judge.max_context_chars == 10000

        sv = cfg.source_validation
        assert sv is not None
        assert sv.expected_experiment_id == exp_id
        assert sv.expected_retriever_type == retriever
        assert sv.expected_orchestration_enabled is False
        assert sv.expected_generator_provider == "ollama"
        assert sv.expected_generator_model == "mistral-small"
        assert sv.expected_top_k == 5
        assert sv.expected_fetch_k == 20
        assert sv.expected_reranker_enabled is False
        assert sv.expected_fallback_to_global is False

    assert b04.source_validation.require_hybrid_diagnostics is False
    assert f04.source_validation.require_hybrid_diagnostics is True


def test_b04_and_f04_are_registered_in_official_mapping():
    mapping = load_yaml_mapping(Path(MAPPING_PATH))["official_experiment_mapping"]

    expected = {
        "B04": {
            "pipeline1": B04_P1_PATH,
            "pipeline2": B04_P2_PATH,
            "pipeline3": B04_P3_PATH,
        },
        "F04": {
            "pipeline1": F04_P1_PATH,
            "pipeline2": F04_P2_PATH,
            "pipeline3": F04_P3_PATH,
        },
    }
    for exp_id, paths in expected.items():
        assert mapping[exp_id] == paths
        assert Path(paths["pipeline1"]).is_file()
        assert Path(paths["pipeline2"]).is_file()
        assert Path(paths["pipeline3"]).is_file()
