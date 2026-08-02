"""Tests for the F00 global production candidate."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluation.source_validation import validate_pipeline1_source
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.orchestrator import _print_f00_startup_summary, _query_with_orchestration_disabled
from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import ElasticsearchHybridRRFRetriever
from src.pipeline1.retrieval.factory import build_retriever
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.run_writer_stage import RunWriterStage
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.config_loader import load_pipeline3_config_payload
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config


F00_P1_PATH = "configs/pipeline1/final_experiments/F00_final_system.yaml"
F00_P2_PATH = "configs/pipeline2/final_experiments/F00_final_system_eval.yaml"
F00_P3_PATH = "configs/pipeline3/final_experiments/F00_final_system_eval.yaml"

B00_P1_PATH = "configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml"
G03_P1_PATH = "configs/pipeline1/final_experiments/G03_gpt55.yaml"
R01_P1_PATH = "configs/pipeline1/final_experiments/R01_adaptive_category_aware.yaml"


def test_f00_experiment_id_is_f00():
    cfg = _f00_cfg()

    assert cfg.experiment.experiment_id == "F00"


def test_f00_full_architecture_matches_global_ground_truth():
    cfg = _f00_cfg()

    assert cfg.chunking.strategy == "sentence"
    assert cfg.chunking.chunk_size == 1024
    assert cfg.chunking.chunk_overlap == 400
    assert cfg.chunking.chunk_size_unit == "tokens"
    assert cfg.chunking.chunk_overlap_unit == "tokens"
    assert cfg.embedding.provider == "mistral"
    assert cfg.embedding.model_name == "mistral-embed"
    assert cfg.embedding.expected_dimension == 1024
    assert cfg.embedding.normalize_embeddings is True
    assert cfg.index.type == "elasticsearch"
    assert cfg.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.fetch_k == 20
    assert cfg.retrieval.hybrid.dense_fetch_k == 20
    assert cfg.retrieval.hybrid.bm25_fetch_k == 20
    assert cfg.retrieval.hybrid.rrf_k == 60
    assert cfg.retrieval.category_routing_validation.enabled is False
    assert cfg.retrieval.fallback_to_global is False
    assert cfg.reranker.enabled is True
    assert cfg.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.reranker.device == "cuda"
    assert cfg.reranker.rerank_top_k == 20
    assert cfg.orchestration.enabled is False
    assert cfg.generation.provider == "openai"
    assert cfg.generation.model_name == "gpt-5.5"
    assert cfg.generation.max_tokens == 2048
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False
    assert cfg.telemetry.estimate_cost is False


def test_f00_factory_builds_global_hybrid_not_adaptive(monkeypatch):
    cfg = _f00_cfg()
    calls: list[str] = []

    class Dense:
        def __init__(self, *args, **kwargs):
            calls.append("dense")

    class BM25:
        def __init__(self, *args, **kwargs):
            calls.append("bm25")

    monkeypatch.setattr("src.pipeline1.retrieval.factory.ElasticsearchDenseRetriever", Dense)
    monkeypatch.setattr("src.pipeline1.retrieval.factory.ElasticsearchBM25Retriever", BM25)

    retriever = build_retriever(cfg.retrieval, embedder=object(), index=object(), chunks=[])

    assert isinstance(retriever, ElasticsearchHybridRRFRetriever)
    assert calls == ["dense", "bm25"]
    assert retriever.dense_fetch_k == 20
    assert retriever.bm25_fetch_k == 20
    assert retriever.rrf_k == 60
    assert "adaptive" not in type(retriever).__name__.lower()


def test_f00_orchestration_stage_is_skipped_by_resolved_config():
    cfg = _f00_cfg()
    query = QueryRecord(question_id="Q001", question="Wie funktioniert das?")

    assert cfg.orchestration.enabled is False
    skipped = _query_with_orchestration_disabled(query)

    assert skipped.cleaned_question == query.question
    assert skipped.detected_category is None
    assert skipped.category_validated is False
    assert skipped.category_validation_reason == "orchestration_disabled"


def test_f00_p2_config_loads_with_global_expectations():
    cfg = EvalConfig.model_validate(load_eval_config_payload(F00_P2_PATH))
    sv = cfg.source_validation

    assert cfg.evaluation.eval_run_id == "F00"
    assert sv.expected_experiment_id == "F00"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "openai"
    assert sv.expected_generator_model == "gpt-5.5"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.expected_fallback_to_global is None
    assert sv.require_hybrid_diagnostics is True
    assert sv.require_reranker_diagnostics is True
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f00_p3_config_loads_with_global_expectations():
    cfg = Pipeline3Config.model_validate(load_pipeline3_config_payload(F00_P3_PATH))
    sv = cfg.source_validation

    assert cfg.pipeline3.run_id == "F00"
    assert cfg.inputs.pipeline1_results_path == "data/runs/pipeline1/F00/results.jsonl"
    assert sv.expected_experiment_id == "F00"
    assert sv.expected_retriever_type == "elasticsearch_hybrid_rrf"
    assert sv.expected_orchestration_enabled is False
    assert sv.expected_generator_provider == "openai"
    assert sv.expected_generator_model == "gpt-5.5"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.require_routing_diagnostics is False
    assert sv.require_routing_reconciliation is False


def test_f00_source_validation_rejects_non_openai_provider():
    manifest = _f00_manifest(generator_provider="ollama")

    with pytest.raises(ValueError, match="generator_provider"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=[_global_row("Q001")],
            source_validation=_source_validation(),
            pipeline_name="Test P2",
        )


def test_f00_source_validation_rejects_wrong_generator_model():
    manifest = _f00_manifest(generator_model="gpt-4o")

    with pytest.raises(ValueError, match="generator_model"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=[_global_row("Q001")],
            source_validation=_source_validation(),
            pipeline_name="Test P2",
        )


def test_f00_source_validation_rejects_routing_diagnostics():
    row = _global_row("Q001")
    row["retrieval_diagnostics"]["routing_decision"] = "accepted"

    with pytest.raises(ValueError, match="adaptive routing diagnostics"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=_f00_manifest(),
            rows=[row],
            source_validation=_source_validation(),
            pipeline_name="Test P2",
        )


def test_f00_startup_summary_prints_global_architecture(capsys):
    cfg = _f00_cfg()

    _print_f00_startup_summary(cfg, docs_count=65, chunk_count=750, question_count=96)
    out = capsys.readouterr().out

    assert "FINAL SYSTEM: F00" in out
    assert "Retrieval strategy: Global" in out
    assert "Retrieval method  : Elasticsearch Hybrid RRF" in out
    assert "Orchestration     : Disabled" in out
    assert "Category routing  : Disabled" in out
    assert "Category validation: Disabled" in out
    assert "Final top_k       : 5" in out
    assert "Generator         : openai / gpt-5.5" in out
    assert "Max tokens        : 2048" in out
    assert "Documents         : 65" in out
    assert "Chunks            : 750" in out
    assert "Questions         : 96" in out
    assert "adaptive category-aware" not in out
    assert "probe_fetch_k" not in out
    assert "llama3.1" not in out
    assert "Prompt V4" not in out


def test_f00_startup_summary_silent_for_other_experiment(capsys):
    cfg = _f00_cfg().model_copy(
        update={"experiment": _f00_cfg().experiment.model_copy(update={"experiment_id": "G03"})}
    )

    _print_f00_startup_summary(cfg, docs_count=10, chunk_count=100, question_count=10)

    assert capsys.readouterr().out == ""


def test_f00_startup_summary_raises_on_wrong_retriever():
    cfg = _f00_cfg().model_copy(
        update={"retrieval": _f00_cfg().retrieval.model_copy(update={"retriever_type": "adaptive_category_aware_hybrid_rrf"})}
    )

    with pytest.raises(RuntimeError, match="elasticsearch_hybrid_rrf"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


def test_f00_startup_summary_raises_on_orchestration_enabled():
    cfg = _f00_cfg().model_copy(
        update={"orchestration": _f00_cfg().orchestration.model_copy(update={"enabled": True})}
    )

    with pytest.raises(RuntimeError, match="orchestration.enabled=True"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


def test_f00_startup_summary_raises_on_routing_validation_enabled():
    retrieval = _f00_cfg().retrieval
    cfg = _f00_cfg().model_copy(
        update={
            "retrieval": retrieval.model_copy(
                update={
                    "category_routing_validation": retrieval.category_routing_validation.model_copy(
                        update={"enabled": True}
                    )
                }
            )
        }
    )

    with pytest.raises(RuntimeError, match="category_routing_validation.enabled=True"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


def test_f00_startup_summary_raises_on_wrong_generator():
    cfg = _f00_cfg().model_copy(
        update={"generation": _f00_cfg().generation.model_copy(update={"model_name": "gpt-4o"})}
    )

    with pytest.raises(RuntimeError, match="gpt-5.5"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


def test_f00_resume_skips_completed_questions():
    cfg = _f00_cfg()
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False

    with tempfile.TemporaryDirectory(prefix="f00_resume_") as tmp:
        run_dir = Path(tmp) / "F00"
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text(
            json.dumps({"question_id": "Q001", "generated_answer": "done"}) + "\n",
            encoding="utf-8",
        )
        stage = RunWriterStage(run_dir, save_csv=False, resume=True)
        out = stage.run()
        stage.close()

    assert "Q001" in out.existing_question_ids


def test_b00_g03_r01_not_modified_by_f00():
    b00 = PipelineConfig.model_validate(load_pipeline_config_payload(B00_P1_PATH))
    g03 = PipelineConfig.model_validate(load_pipeline_config_payload(G03_P1_PATH))
    r01 = PipelineConfig.model_validate(load_pipeline_config_payload(R01_P1_PATH))

    assert b00.retrieval.retriever_type == "adaptive_category_aware_dense"
    assert b00.index.type == "pgvector"
    assert g03.experiment.experiment_id == "G03"
    assert g03.generation.model_name == "gpt-5.5"
    assert g03.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
    assert r01.experiment.experiment_id == "R01"
    assert r01.retrieval.retriever_type == "adaptive_category_aware_hybrid_rrf"


def _f00_cfg() -> PipelineConfig:
    return PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))


class _SourceValidation:
    expected_experiment_id = "F00"
    expected_retriever_type = "elasticsearch_hybrid_rrf"
    expected_orchestration_enabled = False
    expected_generator_provider = "openai"
    expected_generator_model = "gpt-5.5"
    expected_top_k = 5
    expected_fetch_k = 20
    expected_reranker_enabled = True
    expected_fallback_to_global = None
    require_hybrid_diagnostics = True
    require_reranker_diagnostics = True
    require_routing_diagnostics = False
    require_routing_reconciliation = False


def _source_validation() -> _SourceValidation:
    return _SourceValidation()


def _f00_manifest(
    *,
    generator_provider: str = "openai",
    generator_model: str = "gpt-5.5",
) -> dict:
    return {
        "run_id": "F00",
        "run_status": "PASS",
        "expected_questions": 1,
        "successful_questions": 1,
        "failed_questions": 0,
        "orchestration_enabled": False,
        "models": {
            "retriever_type": "elasticsearch_hybrid_rrf",
            "generator_provider": generator_provider,
            "generator_model": generator_model,
            "reranker_enabled": True,
        },
        "config": {
            "retrieval": {
                "retriever_type": "elasticsearch_hybrid_rrf",
                "top_k": 5,
                "fetch_k": 20,
            },
            "reranker": {"enabled": True},
        },
        "resolved_config": {
            "retrieval": {
                "retriever_type": "elasticsearch_hybrid_rrf",
                "top_k": 5,
                "fetch_k": 20,
            },
            "generation": {
                "provider": generator_provider,
                "model_name": generator_model,
            },
            "reranker": {"enabled": True},
        },
    }


def _global_row(question_id: str) -> dict:
    return {
        "question_id": question_id,
        "experiment_id": "F00",
        "retriever_type": "elasticsearch_hybrid_rrf",
        "retrieval_mode": "elasticsearch_hybrid_rrf",
        "reranker_applied": True,
        "generated_answer": "Answer text.",
        "retrieval_diagnostics": {
            "retriever_type": "elasticsearch_hybrid_rrf",
            "configured_retriever_type": "elasticsearch_hybrid_rrf",
            "effective_retriever_type": "elasticsearch_hybrid_rrf",
            "retrieval_scope": "global",
            "dense_candidate_count": 20,
            "bm25_candidate_count": 20,
            "fused_candidate_count": 20,
            "reranker_applied": True,
            "reranked_candidate_count": 20,
            "final_context_count": 5,
            "top_k": 5,
        },
    }
