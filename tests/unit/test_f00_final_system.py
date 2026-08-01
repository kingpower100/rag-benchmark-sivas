"""Tests for F00 final production system.

Verifies:
- F00 P1 config loads and resolves experiment_id to "F00" (not "F00_final_system").
- F00 effective architecture matches every benchmark-phase winner value.
- F00 P2 config loads and rejects P1 results from non-GPT-5.5 generator.
- F00 P3 config loads and rejects P1 results from non-GPT-5.5 generator.
- F00 startup summary validates architecture and raises on misconfiguration.
- Adaptive routing diagnostics carry required flags for source validation.
- RRF fusion fields are populated after category retrieve_with_category.
- Reranking evidence fields are populated in retrieval_diagnostics.
- top_k=5 is enforced in resolved config.
- Resume behaviour: existing question_ids are skipped without re-running.
- Existing configs (B00, G03, R01) are not modified by the F00 files.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evaluation.source_validation import validate_pipeline1_source
from src.pipeline1.config_loader import load_pipeline_config_payload
from src.pipeline1.orchestrator import _print_f00_startup_summary
from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
    AdaptiveCategoryAwareHybridRRFRetriever,
)
from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import (
    ElasticsearchHybridRRFRetriever,
)
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline2.config_loader import load_eval_config_payload
from src.pipeline2.schemas.eval_config_schema import EvalConfig
from src.pipeline3.schemas.pipeline3_config_schema import Pipeline3Config

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

F00_P1_PATH = "configs/pipeline1/final_experiments/F00_final_system.yaml"
F00_P2_PATH = "configs/pipeline2/final_experiments/F00_final_system_eval.yaml"
F00_P3_PATH = "configs/pipeline3/final_experiments/F00_final_system_eval.yaml"

B00_P1_PATH = "configs/pipeline1/final_experiments/B00_sivas_pgvector_reference.yaml"
G03_P1_PATH = "configs/pipeline1/final_experiments/G03_gpt55.yaml"
R01_P1_PATH = "configs/pipeline1/final_experiments/R01_adaptive_category_aware.yaml"


# ===========================================================================
# 1. F00 experiment_id resolves to "F00"
# ===========================================================================

def test_f00_experiment_id_is_F00():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    assert cfg.experiment.experiment_id == "F00", (
        f"Expected experiment_id='F00', got {cfg.experiment.experiment_id!r}. "
        "Update the F00 P1 config."
    )


# ===========================================================================
# 2. F00 architecture matches every benchmark-phase winner
# ===========================================================================

def test_f00_full_architecture():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    # Chunking (C02)
    assert cfg.chunking.strategy == "sentence"
    assert cfg.chunking.chunk_size == 1024
    assert cfg.chunking.chunk_overlap == 400
    # Embedding (E03)
    assert cfg.embedding.model_name == "mistral-embed"
    assert cfg.embedding.provider == "mistral"
    assert cfg.embedding.normalize_embeddings is True
    # Backend (V02 / Elasticsearch)
    assert cfg.index.type == "elasticsearch"
    assert cfg.index.index_name == "sivas_phase4_a00_elasticsearch_dense_mistral_embed"
    # Retrieval (A02 / Hybrid RRF)
    assert cfg.retrieval.retriever_type == "adaptive_category_aware_hybrid_rrf"
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.fetch_k == 20
    assert cfg.retrieval.fallback_to_global is True
    assert cfg.retrieval.hybrid.rrf_k == 60
    assert cfg.retrieval.hybrid.dense_fetch_k == 20
    assert cfg.retrieval.hybrid.bm25_fetch_k == 20
    assert cfg.retrieval.bm25.index_name == "sivas_phase4_a01_bm25_c02_chunks"
    # Routing validation (R01)
    rv = cfg.retrieval.category_routing_validation
    assert rv.enabled is True
    assert rv.probe_fetch_k == 20
    assert rv.minimum_category_share == pytest.approx(0.60)
    assert rv.minimum_category_count == 3
    assert rv.minimum_margin == 2
    # Reranker (A03)
    assert cfg.reranker.enabled is True
    assert cfg.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.reranker.device == "cuda"
    assert cfg.reranker.rerank_top_k == 20
    # Orchestration
    assert cfg.orchestration.enabled is True
    assert cfg.orchestration.model_name == "llama3.1:8b"
    # Generation (G03 / GPT-5.5)
    assert cfg.generation.provider == "openai"
    assert cfg.generation.model_name == "gpt-5.5"
    # Runtime
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False
    # Cost estimation disabled (placeholder pricing)
    assert cfg.telemetry.estimate_cost is False


# ===========================================================================
# 3. F00 P2 config loads and rejects non-GPT-5.5 source
# ===========================================================================

def test_f00_p2_config_loads():
    cfg = EvalConfig.model_validate(load_eval_config_payload(F00_P2_PATH))
    assert cfg.evaluation.eval_run_id == "F00"
    sv = cfg.source_validation
    assert sv is not None
    assert sv.expected_experiment_id == "F00"
    assert sv.expected_retriever_type == "adaptive_category_aware_hybrid_rrf"
    assert sv.expected_orchestration_enabled is True
    assert sv.expected_generator_provider == "openai"
    assert sv.expected_generator_model == "gpt-5.5"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.expected_fallback_to_global is True
    assert sv.require_hybrid_diagnostics is True
    assert sv.require_reranker_diagnostics is True
    assert sv.require_routing_diagnostics is True
    assert sv.require_routing_reconciliation is True


def test_f00_p2_rejects_non_openai_provider():
    sv = _f00_source_validation_config(generator_provider="ollama", generator_model="gpt-5.5")
    manifest = _f00_manifest(generator_provider="ollama", generator_model="gpt-5.5")
    rows = [_routing_row("q1", experiment_id="F00")]
    with pytest.raises(ValueError, match="generator_provider"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=rows,
            source_validation=sv,
            pipeline_name="Test P2",
        )


def test_f00_p2_rejects_wrong_generator_model():
    sv = _f00_source_validation_config(generator_provider="openai", generator_model="gpt-5.5")
    manifest = _f00_manifest(generator_provider="openai", generator_model="gpt-4o")
    rows = [_routing_row("q1", experiment_id="F00")]
    with pytest.raises(ValueError, match="generator_model"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=rows,
            source_validation=sv,
            pipeline_name="Test P2",
        )


# ===========================================================================
# 4. F00 P3 config loads and rejects non-GPT-5.5 source
# ===========================================================================

def test_f00_p3_config_loads():
    cfg = Pipeline3Config.model_validate(
        _load_p3_config(F00_P3_PATH)
    )
    assert cfg.pipeline3.run_id == "F00"
    sv = cfg.source_validation
    assert sv is not None
    assert sv.expected_experiment_id == "F00"
    assert sv.expected_generator_provider == "openai"
    assert sv.expected_generator_model == "gpt-5.5"
    assert sv.expected_top_k == 5
    assert sv.expected_fetch_k == 20
    assert sv.expected_reranker_enabled is True
    assert sv.expected_fallback_to_global is True


def test_f00_p3_rejects_wrong_generator_model():
    sv = _f00_p3_source_validation_config(generator_model="qwen2.5:7b-instruct")
    manifest = _f00_manifest(generator_provider="openai", generator_model="qwen2.5:7b-instruct")
    rows = [_routing_row("q1", experiment_id="F00")]
    with pytest.raises(ValueError, match="generator_model"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=rows,
            source_validation=sv,
            pipeline_name="Test P3",
        )


# ===========================================================================
# 5. Startup summary validates architecture, raises on misconfiguration
# ===========================================================================

def test_f00_startup_summary_prints_for_f00(capsys):
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    _print_f00_startup_summary(cfg, docs_count=10, chunk_count=500, question_count=100)
    out = capsys.readouterr().out
    assert "FINAL SYSTEM: F00" in out
    assert "adaptive_category_aware_hybrid_rrf" in out
    assert "gpt-5.5" in out
    assert "BAAI/bge-reranker-v2-m3" in out
    assert "Questions   : 100" in out
    assert "Chunks      : 500" in out


def test_f00_startup_summary_silent_for_other_experiment(capsys):
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    cfg = cfg.model_copy(
        update={"experiment": cfg.experiment.model_copy(update={"experiment_id": "G03"})}
    )
    _print_f00_startup_summary(cfg, docs_count=10, chunk_count=100, question_count=10)
    out = capsys.readouterr().out
    assert out == ""


def test_f00_startup_summary_raises_on_wrong_retriever():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    cfg = cfg.model_copy(
        update={"retrieval": cfg.retrieval.model_copy(update={"retriever_type": "elasticsearch_hybrid_rrf"})}
    )
    with pytest.raises(RuntimeError, match="retriever_type"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


def test_f00_startup_summary_raises_on_wrong_generator():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    cfg = cfg.model_copy(
        update={"generation": cfg.generation.model_copy(update={"model_name": "gpt-4o"})}
    )
    with pytest.raises(RuntimeError, match="gpt-5.5"):
        _print_f00_startup_summary(cfg, docs_count=1, chunk_count=1, question_count=1)


# ===========================================================================
# 6. Adaptive routing flags required for P2 source validation
# ===========================================================================

def test_f00_routing_flags_in_diagnostics():
    """category route must carry category_filter_applied_dense and _bm25."""
    sv = _f00_source_validation_config()
    manifest = _f00_manifest()
    row = _routing_row(
        "q1",
        scope="category",
        decision="accepted",
        final_mode="category",
        dense_filter=True,
        bm25_filter=True,
    )
    # Must not raise
    validate_pipeline1_source(
        results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
        manifest=manifest,
        rows=[row],
        source_validation=sv,
        pipeline_name="Test routing flags",
    )


def test_f00_routing_flags_missing_dense_filter_raises():
    sv = _f00_source_validation_config()
    manifest = _f00_manifest()
    row = _routing_row(
        "q1",
        scope="category",
        decision="accepted",
        final_mode="category",
        dense_filter=False,   # missing
        bm25_filter=True,
    )
    with pytest.raises(ValueError, match="category route lacks"):
        validate_pipeline1_source(
            results_path=Path("data/runs/pipeline1/F00/results.jsonl"),
            manifest=manifest,
            rows=[row],
            source_validation=sv,
            pipeline_name="Test missing dense filter",
        )


# ===========================================================================
# 7. RRF evidence fields populated after retrieve_with_category
# ===========================================================================

def test_rrf_diagnostics_populated_after_category_retrieve():
    dense = _TrackingRetriever("dense", [_item("d1", 1.0)], [_item("cd1", 0.9)])
    bm25 = _TrackingRetriever("bm25", [_item("b1", 1.0)], [_item("cb1", 0.8)])
    hybrid = ElasticsearchHybridRRFRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fetch_k=4,
        dense_fetch_k=4,
        bm25_fetch_k=4,
        rrf_k=60,
    )
    hybrid.retrieve_with_category("Frage?", 2, "Technik", "kategorie")
    diag = hybrid.last_retrieval_diagnostics
    assert diag["category_filter_applied"] is True
    assert diag["category_filter_applied_dense"] is True
    assert diag["category_filter_applied_bm25"] is True
    assert diag["es_hybrid_dense_candidates"] == 1
    assert diag["es_hybrid_bm25_candidates"] == 1
    assert diag["es_hybrid_fused_candidates"] == 2


# ===========================================================================
# 8. Reranker evidence recorded — confirmed via existing retrieval_stage tests
#    (end-to-end test_accepted_route_calls_category_dense_bm25_rrf_and_reranker
#    already asserts reranked_candidate_count and reranker_applied)
# ===========================================================================

def test_reranker_applied_flag_present_in_diagnostics():
    """Smoke-check the field name; deeper coverage is in test_r01_adaptive_hybrid_runtime."""
    from src.pipeline1.stages.retrieval_stage import RetrievalStage
    from src.pipeline1.schemas.query import QueryRecord
    from src.pipeline1.schemas.chunk import ChunkRecord
    from src.pipeline1.stages.base import StageInput

    cfg = _f00_cfg()
    wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
        _simple_hybrid(dense_items=[_item("d1", 1.0)], bm25_items=[_item("b1", 0.9)]),
        category_field="kategorie",
    )
    reranker = _IdentityReranker()
    query = QueryRecord(
        question_id="q1",
        question="Q?",
        cleaned_question="Q?",
        detected_category="Technik",
        category_validated=True,
    )
    output = RetrievalStage(
        cfg,
        embedder=object(),
        index=object(),
        chunks=_chunks(),
        retriever_factory=lambda *a, **kw: wrapper,
        reranker_factory=lambda *a, **kw: reranker,
    ).run(StageInput({"queries": [query]}))
    diag = output.retrieval_rows[0].retrieval_diagnostics
    assert diag.get("reranker_applied") is True


# ===========================================================================
# 9. top_k=5 enforced in resolved config
# ===========================================================================

def test_f00_top_k_is_5():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    assert cfg.retrieval.top_k == 5


# ===========================================================================
# 10. Resume behaviour: existing IDs are not in pending_queries
# ===========================================================================

def test_f00_resume_skips_completed_questions():
    """RunWriterStage skips question_ids already in results.jsonl (resume=true)."""
    from src.pipeline1.stages.run_writer_stage import RunWriterStage
    from src.pipeline1.schemas.config_schema import PipelineConfig

    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(F00_P1_PATH))
    assert cfg.runtime.resume is True
    assert cfg.runtime.overwrite is False

    with tempfile.TemporaryDirectory(prefix="f00_resume_") as tmp:
        run_dir = Path(tmp) / "F00"
        run_dir.mkdir()
        # Pre-write one completed row
        results = run_dir / "results.jsonl"
        results.write_text(
            json.dumps({"question_id": "q1", "generated_answer": "done"}) + "\n",
            encoding="utf-8",
        )
        stage = RunWriterStage(run_dir, save_csv=False, resume=True)
        out = stage.run()
        stage.close()

    assert "q1" in out.existing_question_ids


# ===========================================================================
# 11. Existing configs are unmodified
# ===========================================================================

def test_b00_not_modified_by_f00():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(B00_P1_PATH))
    assert cfg.experiment.experiment_id == "B00_sivas_pgvector_reference"
    assert cfg.retrieval.retriever_type == "adaptive_category_aware_dense"
    assert cfg.index.type == "pgvector"


def test_g03_not_modified_by_f00():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(G03_P1_PATH))
    assert cfg.experiment.experiment_id == "G03"
    assert cfg.generation.model_name == "gpt-5.5"
    assert cfg.retrieval.retriever_type != "adaptive_category_aware_hybrid_rrf"


def test_r01_not_modified_by_f00():
    cfg = PipelineConfig.model_validate(load_pipeline_config_payload(R01_P1_PATH))
    assert cfg.experiment.experiment_id == "R01"
    assert cfg.retrieval.retriever_type == "adaptive_category_aware_hybrid_rrf"


# ===========================================================================
# Helpers
# ===========================================================================

def _item(chunk_id: str, score: float, category: str = "Technik") -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"ctx-{chunk_id}",
        text=f"text-{chunk_id}",
        score=score,
        dense_score=score,
        bm25_score=None,
        retrieval_source="test",
        metadata={"kategorie": category, "document_id": f"doc-{chunk_id}"},
    )


class _TrackingRetriever:
    def __init__(self, name, global_items, category_items=None):
        self.name = name
        self.global_items = list(global_items)
        self.category_items = list(category_items or [])
        self.last_retrieval_diagnostics: dict = {}
        self.last_bm25_candidates: list = []

    def retrieve(self, question: str, top_k: int):
        return self.global_items[:top_k]

    def retrieve_with_category(self, question: str, top_k: int, category: str, category_field: str):
        if self.name == "dense":
            self.last_retrieval_diagnostics = {"category_filter_applied_dense": True}
        else:
            self.last_retrieval_diagnostics = {"category_filter_applied_bm25": True}
        return self.category_items[:top_k]

    def extract_query_metadata(self, question: str):
        return None


class _IdentityReranker:
    model_name = "identity"
    requested_device = "cpu"
    runtime_device = "cpu"

    def rerank(self, question, items, top_k):
        return [
            item.model_copy(update={"rerank_score": item.score, "ranking_score_type": "rerank_score"})
            for item in items
        ][:top_k]


def _simple_hybrid(dense_items, bm25_items):
    dense = _TrackingRetriever("dense", dense_items, dense_items)
    bm25 = _TrackingRetriever("bm25", bm25_items, bm25_items)
    return ElasticsearchHybridRRFRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fetch_k=4,
        dense_fetch_k=4,
        bm25_fetch_k=4,
        rrf_k=60,
    )


def _chunks():
    from src.pipeline1.schemas.chunk import ChunkRecord
    return [
        ChunkRecord(
            chunk_id="c1",
            document_id="doc1",
            original_context_id="doc1",
            text="Technik text",
            chunk_start=0,
            chunk_end=12,
            metadata={"kategorie": "Technik"},
        )
    ]


def _f00_cfg():
    """Minimal PipelineConfig matching F00 architecture for unit tests."""
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "F00", "output_dir": "runs"},
            "data": {
                "documents_path": "data/raw/kb_documents_fixed.jsonl",
                "questions_path": "data/raw/questions_fixed.jsonl",
            },
            "chunking": {
                "strategy": "sentence",
                "chunk_size": 1024,
                "chunk_overlap": 400,
                "chunk_size_unit": "tokens",
                "chunk_overlap_unit": "tokens",
            },
            "embedding": {
                "provider": "sentence_transformers",
                "model_name": "intfloat/multilingual-e5-large",
                "normalize_embeddings": True,
            },
            "index": {
                "type": "elasticsearch",
                "dense_dim": 1024,
                "index_name": "sivas_phase4_a00_elasticsearch_dense_mistral_embed",
            },
            "retrieval": {
                "retriever_type": "adaptive_category_aware_hybrid_rrf",
                "top_k": 5,
                "fetch_k": 20,
                "fallback_to_global": True,
                "category_routing_validation": {
                    "enabled": True,
                    "probe_fetch_k": 20,
                    "minimum_category_share": 0.60,
                    "minimum_category_count": 3,
                    "minimum_margin": 2,
                },
                "bm25": {
                    "backend": "elasticsearch",
                    "index_name": "sivas_phase4_a01_bm25_c02_chunks",
                },
                "hybrid": {"rrf_k": 60, "dense_fetch_k": 20, "bm25_fetch_k": 20},
            },
            "reranker": {
                "enabled": True,
                "model_name": "BAAI/bge-reranker-v2-m3",
                "device": "cuda",
                "rerank_top_k": 20,
            },
            "orchestration": {
                "enabled": True,
                "model_name": "llama3.1:8b",
                "prompt_path": "orchestration_promptV4.txt",
            },
            "generation": {
                "provider": "openai",
                "model_name": "gpt-5.5",
                "system_prompt": "Use the context.",
            },
            "telemetry": {"estimate_cost": False},
            "runtime": {},
        }
    )


class _MockSourceValidation:
    """Minimal source_validation object for unit-testing validate_pipeline1_source."""
    def __init__(self, **kwargs):
        self.expected_experiment_id = kwargs.get("expected_experiment_id", "F00")
        self.expected_retriever_type = kwargs.get("expected_retriever_type", "adaptive_category_aware_hybrid_rrf")
        self.expected_orchestration_enabled = kwargs.get("expected_orchestration_enabled", True)
        self.expected_generator_provider = kwargs.get("expected_generator_provider", "openai")
        self.expected_generator_model = kwargs.get("expected_generator_model", "gpt-5.5")
        self.expected_top_k = kwargs.get("expected_top_k", 5)
        self.expected_fetch_k = kwargs.get("expected_fetch_k", 20)
        self.expected_reranker_enabled = kwargs.get("expected_reranker_enabled", True)
        self.expected_fallback_to_global = kwargs.get("expected_fallback_to_global", True)
        self.require_hybrid_diagnostics = kwargs.get("require_hybrid_diagnostics", True)
        self.require_reranker_diagnostics = kwargs.get("require_reranker_diagnostics", True)
        self.require_routing_diagnostics = kwargs.get("require_routing_diagnostics", True)
        self.require_routing_reconciliation = kwargs.get("require_routing_reconciliation", True)


def _f00_source_validation_config(**overrides) -> _MockSourceValidation:
    return _MockSourceValidation(**overrides)


def _f00_p3_source_validation_config(**overrides) -> _MockSourceValidation:
    return _MockSourceValidation(**overrides)


def _f00_manifest(
    *,
    generator_provider: str = "openai",
    generator_model: str = "gpt-5.5",
    top_k: int = 5,
    fetch_k: int = 20,
    reranker_enabled: bool = True,
    fallback_to_global: bool = True,
) -> dict:
    return {
        "run_id": "F00",
        "run_status": "PASS",
        "expected_questions": 1,
        "successful_questions": 1,
        "failed_questions": 0,
        "orchestration_enabled": True,
        "models": {
            "retriever_type": "adaptive_category_aware_hybrid_rrf",
            "generator_provider": generator_provider,
            "generator_model": generator_model,
            "reranker_enabled": reranker_enabled,
        },
        "config": {
            "retrieval": {
                "retriever_type": "adaptive_category_aware_hybrid_rrf",
                "top_k": top_k,
                "fetch_k": fetch_k,
                "fallback_to_global": fallback_to_global,
            },
            "reranker": {"enabled": reranker_enabled},
        },
        "category_routing_validation": {
            "category_route_count": 1,
            "global_route_count": 0,
            "fallback_count": 0,
        },
    }


def _routing_row(
    question_id: str,
    *,
    experiment_id: str = "F00",
    scope: str = "category",
    decision: str = "accepted",
    final_mode: str = "category",
    dense_filter: bool = True,
    bm25_filter: bool = True,
    fallback_used: bool = False,
) -> dict:
    return {
        "question_id": question_id,
        "experiment_id": experiment_id,
        "retriever_type": "adaptive_category_aware_hybrid_rrf",
        "retrieval_mode": "adaptive_category_aware_hybrid_rrf",
        "reranker_applied": True,
        "generated_answer": "Answer text.",
        "retrieval_diagnostics": {
            "retriever_type": "adaptive_category_aware_hybrid_rrf",
            "retrieval_scope": scope,
            "routing_decision": decision,
            "final_retrieval_mode": final_mode,
            "fallback_used": fallback_used,
            "category_filter_applied_dense": dense_filter,
            "category_filter_applied_bm25": bm25_filter,
            "dense_candidate_count": 5,
            "bm25_candidate_count": 5,
            "fused_candidate_count": 7,
            "reranker_applied": True,
            "reranked_candidate_count": 5,
            "final_context_count": 5,
            "top_k": 5,
        },
    }


def _load_p3_config(path: str) -> dict:
    from src.pipeline3.config_loader import load_pipeline3_config_payload
    return load_pipeline3_config_payload(path)
