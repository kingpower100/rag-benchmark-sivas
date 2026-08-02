from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.orchestrator import _category_routing_validation_manifest
from src.pipeline1.preflight import run_preflight_checks
from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
    AdaptiveCategoryAwareHybridRRFRetriever,
)
from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Retriever
from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever
from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import ElasticsearchHybridRRFRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.orchestration_stage import OrchestrationStage
from src.pipeline1.stages.retrieval_stage import RetrievalStage


def test_adaptive_hybrid_runs_orchestration_and_passes_category_to_retrieval():
    cfg = _cfg(reranker_enabled=False)
    wrapper, tracking = _adaptive_hybrid(
        dense_global=[
            _item("f1", 1.0, "Finanzen"),
            _item("f2", 0.9, "Finanzen"),
            _item("f3", 0.8, "Finanzen"),
            _item("h1", 0.7, "HR"),
        ],
        bm25_global=[],
        dense_category=[_item("cf1", 1.0, "Finanzen"), _item("cf2", 0.9, "Finanzen")],
        bm25_category=[_item("cb1", 1.0, "Finanzen")],
    )
    orch_calls = {"count": 0}

    orchestration = OrchestrationStage(
        cfg,
        _chunks(),
        generator_factory=lambda config: _OrchestrationGenerator("Finanzen", orch_calls),
    ).run(StageInput({"queries": [QueryRecord(question_id="q1", question="Welche Rechnung?")]}))

    query = orchestration.queries[0]
    assert orch_calls["count"] == 1
    assert query.detected_category == "Finanzen"
    assert query.category_validated is True

    output = _run_retrieval(cfg, wrapper, [query])
    diag = output.retrieval_rows[0].retrieval_diagnostics

    assert diag["predicted_category"] == "Finanzen"
    assert diag["routing_decision"] == "accepted"
    assert diag["final_retrieval_mode"] == "category"
    assert tracking["dense"].category_calls == [("Finanzen", "metadata.kategorie.keyword", 4)]
    assert tracking["bm25"].category_calls == [("Finanzen", "metadata.kategorie.keyword", 4)]


def test_preflight_rejects_adaptive_hybrid_when_orchestration_disabled(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="r01_preflight_") as tmp:
        project_root = _write_inputs(Path(tmp))
        cfg = _cfg(reranker_enabled=False)
        cfg = cfg.model_copy(update={"orchestration": cfg.orchestration.model_copy(update={"enabled": False})})
        monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")

        errors = run_preflight_checks(cfg, project_root)

    assert (
        "retrieval.retriever_type=adaptive_category_aware_hybrid_rrf "
        "requires orchestration.enabled=true"
    ) in errors


def test_accepted_route_calls_category_dense_bm25_rrf_and_reranker():
    cfg = _cfg(reranker_enabled=True, top_k=2, fetch_k=4)
    wrapper, tracking = _adaptive_hybrid(
        dense_global=[
            _item("f1", 1.0, "Finanzen"),
            _item("f2", 0.9, "Finanzen"),
            _item("f3", 0.8, "Finanzen"),
            _item("h1", 0.7, "HR"),
        ],
        bm25_global=[],
        dense_category=[_item("cf1", 1.0, "Finanzen"), _item("cf2", 0.9, "Finanzen")],
        bm25_category=[_item("cf1", 1.2, "Finanzen"), _item("cb1", 1.0, "Finanzen")],
    )
    reranker = _TrackingReranker()
    query = QueryRecord(
        question_id="q1",
        question="Q?",
        cleaned_question="Q?",
        detected_category="Finanzen",
        category_validated=True,
    )

    output = _run_retrieval(cfg, wrapper, [query], reranker=reranker)
    row = output.retrieval_rows[0]
    diag = row.retrieval_diagnostics

    assert tracking["dense"].global_calls == [4]
    assert tracking["bm25"].global_calls == [4]
    assert tracking["dense"].category_calls == [("Finanzen", "metadata.kategorie.keyword", 4)]
    assert tracking["bm25"].category_calls == [("Finanzen", "metadata.kategorie.keyword", 4)]
    assert tracking["hybrid"].fuse_calls == 2
    assert reranker.calls == [3]
    assert diag["final_retrieval_mode"] == "category"
    assert diag["retrieval_mode"] == "adaptive_category_aware_hybrid_rrf"
    assert diag["retriever_type"] == "adaptive_category_aware_hybrid_rrf"
    assert diag["effective_retriever_type"] == "adaptive_category_aware_hybrid_rrf"
    assert diag["retrieval_mode"] != "adaptive_category_aware_dense"
    assert diag["retrieval_scope"] == "category"
    assert diag["fallback_used"] is False
    assert diag["category_filter_applied_dense"] is True
    assert diag["category_filter_applied_bm25"] is True
    assert diag["dense_candidate_count"] == 2
    assert diag["bm25_candidate_count"] == 2
    assert diag["fused_candidate_count"] == 3
    assert diag["reranked_candidate_count"] == 2
    assert diag["final_context_count"] == 2
    assert len(row.retrieved) == cfg.retrieval.top_k


def test_rejected_route_uses_global_dense_bm25_rrf_and_records_reason():
    cfg = _cfg(reranker_enabled=False, top_k=2, fetch_k=4)
    wrapper, tracking = _adaptive_hybrid(
        dense_global=[
            _item("f1", 1.0, "Finanzen"),
            _item("h1", 0.9, "HR"),
            _item("h2", 0.8, "HR"),
            _item("h3", 0.7, "HR"),
        ],
        bm25_global=[_item("g1", 1.0, "HR")],
        dense_category=[_item("cf1", 1.0, "Finanzen")],
        bm25_category=[_item("cb1", 1.0, "Finanzen")],
    )
    query = QueryRecord(
        question_id="q1",
        question="Q?",
        cleaned_question="Q?",
        detected_category="Finanzen",
        category_validated=True,
    )

    output = _run_retrieval(cfg, wrapper, [query])
    diag = output.retrieval_rows[0].retrieval_diagnostics

    assert tracking["dense"].global_calls == [4, 4]
    assert tracking["bm25"].global_calls == [4, 4]
    assert tracking["dense"].category_calls == []
    assert tracking["bm25"].category_calls == []
    assert tracking["hybrid"].fuse_calls == 2
    assert diag["routing_decision"] == "rejected"
    assert diag["final_retrieval_mode"] == "global"
    assert diag["retrieval_scope"] == "global"
    assert diag["fallback_used"] is True
    assert "category" in diag["fallback_reason"]


def test_adaptive_hybrid_manifest_statistics_reconcile():
    cfg = _cfg(reranker_enabled=False)
    with tempfile.TemporaryDirectory(prefix="r01_manifest_") as tmp:
        run_dir = Path(tmp) / "R01"
        run_dir.mkdir()
        rows = [
            _manifest_row("q1", "Finanzen", True, "accepted", "category", False, None),
            _manifest_row("q2", "Finanzen", True, "rejected", "global", True, "category_share_below_threshold"),
            _manifest_row("q3", None, False, "rejected", "global", True, "invalid_category_global_fallback"),
        ]
        (run_dir / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        summary = _category_routing_validation_manifest(cfg, run_dir)

    assert summary["enabled"] is True
    assert summary["total_questions"] == 3
    assert summary["orchestration_attempted_count"] == 3
    assert summary["orchestration_success_count"] == 3
    assert summary["valid_category_count"] == 2
    assert summary["invalid_category_count"] == 1
    assert summary["category_route_count"] == 1
    assert summary["global_route_count"] == 2
    assert summary["fallback_count"] == 2
    assert summary["unknown_category_count"] == 1
    assert summary["routed_question_count"] == 3
    assert summary["category_route_count"] + summary["global_route_count"] == summary["routed_question_count"]
    assert summary["routing_acceptance_rate"] == pytest.approx(1 / 3)
    assert summary["category_route_rate"] == pytest.approx(1 / 3)
    assert summary["global_route_rate"] == pytest.approx(2 / 3)
    assert summary["fallback_rate"] == pytest.approx(2 / 3)


def test_elasticsearch_hybrid_rrf_r00_mode_does_not_use_orchestration_or_category_scope():
    cfg = _cfg(retriever_type="elasticsearch_hybrid_rrf", orchestration_enabled=False, reranker_enabled=False)
    hybrid, tracking = _global_hybrid(
        dense_global=[_item("d1", 1.0, "Finanzen")],
        bm25_global=[_item("b1", 1.0, "HR")],
    )
    query = QueryRecord(
        question_id="q1",
        question="Q?",
        category_validation_reason="orchestration_disabled",
    )

    output = _run_retrieval(cfg, hybrid, [query])
    diag = output.retrieval_rows[0].retrieval_diagnostics

    assert tracking["dense"].global_calls == [4]
    assert tracking["bm25"].global_calls == [4]
    assert tracking["dense"].category_calls == []
    assert tracking["bm25"].category_calls == []
    assert tracking["hybrid"].fuse_calls == 1
    assert diag["orchestration_status"] == "disabled"
    assert diag["retriever_type"] == "elasticsearch_hybrid_rrf"
    assert diag["retrieval_scope"] == "global"
    assert diag["category_filter_applied"] is False
    assert diag["fallback_used"] is False


def test_category_dense_retrieval_fails_closed_without_filtered_search():
    retriever = ElasticsearchDenseRetriever.__new__(ElasticsearchDenseRetriever)
    retriever.fetch_k = 4
    retriever.embedder = _Embedder()
    retriever.index = _UnfilteredIndex()
    retriever.chunks = []

    with pytest.raises(RuntimeError, match="refusing unfiltered fallback"):
        retriever.retrieve_with_category("Q?", 2, "Finanzen", "kategorie")

    assert retriever.index.global_calls == 0


def test_category_bm25_retrieval_fails_closed_without_search_support():
    retriever = ElasticsearchBM25Retriever.__new__(ElasticsearchBM25Retriever)
    retriever.client = object()

    with pytest.raises(RuntimeError, match="refusing unfiltered fallback"):
        retriever.retrieve_with_category("Q?", 2, "Finanzen", "kategorie")


class _OrchestrationGenerator:
    def __init__(self, category: str, calls: dict[str, int]) -> None:
        self.category = category
        self.calls = calls

    def generate(self, prompt):
        self.calls["count"] += 1
        return GenerationResult(
            answer=json.dumps({"cleaned_question": "Welche Rechnung?", "detected_category": self.category}),
            input_tokens=1,
            output_tokens=1,
        )


class _TrackingRetriever:
    def __init__(self, name, global_items, category_items):
        self.name = name
        self.global_items = list(global_items)
        self.category_items = list(category_items)
        self.global_calls: list[int] = []
        self.category_calls: list[tuple[str, str, int]] = []
        self.last_retrieval_diagnostics = {}
        self.last_bm25_candidates = []

    def retrieve(self, question: str, top_k: int):
        self.global_calls.append(top_k)
        return self.global_items[:top_k]

    def retrieve_with_category(self, question: str, top_k: int, category: str, category_field: str):
        self.category_calls.append((category, category_field, top_k))
        if self.name == "dense":
            self.last_retrieval_diagnostics = {"category_filter_applied_dense": True}
        else:
            self.last_retrieval_diagnostics = {"category_filter_applied_bm25": True}
        return self.category_items[:top_k]

    def extract_query_metadata(self, question: str):
        return None


class _TrackingHybrid(ElasticsearchHybridRRFRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_calls = 0

    def _fuse(self, dense, bm25):
        self.fuse_calls += 1
        return super()._fuse(dense, bm25)


class _TrackingReranker:
    def __init__(self, *args, **kwargs):
        self.model_name = "fake-reranker"
        self.requested_device = "cpu"
        self.runtime_device = "cpu"
        self.calls: list[int] = []

    def rerank(self, question: str, items: list[RetrievalItem], top_k: int):
        self.calls.append(len(items))
        return [
            item.model_copy(
                update={
                    "score": float(len(items) - index),
                    "rerank_score": float(len(items) - index),
                    "ranking_score_type": "rerank_score",
                }
            )
            for index, item in enumerate(items)
        ][:top_k]


def _adaptive_hybrid(dense_global, bm25_global, dense_category, bm25_category):
    hybrid, tracking = _global_hybrid(dense_global, bm25_global, dense_category, bm25_category)
    return AdaptiveCategoryAwareHybridRRFRetriever(hybrid, category_field="kategorie"), tracking


def _global_hybrid(dense_global, bm25_global, dense_category=None, bm25_category=None):
    dense = _TrackingRetriever("dense", dense_global, dense_category or [])
    bm25 = _TrackingRetriever("bm25", bm25_global, bm25_category or [])
    hybrid = _TrackingHybrid(
        dense_retriever=dense,
        bm25_retriever=bm25,
        fetch_k=4,
        dense_fetch_k=4,
        bm25_fetch_k=4,
        rrf_k=60,
    )
    return hybrid, {"dense": dense, "bm25": bm25, "hybrid": hybrid}


class _Embedder:
    def encode_query(self, text):
        return [0.1, 0.2]


class _UnfilteredIndex:
    def __init__(self) -> None:
        self.global_calls = 0

    def search_hits(self, *args, **kwargs):
        self.global_calls += 1
        return []


def _run_retrieval(cfg, retriever, queries, reranker=None):
    return RetrievalStage(
        cfg,
        embedder=object(),
        index=object(),
        chunks=_chunks(),
        retriever_factory=lambda *args, **kwargs: retriever,
        reranker_factory=(lambda *args, **kwargs: reranker or _TrackingReranker()),
    ).run(StageInput({"queries": queries}))


def _item(chunk_id: str, score: float, category: str):
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


def _chunks():
    return [
        ChunkRecord(
            chunk_id="c-fin",
            document_id="doc-fin",
            original_context_id="doc-fin",
            text="Finanzen alpha",
            chunk_start=0,
            chunk_end=14,
            metadata={"kategorie": "Finanzen"},
        ),
        ChunkRecord(
            chunk_id="c-hr",
            document_id="doc-hr",
            original_context_id="doc-hr",
            text="HR alpha",
            chunk_start=0,
            chunk_end=8,
            metadata={"kategorie": "HR"},
        ),
    ]


def _cfg(
    *,
    retriever_type: str = "adaptive_category_aware_hybrid_rrf",
    orchestration_enabled: bool = True,
    reranker_enabled: bool,
    top_k: int = 2,
    fetch_k: int = 4,
):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "R01", "output_dir": "runs"},
            "data": {"documents_path": "data/raw/kb_documents_fixed.jsonl", "questions_path": "data/raw/questions_fixed.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "elasticsearch", "dense_dim": 2},
            "retrieval": {
                "retriever_type": retriever_type,
                "top_k": top_k,
                "fetch_k": fetch_k,
                "category_routing_validation": {
                    "enabled": True,
                    "probe_fetch_k": fetch_k,
                    "minimum_category_share": 0.60,
                    "minimum_category_count": 3,
                    "minimum_margin": 2,
                },
                "bm25": {"backend": "local"},
                "hybrid": {"rrf_k": 60, "dense_fetch_k": fetch_k, "bm25_fetch_k": fetch_k},
            },
            "reranker": {
                "enabled": reranker_enabled,
                "model_name": "fake-reranker" if reranker_enabled else None,
                "device": "cpu",
                "rerank_top_k": fetch_k if reranker_enabled else None,
            },
            "orchestration": {"enabled": orchestration_enabled, "model_name": "llama3.1:8b", "fixed": True},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )


def _write_inputs(project_root: Path) -> Path:
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "kb_documents_fixed.jsonl").write_text(
        '{"doc_key":"doc-1","doc_name":"a.md","text":"alpha","kategorie":"Finanzen"}\n',
        encoding="utf-8",
    )
    (data_dir / "questions_fixed.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
    return project_root


def _manifest_row(qid, category, valid, decision, final_mode, fallback_used, fallback_reason):
    return {
        "question_id": qid,
        "detected_category": category,
        "category_validated": valid,
        "orchestration_error": None,
        "retrieval_diagnostics": {
            "retriever_type": "adaptive_category_aware_hybrid_rrf",
            "orchestration_status": "enabled",
            "predicted_category": category,
            "category_validated": valid,
            "routing_decision": decision,
            "routing_accepted": decision == "accepted",
            "decision_reason": fallback_reason or "thresholds_satisfied",
            "final_retrieval_mode": final_mode,
            "retrieval_scope": final_mode,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
        },
        "error": None,
    }
