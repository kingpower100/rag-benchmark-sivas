"""
Tests for the R01 Elasticsearch category-filter field fix.

Verifies:
1. Dense uses metadata.kategorie.keyword, not metadata.kategorie
2. BM25 uses metadata.kategorie.keyword, not metadata.kategorie
3. Config auto-derives category_filter_field from category_field
4. Empty-category fallback: 0 category results → global called, reason recorded
5. Non-empty category route: fallback_used=False, final_retrieval_mode=category
6. Reranker reranker_applied=False when zero candidates
7. Diagnostics preserved in empty-category fallback
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
    AdaptiveCategoryAwareHybridRRFRetriever,
)
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.stages.retrieval_stage import RetrievalStage, retrieve_top_k_unique_contexts
from src.pipeline1.stages.base import StageInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(chunk_id: str, score: float, category: str = "Finanzen") -> RetrievalItem:
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


class _FieldCapturingIndex:
    """Fake ES index that records the filter field passed to search_hits_filtered."""
    def __init__(self):
        self.captured_field: str | None = None

    def search_hits_filtered(self, query_vec, top_k, category_field, category):
        self.captured_field = category_field
        return []

    def search_hits(self, query_vec, top_k):
        return []


class _StubEmbedder:
    def encode_query(self, text):
        return [0.0]


class _StubBM25Client:
    """Fake ES BM25 client that records the filter field in query body."""
    def __init__(self, items=None):
        self.items = items or []
        self.captured_filter_field: str | None = None

    def search(self, index, size, query):
        filters = query.get("bool", {}).get("filter", [])
        for f in filters:
            term_keys = list(f.get("term", {}).keys())
            if term_keys:
                self.captured_filter_field = term_keys[0]
        return {"hits": {"hits": []}}

    def info(self):
        return {}

    @property
    def indices(self):
        return _FakeIndices()


class _FakeIndices:
    def exists(self, index):
        return True

    def refresh(self, index):
        pass


class _TrackingRetriever:
    """Minimal stub that tracks category_field received in retrieve_with_category."""
    def __init__(self, category_items=None, global_items=None):
        self._category_items = category_items or []
        self._global_items = global_items or []
        self.category_field_received: str | None = None
        self.category_calls: int = 0
        self.global_calls: int = 0
        self.last_retrieval_diagnostics: dict = {}
        self.last_dense_candidates: list = []
        self.last_bm25_candidates: list = []
        self.last_fused_candidates: list = []

    def retrieve(self, question: str, top_k: int):
        self.global_calls += 1
        self.last_retrieval_diagnostics = {}
        return self._global_items[:top_k]

    def retrieve_with_category(self, question: str, top_k: int, category: str, category_field: str):
        self.category_calls += 1
        self.category_field_received = category_field
        self.last_retrieval_diagnostics = {
            "category_filter_applied_dense": True,
            "category_filter_applied_bm25": True,
        }
        return self._category_items[:top_k]

    def set_active_category(self, category):
        pass

    def retrieve_global_probe(self, question: str, probe_fetch_k: int):
        return []

    def extract_query_metadata(self, question: str):
        return None


class _TrackingReranker:
    def __init__(self):
        self.model_name = "fake-reranker"
        self.requested_device = "cpu"
        self.runtime_device = "cpu"
        self.calls: list[int] = []

    def rerank(self, question: str, items: list, top_k: int):
        self.calls.append(len(items))
        return items[:top_k]


def _cfg(top_k: int = 2, fetch_k: int = 4, reranker_enabled: bool = False) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "TEST", "output_dir": "runs"},
            "data": {"documents_path": "data/raw/kb_documents_fixed.jsonl", "questions_path": "data/raw/questions_fixed.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "elasticsearch", "dense_dim": 2},
            "retrieval": {
                "retriever_type": "adaptive_category_aware_hybrid_rrf",
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
            },
            "orchestration": {"enabled": True, "model_name": "llama3.1:8b", "fixed": True},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )


def _run(cfg, retriever, queries, reranker=None):
    return RetrievalStage(
        cfg,
        embedder=object(),
        index=object(),
        chunks=[],
        retriever_factory=lambda *a, **kw: retriever,
        reranker_factory=(lambda *a, **kw: reranker) if reranker else (lambda *a, **kw: None),
    ).run(StageInput({"queries": queries}))


def _accepted_query(category: str = "Finanzen") -> QueryRecord:
    return QueryRecord(
        question_id="q1",
        question="Welche Rechnung?",
        cleaned_question="Welche Rechnung?",
        detected_category=category,
        category_validated=True,
    )


# ---------------------------------------------------------------------------
# Test 1 — Dense retriever uses the full ES filter field directly
# ---------------------------------------------------------------------------

class TestDenseUsesKeywordField:
    def test_dense_passes_full_field_to_index(self):
        from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever
        from src.pipeline1.schemas.chunk import ChunkRecord

        index = _FieldCapturingIndex()
        retriever = ElasticsearchDenseRetriever(
            embedder=_StubEmbedder(),
            index=index,
            chunks=[],
            top_k=2,
            fetch_k=4,
            metadata_boosting=type("MB", (), {"enabled": False})(),
            metadata_filtering=type("MF", (), {"enabled": False})(),
        )
        retriever.retrieve_with_category("Q?", 2, "Finanzen", "metadata.kategorie.keyword")
        assert index.captured_field == "metadata.kategorie.keyword", (
            f"Expected 'metadata.kategorie.keyword', got {index.captured_field!r}"
        )

    def test_dense_does_not_prepend_metadata_prefix(self):
        from src.pipeline1.retrieval.elasticsearch_dense_retriever import ElasticsearchDenseRetriever

        index = _FieldCapturingIndex()
        retriever = ElasticsearchDenseRetriever(
            embedder=_StubEmbedder(),
            index=index,
            chunks=[],
            top_k=2,
            fetch_k=4,
            metadata_boosting=type("MB", (), {"enabled": False})(),
            metadata_filtering=type("MF", (), {"enabled": False})(),
        )
        retriever.retrieve_with_category("Q?", 2, "Finanzen", "metadata.kategorie.keyword")
        # The field must NOT be double-prefixed
        assert index.captured_field != "metadata.metadata.kategorie.keyword"
        assert "metadata.metadata" not in (index.captured_field or "")


# ---------------------------------------------------------------------------
# Test 2 — BM25 retriever uses the full ES filter field directly
# ---------------------------------------------------------------------------

class TestBM25UsesKeywordField:
    def test_bm25_passes_full_field_in_term_filter(self):
        from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Retriever

        client = _StubBM25Client()
        retriever = ElasticsearchBM25Retriever.__new__(ElasticsearchBM25Retriever)
        retriever.client = client
        retriever.index_name = "test-index"
        retriever.host = "http://localhost:9200"
        retriever.analyzer = "german"
        retriever.chunk_by_id = {}
        retriever.last_bm25_candidates = []
        retriever.last_retrieval_diagnostics = {}

        retriever.retrieve_with_category("Q?", 2, "Finanzen", "metadata.kategorie.keyword")
        assert client.captured_filter_field == "metadata.kategorie.keyword", (
            f"Expected 'metadata.kategorie.keyword', got {client.captured_filter_field!r}"
        )

    def test_bm25_does_not_prepend_metadata_prefix(self):
        from src.pipeline1.retrieval.elasticsearch_bm25_retriever import ElasticsearchBM25Retriever

        client = _StubBM25Client()
        retriever = ElasticsearchBM25Retriever.__new__(ElasticsearchBM25Retriever)
        retriever.client = client
        retriever.index_name = "test-index"
        retriever.host = "http://localhost:9200"
        retriever.analyzer = "german"
        retriever.chunk_by_id = {}
        retriever.last_bm25_candidates = []
        retriever.last_retrieval_diagnostics = {}

        retriever.retrieve_with_category("Q?", 2, "Finanzen", "metadata.kategorie.keyword")
        assert "metadata.metadata" not in (client.captured_filter_field or "")


# ---------------------------------------------------------------------------
# Test 3 — Config auto-derives category_filter_field
# ---------------------------------------------------------------------------

class TestConfigAutoDerivesFilterField:
    def test_default_category_field_derives_keyword_filter(self):
        cfg = _cfg()
        assert cfg.retrieval.category_field == "kategorie"
        assert cfg.retrieval.category_filter_field == "metadata.kategorie.keyword"

    def test_explicit_category_filter_field_is_preserved(self):
        from src.pipeline1.schemas.config_schema import RetrievalConfig

        rcfg = RetrievalConfig.model_validate({
            "retriever_type": "adaptive_category_aware_hybrid_rrf",
            "top_k": 2,
            "fetch_k": 4,
            "category_filter_field": "metadata.custom_field.keyword",
        })
        assert rcfg.category_filter_field == "metadata.custom_field.keyword"

    def test_adaptive_wrapper_inherits_filter_field_from_config(self):
        from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
            AdaptiveCategoryAwareHybridRRFRetriever,
        )
        stub = _TrackingRetriever()
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub,
            category_field="kategorie",
            category_filter_field="metadata.kategorie.keyword",
        )
        assert wrapper.category_filter_field == "metadata.kategorie.keyword"

    def test_adaptive_wrapper_auto_derives_filter_field_when_not_provided(self):
        from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
            AdaptiveCategoryAwareHybridRRFRetriever,
        )
        stub = _TrackingRetriever()
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(stub, category_field="kategorie")
        assert wrapper.category_filter_field == "metadata.kategorie.keyword"


# ---------------------------------------------------------------------------
# Test 4 — Empty-category fallback: 0 category results → global called
# ---------------------------------------------------------------------------

class TestEmptyCategoryFallback:
    def test_global_called_when_category_returns_zero(self):
        global_items = [_item("g1", 0.9)]
        stub = _TrackingRetriever(category_items=[], global_items=global_items)
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        wrapper.set_active_category("Finanzen")
        results = wrapper.retrieve("Q?", 2)
        assert stub.category_calls == 1, "Category retrieve must have been called"
        assert stub.global_calls == 1, "Global retrieve must have been called as fallback"
        assert len(results) == 1
        assert results[0].chunk_id == "g1"

    def test_category_retrieval_empty_flag_set(self):
        stub = _TrackingRetriever(category_items=[], global_items=[_item("g1", 0.9)])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        wrapper.set_active_category("Finanzen")
        wrapper.retrieve("Q?", 2)
        assert wrapper.category_retrieval_empty is True

    def test_fallback_reason_in_retrieval_stage_diagnostics(self):
        cfg = _cfg(top_k=2, fetch_k=4)
        # probe_fetch_k=4; need ≥3 Finanzen items with ≥60% share and ≥2 margin to pass routing
        global_items = [_item(f"p{i}", 1.0 - i * 0.1) for i in range(4)]
        stub = _TrackingRetriever(category_items=[], global_items=global_items)
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        output = _run(cfg, wrapper, [_accepted_query()])
        diag = output.retrieval_rows[0].retrieval_diagnostics
        assert diag["fallback_used"] is True
        assert diag["fallback_reason"] == "empty_category_retrieval"
        assert diag["final_retrieval_mode"] == "global"

    def test_filter_evidence_preserved_in_empty_fallback_diagnostics(self):
        stub = _TrackingRetriever(category_items=[], global_items=[_item("g1", 0.9)])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        wrapper.set_active_category("Finanzen")
        wrapper.retrieve("Q?", 2)
        diag = wrapper.last_retrieval_diagnostics
        assert diag["category_retrieval_empty"] is True
        assert diag["category_fallback_used"] is True
        assert diag.get("category_filter_applied_dense") is True
        assert diag.get("category_filter_applied_bm25") is True


# ---------------------------------------------------------------------------
# Test 5 — Non-empty category route: no fallback, scope=category
# ---------------------------------------------------------------------------

class TestSuccessfulCategoryRoute:
    def test_no_fallback_when_category_returns_results(self):
        category_items = [_item("c1", 1.0), _item("c2", 0.9)]
        stub = _TrackingRetriever(category_items=category_items, global_items=[])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        wrapper.set_active_category("Finanzen")
        results = wrapper.retrieve("Q?", 2)
        assert stub.global_calls == 0, "Global must not be called when category returns results"
        assert wrapper.category_retrieval_empty is False
        assert len(results) == 2

    def test_category_route_scope_in_stage_diagnostics(self):
        cfg = _cfg(top_k=2, fetch_k=4)
        # global_items needed for probe (probe_fetch_k=4); ≥3 Finanzen for routing acceptance
        global_items = [_item(f"p{i}", 1.0 - i * 0.1) for i in range(4)]
        category_items = [_item("c1", 1.0), _item("c2", 0.9)]
        stub = _TrackingRetriever(category_items=category_items, global_items=global_items)
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        output = _run(cfg, wrapper, [_accepted_query()])
        diag = output.retrieval_rows[0].retrieval_diagnostics
        assert diag["fallback_used"] is False
        assert diag["retrieval_scope"] == "category"

    def test_filter_field_passed_to_hybrid_in_accepted_route(self):
        stub = _TrackingRetriever(category_items=[_item("c1", 1.0)], global_items=[])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(
            stub, category_field="kategorie", category_filter_field="metadata.kategorie.keyword"
        )
        wrapper.set_active_category("Finanzen")
        wrapper.retrieve("Q?", 2)
        assert stub.category_field_received == "metadata.kategorie.keyword", (
            f"Expected 'metadata.kategorie.keyword', got {stub.category_field_received!r}"
        )


# ---------------------------------------------------------------------------
# Test 6 — Reranker is not applied when no candidates exist
# ---------------------------------------------------------------------------

class TestRerankerSkippedOnNoCandidates:
    def test_reranker_applied_false_when_no_candidates(self):
        class _EmptyRetriever:
            last_retrieval_diagnostics: dict = {}
            last_dense_candidates: list = []
            last_bm25_candidates: list = []
            last_fused_candidates: list = []
            def retrieve(self, question, top_k):
                return []
            def extract_query_metadata(self, question):
                return None

        reranker = _TrackingReranker()
        raw, retrieved, warnings, reranker_used, diagnostics = retrieve_top_k_unique_contexts(
            "Q?", _EmptyRetriever(), reranker, top_k=2, fetch_k=4, max_candidates=0
        )
        assert reranker_used is False
        assert diagnostics["reranker_applied"] is False
        assert diagnostics.get("reranker_skipped_reason") == "no_candidates"
        assert reranker.calls == [], "Reranker must not be called with empty candidate list"

    def test_reranker_applied_true_when_candidates_exist(self):
        class _OneItemRetriever:
            last_retrieval_diagnostics: dict = {}
            last_dense_candidates: list = []
            last_bm25_candidates: list = []
            last_fused_candidates: list = []
            def retrieve(self, question, top_k):
                return [_item("x1", 1.0)]
            def extract_query_metadata(self, question):
                return None

        reranker = _TrackingReranker()
        raw, retrieved, warnings, reranker_used, diagnostics = retrieve_top_k_unique_contexts(
            "Q?", _OneItemRetriever(), reranker, top_k=1, fetch_k=4, max_candidates=10
        )
        assert reranker_used is True
        assert diagnostics["reranker_applied"] is True
        assert diagnostics.get("reranker_skipped_reason") is None
        assert reranker.calls == [1]
