"""
Unit tests for AdaptiveCategoryAwareHybridRRFRetriever and the
adaptive_category_aware_hybrid_rrf retriever_type.

Tests 1–9 cover:
  1. AdaptiveCategoryAwareHybridRRFRetriever exposes set_active_category
  2. ElasticsearchHybridRRFRetriever still has no set_active_category (regression guard)
  3. retrieve_with_category on ElasticsearchHybridRRFRetriever filters the dense leg
  4. retrieve_with_category on ElasticsearchHybridRRFRetriever filters the BM25 leg
  5. retrieve_with_category fuses both legs using the same RRF logic as global retrieve
  6. retrieve_global_probe calls global Hybrid RRF (no category filter)
  7. set_active_category(None) triggers global retrieve path, not filtered
  8. Schema accepts "adaptive_category_aware_hybrid_rrf" and enforces ES index requirement
  9. Factory builds AdaptiveCategoryAwareHybridRRFRetriever for the new retriever_type

No running Elasticsearch server or real ES client is required.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.pipeline1.retrieval.adaptive_category_aware_hybrid_rrf_retriever import (
    AdaptiveCategoryAwareHybridRRFRetriever,
)
from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import (
    ElasticsearchHybridRRFRetriever,
)
from src.pipeline1.schemas.retrieval import RetrievalItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(chunk_id: str, score: float = 1.0, category: str | None = None) -> RetrievalItem:
    meta: dict = {}
    if category is not None:
        meta["kategorie"] = category
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"ctx-{chunk_id}",
        text=f"text-{chunk_id}",
        score=score,
        dense_score=score,
        retrieval_source="elasticsearch_dense",
        metadata=meta,
    )


class _StubDense:
    """Stub dense sub-retriever that records category-filtered calls."""

    def __init__(self, global_items, category_items=None):
        self._global = global_items
        self._category = category_items if category_items is not None else []
        self.last_category_call: tuple | None = None
        self.last_retrieval_diagnostics: dict = {}

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        return self._global[:top_k]

    def retrieve_with_category(
        self, question: str, top_k: int, category: str, category_field: str
    ) -> list[RetrievalItem]:
        self.last_category_call = (category, category_field)
        return self._category[:top_k]

    def extract_query_metadata(self, question: str):
        return None


class _StubBM25:
    """Stub BM25 sub-retriever that records category-filtered calls."""

    def __init__(self, global_items, category_items=None):
        self._global = global_items
        self._category = category_items if category_items is not None else []
        self.last_category_call: tuple | None = None
        self.last_bm25_candidates: list = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
        return self._global[:top_k]

    def retrieve_with_category(
        self, question: str, top_k: int, category: str, category_field: str
    ) -> list[RetrievalItem]:
        self.last_category_call = (category, category_field)
        return self._category[:top_k]


def _build_hybrid(
    dense_global=None,
    bm25_global=None,
    dense_category=None,
    bm25_category=None,
    rrf_k: int = 60,
    fetch_k: int = 20,
) -> tuple[ElasticsearchHybridRRFRetriever, _StubDense, _StubBM25]:
    dense_stub = _StubDense(dense_global or [], dense_category or [])
    bm25_stub = _StubBM25(bm25_global or [], bm25_category or [])
    hybrid = ElasticsearchHybridRRFRetriever(
        dense_retriever=dense_stub,
        bm25_retriever=bm25_stub,
        fetch_k=fetch_k,
        rrf_k=rrf_k,
    )
    return hybrid, dense_stub, bm25_stub


# ---------------------------------------------------------------------------
# Test 1 — AdaptiveCategoryAwareHybridRRFRetriever exposes set_active_category
# ---------------------------------------------------------------------------

class TestWrapperExposesSetActiveCategory:
    def test_has_set_active_category(self):
        hybrid, _, _ = _build_hybrid()
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        assert hasattr(wrapper, "set_active_category"), (
            "AdaptiveCategoryAwareHybridRRFRetriever must expose set_active_category"
        )

    def test_set_active_category_updates_internal_state(self):
        hybrid, _, _ = _build_hybrid()
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        wrapper.set_active_category("Finanzen")
        assert wrapper.active_category == "Finanzen"

    def test_set_active_category_none_clears_state(self):
        hybrid, _, _ = _build_hybrid()
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        wrapper.set_active_category("Finanzen")
        wrapper.set_active_category(None)
        assert wrapper.active_category is None


# ---------------------------------------------------------------------------
# Test 2 — ElasticsearchHybridRRFRetriever still has no set_active_category
# ---------------------------------------------------------------------------

class TestExistingRetrieverNoRegression:
    def test_elasticsearch_hybrid_rrf_retriever_has_no_set_active_category(self):
        hybrid, _, _ = _build_hybrid()
        assert not hasattr(hybrid, "set_active_category"), (
            "ElasticsearchHybridRRFRetriever must not implement set_active_category — "
            "regression detected after adding retrieve_with_category"
        )

    def test_elasticsearch_hybrid_rrf_retrieve_still_works_globally(self):
        dense_items = [_item("d1", 0.9), _item("d2", 0.8)]
        bm25_items = [_item("b1", 1.5)]
        hybrid, _, _ = _build_hybrid(dense_global=dense_items, bm25_global=bm25_items)
        results = hybrid.retrieve("Was kostet das Produkt?", top_k=5)
        ids = {r.chunk_id for r in results}
        assert {"d1", "d2", "b1"} == ids


# ---------------------------------------------------------------------------
# Test 3 — retrieve_with_category sends a category filter to the dense leg
# ---------------------------------------------------------------------------

class TestCategoryFilterSentToDenseLeg:
    def test_dense_leg_receives_category(self):
        dense_cat = [_item("cat-d1", 0.9, category="Finanzen")]
        hybrid, dense_stub, _ = _build_hybrid(dense_category=dense_cat)
        hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        assert dense_stub.last_category_call is not None, "Dense stub was not called with category filter"
        assert dense_stub.last_category_call[0] == "Finanzen"

    def test_dense_leg_receives_correct_category_field(self):
        dense_cat = [_item("cat-d1", 0.9)]
        hybrid, dense_stub, _ = _build_hybrid(dense_category=dense_cat)
        hybrid.retrieve_with_category("query", top_k=5, category="Logistik", category_field="kategorie")
        assert dense_stub.last_category_call[1] == "kategorie"

    def test_dense_leg_items_appear_in_fused_output(self):
        dense_cat = [_item("cat-dense", 0.9, category="Finanzen")]
        bm25_cat = []
        hybrid, _, _ = _build_hybrid(dense_category=dense_cat, bm25_category=bm25_cat)
        results = hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        ids = {r.chunk_id for r in results}
        assert "cat-dense" in ids


# ---------------------------------------------------------------------------
# Test 4 — retrieve_with_category sends a category filter to the BM25 leg
# ---------------------------------------------------------------------------

class TestCategoryFilterSentToBM25Leg:
    def test_bm25_leg_receives_category(self):
        bm25_cat = [_item("cat-b1", 1.5, category="Finanzen")]
        hybrid, _, bm25_stub = _build_hybrid(bm25_category=bm25_cat)
        hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        assert bm25_stub.last_category_call is not None, "BM25 stub was not called with category filter"
        assert bm25_stub.last_category_call[0] == "Finanzen"

    def test_bm25_leg_receives_correct_category_field(self):
        bm25_cat = [_item("cat-b1", 1.5)]
        hybrid, _, bm25_stub = _build_hybrid(bm25_category=bm25_cat)
        hybrid.retrieve_with_category("query", top_k=5, category="Logistik", category_field="kategorie")
        assert bm25_stub.last_category_call[1] == "kategorie"

    def test_bm25_leg_items_appear_in_fused_output(self):
        dense_cat = []
        bm25_cat = [_item("cat-bm25", 1.5, category="Finanzen")]
        hybrid, _, _ = _build_hybrid(dense_category=dense_cat, bm25_category=bm25_cat)
        results = hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        ids = {r.chunk_id for r in results}
        assert "cat-bm25" in ids


# ---------------------------------------------------------------------------
# Test 5 — retrieve_with_category fuses both legs via RRF (same as global)
# ---------------------------------------------------------------------------

class TestCategoryFilteredRRFFusion:
    def test_chunk_in_both_legs_has_highest_rrf_score(self):
        shared = _item("shared", 0.9, category="Finanzen")
        dense_cat = [shared, _item("dense-only", 0.8, category="Finanzen")]
        bm25_cat = [shared, _item("bm25-only", 1.5, category="Finanzen")]
        hybrid, _, _ = _build_hybrid(dense_category=dense_cat, bm25_category=bm25_cat, rrf_k=60)
        results = hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        by_id = {r.chunk_id: r for r in results}
        assert "shared" in by_id
        assert by_id["shared"].rrf_score > by_id.get("dense-only", MagicMock(rrf_score=-1)).rrf_score
        assert by_id["shared"].rrf_score > by_id.get("bm25-only", MagicMock(rrf_score=-1)).rrf_score

    def test_rrf_score_formula_matches_global_formula(self):
        rrf_k = 60
        item = _item("c1", 0.9, category="Finanzen")
        hybrid, _, _ = _build_hybrid(
            dense_category=[item], bm25_category=[item], rrf_k=rrf_k
        )
        results = hybrid.retrieve_with_category("query", top_k=5, category="Finanzen", category_field="kategorie")
        expected = 2 * (1.0 / (rrf_k + 1))
        assert results, "Expected at least one result"
        assert abs(results[0].rrf_score - expected) < 1e-9

    def test_top_k_respected_in_category_filtered_retrieve(self):
        items = [_item(f"c{i}", float(10 - i), category="Finanzen") for i in range(8)]
        hybrid, _, _ = _build_hybrid(dense_category=items, bm25_category=items)
        results = hybrid.retrieve_with_category("query", top_k=3, category="Finanzen", category_field="kategorie")
        assert len(results) <= 3


# ---------------------------------------------------------------------------
# Test 6 — retrieve_global_probe calls global Hybrid RRF without category filter
# ---------------------------------------------------------------------------

class TestGlobalProbeIsUnfiltered:
    def test_probe_does_not_call_dense_retrieve_with_category(self):
        dense_global = [_item("global-d1", 0.9)]
        hybrid, dense_stub, bm25_stub = _build_hybrid(
            dense_global=dense_global,
            bm25_global=[_item("global-b1", 1.5)],
        )
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid, category_field="kategorie")
        wrapper.set_active_category("Finanzen")
        wrapper.retrieve_global_probe("query", probe_fetch_k=10)
        assert dense_stub.last_category_call is None, (
            "retrieve_global_probe must not activate category filtering on the dense leg"
        )

    def test_probe_does_not_call_bm25_retrieve_with_category(self):
        hybrid, dense_stub, bm25_stub = _build_hybrid(
            dense_global=[_item("g1", 0.9)],
            bm25_global=[_item("g2", 1.5)],
        )
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        wrapper.set_active_category("Logistik")
        wrapper.retrieve_global_probe("query", probe_fetch_k=10)
        assert bm25_stub.last_category_call is None, (
            "retrieve_global_probe must not activate category filtering on the BM25 leg"
        )

    def test_probe_clears_active_category_before_retrieve(self):
        hybrid, dense_stub, _ = _build_hybrid(dense_global=[_item("g1", 0.9)])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        wrapper.set_active_category("Finanzen")
        wrapper.retrieve_global_probe("query", probe_fetch_k=5)
        assert wrapper.active_category is None, (
            "retrieve_global_probe must reset active_category to None"
        )

    def test_probe_returns_global_results(self):
        global_items = [_item("g1", 0.9), _item("g2", 0.8)]
        hybrid, _, _ = _build_hybrid(dense_global=global_items, bm25_global=[])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        results = wrapper.retrieve_global_probe("query", probe_fetch_k=5)
        ids = {r.chunk_id for r in results}
        assert "g1" in ids
        assert "g2" in ids


# ---------------------------------------------------------------------------
# Test 7 — set_active_category(None) triggers global retrieve, not filtered
# ---------------------------------------------------------------------------

class TestGlobalRetrieveWhenNoCategorySet:
    def test_global_path_used_when_no_active_category(self):
        global_items = [_item("global-only", 0.9)]
        cat_items = [_item("cat-only", 0.95)]
        hybrid, dense_stub, bm25_stub = _build_hybrid(
            dense_global=global_items,
            bm25_global=[],
            dense_category=cat_items,
            bm25_category=[],
        )
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        # No active category — must use global retrieve
        results = wrapper.retrieve("query", top_k=5)
        ids = {r.chunk_id for r in results}
        assert "global-only" in ids, "Global items must appear when no category is active"
        assert dense_stub.last_category_call is None, (
            "Dense leg must not be called with a category filter when active_category is None"
        )

    def test_diagnostics_show_no_category_filter_when_inactive(self):
        hybrid, _, _ = _build_hybrid(dense_global=[_item("g1", 0.9)])
        wrapper = AdaptiveCategoryAwareHybridRRFRetriever(hybrid)
        wrapper.retrieve("query", top_k=5)
        diag = wrapper.last_retrieval_diagnostics
        assert diag["category_filter_applied"] is False
        assert diag["detected_category"] is None


# ---------------------------------------------------------------------------
# Test 8 — Schema accepts "adaptive_category_aware_hybrid_rrf" and validates
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def _base_payload(self, retriever_type: str, index_type: str = "elasticsearch", orchestration_enabled: bool = True):
        payload: dict = {
            "experiment": {"experiment_id": "test-achrrf", "output_dir": "runs"},
            "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": index_type},
            "retrieval": {
                "retriever_type": retriever_type,
                "top_k": 5,
                "fetch_k": 20,
                "bm25": {"backend": "local"},
                "hybrid": {"rrf_k": 60, "dense_fetch_k": 20, "bm25_fetch_k": 20},
            },
            "reranker": {"enabled": False},
            "orchestration": {"enabled": orchestration_enabled, "model_name": "llama3.1:8b"},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
            "telemetry": {},
            "runtime": {},
        }
        if index_type == "pgvector":
            payload["index"]["pgvector"] = {"dsn_env": "PG"}
        return payload

    def test_schema_accepts_adaptive_category_aware_hybrid_rrf(self):
        from src.pipeline1.schemas.config_schema import PipelineConfig

        cfg = PipelineConfig.model_validate(
            self._base_payload("adaptive_category_aware_hybrid_rrf")
        )
        assert cfg.retrieval.retriever_type == "adaptive_category_aware_hybrid_rrf"

    def test_schema_rejects_faiss_index_for_adaptive_category_aware_hybrid_rrf(self):
        from pydantic import ValidationError
        from src.pipeline1.schemas.config_schema import PipelineConfig

        with pytest.raises((ValidationError, ValueError)):
            PipelineConfig.model_validate(
                self._base_payload("adaptive_category_aware_hybrid_rrf", index_type="faiss")
            )

    def test_schema_rejects_orchestration_disabled_for_adaptive_category_aware_hybrid_rrf(self):
        from pydantic import ValidationError
        from src.pipeline1.schemas.config_schema import PipelineConfig

        with pytest.raises((ValidationError, ValueError)):
            PipelineConfig.model_validate(
                self._base_payload(
                    "adaptive_category_aware_hybrid_rrf",
                    orchestration_enabled=False,
                )
            )

    def test_existing_retriever_types_still_validate(self):
        from src.pipeline1.schemas.config_schema import PipelineConfig

        cfg = PipelineConfig.model_validate(
            self._base_payload("elasticsearch_hybrid_rrf", orchestration_enabled=False)
        )
        assert cfg.retrieval.retriever_type == "elasticsearch_hybrid_rrf"


# ---------------------------------------------------------------------------
# Test 9 — Factory builds AdaptiveCategoryAwareHybridRRFRetriever
# ---------------------------------------------------------------------------

class TestFactory:
    def _make_es_index(self):
        from src.pipeline1.indexing.elasticsearch_index import ElasticsearchIndex

        idx = MagicMock(spec=ElasticsearchIndex)
        idx.metric = "cosine"
        idx.uses_external_storage = True
        idx.__class__ = ElasticsearchIndex
        return idx

    def _make_chunks(self):
        chunk = MagicMock()
        chunk.chunk_id = "c1"
        chunk.metadata = {}
        chunk.text = "text"
        chunk.document_id = "doc1"
        chunk.original_context_id = "ctx1"
        return [chunk]

    def test_factory_builds_adaptive_category_aware_hybrid_rrf_retriever(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="adaptive_category_aware_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=60, dense_fetch_k=20, bm25_fetch_k=20),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert isinstance(retriever, AdaptiveCategoryAwareHybridRRFRetriever)

    def test_factory_wraps_elasticsearch_hybrid_rrf_retriever(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="adaptive_category_aware_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=60),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert isinstance(retriever.hybrid_retriever, ElasticsearchHybridRRFRetriever)

    def test_factory_sets_correct_category_field(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="adaptive_category_aware_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            category_field="kategorie",
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=60),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert retriever.category_field == "kategorie"

    def test_factory_preserves_rrf_k_in_hybrid_sub_retriever(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="adaptive_category_aware_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=60, dense_fetch_k=20, bm25_fetch_k=20),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert retriever.hybrid_retriever.rrf_k == 60
        assert retriever.hybrid_retriever.dense_fetch_k == 20
        assert retriever.hybrid_retriever.bm25_fetch_k == 20
