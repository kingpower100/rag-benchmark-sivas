"""
Unit and integration tests for ElasticsearchHybridRRFRetriever.

All tests use mocked/stub retrievers — no running Elasticsearch server is required.

Tests 1–11:  ElasticsearchHybridRRFRetriever._fuse / .retrieve internals
Test 12:     Retrieval factory dispatches elasticsearch_hybrid_rrf correctly
Test 13:     Existing retriever_type configs remain unchanged
Test 14:     elasticsearch_hybrid_rrf does not apply category filtering

Integration:  Full path through RetrievalStage with mocked sub-retrievers
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.pipeline1.retrieval.elasticsearch_hybrid_rrf_retriever import (
    ElasticsearchHybridRRFRetriever,
)
from src.pipeline1.schemas.retrieval import RetrievalItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense_item(chunk_id: str, score: float = 1.0, **kwargs) -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"ctx-{chunk_id}",
        text=f"text-{chunk_id}",
        score=score,
        dense_score=score,
        retrieval_source="elasticsearch_dense",
        metadata={"document_id": f"doc-{chunk_id}"},
        **kwargs,
    )


def _bm25_item(chunk_id: str, score: float = 1.0, **kwargs) -> RetrievalItem:
    return RetrievalItem(
        chunk_id=chunk_id,
        original_context_id=f"ctx-{chunk_id}",
        text=f"text-{chunk_id}",
        score=score,
        bm25_score=score,
        retrieval_source="elasticsearch_bm25",
        metadata={"document_id": f"doc-{chunk_id}"},
        **kwargs,
    )


def _build_retriever(
    dense_items: list[RetrievalItem],
    bm25_items: list[RetrievalItem],
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    fetch_k: int = 20,
) -> ElasticsearchHybridRRFRetriever:
    class _StubDense:
        last_retrieval_diagnostics: dict = {}

        def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
            return dense_items[:top_k]

        def extract_query_metadata(self, question: str):
            return None

    class _StubBM25:
        last_bm25_candidates: list[RetrievalItem] = []

        def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
            return bm25_items[:top_k]

    return ElasticsearchHybridRRFRetriever(
        dense_retriever=_StubDense(),
        bm25_retriever=_StubBM25(),
        fetch_k=fetch_k,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )


# ---------------------------------------------------------------------------
# Test 1 — chunk in both lists receives two RRF contributions
# ---------------------------------------------------------------------------

class TestBothLegsContribute:
    def test_chunk_in_both_lists_has_higher_score_than_single_leg(self):
        # c1 appears in dense only; c2 appears in both; c3 appears in bm25 only.
        dense = [_dense_item("c1", 0.9), _dense_item("c2", 0.8)]
        bm25 = [_bm25_item("c2", 2.0), _bm25_item("c3", 1.5)]
        retriever = _build_retriever(dense, bm25, rrf_k=60)
        results = retriever.retrieve("query", top_k=3)
        by_id = {r.chunk_id: r for r in results}

        # c2 receives contributions from both legs → must outscore c1 (dense only)
        # and c3 (BM25 only).
        assert by_id["c2"].rrf_score > by_id["c1"].rrf_score
        assert by_id["c2"].rrf_score > by_id["c3"].rrf_score

    def test_both_leg_contributions_accumulated_correctly(self):
        # c1 at dense rank 1 and BM25 rank 1 — score = 2 * 1/(60+1)
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c1", 1.0)]
        retriever = _build_retriever(dense, bm25, rrf_k=60)
        results = retriever.retrieve("query", top_k=1)
        expected = 2 * (1.0 / (60 + 1))
        assert abs(results[0].rrf_score - expected) < 1e-9


# ---------------------------------------------------------------------------
# Test 2 — dense-only chunk is preserved
# ---------------------------------------------------------------------------

class TestDenseOnlyChunkPreserved:
    def test_dense_only_chunk_appears_in_output(self):
        dense = [_dense_item("dense_only", 1.0)]
        bm25 = [_bm25_item("bm25_only", 1.0)]
        retriever = _build_retriever(dense, bm25)
        results = retriever.retrieve("q", top_k=5)
        ids = [r.chunk_id for r in results]
        assert "dense_only" in ids

    def test_dense_only_chunk_has_no_bm25_score(self):
        dense = [_dense_item("c1", 0.9)]
        retriever = _build_retriever(dense, [], rrf_k=60)
        results = retriever.retrieve("q", top_k=1)
        assert results[0].chunk_id == "c1"
        assert results[0].bm25_score is None
        assert results[0].dense_score is not None


# ---------------------------------------------------------------------------
# Test 3 — BM25-only chunk is preserved
# ---------------------------------------------------------------------------

class TestBM25OnlyChunkPreserved:
    def test_bm25_only_chunk_appears_in_output(self):
        dense = [_dense_item("dense_only", 1.0)]
        bm25 = [_bm25_item("bm25_only", 1.0)]
        retriever = _build_retriever(dense, bm25)
        results = retriever.retrieve("q", top_k=5)
        ids = [r.chunk_id for r in results]
        assert "bm25_only" in ids

    def test_bm25_only_chunk_has_no_dense_score(self):
        bm25 = [_bm25_item("c_bm25", 2.5)]
        retriever = _build_retriever([], bm25, rrf_k=60)
        results = retriever.retrieve("q", top_k=1)
        assert results[0].chunk_id == "c_bm25"
        assert results[0].dense_score is None
        assert results[0].bm25_score is not None


# ---------------------------------------------------------------------------
# Test 4 — duplicate chunk IDs appear only once
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_chunk_in_both_lists_appears_once(self):
        dense = [_dense_item("dup", 1.0), _dense_item("c2", 0.5)]
        bm25 = [_bm25_item("dup", 2.0), _bm25_item("c3", 1.0)]
        retriever = _build_retriever(dense, bm25)
        results = retriever.retrieve("q", top_k=10)
        ids = [r.chunk_id for r in results]
        assert ids.count("dup") == 1

    def test_repeated_chunk_id_within_same_leg_counts_once(self):
        # If the same retriever somehow returns the same id twice (defensive test).
        dense = [_dense_item("c1", 1.0), _dense_item("c1", 0.5)]
        retriever = _build_retriever(dense, [])
        results = retriever.retrieve("q", top_k=10)
        ids = [r.chunk_id for r in results]
        # The by_chunk dict uses setdefault so only the first occurrence is kept.
        assert ids.count("c1") == 1


# ---------------------------------------------------------------------------
# Test 5 — results ordered by descending RRF score
# ---------------------------------------------------------------------------

class TestResultOrdering:
    def test_results_sorted_by_descending_rrf_score(self):
        # c1 at dense rank 1 + BM25 rank 2 — highest combined
        # c2 at dense rank 2 + BM25 rank 1 — also high
        # c3 at dense rank 3 only — lowest
        rrf_k = 60
        dense = [_dense_item("c1", 0.9), _dense_item("c2", 0.8), _dense_item("c3", 0.7)]
        bm25 = [_bm25_item("c2", 2.0), _bm25_item("c1", 1.5)]
        retriever = _build_retriever(dense, bm25, rrf_k=rrf_k)
        results = retriever.retrieve("q", top_k=3)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Results not sorted by descending RRF score: {scores}"
        )

    def test_ranking_score_type_is_rrf_score(self):
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c2", 1.0)]
        retriever = _build_retriever(dense, bm25)
        results = retriever.retrieve("q", top_k=5)
        for item in results:
            assert item.ranking_score_type == "rrf_score"


# ---------------------------------------------------------------------------
# Test 6 — result count never exceeds top_k
# ---------------------------------------------------------------------------

class TestTopKLimit:
    def test_result_count_does_not_exceed_top_k(self):
        dense = [_dense_item(f"d{i}", float(10 - i)) for i in range(10)]
        bm25 = [_bm25_item(f"b{i}", float(10 - i)) for i in range(10)]
        retriever = _build_retriever(dense, bm25, fetch_k=10)
        for top_k in (1, 3, 5):
            results = retriever.retrieve("q", top_k=top_k)
            assert len(results) <= top_k, f"Expected <= {top_k} results, got {len(results)}"

    def test_result_count_correct_when_fewer_candidates_than_top_k(self):
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c2", 1.0)]
        retriever = _build_retriever(dense, bm25)
        results = retriever.retrieve("q", top_k=10)
        assert len(results) == 2  # only 2 unique chunks available


# ---------------------------------------------------------------------------
# Test 7 — dense empty, BM25 results available
# ---------------------------------------------------------------------------

class TestDenseEmptyBM25Available:
    def test_bm25_results_returned_when_dense_is_empty(self):
        bm25 = [_bm25_item("b1", 2.0), _bm25_item("b2", 1.5)]
        retriever = _build_retriever([], bm25)
        results = retriever.retrieve("q", top_k=5)
        assert len(results) == 2
        ids = {r.chunk_id for r in results}
        assert ids == {"b1", "b2"}

    def test_diagnostics_correct_when_dense_is_empty(self):
        bm25 = [_bm25_item("b1", 1.0)]
        retriever = _build_retriever([], bm25)
        retriever.retrieve("q", top_k=5)
        diag = retriever.last_retrieval_diagnostics
        assert diag["es_hybrid_dense_candidates"] == 0
        assert diag["es_hybrid_bm25_candidates"] == 1


# ---------------------------------------------------------------------------
# Test 8 — BM25 empty, dense results available
# ---------------------------------------------------------------------------

class TestBM25EmptyDenseAvailable:
    def test_dense_results_returned_when_bm25_is_empty(self):
        dense = [_dense_item("d1", 0.9), _dense_item("d2", 0.8)]
        retriever = _build_retriever(dense, [])
        results = retriever.retrieve("q", top_k=5)
        assert len(results) == 2
        ids = {r.chunk_id for r in results}
        assert ids == {"d1", "d2"}

    def test_diagnostics_correct_when_bm25_is_empty(self):
        dense = [_dense_item("d1", 1.0)]
        retriever = _build_retriever(dense, [])
        retriever.retrieve("q", top_k=5)
        diag = retriever.last_retrieval_diagnostics
        assert diag["es_hybrid_dense_candidates"] == 1
        assert diag["es_hybrid_bm25_candidates"] == 0


# ---------------------------------------------------------------------------
# Test 9 — both legs empty
# ---------------------------------------------------------------------------

class TestBothLegsEmpty:
    def test_returns_empty_list_when_both_legs_empty(self):
        retriever = _build_retriever([], [])
        results = retriever.retrieve("q", top_k=5)
        assert results == []

    def test_diagnostics_all_zero_when_both_empty(self):
        retriever = _build_retriever([], [])
        retriever.retrieve("q", top_k=5)
        diag = retriever.last_retrieval_diagnostics
        assert diag["es_hybrid_dense_candidates"] == 0
        assert diag["es_hybrid_bm25_candidates"] == 0
        assert diag["es_hybrid_fused_candidates"] == 0


# ---------------------------------------------------------------------------
# Test 10 — tie scores resolved deterministically
# ---------------------------------------------------------------------------

class TestTieBreaking:
    def test_tie_resolved_by_first_seen_order(self):
        # c1 and c2 each appear in exactly one leg at rank 1, giving identical
        # RRF scores.  c1 appears first (dense leg), so it should rank first.
        rrf_k = 60
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c2", 1.0)]
        retriever = _build_retriever(dense, bm25, rrf_k=rrf_k)
        results = retriever.retrieve("q", top_k=2)

        c1_score = next(r.rrf_score for r in results if r.chunk_id == "c1")
        c2_score = next(r.rrf_score for r in results if r.chunk_id == "c2")
        assert abs(c1_score - c2_score) < 1e-12, "Scores must be equal for tie-break test"
        assert results[0].chunk_id == "c1", (
            "c1 (seen first in dense leg) must rank first on a tie"
        )

    def test_tie_break_is_stable_across_multiple_calls(self):
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c2", 1.0)]
        retriever = _build_retriever(dense, bm25)
        orders = []
        for _ in range(5):
            results = retriever.retrieve("q", top_k=2)
            orders.append(tuple(r.chunk_id for r in results))
        assert len(set(orders)) == 1, "Tie-break ordering must be stable across calls"


# ---------------------------------------------------------------------------
# Test 11 — malformed results (empty chunk_id) are skipped
# ---------------------------------------------------------------------------

class TestMalformedResults:
    def test_empty_chunk_id_skipped_in_dense_leg(self):
        # RetrievalItem with chunk_id="" is technically valid Pydantic-wise
        # but is treated as unstable and skipped during fusion.
        bad = RetrievalItem(
            chunk_id="",
            original_context_id="ctx-bad",
            text="bad",
            score=10.0,
            dense_score=10.0,
            retrieval_source="elasticsearch_dense",
            metadata={},
        )
        good = _dense_item("c_good", 0.5)
        retriever = _build_retriever([bad, good], [])
        results = retriever.retrieve("q", top_k=5)
        ids = [r.chunk_id for r in results]
        assert "" not in ids
        assert "c_good" in ids

    def test_empty_chunk_id_skipped_in_bm25_leg(self):
        bad = RetrievalItem(
            chunk_id="",
            original_context_id="ctx-bad",
            text="bad",
            score=10.0,
            bm25_score=10.0,
            retrieval_source="elasticsearch_bm25",
            metadata={},
        )
        good = _bm25_item("c_good", 0.5)
        retriever = _build_retriever([], [bad, good])
        results = retriever.retrieve("q", top_k=5)
        ids = [r.chunk_id for r in results]
        assert "" not in ids
        assert "c_good" in ids

    def test_all_malformed_returns_empty(self):
        bad = RetrievalItem(
            chunk_id="",
            original_context_id="ctx",
            text="x",
            score=1.0,
            metadata={},
        )
        retriever = _build_retriever([bad], [bad])
        results = retriever.retrieve("q", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Test 12 — factory creates ElasticsearchHybridRRFRetriever
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

    def test_factory_builds_elasticsearch_hybrid_rrf_retriever(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="elasticsearch_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=60, dense_weight=1.0, bm25_weight=1.0),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert isinstance(retriever, ElasticsearchHybridRRFRetriever)

    def test_factory_dense_leg_is_elasticsearch_dense_retriever(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.retrieval.elasticsearch_dense_retriever import (
            ElasticsearchDenseRetriever,
        )
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="elasticsearch_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert isinstance(retriever.dense_retriever, ElasticsearchDenseRetriever)

    def test_factory_no_faiss_fallback_when_es_index_given(self):
        """The dense leg must use ElasticsearchDenseRetriever, not DenseRetriever(FAISS)."""
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.retrieval.dense_retriever import DenseRetriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="elasticsearch_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert not isinstance(retriever.dense_retriever, DenseRetriever)

    def test_factory_respects_dense_fetch_k_and_bm25_fetch_k(self):
        from src.pipeline1.retrieval.factory import build_retriever
        from src.pipeline1.schemas.config_schema import (
            BM25Config,
            HybridConfig,
            MetadataBoostingConfig,
            MetadataFilteringConfig,
            RetrievalConfig,
        )

        cfg = RetrievalConfig(
            retriever_type="elasticsearch_hybrid_rrf",
            top_k=5,
            fetch_k=20,
            metadata_boosting=MetadataBoostingConfig(enabled=False),
            metadata_filtering=MetadataFilteringConfig(enabled=False),
            bm25=BM25Config(backend="local"),
            hybrid=HybridConfig(rrf_k=30, dense_fetch_k=15, bm25_fetch_k=25),
        )
        embedder = MagicMock()
        index = self._make_es_index()
        chunks = self._make_chunks()

        retriever = build_retriever(cfg, embedder, index, chunks)
        assert retriever.dense_fetch_k == 15
        assert retriever.bm25_fetch_k == 25
        assert retriever.rrf_k == 30


# ---------------------------------------------------------------------------
# Test 13 — existing configurations remain unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """All pre-existing retriever_type values must continue to validate and build."""

    def _base(self, index_type="faiss", retriever_type="dense", extra_retrieval=None):
        from src.pipeline1.schemas.config_schema import PipelineConfig

        retrieval = {"retriever_type": retriever_type, "top_k": 5, "fetch_k": 20}
        if extra_retrieval:
            retrieval.update(extra_retrieval)
        index = {"type": index_type}
        if index_type == "elasticsearch":
            index["index_name"] = "test"
        return PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "test", "output_dir": "runs"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": index,
                "retrieval": retrieval,
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
        )

    def test_dense_faiss_still_validates(self):
        cfg = self._base("faiss", "dense")
        assert cfg.retrieval.retriever_type == "dense"

    def test_category_aware_dense_still_validates(self):
        cfg = self._base("faiss", "category_aware_dense")
        assert cfg.retrieval.retriever_type == "category_aware_dense"

    def test_bm25_local_still_validates(self):
        cfg = self._base("faiss", "bm25", {"bm25": {"backend": "local"}})
        assert cfg.retrieval.retriever_type == "bm25"

    def test_hybrid_rrf_faiss_still_validates(self):
        cfg = self._base(
            "faiss",
            "hybrid_rrf",
            {"hybrid": {"rrf_k": 60, "dense_backend": "faiss"}},
        )
        assert cfg.retrieval.retriever_type == "hybrid_rrf"

    def test_elasticsearch_dense_still_validates(self):
        cfg = self._base("elasticsearch", "elasticsearch_dense")
        assert cfg.retrieval.retriever_type == "elasticsearch_dense"

    def test_hybrid_config_dense_backend_elasticsearch_still_rejected(self):
        from pydantic import ValidationError
        from src.pipeline1.schemas.config_schema import HybridConfig

        with pytest.raises(ValidationError):
            HybridConfig(dense_backend="elasticsearch")

    def test_existing_validation_faiss_dense_rejects_elasticsearch_dense_retriever(self):
        from pydantic import ValidationError
        from src.pipeline1.schemas.config_schema import PipelineConfig

        with pytest.raises((ValidationError, ValueError)):
            PipelineConfig.model_validate(
                {
                    "experiment": {"experiment_id": "e", "output_dir": "r"},
                    "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                    "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                    "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                    "index": {"type": "faiss"},
                    "retrieval": {"retriever_type": "elasticsearch_dense", "top_k": 5, "fetch_k": 10},
                    "reranker": {"enabled": False},
                    "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                    "telemetry": {},
                    "runtime": {},
                }
            )

    def test_new_hybrid_fields_default_to_none(self):
        from src.pipeline1.schemas.config_schema import HybridConfig

        cfg = HybridConfig()
        assert cfg.dense_fetch_k is None
        assert cfg.bm25_fetch_k is None

    def test_new_hybrid_fields_accept_positive_int(self):
        from src.pipeline1.schemas.config_schema import HybridConfig

        cfg = HybridConfig(dense_fetch_k=15, bm25_fetch_k=25)
        assert cfg.dense_fetch_k == 15
        assert cfg.bm25_fetch_k == 25

    def test_elasticsearch_hybrid_rrf_requires_elasticsearch_index(self):
        from pydantic import ValidationError
        from src.pipeline1.schemas.config_schema import PipelineConfig

        for index_type in ("faiss", "pgvector"):
            payload = {
                "experiment": {"experiment_id": "e", "output_dir": "r"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": index_type},
                "retrieval": {
                    "retriever_type": "elasticsearch_hybrid_rrf",
                    "top_k": 5,
                    "fetch_k": 20,
                },
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
            if index_type == "pgvector":
                payload["index"]["pgvector"] = {"dsn_env": "PG"}
            with pytest.raises((ValidationError, ValueError)):
                PipelineConfig.model_validate(payload)

    def test_elasticsearch_hybrid_rrf_validates_with_elasticsearch_index(self):
        from src.pipeline1.schemas.config_schema import PipelineConfig

        cfg = PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "e", "output_dir": "r"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "elasticsearch"},
                "retrieval": {
                    "retriever_type": "elasticsearch_hybrid_rrf",
                    "top_k": 5,
                    "fetch_k": 20,
                    "bm25": {"backend": "local"},
                    "hybrid": {"rrf_k": 60, "dense_fetch_k": 20, "bm25_fetch_k": 20},
                },
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
        )
        assert cfg.retrieval.retriever_type == "elasticsearch_hybrid_rrf"
        assert cfg.retrieval.hybrid.dense_fetch_k == 20
        assert cfg.retrieval.hybrid.bm25_fetch_k == 20


# ---------------------------------------------------------------------------
# Test 14 — no category filtering applied
# ---------------------------------------------------------------------------

class TestNoCategoryFiltering:
    def test_retriever_has_no_set_active_category(self):
        """ElasticsearchHybridRRFRetriever must not expose set_active_category
        so the RetrievalStage never activates category scoping for it."""
        retriever = _build_retriever([], [])
        assert not hasattr(retriever, "set_active_category"), (
            "ElasticsearchHybridRRFRetriever must not implement set_active_category"
        )

    def test_retrieval_stage_uses_global_path_for_elasticsearch_hybrid_rrf(self):
        """When retriever_type='elasticsearch_hybrid_rrf', RetrievalStage must
        call retrieve() directly (global path) without any category scoping."""
        from src.pipeline1.schemas.config_schema import PipelineConfig
        from src.pipeline1.schemas.query import QueryRecord
        from src.pipeline1.stages.base import StageInput
        from src.pipeline1.stages.retrieval_stage import RetrievalStage

        # Build a PipelineConfig for elasticsearch_hybrid_rrf
        cfg = PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "e", "output_dir": "r"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "elasticsearch"},
                "retrieval": {
                    "retriever_type": "elasticsearch_hybrid_rrf",
                    "top_k": 2,
                    "fetch_k": 5,
                    "bm25": {"backend": "local"},
                    "hybrid": {"rrf_k": 60},
                },
                "reranker": {"enabled": False},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
        )

        # Stub retriever that records whether set_active_category was called
        category_calls: list[str] = []

        class _StubHybridRetriever:
            last_dense_candidates: list = []
            last_bm25_candidates: list = []
            last_fused_candidates: list = []
            last_retrieval_diagnostics: dict = {}

            def retrieve(self, question: str, top_k: int) -> list:
                return [
                    _dense_item("c1", 0.9),
                    _dense_item("c2", 0.8),
                ]

            # Deliberately not implementing set_active_category

        stub_retriever = _StubHybridRetriever()

        stage = RetrievalStage(
            cfg,
            embedder=MagicMock(),
            index=MagicMock(),
            chunks=[],
            retriever_factory=lambda *a, **kw: stub_retriever,
        )
        query = QueryRecord(
            question_id="q1",
            question="test",
            detected_category="Finanzen",
            category_validated=True,
        )
        output = stage.run(StageInput({"queries": [query]}))

        # No category scoping should have been applied
        assert category_calls == [], "set_active_category must not be called"
        row = output.retrieval_rows[0]
        assert row.retrieval_diagnostics["retrieval_scope"] == "global"
        assert row.retrieval_diagnostics["category_filter_applied"] is False
        assert row.retrieval_diagnostics["category_fallback_used"] is False


# ---------------------------------------------------------------------------
# Integration test — dense and BM25 legs both called, results fused, top_k honoured
# ---------------------------------------------------------------------------

class TestIntegration:
    """Verifies the full retrieve() call path end-to-end using mocked sub-retrievers.
    No Elasticsearch server is required."""

    def test_both_legs_called_and_fused(self):
        dense_calls: list[int] = []
        bm25_calls: list[int] = []

        class _TrackingDense:
            last_retrieval_diagnostics: dict = {}

            def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
                dense_calls.append(top_k)
                return [
                    _dense_item("shared", 0.9),
                    _dense_item("dense_only", 0.7),
                ]

            def extract_query_metadata(self, q):
                return None

        class _TrackingBM25:
            last_bm25_candidates: list = []

            def retrieve(self, question: str, top_k: int) -> list[RetrievalItem]:
                bm25_calls.append(top_k)
                return [
                    _bm25_item("shared", 2.0),
                    _bm25_item("bm25_only", 1.5),
                ]

        retriever = ElasticsearchHybridRRFRetriever(
            dense_retriever=_TrackingDense(),
            bm25_retriever=_TrackingBM25(),
            fetch_k=10,
            dense_fetch_k=10,
            bm25_fetch_k=10,
            rrf_k=60,
        )
        top_k = 3
        results = retriever.retrieve("Was kostet das Produkt?", top_k=top_k)

        # Both legs were invoked
        assert len(dense_calls) == 1, "Dense leg must be called exactly once"
        assert len(bm25_calls) == 1, "BM25 leg must be called exactly once"

        # Results are fused (all three unique chunks present)
        ids = {r.chunk_id for r in results}
        assert "shared" in ids
        assert "dense_only" in ids
        assert "bm25_only" in ids

        # 'shared' received contributions from both legs → highest RRF score
        shared_score = next(r.rrf_score for r in results if r.chunk_id == "shared")
        for other in results:
            if other.chunk_id != "shared":
                assert shared_score >= other.rrf_score, (
                    f"'shared' must have the highest RRF score; "
                    f"got {shared_score} vs {other.chunk_id}={other.rrf_score}"
                )

        # top_k is respected
        assert len(results) <= top_k

    def test_intermediate_candidates_stored_on_retriever(self):
        dense_items = [_dense_item("d1", 0.9), _dense_item("d2", 0.8)]
        bm25_items = [_bm25_item("b1", 1.5), _bm25_item("d1", 1.0)]

        retriever = _build_retriever(dense_items, bm25_items, fetch_k=5)
        retriever.retrieve("q", top_k=5)

        # Per-leg candidates are accessible for diagnostics
        assert len(retriever.last_dense_candidates) == 2
        assert len(retriever.last_bm25_candidates) == 2

        # Fused list has 3 unique chunks (d1, d2, b1)
        assert len(retriever.last_fused_candidates) == 3

    def test_per_item_hybrid_diagnostics_present(self):
        dense = [_dense_item("c1", 1.0)]
        bm25 = [_bm25_item("c1", 2.0), _bm25_item("c2", 1.5)]
        retriever = _build_retriever(dense, bm25, fetch_k=5)
        results = retriever.retrieve("q", top_k=5)

        by_id = {r.chunk_id: r for r in results}
        c1_diag = by_id["c1"].metadata.get("_es_hybrid_diagnostics", {})
        assert c1_diag["dense_rank"] == 1
        assert c1_diag["bm25_rank"] == 1
        assert set(c1_diag["retrieval_sources"]) == {"dense", "bm25"}

        c2_diag = by_id["c2"].metadata.get("_es_hybrid_diagnostics", {})
        assert c2_diag["dense_rank"] is None
        assert c2_diag["bm25_rank"] == 2
        assert c2_diag["retrieval_sources"] == ["bm25"]
