"""
Regression tests covering:
  1. dense type searches global FAISS (no category scope)
  2. category_aware_dense with embeddings → true category-specific FAISS index
  3. category_aware_dense without embeddings → post-hoc filter fallback
  4. Invalid category triggers global fallback with fallback_reason
  5. Insufficient category results trigger controlled fallback
  6. hybrid_rrf fuses dense and BM25 rankings correctly
  7. New diagnostic fields (retriever_type, retrieval_scope, category_index_used,
     fallback_used, fallback_reason) are present in every output row
  8. All four YAML-activatable retriever_type values remain selectable via config
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pipeline1.retrieval.bm25_retriever import BM25Retriever
from src.pipeline1.retrieval.category_aware_dense_retriever import CategoryAwareDenseRetriever
from src.pipeline1.retrieval.hybrid_rrf_retriever import HybridRRFRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.retrieval_stage import RetrievalStage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str, category: str | None = None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"ctx-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={
            "document_id": f"doc-{chunk_id}",
            "file_name": f"{chunk_id}.txt",
            "kategorie": category,
        },
    )


def _cfg(
    retriever_type: str = "dense",
    index_type: str = "faiss",
    top_k: int = 1,
    fetch_k: int = 4,
):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": index_type, "metric": "cosine"},
            "retrieval": {"retriever_type": retriever_type, "top_k": top_k, "fetch_k": fetch_k},
            "reranker": {"enabled": False, "model_name": None, "device": "cpu"},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
            "telemetry": {},
            "runtime": {},
        }
    )


class _Embedder:
    """Stub embedder; returns a constant unit vector so FAISS can operate."""
    def encode_query(self, question: str):
        return np.ones(2, dtype="float32")


class _GlobalIndex:
    """Returns first N chunks in order regardless of query."""
    def __init__(self, n: int = 4):
        self._n = n

    def search(self, query_embedding, top_k):
        k = min(top_k, self._n)
        scores = np.linspace(1.0, 0.5, self._n, dtype="float32")[:k]
        indices = np.arange(self._n, dtype="int64")[:k]
        return scores, indices


# ---------------------------------------------------------------------------
# 1. dense — global FAISS, no category scope
# ---------------------------------------------------------------------------

class TestDenseRetrieverGlobalScope:
    def test_dense_returns_chunks_from_global_index(self):
        cfg = _cfg(retriever_type="dense", top_k=2, fetch_k=4)
        chunks = [
            _chunk("c1", "alpha", "Einkauf"),
            _chunk("c2", "beta", "Finanzen"),
            _chunk("c3", "gamma", "Personal"),
        ]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(3), chunks).run(
            StageInput({"queries": [QueryRecord(question_id="q1", question="alpha")]})
        )
        row = output.retrieval_rows[0]
        # Global index returns c1, c2 (first two)
        assert [i.chunk_id for i in row.retrieved] == ["c1", "c2"]

    def test_dense_diagnostics_retriever_type(self):
        cfg = _cfg(retriever_type="dense", top_k=1, fetch_k=2)
        chunks = [_chunk("c1", "text", "Einkauf")]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(1), chunks).run(
            StageInput({"queries": [QueryRecord(question_id="q1", question="text")]})
        )
        diag = output.retrieval_rows[0].retrieval_diagnostics
        assert diag["retriever_type"] == "dense"
        assert diag["retrieval_scope"] == "global"
        assert diag["category_index_used"] is False
        assert diag["fallback_used"] is False
        assert diag["fallback_reason"] is None

    def test_dense_ignores_detected_category(self):
        """dense retriever must not restrict results to a specific category."""
        cfg = _cfg(retriever_type="dense", top_k=2, fetch_k=4)
        chunks = [_chunk("c1", "alpha", "Einkauf"), _chunk("c2", "beta", "Finanzen")]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(2), chunks).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        # Global index returns both; dense never filters by category
        assert {i.chunk_id for i in output.retrieval_rows[0].retrieved} == {"c1", "c2"}


# ---------------------------------------------------------------------------
# 2. category_aware_dense — true per-category FAISS index (embeddings provided)
# ---------------------------------------------------------------------------

class TestCategoryAwareDenseTrueFaiss:
    """When pre-computed embeddings are passed, the retriever must build and
    search dedicated per-category FAISS indexes, not fall back to post-hoc
    global filtering."""

    @pytest.fixture
    def setup(self):
        """Three chunks: one Einkauf, two Finanzen.  Each has a distinct 2-D embedding."""
        chunks = [
            _chunk("c_einkauf", "Einkauf text", "Einkauf"),
            _chunk("c_fin1",    "Finanzen text 1", "Finanzen"),
            _chunk("c_fin2",    "Finanzen text 2", "Finanzen"),
        ]
        # Embeddings shape [3, 2]; rows correspond to chunks above
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype="float32"
        )
        return chunks, embeddings

    def test_true_category_faiss_used(self, setup):
        chunks, embeddings = setup
        cfg = _cfg(retriever_type="category_aware_dense", top_k=2, fetch_k=4)

        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(3), chunks, embeddings=embeddings).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="finanzen",
                        cleaned_question="finanzen",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        row = output.retrieval_rows[0]
        diag = row.retrieval_diagnostics

        # True FAISS category index was used
        assert diag["category_index_used"] is True, "Expected true FAISS category index"
        assert diag["retrieval_scope"] == "category"
        assert diag["fallback_used"] is False
        assert diag["fallback_reason"] is None
        assert diag["retriever_type"] == "category_aware_dense"

    def test_only_category_chunks_returned(self, setup):
        """With a true category index, only chunks from the searched category
        can appear in results."""
        chunks, embeddings = setup
        cfg = _cfg(retriever_type="category_aware_dense", top_k=2, fetch_k=4)

        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(3), chunks, embeddings=embeddings).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="finanzen",
                        cleaned_question="finanzen",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        retrieved = output.retrieval_rows[0].retrieved
        for item in retrieved:
            assert item.metadata.get("kategorie") == "Finanzen", (
                f"Chunk {item.chunk_id} has category {item.metadata.get('kategorie')}, "
                "expected Finanzen"
            )

    def test_einkauf_category_searches_only_einkauf_index(self, setup):
        chunks, embeddings = setup
        cfg = _cfg(retriever_type="category_aware_dense", top_k=1, fetch_k=4)

        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(3), chunks, embeddings=embeddings).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="einkauf",
                        cleaned_question="einkauf",
                        detected_category="Einkauf",
                        category_validated=True,
                    )
                ]
            })
        )
        retrieved = output.retrieval_rows[0].retrieved
        assert all(i.metadata.get("kategorie") == "Einkauf" for i in retrieved)


# ---------------------------------------------------------------------------
# 3. category_aware_dense without embeddings → post-hoc filter
# ---------------------------------------------------------------------------

class TestCategoryAwareDensePostHocFilter:
    """When embeddings are NOT provided, the retriever must fall back to a global
    search followed by a post-hoc category filter."""

    def test_post_hoc_filter_used_when_no_embeddings(self):
        cfg = _cfg(retriever_type="category_aware_dense", top_k=1, fetch_k=4)
        chunks = [
            _chunk("c1", "alpha", "Einkauf"),
            _chunk("c2", "alpha", "Finanzen"),
        ]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(2), chunks).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha",
                        cleaned_question="alpha",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        row = output.retrieval_rows[0]
        diag = row.retrieval_diagnostics
        # No embeddings → per-category FAISS not built → post-hoc filter
        assert diag["category_index_used"] is False
        assert diag["retrieval_scope"] in ("category", "global")  # filter applied
        assert [i.chunk_id for i in row.retrieved] == ["c2"]


# ---------------------------------------------------------------------------
# 4. Invalid / unvalidated category → global fallback
# ---------------------------------------------------------------------------

class TestCategoryAwareDenseInvalidCategoryFallback:
    def test_invalid_category_falls_back_to_global(self):
        cfg = _cfg(retriever_type="category_aware_dense", top_k=2, fetch_k=4)
        chunks = [_chunk("c1", "alpha", "Einkauf"), _chunk("c2", "beta", "Finanzen")]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(2), chunks).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="test",
                        cleaned_question="test",
                        detected_category="UnknownCat",
                        category_validated=False,
                        category_validation_reason="detected_category not found in KB category list",
                    )
                ]
            })
        )
        row = output.retrieval_rows[0]
        diag = row.retrieval_diagnostics
        assert diag["retrieval_scope"] == "global"
        assert diag["fallback_used"] is True
        assert diag["fallback_reason"] == "invalid_category_global_fallback"
        assert diag["category_index_used"] is False
        assert diag["retriever_type"] == "category_aware_dense"

    def test_no_detected_category_falls_back_to_global(self):
        cfg = _cfg(retriever_type="category_aware_dense", top_k=2, fetch_k=4)
        chunks = [_chunk("c1", "alpha", "Einkauf"), _chunk("c2", "beta", "Finanzen")]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(2), chunks).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="test",
                        cleaned_question="test",
                    )
                ]
            })
        )
        diag = output.retrieval_rows[0].retrieval_diagnostics
        assert diag["retrieval_scope"] == "global"
        assert diag["fallback_used"] is True
        assert diag["fallback_reason"] == "invalid_category_global_fallback"

    def test_fallback_reason_is_none_when_no_fallback(self):
        """When category retrieval succeeds, fallback_reason must be None."""
        cfg = _cfg(retriever_type="category_aware_dense", top_k=1, fetch_k=4)
        chunks = [
            _chunk("c1", "alpha", "Einkauf"),
            _chunk("c2", "beta", "Finanzen"),
            _chunk("c3", "gamma", "Finanzen"),
        ]
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype="float32"
        )
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(3), chunks, embeddings=embeddings).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="finanzen",
                        cleaned_question="finanzen",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        diag = output.retrieval_rows[0].retrieval_diagnostics
        assert diag["fallback_used"] is False
        assert diag["fallback_reason"] is None


# ---------------------------------------------------------------------------
# 5. Insufficient category results → controlled fallback
# ---------------------------------------------------------------------------

class TestCategoryAwareDenseInsufficientResultsFallback:
    def test_insufficient_category_results_trigger_global_fallback(self):
        """When the category search returns fewer chunks than top_k, the
        pipeline must fall back to global search and set fallback_reason."""
        cfg = _cfg(retriever_type="category_aware_dense", top_k=3, fetch_k=4)
        chunks = [
            _chunk("c1", "alpha", "Einkauf"),
            _chunk("c2", "alpha", "Finanzen"),   # only 1 Finanzen chunk
            _chunk("c3", "alpha", "Personal"),
            _chunk("c4", "alpha", "Personal"),
        ]
        output = RetrievalStage(cfg, _Embedder(), _GlobalIndex(4), chunks).run(
            StageInput({
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha",
                        cleaned_question="alpha",
                        detected_category="Finanzen",
                        category_validated=True,
                    )
                ]
            })
        )
        row = output.retrieval_rows[0]
        diag = row.retrieval_diagnostics
        assert diag["category_fallback_used"] is True
        assert diag["fallback_used"] is True
        assert diag["fallback_reason"] == "insufficient_category_results_global_fallback"
        assert diag["retrieval_scope"] == "global"
        assert diag["retrieval_mode"] == "global_fallback"
        assert diag["number_of_category_results"] < 3  # triggered fallback


# ---------------------------------------------------------------------------
# 6. hybrid_rrf — dense + BM25 fusion
# ---------------------------------------------------------------------------

class TestHybridRRFRetriever:
    """HybridRRFRetriever must fuse dense and BM25 ranked lists via RRF."""

    def _make_retrieval_item(self, chunk_id: str, score: float) -> RetrievalItem:
        return RetrievalItem(
            chunk_id=chunk_id,
            original_context_id=f"ctx-{chunk_id}",
            text=chunk_id,
            score=score,
            dense_score=score,
        )

    def test_rrf_fuses_dense_and_bm25_by_rank(self):
        """Items appearing in both lists receive higher RRF scores."""
        dense_items = [
            self._make_retrieval_item("c1", 0.9),
            self._make_retrieval_item("c2", 0.8),
            self._make_retrieval_item("c3", 0.7),
        ]
        bm25_items = [
            self._make_retrieval_item("c3", 2.0),   # top BM25 hit
            self._make_retrieval_item("c1", 1.5),   # second BM25 hit
            self._make_retrieval_item("c4", 0.5),   # only in BM25
        ]

        class _StubDense:
            last_retrieval_diagnostics = {}
            def retrieve(self, q, k): return dense_items[:k]
            def extract_query_metadata(self, q): return None

        class _StubBM25:
            last_bm25_candidates = []
            def retrieve(self, q, k): return bm25_items[:k]

        retriever = HybridRRFRetriever(
            dense_retriever=_StubDense(),
            bm25_retriever=_StubBM25(),
            fetch_k=3,
            rrf_k=60,
            dense_weight=1.0,
            bm25_weight=1.0,
        )
        results = retriever.retrieve("query", top_k=4)
        ids = [r.chunk_id for r in results]

        # c1 appears at rank 1 (dense) and rank 2 (BM25) → highest combined score
        # c3 appears at rank 3 (dense) and rank 1 (BM25) → also high
        # c2 appears only in dense → lower
        # c4 appears only in BM25 → lower
        assert "c1" in ids, "c1 must appear in fused results"
        assert "c3" in ids, "c3 must appear in fused results"
        # c1 and c3 both appear in both lists; they must outscore c4 (only in BM25)
        c1_score = next(r.score for r in results if r.chunk_id == "c1")
        c4_score = next(r.score for r in results if r.chunk_id == "c4")
        assert c1_score > c4_score, "c1 (in both lists) must outscore c4 (BM25 only)"

    def test_rrf_deduplicates_by_chunk_id(self):
        item = self._make_retrieval_item("c1", 1.0)

        class _StubBoth:
            last_retrieval_diagnostics = {}
            last_bm25_candidates = []
            def retrieve(self, q, k): return [item]
            def extract_query_metadata(self, q): return None

        retriever = HybridRRFRetriever(
            dense_retriever=_StubBoth(),
            bm25_retriever=_StubBoth(),
            fetch_k=2,
        )
        results = retriever.retrieve("q", top_k=5)
        assert [r.chunk_id for r in results].count("c1") == 1, "Duplicate chunk_id must be collapsed"

    def test_rrf_scores_stored_as_rrf_score(self):
        items = [self._make_retrieval_item("c1", 1.0)]

        class _Stub:
            last_retrieval_diagnostics = {}
            last_bm25_candidates = []
            def retrieve(self, q, k): return items
            def extract_query_metadata(self, q): return None

        retriever = HybridRRFRetriever(
            dense_retriever=_Stub(),
            bm25_retriever=_Stub(),
            fetch_k=2,
        )
        results = retriever.retrieve("q", top_k=1)
        assert results[0].ranking_score_type == "rrf_score"
        assert results[0].rrf_score is not None and results[0].rrf_score > 0


# ---------------------------------------------------------------------------
# 7. New diagnostic fields present in every output row
# ---------------------------------------------------------------------------

class TestNewDiagnosticFields:
    """All new diagnostic fields must be present regardless of retriever_type."""

    REQUIRED_FIELDS = [
        "retriever_type",
        "retrieval_scope",
        "category_index_used",
        "fallback_used",
        "fallback_reason",
    ]

    def _run(self, retriever_type: str):
        cfg = _cfg(retriever_type=retriever_type, top_k=1, fetch_k=2)
        chunks = [_chunk("c1", "text", "Einkauf")]
        return RetrievalStage(cfg, _Embedder(), _GlobalIndex(1), chunks).run(
            StageInput({"queries": [QueryRecord(question_id="q1", question="text")]})
        )

    @pytest.mark.parametrize("retriever_type", ["dense", "category_aware_dense"])
    def test_all_new_fields_present(self, retriever_type):
        output = self._run(retriever_type)
        diag = output.retrieval_rows[0].retrieval_diagnostics
        for field in self.REQUIRED_FIELDS:
            assert field in diag, f"Missing diagnostic field '{field}' for retriever_type={retriever_type}"

    @pytest.mark.parametrize("retriever_type", ["dense", "category_aware_dense"])
    def test_existing_fields_not_removed(self, retriever_type):
        """Legacy diagnostic fields must be preserved (no renames or removals)."""
        output = self._run(retriever_type)
        diag = output.retrieval_rows[0].retrieval_diagnostics
        legacy_fields = [
            "retrieval_mode",
            "category_filter_applied",
            "category_fallback_used",
            "number_of_category_results",
            "number_of_global_fallback_results",
            "retrieved_chunks",
            "retrieval_scores",
        ]
        for field in legacy_fields:
            assert field in diag, f"Legacy field '{field}' was removed for retriever_type={retriever_type}"


# ---------------------------------------------------------------------------
# 8. YAML-only activation of all four retriever_type values
# ---------------------------------------------------------------------------

class TestYAMLOnlyRetrieverActivation:
    """Switching between retriever types must require only a YAML change."""

    @pytest.mark.parametrize("retriever_type", ["dense", "bm25", "hybrid_rrf", "category_aware_dense"])
    def test_retriever_type_activatable_via_config_only(self, retriever_type):
        cfg = _cfg(retriever_type=retriever_type)
        assert cfg.retrieval.retriever_type == retriever_type

    def test_dense_config_validates(self):
        cfg = _cfg(retriever_type="dense")
        assert cfg.retrieval.retriever_type == "dense"

    def test_bm25_local_config_validates(self):
        cfg = PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "exp", "output_dir": "runs"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "faiss", "metric": "cosine"},
                "retrieval": {
                    "retriever_type": "bm25",
                    "top_k": 5,
                    "fetch_k": 10,
                    "bm25": {"backend": "local", "k1": 1.5, "b": 0.75},
                },
                "reranker": {"enabled": False, "model_name": None, "device": "cpu"},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
        )
        assert cfg.retrieval.retriever_type == "bm25"
        assert cfg.retrieval.bm25.backend == "local"

    def test_hybrid_rrf_config_validates(self):
        cfg = PipelineConfig.model_validate(
            {
                "experiment": {"experiment_id": "exp", "output_dir": "runs"},
                "data": {"documents_path": "d.jsonl", "questions_path": "q.jsonl"},
                "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
                "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
                "index": {"type": "faiss", "metric": "cosine"},
                "retrieval": {
                    "retriever_type": "hybrid_rrf",
                    "top_k": 5,
                    "fetch_k": 20,
                    "bm25": {"backend": "local"},
                    "hybrid": {"rrf_k": 60, "dense_weight": 1.0, "bm25_weight": 1.0},
                },
                "reranker": {"enabled": False, "model_name": None, "device": "cpu"},
                "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "ctx"},
                "telemetry": {},
                "runtime": {},
            }
        )
        assert cfg.retrieval.retriever_type == "hybrid_rrf"
        assert cfg.retrieval.hybrid.rrf_k == 60
