import numpy as np

from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.retrieval_stage import RetrievalStage


def test_retrieval_stage_returns_retrieval_item_compatible_outputs():
    cfg = _cfg()
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]

    output = RetrievalStage(cfg, _Embedder(), _FaissIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert output.attempted == 1
    assert row.query.question_id == "q1"
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c2"]
    assert [item.chunk_id for item in row.retrieved] == ["c1"]
    assert row.retrieved[0].original_context_id == "ctx-c1"
    assert row.retrieved[0].metadata["document_id"] == "doc-c1"
    assert row.retrieval_time_ms >= 0


def test_retrieval_stage_preserves_raw_vs_final_ids_after_dedupe():
    cfg = _cfg(fetch_k=2, top_k=2)
    chunks = [_chunk("c1", "alpha"), _chunk("c1", "alpha duplicate")]

    output = RetrievalStage(cfg, _Embedder(), _DuplicateIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c1"]
    assert [item.chunk_id for item in row.retrieved] == ["c1"]
    assert row.retrieval_warnings == ["Only 1 unique chunks were available after deduplication; requested top_k=2."]


def test_retrieval_stage_reranker_enabled_path_still_works():
    cfg = _cfg(top_k=2, fetch_k=2, reranker_enabled=True, final_top_k=1)
    chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]

    output = RetrievalStage(
        cfg,
        _Embedder(),
        _FaissIndex(),
        chunks,
        reranker_factory=lambda model_name, device: _ReverseReranker(),
    ).run(StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]}))

    row = output.retrieval_rows[0]
    assert output.final_top_k == 1
    assert row.reranker_used is True
    assert [item.chunk_id for item in row.raw_retrieved] == ["c1", "c2"]
    assert [item.chunk_id for item in row.retrieved] == ["c2"]
    assert row.retrieved[0].rerank_score == 1.0


def test_retrieval_stage_elasticsearch_dense_retriever_works():
    cfg = _cfg(retriever_type="elasticsearch_dense", index_type="elasticsearch", top_k=1, fetch_k=2)
    chunks = [_chunk("c1", "alpha")]

    output = RetrievalStage(cfg, _Embedder(), _ElasticsearchDenseIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert row.retrieved[0].chunk_id == "c1"
    assert row.retrieved[0].retrieval_source == "elasticsearch_dense"
    assert row.retrieved[0].dense_score == 0.75
    assert row.raw_dense_retrieved[0].chunk_id == "c1"


def test_retrieval_stage_uses_cleaned_question_and_detected_category():
    cfg = _cfg(retriever_type="category_aware_dense", top_k=1, fetch_k=2)
    chunks = [_chunk("c1", "alpha", "Einkauf"), _chunk("c2", "alpha", "Finanzen")]
    embedder = _RecordingEmbedder()

    output = RetrievalStage(cfg, embedder, _FaissIndex(), chunks).run(
        StageInput(
            {
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="dirty alpha?",
                        cleaned_question="clean alpha?",
                        detected_category="Finanzen",
                        category_confidence=0.8,
                    )
                ]
            }
        )
    )

    row = output.retrieval_rows[0]
    assert embedder.last_query == "clean alpha?"
    assert [item.chunk_id for item in row.retrieved] == ["c2"]
    assert row.retrieval_diagnostics["detected_category"] == "Finanzen"


def test_category_aware_retrieval_fills_remaining_slots_from_global_candidates():
    cfg = _cfg(retriever_type="category_aware_dense", top_k=3, fetch_k=4)
    chunks = [
        _chunk("c1", "alpha", "Einkauf"),
        _chunk("c2", "alpha", "Finanzen"),
        _chunk("c3", "alpha", "Personal"),
        _chunk("c4", "alpha", "Finanzen"),
    ]

    output = RetrievalStage(cfg, _Embedder(), _FourIndex(), chunks).run(
        StageInput(
            {
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha?",
                        cleaned_question="alpha?",
                        detected_category="Finanzen",
                        category_confidence=0.9,
                    )
                ]
            }
        )
    )

    row = output.retrieval_rows[0]
    assert [item.chunk_id for item in row.retrieved] == ["c2", "c4", "c1"]
    assert row.retrieval_diagnostics["retrieved_chunks"] == ["c2", "c4", "c1"]
    assert row.retrieval_diagnostics["retrieved_documents"] == ["doc-c2", "doc-c4", "doc-c1"]
    assert row.retrieval_diagnostics["retrieved_categories"] == ["Finanzen", "Finanzen", "Einkauf"]
    assert row.retrieval_diagnostics["category_filter_applied"] is True
    assert row.retrieval_diagnostics["category_fallback_used"] is True


def test_category_aware_retrieval_falls_back_to_global_when_category_missing():
    cfg = _cfg(retriever_type="category_aware_dense", top_k=2, fetch_k=2)
    chunks = [_chunk("c1", "alpha", "Einkauf"), _chunk("c2", "alpha", "Finanzen")]

    output = RetrievalStage(cfg, _Embedder(), _FaissIndex(), chunks).run(
        StageInput({"queries": [QueryRecord(question_id="q1", question="alpha?", cleaned_question="alpha?")]})
    )

    row = output.retrieval_rows[0]
    assert [item.chunk_id for item in row.retrieved] == ["c1", "c2"]
    assert row.retrieval_diagnostics["category_filter_applied"] is False
    assert row.retrieval_diagnostics["category_fallback_used"] is False


def test_sivas_chunk_metadata_is_available_during_retrieval():
    cfg = _cfg(retriever_type="category_aware_dense", top_k=1, fetch_k=1)
    chunks = [
        ChunkRecord(
            chunk_id="doc-a:chunk:0001",
            document_id="doc-a",
            original_context_id="doc-a",
            text="alpha",
            chunk_start=0,
            chunk_end=1,
            metadata={
                "doc_id": 1,
                "doc_key": "doc-a",
                "doc_name": "a.md",
                "kategorie": "Einkauf",
                "wissensart": "FAQ",
                "titel": "Bestellung",
                "quellpfad": "kb/a.md",
                "sprache": "de",
            },
        )
    ]

    output = RetrievalStage(cfg, _Embedder(), _SingleIndex(), chunks).run(
        StageInput(
            {
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha?",
                        cleaned_question="alpha?",
                        detected_category="Einkauf",
                    )
                ]
            }
        )
    )

    metadata = output.retrieval_rows[0].retrieved[0].metadata
    assert output.retrieval_rows[0].retrieved[0].chunk_id == "doc-a:chunk:0001"
    assert metadata["doc_id"] == 1
    assert metadata["doc_key"] == "doc-a"
    assert metadata["doc_name"] == "a.md"
    assert metadata["kategorie"] == "Einkauf"
    assert metadata["wissensart"] == "FAQ"
    assert metadata["titel"] == "Bestellung"
    assert metadata["quellpfad"] == "kb/a.md"
    assert metadata["sprache"] == "de"


class _Embedder:
    def encode_query(self, question):
        return np.ones(2, dtype="float32")


class _RecordingEmbedder(_Embedder):
    last_query = None

    def encode_query(self, question):
        self.last_query = question
        return super().encode_query(question)


class _FaissIndex:
    def search(self, query_embedding, top_k):
        return np.array([1.0, 0.9], dtype="float32")[:top_k], np.array([0, 1], dtype="int64")[:top_k]


class _DuplicateIndex:
    def search(self, query_embedding, top_k):
        return np.array([1.0, 0.9], dtype="float32")[:top_k], np.array([0, 1], dtype="int64")[:top_k]


class _FourIndex:
    def search(self, query_embedding, top_k):
        return (
            np.array([1.0, 0.9, 0.8, 0.7], dtype="float32")[:top_k],
            np.array([0, 1, 2, 3], dtype="int64")[:top_k],
        )


class _SingleIndex:
    def search(self, query_embedding, top_k):
        return np.array([1.0], dtype="float32")[:top_k], np.array([0], dtype="int64")[:top_k]


class _ElasticsearchDenseIndex:
    text_field = "text"

    def search_hits(self, query_vec, candidate_k):
        return [
            {
                "_id": "c1",
                "_score": 1.75,
                "_source": {
                    "chunk_id": "c1",
                    "document_id": "doc-c1",
                    "original_context_id": "ctx-c1",
                    "text": "alpha",
                    "metadata": {"document_id": "doc-c1", "file_name": "c1.txt"},
                },
            }
        ]


class _ReverseReranker:
    requested_device = "cpu"
    runtime_device = "cpu"

    def rerank(self, question, items, top_k):
        reranked = []
        for score, item in enumerate(reversed(items), start=1):
            reranked.append(item.model_copy(update={"score": float(score), "rerank_score": float(score)}))
        return reranked[:top_k]


def _chunk(chunk_id: str, text: str, category: str | None = None):
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        original_context_id=f"ctx-{chunk_id}",
        text=text,
        chunk_start=0,
        chunk_end=len(text),
        metadata={"document_id": f"doc-{chunk_id}", "file_name": f"{chunk_id}.txt", "kategorie": category},
    )


def _cfg(
    retriever_type: str = "dense",
    index_type: str = "faiss",
    top_k: int = 1,
    fetch_k: int = 2,
    reranker_enabled: bool = False,
    final_top_k: int | None = None,
):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": index_type, "metric": "cosine"},
            "retrieval": {"retriever_type": retriever_type, "top_k": top_k, "fetch_k": fetch_k},
            "reranker": {
                "enabled": reranker_enabled,
                "model_name": "fake-reranker" if reranker_enabled else None,
                "device": "cpu",
                "final_top_k": final_top_k,
            },
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )
