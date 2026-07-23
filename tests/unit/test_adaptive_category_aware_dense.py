import json
from pathlib import Path

import numpy as np

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.orchestrator import run_pipeline
from src.pipeline1.retrieval.adaptive_category_aware_dense_retriever import AdaptiveCategoryAwareDenseRetriever
from src.pipeline1.retrieval.pgvector_dense_retriever import PgvectorDenseRetriever
from src.pipeline1.schemas.chunk import ChunkRecord
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.retrieval_stage import RetrievalStage


def test_valid_category_with_strong_support_routes_to_final_category_retrieval():
    cfg = _cfg(top_k=2, fetch_k=5, probe_fetch_k=5)
    chunks = [
        _chunk("f1", "alpha", "Finanzen"),
        _chunk("f2", "alpha", "Finanzen"),
        _chunk("f3", "alpha", "Finanzen"),
        _chunk("h1", "alpha", "HR"),
        _chunk("e1", "alpha", "Einkauf"),
    ]
    index = _SequenceIndex([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert index.requests == [5, 5]
    assert [item.chunk_id for item in row.retrieved] == ["f1", "f2"]
    assert row.retrieval_diagnostics["routing_decision"] == "accepted"
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "category"
    assert row.retrieval_diagnostics["predicted_category_count"] == 3
    assert row.retrieval_diagnostics["predicted_category_share"] == 0.6
    assert row.retrieval_diagnostics["support_margin"] == 2


def test_probe_fetch_k_less_than_retrieval_fetch_k_uses_exact_probe_cap():
    cfg = _cfg(top_k=2, fetch_k=6, probe_fetch_k=3, min_share=0.67, min_count=2, min_margin=1)
    chunks = [
        _chunk("f1", "alpha", "Finanzen"),
        _chunk("f2", "alpha", "Finanzen"),
        _chunk("h1", "alpha", "HR"),
        _chunk("f3", "alpha", "Finanzen"),
        _chunk("h2", "alpha", "HR"),
        _chunk("e1", "alpha", "Einkauf"),
    ]
    index = _SequenceIndex([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert index.requests == [3, 6]
    assert row.retrieval_diagnostics["probe_fetch_k"] == 3
    assert row.retrieval_diagnostics["probe_candidate_ids"] == ["f1", "f2", "h1"]


def test_probe_fetch_k_greater_than_retrieval_fetch_k_uses_exact_probe_cap():
    cfg = _cfg(top_k=2, fetch_k=3, probe_fetch_k=5)
    chunks = [
        _chunk("f1", "alpha", "Finanzen"),
        _chunk("f2", "alpha", "Finanzen"),
        _chunk("f3", "alpha", "Finanzen"),
        _chunk("h1", "alpha", "HR"),
        _chunk("e1", "alpha", "Einkauf"),
    ]
    index = _SequenceIndex([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert index.requests == [5, 3]
    assert row.retrieval_diagnostics["probe_fetch_k"] == 5
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "category"


def test_valid_category_with_weak_support_routes_to_final_global_retrieval():
    cfg = _cfg(top_k=2, fetch_k=5, probe_fetch_k=5)
    chunks = [
        _chunk("f1", "alpha", "Finanzen"),
        _chunk("h1", "alpha", "HR"),
        _chunk("h2", "alpha", "HR"),
        _chunk("h3", "alpha", "HR"),
        _chunk("f2", "alpha", "Finanzen"),
    ]
    index = _SequenceIndex([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert row.retrieval_diagnostics["routing_decision"] == "rejected"
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "global"
    assert row.retrieval_diagnostics["strongest_competing_category"] == "HR"
    assert row.retrieval_diagnostics["competing_category_count"] == 3
    assert row.retrieval_diagnostics["fallback_used"] is True


def test_invalid_category_immediately_uses_global_without_probe():
    cfg = _cfg(top_k=2, fetch_k=4, probe_fetch_k=20)
    chunks = [_chunk("a", "alpha", "Finanzen"), _chunk("b", "alpha", "HR")]
    index = _SequenceIndex([[0, 1]])

    row = _run_row(cfg, chunks, index, detected_category="Unknown", category_validated=False)

    assert index.requests == [4]
    assert row.retrieval_diagnostics["routing_decision"] == "rejected"
    assert row.retrieval_diagnostics["decision_reason"] == "invalid_or_missing_category"
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "global"


def test_missing_category_immediately_uses_global_without_probe():
    cfg = _cfg(top_k=1, fetch_k=2, probe_fetch_k=20)
    chunks = [_chunk("a", "alpha", "Finanzen")]
    index = _SequenceIndex([[0]])

    row = _run_row(cfg, chunks, index, detected_category=None, category_validated=False)

    assert index.requests == [2]
    assert row.retrieval_diagnostics["predicted_category"] is None
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "global"


def test_empty_probe_rejects_to_global():
    cfg = _cfg(top_k=1, fetch_k=2, probe_fetch_k=3)
    chunks = [_chunk("a", "alpha", "Finanzen"), _chunk("b", "alpha", "HR")]
    index = _SequenceIndex([[], [0, 1]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert row.retrieval_diagnostics["total_probe_candidates"] == 0
    assert row.retrieval_diagnostics["decision_reason"] == "empty_global_probe"
    assert row.retrieval_diagnostics["final_retrieval_mode"] == "global"


def test_tie_between_categories_is_deterministic_and_rejected_by_margin():
    cfg = _cfg(top_k=1, fetch_k=4, probe_fetch_k=4, min_share=0.5, min_count=2, min_margin=1)
    chunks = [
        _chunk("f1", "alpha", "Finanzen"),
        _chunk("h1", "alpha", "HR"),
        _chunk("f2", "alpha", "Finanzen"),
        _chunk("h2", "alpha", "HR"),
    ]
    index = _SequenceIndex([[0, 1, 2, 3], [0, 1, 2, 3]])

    row = _run_row(cfg, chunks, index, detected_category="Finanzen", category_validated=True)

    assert row.retrieval_diagnostics["strongest_competing_category"] == "HR"
    assert row.retrieval_diagnostics["support_margin"] == 0
    assert row.retrieval_diagnostics["decision_reason"] == "support_margin_below_threshold"


def test_threshold_failures_are_reported():
    cases = [
        (0.61, 1, 0, "category_share_below_threshold"),
        (0.0, 3, 0, "category_count_below_threshold"),
        (0.0, 1, 2, "support_margin_below_threshold"),
    ]
    for min_share, min_count, min_margin, reason in cases:
        cfg = _cfg(top_k=1, fetch_k=5, probe_fetch_k=5, min_share=min_share, min_count=min_count, min_margin=min_margin)
        chunks = [
            _chunk("f1", "alpha", "Finanzen"),
            _chunk("f2", "alpha", "Finanzen"),
            _chunk("h1", "alpha", "HR"),
            _chunk("h2", "alpha", "HR"),
            _chunk("h3", "alpha", "HR"),
        ]
        row = _run_row(cfg, chunks, _SequenceIndex([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]), "Finanzen", True)
        assert reason in row.retrieval_diagnostics["decision_reason"]
        assert row.retrieval_diagnostics["routing_decision"] == "rejected"


def test_diagnostics_include_required_probe_and_final_fields():
    cfg = _cfg(top_k=1, fetch_k=3, probe_fetch_k=3, min_share=0.34, min_count=1, min_margin=0)
    chunks = [_chunk("f1", "alpha", "Finanzen"), _chunk("h1", "alpha", "HR"), _chunk("f2", "alpha", "Finanzen")]

    row = _run_row(cfg, chunks, _SequenceIndex([[0, 1, 2], [0, 1, 2]]), "Finanzen", True)
    diag = row.retrieval_diagnostics

    for field in (
        "predicted_category",
        "category_validated",
        "probe_fetch_k",
        "probe_candidate_ids",
        "probe_candidate_categories",
        "probe_candidate_scores",
        "predicted_category_count",
        "predicted_category_share",
        "competing_category",
        "competing_category_count",
        "support_margin",
        "routing_thresholds",
        "routing_decision",
        "decision_reason",
        "final_retrieval_mode",
        "final_chunk_ids",
        "average_similarity_by_category",
    ):
        assert field in diag
    assert len(diag["probe_candidate_ids"]) == len(diag["probe_candidate_categories"]) == len(diag["probe_candidate_scores"])
    assert list(zip(diag["probe_candidate_ids"], diag["probe_candidate_categories"], diag["probe_candidate_scores"])) == [
        ("f1", "Finanzen", 1.0),
        ("h1", "HR", 0.9900000095367432),
        ("f2", "Finanzen", 0.9800000190734863),
    ]
    assert diag["probe_score_semantics"] == "higher_is_better"


def test_factory_builds_adaptive_faiss_retriever():
    from src.pipeline1.retrieval.factory import build_retriever

    cfg = _cfg().retrieval
    retriever = build_retriever(cfg, _Embedder(), _SequenceIndex([[0]]), [_chunk("c1", "alpha", "Finanzen")])
    assert isinstance(retriever, AdaptiveCategoryAwareDenseRetriever)
    assert retriever._pgvector_mode is False


def test_adaptive_pgvector_switches_between_global_probe_and_category_final():
    chunks = [_chunk("f1", "alpha", "Finanzen"), _chunk("h1", "alpha", "HR")]
    index = _PgvectorLikeIndex(chunks)
    dense = PgvectorDenseRetriever(
        embedder=_Embedder(),
        index=index,
        chunks=chunks,
        fetch_k=2,
        metadata_boosting=_NoMetadata(),
        metadata_filtering=_NoMetadata(strict=False, strict_year_match=False, strict_year_month_match=False),
    )
    retriever = AdaptiveCategoryAwareDenseRetriever(dense)

    retriever.set_active_category(None)
    probe = retriever.retrieve("alpha", 2)
    retriever.set_active_category("Finanzen")
    final = retriever.retrieve("alpha", 2)

    assert [item.chunk_id for item in probe] == ["f1", "h1"]
    assert [item.chunk_id for item in final] == ["f1"]
    assert index.calls == [("global", None, 2), ("category", "Finanzen", 2)]


def test_adaptive_pgvector_global_probe_uses_exact_probe_cap_independent_of_fetch_k():
    chunks = [_chunk("f1", "alpha", "Finanzen"), _chunk("f2", "alpha", "Finanzen"), _chunk("h1", "alpha", "HR")]
    index = _PgvectorLikeIndex(chunks)
    dense = PgvectorDenseRetriever(
        embedder=_Embedder(),
        index=index,
        chunks=chunks,
        fetch_k=7,
        metadata_boosting=_NoMetadata(),
        metadata_filtering=_NoMetadata(strict=False, strict_year_match=False, strict_year_month_match=False),
    )
    retriever = AdaptiveCategoryAwareDenseRetriever(dense)

    probe = retriever.retrieve_global_probe("alpha", 2)

    assert [item.chunk_id for item in probe] == ["f1", "f2"]
    assert index.calls == [("global", None, 2)]


def test_manifest_persists_adaptive_routing_summary(tmp_path, monkeypatch):
    project_root = tmp_path
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "kb_documents_fixed.jsonl").write_text(
        (
            '{"doc_key":"doc-1","doc_name":"a.md","text":"alpha","kategorie":"Finanzen"}\n'
            '{"doc_key":"doc-2","doc_name":"b.md","text":"alpha","kategorie":"Finanzen"}\n'
            '{"doc_key":"doc-3","doc_name":"c.md","text":"alpha","kategorie":"Finanzen"}\n'
            '{"doc_key":"doc-4","doc_name":"d.md","text":"alpha","kategorie":"HR"}\n'
        ),
        encoding="utf-8",
    )
    (data_dir / "questions_fixed.jsonl").write_text('{"question_id":"q1","frage":"Q?"}\n', encoding="utf-8")
    cfg_path = project_root / "config.yaml"
    cfg_path.write_text(_manifest_config_yaml(), encoding="utf-8")
    monkeypatch.setenv("PIPELINE1_SKIP_OLLAMA_PREFLIGHT", "1")
    monkeypatch.setattr("src.pipeline1.orchestrator._project_root", lambda: project_root)
    monkeypatch.setattr("src.pipeline1.orchestrator.build_embedder", lambda config: _Embedder())
    monkeypatch.setattr("src.pipeline1.orchestrator.build_index", lambda config: _SequenceIndex([[0, 1, 2, 3], [0, 1, 2, 3]]))
    monkeypatch.setattr("src.pipeline1.orchestrator.build_generator", lambda config: _FakeGenerator())

    run_dir = run_pipeline(str(cfg_path))

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    routing = manifest["category_routing_validation"]
    assert routing["enabled"] is True
    assert routing["probe_fetch_k"] == 4
    assert routing["number_accepted"] == 1
    assert routing["number_rejected"] == 0
    assert routing["number_invalid_categories"] == 0
    assert routing["number_global_fallbacks"] == 0


class _Embedder:
    def encode_query(self, question):
        return np.ones(2, dtype="float32")

    def encode_texts(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype="float32")


class _SequenceIndex:
    metric = "cosine"

    def __init__(self, sequences):
        self.sequences = [list(seq) for seq in sequences]
        self.requests = []
        self._call = 0

    def build(self, embeddings):
        self.embeddings = embeddings

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake", encoding="utf-8")

    def load(self, path):
        pass

    def search(self, query_embedding, top_k):
        self.requests.append(top_k)
        seq = self.sequences[min(self._call, len(self.sequences) - 1)]
        self._call += 1
        idxs = np.array(seq[:top_k], dtype="int64")
        scores = np.array([1.0 - (i * 0.01) for i in range(len(idxs))], dtype="float32")
        return scores, idxs


class _PgvectorLikeIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query_embedding, top_k):
        self.calls.append(("global", None, top_k))
        return [chunk.chunk_id for chunk in self.chunks[:top_k]], [1.0 - i * 0.1 for i in range(min(top_k, len(self.chunks)))]

    def search_category(self, query_embedding, top_k, category, category_field="kategorie"):
        self.calls.append(("category", category, top_k))
        matches = [chunk for chunk in self.chunks if chunk.metadata.get(category_field) == category]
        return [chunk.chunk_id for chunk in matches[:top_k]], [1.0 - i * 0.1 for i in range(min(top_k, len(matches)))]


class _NoMetadata:
    def __init__(self, enabled=False, strict=False, strict_year_match=False, strict_year_month_match=False):
        self.enabled = enabled
        self.strict = strict
        self.strict_year_match = strict_year_match
        self.strict_year_month_match = strict_year_month_match
        self.company_weight = 0.0
        self.year_weight = 0.0
        self.month_weight = 0.0
        self.year_month_weight = 0.0
        self.wrong_year_penalty = 0.0
        self.symbol_weight = 0.0
        self.file_name_weight = 0.0


class _FakeGenerator:
    def generate(self, prompt):
        if "detected_category" in prompt and "cleaned_question" in prompt:
            return GenerationResult(
                answer='{"cleaned_question":"Q?","detected_category":"Finanzen"}',
                input_tokens=5,
                output_tokens=5,
            )
        return GenerationResult(answer="answer", input_tokens=5, output_tokens=1)


def _run_row(cfg, chunks, index, detected_category, category_validated):
    return RetrievalStage(cfg, _Embedder(), index, chunks).run(
        StageInput(
            {
                "queries": [
                    QueryRecord(
                        question_id="q1",
                        question="alpha?",
                        cleaned_question="alpha?",
                        detected_category=detected_category,
                        category_validated=category_validated,
                    )
                ]
            }
        )
    ).retrieval_rows[0]


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
    top_k=2,
    fetch_k=5,
    probe_fetch_k=5,
    min_share=0.60,
    min_count=3,
    min_margin=2,
):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {
                "retriever_type": "adaptive_category_aware_dense",
                "top_k": top_k,
                "fetch_k": fetch_k,
                "category_routing_validation": {
                    "enabled": True,
                    "probe_fetch_k": probe_fetch_k,
                    "minimum_category_share": min_share,
                    "minimum_category_count": min_count,
                    "minimum_margin": min_margin,
                },
            },
            "reranker": {"enabled": False},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )


def _manifest_config_yaml():
    return """
experiment:
  experiment_id: "adaptive_manifest"
  random_seed: 42
  output_dir: "runs"
data:
  dataset_schema: "sivas"
  documents_path: "data/raw/kb_documents_fixed.jsonl"
  questions_path: "data/raw/questions_fixed.jsonl"
  question_id_field: "question_id"
  question_field: "frage"
  document_text_field: "text"
  allow_document_text_fallback: false
  allow_unsafe_query_fields: false
chunking:
  strategy: "fixed_word"
  chunk_size: 10
  chunk_overlap: 0
  allow_word_fallback: false
embedding:
  provider: "sentence_transformers"
  model_name: "fake"
  normalize_embeddings: true
  batch_size: 2
  device: "cpu"
index:
  type: "faiss"
  metric: "cosine"
  dense_dim: 2
retrieval:
  retriever_type: "adaptive_category_aware_dense"
  top_k: 1
  fetch_k: 4
  category_routing_validation:
    enabled: true
    probe_fetch_k: 4
    minimum_category_share: 0.6
    minimum_category_count: 3
    minimum_margin: 2
reranker:
  enabled: false
generation:
  provider: "ollama"
  model_name: "fake"
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 8
  timeout_s: 10
  system_prompt: "Use context."
telemetry:
  estimate_cost: false
runtime:
  save_csv: true
  log_level: "INFO"
  resume: false
  overwrite: true
"""
