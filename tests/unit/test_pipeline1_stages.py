import json

import numpy as np

from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.chunking_stage import ChunkingStage
from src.pipeline1.stages.document_stage import DocumentStage
from src.pipeline1.stages.embedding_stage import EmbeddingStage
from src.pipeline1.stages.run_writer_stage import RunWriterStage


class _FakeEmbedder:
    def encode_texts(self, texts, show_progress=False):
        return np.ones((len(texts), 2), dtype="float32")


def test_document_stage_loads_jsonl_documents(tmp_path):
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text('{"doc_key":"ctx1","doc_name":"ctx1.md","text":"alpha","kategorie":"ERP"}\n', encoding="utf-8")
    cfg = _cfg(tmp_path)

    output = DocumentStage(cfg, docs_path).run()

    assert len(output.documents) == 1
    assert output.documents[0].original_context_id == "ctx1"
    assert output.document_input_info["source_type"] == "jsonl"
    assert output.document_input_info["path"] == str(docs_path)


def test_chunking_stage_builds_and_reuses_chunk_cache(tmp_path):
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text('{"doc_key":"ctx1","doc_name":"ctx1.md","text":"alpha beta","kategorie":"ERP"}\n', encoding="utf-8")
    cfg = _cfg(tmp_path)
    docs = DocumentStage(cfg, docs_path).run().documents
    cache_dir = tmp_path / "data" / "processed"

    first = ChunkingStage(cfg, tmp_path, cache_dir, docs_path).run(StageInput({"documents": docs}))
    second = ChunkingStage(cfg, tmp_path, cache_dir, docs_path).run(StageInput({"documents": docs}))

    assert len(first.chunks) == 1
    assert first.cache_status == "built"
    assert second.cache_status == "loaded"
    assert first.chunks_key == second.chunks_key
    assert first.chunk_diagnostics["total_chunks"] == 1
    assert first.chunks_path.exists()


def test_embedding_stage_builds_and_validates_npy_cache(tmp_path):
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text('{"doc_key":"ctx1","doc_name":"ctx1.md","text":"alpha beta","kategorie":"ERP"}\n', encoding="utf-8")
    cfg = _cfg(tmp_path)
    docs = DocumentStage(cfg, docs_path).run().documents
    cache_dir = tmp_path / "data" / "processed"
    chunks = ChunkingStage(cfg, tmp_path, cache_dir, docs_path).run(StageInput({"documents": docs}))

    factory = lambda config: _FakeEmbedder()
    first = EmbeddingStage(cfg, cache_dir, embedder_factory=factory).run(
        StageInput({"chunks": chunks.chunks, "chunks_key": chunks.chunks_key})
    )
    second = EmbeddingStage(cfg, cache_dir, embedder_factory=factory).run(
        StageInput({"chunks": chunks.chunks, "chunks_key": chunks.chunks_key})
    )

    assert first.cache_status == "built"
    assert second.cache_status == "validated"
    assert first.embeddings.shape == (1, 2)
    assert first.embeddings_path.exists()
    assert first.embeddings_path.with_suffix(first.embeddings_path.suffix + ".meta.json").exists()


def test_run_writer_stage_preserves_result_schema_and_counts(tmp_path):
    stage = RunWriterStage(tmp_path, save_csv=True, resume=False)
    output = stage.run()

    stage.write(_record())
    stage.close()

    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    counts = RunWriterStage.output_row_counts(tmp_path)
    assert output.existing_question_ids == set()
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["retrieved_original_context_ids"] == ["ctx1"]
    assert counts["results.jsonl"] == 1
    assert counts["results.csv"] == 1


def _cfg(tmp_path):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": str(tmp_path / "runs")},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake", "device": "cpu"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
            "reranker": {"enabled": False},
            "generation": {"provider": "ollama", "model_name": "fake", "system_prompt": "Use context."},
            "telemetry": {},
            "runtime": {},
        }
    )


def _record():
    return {
        "experiment_id": "exp",
        "question_id": "q1",
        "uid": "q1",
        "question": "Q?",
        "generated_answer": "1",
        "retrieved_chunk_ids": ["c1"],
        "retrieved_original_context_ids": ["ctx1"],
        "retrieved_context_texts": ["alpha"],
        "retrieval_scores": [1.0],
        "top_k": 1,
        "chunking_strategy": "fixed_word",
        "chunk_size": 10,
        "chunk_overlap": 0,
        "embedding_model": "fake",
        "retriever_type": "dense",
        "reranker_used": False,
        "llm_model": "fake",
        "retrieval_time_ms": 1.0,
        "generation_time_ms": 1.0,
        "total_latency_ms": 2.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }
