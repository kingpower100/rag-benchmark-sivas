"""Smoke test: validate that OutputRecord.to_export_record() produces
exactly the required top-level fields defined in the benchmark target schema,
and that the retrieved_chunks list has the correct per-chunk structure.
"""
from __future__ import annotations

import pytest

from src.pipeline1.schemas.output_record import OutputRecord

TARGET_TOP_LEVEL_FIELDS = {
    "question_id",
    "question",
    "clean_question",
    "detected_category",
    "category_validated",
    "category_validation_reason",
    "retrieval_mode",
    "category_filter_applied",
    "category_fallback_used",
    "retrieved_chunks",
    "answer",
    "config_id",
    "embedding_model",
    "generation_model",
    "retrieval_k",
}

TARGET_CHUNK_FIELDS = {"rank", "chunk_id", "doc_id", "doc_name", "category", "score", "chunk_text"}


def _make_record(**overrides) -> OutputRecord:
    defaults: dict = {
        "experiment_id": "cfg_001",
        "question_id": "Q001",
        "question": "Was ist ein Arbeitsplan?",
        "cleaned_question": "Was ist ein Arbeitsplan?",
        "detected_category": "Arbeitsplan",
        "category_validated": True,
        "category_validation_reason": None,
        "retrieval_mode": "category_aware_dense",
        "generated_answer": "Ein Arbeitsplan definiert die Abfolge der Arbeitsgänge.",
        "retrieved_chunk_ids": ["chunk_001"],
        "retrieved_original_context_ids": ["doc_key_001"],
        "retrieved_context_texts": ["Arbeitsplan definiert die einzelnen Arbeitsgänge."],
        "retrieval_scores": [0.91],
        "top_k": 5,
        "chunking_strategy": "fixed_token",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "retriever_type": "category_aware_dense",
        "reranker_used": False,
        "llm_model": "qwen2.5:7b",
        "retrieval_time_ms": 12.5,
        "generation_time_ms": 345.0,
        "total_latency_ms": 357.5,
        "input_tokens": 512,
        "output_tokens": 16,
        "total_tokens": 528,
    }
    defaults.update(overrides)
    return OutputRecord(**defaults)


def test_export_record_contains_all_target_fields():
    record = _make_record()
    export = record.to_export_record()
    missing = TARGET_TOP_LEVEL_FIELDS - set(export)
    assert not missing, f"to_export_record() is missing target fields: {sorted(missing)}"


def test_export_record_has_no_extra_undeclared_target_fields():
    """The 11 target fields must all be present; extra compat fields are allowed."""
    record = _make_record()
    export = record.to_export_record()
    for field in TARGET_TOP_LEVEL_FIELDS:
        assert field in export, f"Missing target field: {field!r}"


def test_export_record_field_renames():
    record = _make_record()
    export = record.to_export_record()
    assert export["clean_question"] == record.cleaned_question
    assert export["category_validated"] is True
    assert export["category_validation_reason"] is None
    assert export["retrieval_mode"] == "category_aware_dense"
    assert export["answer"] == record.generated_answer
    assert export["config_id"] == record.experiment_id
    assert export["generation_model"] == record.llm_model
    assert export["retrieval_k"] == record.top_k


def test_export_record_includes_reranker_timing_breakdown():
    record = _make_record(
        reranker_used=True,
        retriever_time_ms=10.0,
        rerank_time_ms=2.0,
        retrieval_pipeline_time_ms=13.0,
        reranker_applied=True,
        reranker_candidate_count=20,
        reranker_output_count=20,
        reranker_model_name="cross-encoder",
    )
    export = record.to_export_record()

    assert export["retriever_time_ms"] == 10.0
    assert export["rerank_time_ms"] == 2.0
    assert export["retrieval_pipeline_time_ms"] == 13.0
    assert export["reranker_applied"] is True
    assert export["reranker_candidate_count"] == 20
    assert export["reranker_output_count"] == 20
    assert export["reranker_model_name"] == "cross-encoder"


def test_export_record_does_not_emit_category_confidence():
    record = _make_record()
    export = record.to_export_record()
    assert "category_confidence" not in export


def test_export_record_retrieved_chunks_structure():
    record = _make_record(
        retrieved_chunk_ids=["chunk_001", "chunk_002"],
        retrieved_original_context_ids=["doc_001", "doc_002"],
        retrieved_context_texts=["Chunk text one.", "Chunk text two."],
        retrieval_scores=[0.91, 0.84],
        retrieved_chunk_metadata=[
            {"doc_id": 15, "doc_name": "handbuch.md", "kategorie": "Arbeitsplan"},
            {"doc_id": 16, "doc_name": "glossar.md", "kategorie": "Arbeitsplan"},
        ],
        retrieved_categories=["Arbeitsplan", "Arbeitsplan"],
        top_k=5,
    )
    export = record.to_export_record()
    chunks = export["retrieved_chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) == 2

    first = chunks[0]
    assert set(first.keys()) == TARGET_CHUNK_FIELDS
    assert first["rank"] == 1
    assert first["chunk_id"] == "chunk_001"
    assert first["doc_id"] == 15
    assert first["doc_name"] == "handbuch.md"
    assert first["category"] == "Arbeitsplan"
    assert first["score"] == pytest.approx(0.91)
    assert first["chunk_text"] == "Chunk text one."

    second = chunks[1]
    assert second["rank"] == 2
    assert second["chunk_id"] == "chunk_002"


def test_export_record_empty_retrieval():
    record = _make_record(
        retrieved_chunk_ids=[],
        retrieved_original_context_ids=[],
        retrieved_context_texts=[],
        retrieval_scores=[],
    )
    export = record.to_export_record()
    assert export["retrieved_chunks"] == []


def test_export_record_pipeline2_compat_fields_present():
    """Fields that pipeline2/orchestrator.py reads by original name must survive."""
    p2_fields = {
        "experiment_id",
        "generated_answer",
        "llm_model",
        "retrieved_original_context_ids",
        "retrieved_file_names",
        "raw_retrieved_file_names",
        "retrieved_chunk_metadata",
        "retrieval_time_ms",
        "generation_time_ms",
        "total_latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
        "error",
    }
    record = _make_record()
    export = record.to_export_record()
    missing = p2_fields - set(export)
    assert not missing, f"Pipeline 2 compat fields missing from export: {sorted(missing)}"


def test_export_record_is_json_serialisable():
    import json
    record = _make_record()
    export = record.to_export_record()
    serialised = json.dumps(export, ensure_ascii=False)
    parsed = json.loads(serialised)
    assert parsed["question_id"] == "Q001"
    assert parsed["answer"] == record.generated_answer
    assert isinstance(parsed["retrieved_chunks"], list)
