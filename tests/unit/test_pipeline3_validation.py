from __future__ import annotations

import pytest

from src.pipeline3.stages.validation_stage import (
    ValidationError,
    _extract_context_texts,
    _resolve_id,
    _resolve_qa_answer,
    build_qa_index,
    validate_inputs,
)


def _make_rag_row(question_id="q1", answer="Some answer", contexts=None):
    return {
        "question_id": question_id,
        "experiment_id": "exp1",
        "question": "What is the price?",
        "generated_answer": answer,
        "retrieved_context_texts": contexts or ["Context text here."],
    }


def _make_qa_row(question_id="q1", answer="Correct answer"):
    return {"question_id": question_id, "answer": answer}


def _make_questions_row(question_id="q1"):
    return {"question_id": question_id, "question": "What is the price?"}


def test_valid_inputs_pass():
    report = validate_inputs(
        [_make_rag_row()],
        [_make_qa_row()],
        [_make_questions_row()],
    )
    assert report.passed is True
    assert len(report.errors) == 0


def test_empty_rag_rows_raises():
    with pytest.raises(ValidationError, match="zero rows"):
        validate_inputs([], [_make_qa_row()], [_make_questions_row()])


def test_missing_qa_for_rag_row_raises():
    with pytest.raises(ValidationError):
        validate_inputs(
            [_make_rag_row("q1")],
            [_make_qa_row("q2")],
            [_make_questions_row("q1")],
        )


def test_duplicate_question_ids_raises():
    with pytest.raises(ValidationError, match="Duplicate"):
        validate_inputs(
            [_make_rag_row("q1"), _make_rag_row("q1")],
            [_make_qa_row("q1")],
            [_make_questions_row("q1")],
        )


def test_missing_answer_is_warning_not_error():
    report = validate_inputs(
        [_make_rag_row("q1", answer="")],
        [_make_qa_row("q1")],
        [_make_questions_row("q1")],
    )
    assert report.passed is True
    assert any("Missing generated_answer" in w for w in report.warnings)


def test_missing_context_is_warning_not_error():
    row = {
        "question_id": "q1",
        "generated_answer": "Some answer",
        "retrieved_context_texts": [],
    }
    report = validate_inputs(
        [row],
        [_make_qa_row("q1")],
        [_make_questions_row("q1")],
    )
    assert report.passed is True
    assert any("Missing retrieved contexts" in w for w in report.warnings)


def test_build_qa_index_keys_by_question_id():
    qa_rows = [
        {"question_id": "q1", "answer": "A"},
        {"question_id": "q2", "answer": "B"},
    ]
    index = build_qa_index(qa_rows)
    assert "q1" in index
    assert "q2" in index


def test_extract_context_texts_from_direct_field():
    row = {"retrieved_context_texts": ["text1", "text2"]}
    texts = _extract_context_texts(row)
    assert texts == ["text1", "text2"]


def test_extract_context_texts_from_chunk_texts_field():
    row = {"retrieved_chunk_texts": ["chunk1"]}
    texts = _extract_context_texts(row)
    assert texts == ["chunk1"]


def test_extract_context_texts_from_chunks_dicts():
    row = {
        "retrieved_chunks": [
            {"chunk_id": "c1", "chunk_text": "chunk text one"},
            {"chunk_id": "c2", "chunk_text": "chunk text two"},
        ]
    }
    texts = _extract_context_texts(row)
    assert len(texts) == 2
    assert "chunk text one" in texts


def test_extract_context_texts_empty_when_no_field():
    row = {"question_id": "q1"}
    texts = _extract_context_texts(row)
    assert texts == []


def test_resolve_id_prefers_question_id():
    row = {"question_id": "q1", "uid": "u1", "id": "i1"}
    assert _resolve_id(row) == "q1"


def test_resolve_id_falls_back_to_uid():
    row = {"uid": "u1"}
    assert _resolve_id(row) == "u1"


def test_resolve_id_falls_back_to_id():
    row = {"id": "i1"}
    assert _resolve_id(row) == "i1"


def test_resolve_id_returns_empty_when_missing():
    row = {"question": "text only"}
    assert _resolve_id(row) == ""


def test_resolve_qa_answer_tries_multiple_keys():
    assert _resolve_qa_answer({"answer": "A"}) == "A"
    assert _resolve_qa_answer({"gold_answer": "B"}) == "B"
    assert _resolve_qa_answer({"referenzantwort": "C"}) == "C"
    assert _resolve_qa_answer({"ground_truth_answer": "D"}) == "D"


def test_resolve_qa_answer_returns_empty_when_missing():
    assert _resolve_qa_answer({}) == ""


def test_validation_stats_populated():
    report = validate_inputs(
        [_make_rag_row("q1")],
        [_make_qa_row("q1")],
        [_make_questions_row("q1")],
    )
    assert report.stats["rag_rows"] == 1
    assert report.stats["qa_rows"] == 1
    assert report.stats["questions_rows"] == 1
