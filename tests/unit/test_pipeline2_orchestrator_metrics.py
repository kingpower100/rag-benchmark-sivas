import pytest

from src.pipeline2.orchestrator import EvaluationOrchestrator, _index_by_id, _validate_pipeline1_questions_have_qa
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def test_numeric_accuracy_flag_is_respected():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2},
            "answer_quality": {"enable_numeric_accuracy": False},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "100",
            "question": "What is revenue?",
            "retrieved_original_context_ids": ["c1"],
            "raw_retrieved_original_context_ids": ["c1", "c1"],
            "retrieval_time_ms": 4,
            "generation_time_ms": 8,
            "total_latency_ms": 12,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "estimated_cost": 0.0,
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["c1"]},
        cfg,
    )

    assert evaluated[0]["numeric_accuracy"] is None
    assert evaluated[0]["hit_at_1"] == 1.0
    assert evaluated[0]["hit_at_3"] == 1.0
    assert evaluated[0]["context_precision_at_3"] == 1 / 3
    assert evaluated[0]["ndcg_at_3"] == 1.0
    assert evaluated[0]["raw_duplicate_rate"] == 0.5
    assert evaluated[0]["non_empty_answer_rate"] == 1.0
    assert evaluated[0]["answer_coverage_rate"] == 1.0
    assert evaluated[0]["exact_match"] == 1.0
    assert evaluated[0]["numeric_parse_success"] == 1.0
    assert evaluated[0]["abstention_rate"] == 0.0
    assert evaluated[0]["answer_relevancy_score"] == 0.0
    assert evaluated[0]["retrieval_time_ms"] == 4
    assert evaluated[0]["generation_time_ms"] == 8
    assert evaluated[0]["input_tokens"] == 2
    assert evaluated[0]["output_tokens"] == 1
    assert evaluated[0]["id_alignment_ok"] is True


def test_missing_retrieved_original_context_ids_does_not_fallback_to_other_ids():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "100",
            "question": "Q?",
            "retrieved_context_ids": ["c1"],
            "retrieved_chunk_ids": ["chunk1"],
            "total_latency_ms": 12,
            "total_tokens": 3,
            "estimated_cost": 0.0,
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["c1"]},
        cfg,
    )

    assert evaluated[0]["retrieved_original_context_ids"] == []
    assert evaluated[0]["gold_context_ids"] == ["c1"]
    assert evaluated[0]["hit_at_1"] == 0.0
    assert evaluated[0]["recall_at_1"] == 0.0
    assert evaluated[0]["context_precision_at_1"] == 0.0
    assert evaluated[0]["mrr_at_1"] == 0.0
    assert evaluated[0]["ndcg_at_1"] == 0.0
    assert evaluated[0]["id_alignment_ok"] is False
    assert "missing retrieved_original_context_ids" in evaluated[0]["evaluation_errors"][0]


def test_retrieval_metrics_can_use_file_names_for_officeqa_gold_ids():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2, "ks": [1, 2]},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "100",
            "question": "Q?",
            "retrieved_original_context_ids": ["chunk_a", "chunk_b"],
            "raw_retrieved_original_context_ids": ["chunk_a", "chunk_b"],
            "retrieved_file_names": ["treasury_bulletin_1944_01.txt", "treasury_bulletin_1944_01.txt"],
            "raw_retrieved_file_names": ["treasury_bulletin_1944_01.txt", "treasury_bulletin_1944_01.txt"],
            "retrieved_document_ids": ["treasury_bulletin_1944_01.txt", "treasury_bulletin_1944_01.txt"],
            "raw_retrieved_document_ids": ["treasury_bulletin_1944_01.txt", "treasury_bulletin_1944_01.txt"],
            "total_latency_ms": 12,
            "total_tokens": 3,
            "estimated_cost": 0.0,
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["treasury_bulletin_1944_01.txt"]},
        cfg,
    )

    assert evaluated[0]["retrieval_eval_ids"] == [
        "treasury_bulletin_1944_01.txt",
        "treasury_bulletin_1944_01.txt",
    ]
    assert evaluated[0]["hit_at_1"] == 1.0
    assert evaluated[0]["recall_at_1"] == 1.0
    assert evaluated[0]["raw_duplicate_rate"] == 0.5


def test_missing_gold_contexts_fail_evaluation():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [{"question_id": "q_missing", "experiment_id": "exp", "generated_answer": "100", "retrieved_original_context_ids": ["c1"]}]

    with pytest.raises(ValueError, match="Missing 1 question"):
        EvaluationOrchestrator()._evaluate_rows(rows, {"q_missing": {"id": "q_missing", "answer": "100"}}, {}, cfg)


def test_pipeline1_error_row_is_retained_and_scores_zero_for_answer_metrics():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2, "ks": [1, 2]},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "100",
            "question": "Q?",
            "retrieved_original_context_ids": [],
            "error": "generation failed",
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["c1"]},
        cfg,
    )

    assert len(evaluated) == 1
    assert evaluated[0]["pipeline_success"] == 0.0
    assert evaluated[0]["pipeline1_error"] == "generation failed"
    assert evaluated[0]["numeric_accuracy"] == 0.0
    assert evaluated[0]["exact_match"] == 0.0
    assert evaluated[0]["non_empty_answer_rate"] == 0.0
    assert evaluated[0]["numeric_parse_success"] == 0.0
    assert evaluated[0]["answer_match_status"] == "pipeline1_error"
    assert evaluated[0]["hit_at_1"] == 0.0
    assert evaluated[0]["recall_at_1"] == 0.0
    assert evaluated[0]["mrr_at_1"] == 0.0


def test_qa_index_supports_uid_only_rows_and_numeric_answer_resolution():
    qa_by_id = _index_by_id(
        [
            {
                "uid": "UID0002",
                "question": "Q?",
                "answer": "507",
                "source_files": "treasury_bulletin_1944_01.txt",
            }
        ]
    )
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 1, "ks": [1]},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [
        {
            "question_id": "UID0002",
            "experiment_id": "exp",
            "generated_answer": "507",
            "question": "Q?",
            "retrieved_original_context_ids": ["treasury_bulletin_1944_01.txt"],
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        qa_by_id,
        {"UID0002": ["treasury_bulletin_1944_01.txt"]},
        cfg,
    )

    assert evaluated[0]["ground_truth_answer"] == "507"
    assert evaluated[0]["numeric_accuracy"] == 1.0
    assert evaluated[0]["exact_match"] == 1.0
    assert evaluated[0]["answer_match_status"] == "match"


def test_qa_index_still_supports_id_only_rows():
    qa_by_id = _index_by_id([{"id": "q1", "answer": "100"}])

    assert qa_by_id["q1"]["answer"] == "100"


def test_qa_index_still_supports_question_id_only_rows():
    qa_by_id = _index_by_id([{"question_id": "q1", "answer": "100"}])

    assert qa_by_id["q1"]["answer"] == "100"


def test_qa_index_supports_mixed_id_styles():
    qa_by_id = _index_by_id(
        [
            {"uid": "u1", "answer": "1"},
            {"id": "i1", "answer": "2"},
            {"question_id": "q1", "answer": "3"},
        ]
    )

    assert sorted(qa_by_id) == ["i1", "q1", "u1"]


def test_qa_index_prefers_uid_over_other_id_fields():
    qa_by_id = _index_by_id([{"uid": "u1", "id": "legacy", "question_id": "q1", "answer": "1"}])

    assert sorted(qa_by_id) == ["u1"]


def test_qa_index_duplicate_ids_fail():
    with pytest.raises(ValueError, match="duplicate resolved IDs: q1"):
        _index_by_id([{"uid": "q1", "answer": "1"}, {"id": "q1", "answer": "2"}])


def test_qa_index_missing_ids_fail():
    with pytest.raises(ValueError, match="missing uid/id/question_id"):
        _index_by_id([{"question": "Q?", "answer": "1"}])


def test_qa_index_empty_answer_fails():
    with pytest.raises(ValueError, match="empty answer fields"):
        _index_by_id([{"uid": "q1", "answer": ""}])


def test_missing_pipeline1_question_ids_in_qa_fail():
    rows = [{"question_id": "q_missing", "generated_answer": "1"}]

    with pytest.raises(ValueError, match="missing answers"):
        _validate_pipeline1_questions_have_qa(rows, {"q1": {"answer": "1"}})
