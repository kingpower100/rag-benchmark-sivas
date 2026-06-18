import pytest

from src.pipeline2.orchestrator import (
    EvaluationOrchestrator,
    _raise_on_failure_threshold,
    _run_validity_by_experiment,
    _index_by_id,
    _gold_by_question,
    _merge_gold_with_qa_fallback,
    _validate_pipeline1_questions_have_qa,
    _validate_pipeline1_questions_have_gold_contexts,
    _validate_no_duplicate_pipeline1_question_ids,
    summarize_by_difficulty,
)
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def test_numeric_accuracy_flag_is_respected():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
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
    assert evaluated[0]["context_precision_at_3"] == 1.0
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


def test_missing_configured_retrieval_eval_field_raises_clear_error():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
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

    with pytest.raises(ValueError, match="retrieval_eval_field='retrieved_original_context_ids' is missing"):
        EvaluationOrchestrator()._evaluate_rows(
            rows,
            {"q1": {"id": "q1", "answer": "100"}},
            {"q1": ["c1"]},
            cfg,
        )


def test_retrieval_metrics_can_use_file_names_for_sivas_gold_ids():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_file_names"},
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
            "retrieved_file_names": ["sivas_manual_01.md", "sivas_manual_01.md"],
            "raw_retrieved_file_names": ["sivas_manual_01.md", "sivas_manual_01.md"],
            "retrieved_document_ids": ["doc-key-1", "doc-key-1"],
            "raw_retrieved_document_ids": ["doc-key-1", "doc-key-1"],
            "total_latency_ms": 12,
            "total_tokens": 3,
            "estimated_cost": 0.0,
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["sivas_manual_01.md"]},
        cfg,
    )

    assert evaluated[0]["retrieval_eval_ids"] == [
        "sivas_manual_01.md",
        "sivas_manual_01.md",
    ]
    assert evaluated[0]["hit_at_1"] == 1.0
    assert evaluated[0]["recall_at_1"] == 1.0
    assert evaluated[0]["raw_duplicate_rate"] == 0.5


def test_retrieval_evaluation_does_not_switch_fields_based_on_gold_overlap():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_file_names"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 1, "ks": [1]},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "100",
            "question": "Q?",
            "retrieved_original_context_ids": ["gold_doc"],
            "retrieved_file_names": ["wrong_doc.txt"],
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"q1": {"id": "q1", "answer": "100"}},
        {"q1": ["gold_doc"]},
        cfg,
    )

    assert evaluated[0]["retrieval_eval_ids"] == ["wrong_doc.txt"]
    assert evaluated[0]["hit_at_1"] == 0.0


def test_missing_gold_contexts_fail_evaluation():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2},
            "answer_quality": {"enable_numeric_accuracy": True},
        }
    )
    rows = [{"question_id": "q_missing", "experiment_id": "exp", "generated_answer": "100", "retrieved_original_context_ids": ["c1"]}]

    with pytest.raises(ValueError, match="Missing gold context for question q_missing"):
        EvaluationOrchestrator()._evaluate_rows(rows, {"q_missing": {"id": "q_missing", "answer": "100"}}, {}, cfg)


def test_gold_context_validation_fails_before_summary_denominator_can_change():
    rows = [{"question_id": "q_missing", "experiment_id": "exp"}]

    with pytest.raises(ValueError, match="Missing gold context for question q_missing"):
        _validate_pipeline1_questions_have_gold_contexts(rows, {})


def test_same_question_ids_across_experiments_are_allowed():
    rows = [
        {"question_id": "q1", "experiment_id": "exp_a"},
        {"question_id": "q1", "experiment_id": "exp_b"},
    ]

    _validate_no_duplicate_pipeline1_question_ids(rows)


def test_duplicate_question_ids_inside_same_experiment_still_fail():
    rows = [
        {"question_id": "q1", "experiment_id": "exp_a"},
        {"question_id": "q1", "experiment_id": "exp_a"},
    ]

    with pytest.raises(ValueError, match="exp_a:q1"):
        _validate_no_duplicate_pipeline1_question_ids(rows)


def test_pipeline1_error_row_is_retained_and_scores_zero_for_answer_metrics():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
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
                "source_files": "sivas_manual_01.md",
            }
        ]
    )
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
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
            "retrieved_original_context_ids": ["sivas_manual_01.md"],
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        qa_by_id,
        {"UID0002": ["sivas_manual_01.md"]},
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


def test_qa_index_allows_empty_answers_for_retrieval_only_mode():
    qa_by_id = _index_by_id([{"uid": "q1", "question": "Q?", "source_files": ["sivas_manual_02.md"]}], require_answer=False)

    assert qa_by_id["q1"]["source_files"] == ["sivas_manual_02.md"]


def test_pipeline2_joins_difficulty():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
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
            "retrieved_original_context_ids": ["sivas_manual_01.md"],
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"UID0002": {"uid": "UID0002", "answer": "507", "difficulty": "easy"}},
        {"UID0002": ["sivas_manual_01.md"]},
        cfg,
    )

    assert evaluated[0]["uid"] == "UID0002"
    assert evaluated[0]["difficulty"] == "easy"


def test_retrieval_only_mode_skips_answer_metrics_without_generated_answer():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_only": True, "retrieval_eval_field": "retrieved_original_context_ids"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 1, "ks": [1]},
        }
    )
    rows = [
        {
            "question_id": "UID0001",
            "experiment_id": "exp",
            "question": "Q?",
            "retrieved_original_context_ids": ["sivas_manual_02"],
        }
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {"UID0001": {"uid": "UID0001", "difficulty": "hard", "source_files": ["sivas_manual_02.md"]}},
        {"UID0001": ["sivas_manual_02"]},
        cfg,
    )

    assert evaluated[0]["hit_at_1"] == 1.0
    assert evaluated[0]["numeric_accuracy"] is None
    assert evaluated[0]["exact_match"] is None
    assert evaluated[0]["answer_match_status"] == "skipped_retrieval_only"


def test_source_files_fallback_from_qa_is_disabled():
    with pytest.raises(RuntimeError, match="fallback.*disabled"):
        _merge_gold_with_qa_fallback(
            {},
            {"UID0001": {"uid": "UID0001", "source_files": ["sivas_manual_02.md"]}},
        )


def test_gold_context_index_supports_uid_and_rejects_duplicates():
    assert _gold_by_question([{"uid": "UID0001", "context_id": ["doc.txt"]}]) == {"UID0001": ["doc.txt"]}

    with pytest.raises(ValueError, match="duplicate resolved IDs"):
        _gold_by_question([
            {"id": "UID0001", "context_id": ["doc.txt"]},
            {"uid": "UID0001", "context_id": ["other.txt"]},
        ])


def test_difficulty_summary_includes_all_and_each_difficulty():
    rows = [
        {"difficulty": "easy", "hit_at_1": 1.0, "recall_at_1": 1.0, "mrr_at_1": 1.0, "ndcg_at_1": 1.0, "exact_match": 1.0, "numeric_accuracy": 1.0, "total_latency_ms": 10, "total_tokens": 5},
        {"difficulty": "hard", "hit_at_1": 0.0, "recall_at_1": 0.0, "mrr_at_1": 0.0, "ndcg_at_1": 0.0, "exact_match": 0.0, "numeric_accuracy": 0.0, "total_latency_ms": 20, "total_tokens": 7},
    ]

    summary = summarize_by_difficulty(rows)

    assert [row["difficulty"] for row in summary] == ["all", "easy", "hard"]
    assert summary[0]["n_questions"] == 2
    assert summary[0]["mean_hit_at_1"] == 0.5


def test_generation_failure_threshold_validity_counts_failures():
    rows = [{"experiment_id": "exp", "generation_failed": False} for _ in range(97)]
    rows.extend({"experiment_id": "exp", "generation_failed": True} for _ in range(3))

    stats = _run_validity_by_experiment(rows, 0.05)["exp"]

    assert stats["total_questions"] == 100
    assert stats["generation_failure_count"] == 3
    assert stats["generation_failure_rate"] == pytest.approx(0.03)
    assert stats["pipeline_success_rate"] == pytest.approx(0.97)
    assert stats["run_valid"] is True


def test_generation_failure_threshold_marks_invalid_when_exceeded():
    rows = [{"experiment_id": "exp", "generation_failed": False} for _ in range(94)]
    rows.extend({"experiment_id": "exp", "generation_failed": True} for _ in range(6))

    stats = _run_validity_by_experiment(rows, 0.05)["exp"]

    assert stats["generation_failure_count"] == 6
    assert stats["generation_failure_rate"] == pytest.approx(0.06)
    assert stats["run_valid"] is False


def test_generation_failure_threshold_accepts_zero_failures():
    rows = [{"experiment_id": "exp", "generation_failed": False} for _ in range(100)]

    stats = _run_validity_by_experiment(rows, 0.05)["exp"]

    assert stats["generation_failure_count"] == 0
    assert stats["run_valid"] is True


def test_strict_generation_failure_threshold_raises_when_exceeded():
    stats = {"exp": {"generation_failure_rate": 0.06, "run_valid": False}}

    with pytest.raises(RuntimeError, match="Generation failure rate exceeded"):
        _raise_on_failure_threshold(stats, 0.05)


def test_missing_pipeline1_question_ids_in_qa_fail():
    rows = [{"question_id": "q_missing", "generated_answer": "1"}]

    with pytest.raises(ValueError, match="missing answers"):
        _validate_pipeline1_questions_have_qa(rows, {"q1": {"answer": "1"}})
