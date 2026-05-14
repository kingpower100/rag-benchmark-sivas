import pytest

from src.pipeline2.orchestrator import EvaluationOrchestrator
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
            "retrieved_original_context_ids": ["c1"],
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
    assert evaluated[0]["answer_coverage_rate"] == 1.0
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
    assert evaluated[0]["id_alignment_ok"] is False
    assert "missing retrieved_original_context_ids" in evaluated[0]["evaluation_errors"][0]


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
