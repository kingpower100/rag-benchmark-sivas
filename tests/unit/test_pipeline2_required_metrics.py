import json
import shutil
from pathlib import Path

import pytest

from src.pipeline2.aggregation.summarizer import summarize_by_experiment
from src.pipeline2.io.jsonl import read_jsonl
from src.pipeline2.metrics.efficiency_metrics import compute_efficiency_metrics
from src.pipeline2.metrics.embedding_similarity import DeterministicHashEmbedder, compute_embedding_similarity
from src.pipeline2.orchestrator import EvaluationOrchestrator
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def test_embedding_similarity_uses_configurable_deterministic_embedder():
    embedder = DeterministicHashEmbedder(model_name="test-model", dimensions=64)

    assert compute_embedding_similarity("net income", "net income", embedder) == pytest.approx(1.0)
    assert compute_embedding_similarity("", "net income", embedder) == 0.0


def test_total_latency_is_component_sum_with_rerank():
    metrics = compute_efficiency_metrics(
        {"retrieval_time_ms": 10, "rerank_time_ms": 2.5, "generation_time_ms": 20, "total_latency_ms": 999}
    )

    assert metrics["retrieval_time_ms"] == 10
    assert metrics["rerank_time_ms"] == 2.5
    assert metrics["generation_time_ms"] == 20
    assert metrics["total_latency_ms"] == 32.5


def test_summary_aggregates_semantic_latency_and_reliability_denominators():
    rows = [
        {
            "experiment_id": "exp",
            "embedding_similarity": 1.0,
            "bertscore_f1": 1.0,
            "total_latency_ms": 10,
            "generation_failed": False,
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "embedding_similarity": 0.0,
            "bertscore_f1": 0.0,
            "total_latency_ms": 0,
            "generation_failed": True,
            "evaluation_errors": ["missing gold context"],
        },
    ]

    summary = summarize_by_experiment(rows)[0]

    assert summary["n_questions"] == 2
    assert summary["mean_embedding_similarity"] == 0.5
    assert summary["mean_bertscore_f1"] == 0.5
    assert summary["mean_total_latency_ms"] == 5
    assert summary["pipeline_success_rate"] == 0.5
    assert summary["eval_success_rate"] == 0.5


def test_orchestrator_emits_generation_metrics_for_edge_cases():
    cfg = EvalConfig.model_validate(
        {
            "evaluation": {"eval_run_id": "eval", "retrieval_eval_field": "retrieved_original_context_ids"},
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 2, "ks": [1, 2]},
            "embedding_similarity": {"provider": "deterministic_hash", "model_name": "unit-test", "dimensions": 64},
        }
    )
    rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "generated_answer": "2.602 billion dollars",
            "question": "Q?",
            "retrieved_original_context_ids": ["c2", "c1", "c1"],
            "retrieval_time_ms": 1,
            "rerank_time_ms": 2,
            "generation_time_ms": 3,
        },
        {
            "question_id": "q2",
            "experiment_id": "exp",
            "generated_answer": "UNKNOWN",
            "question": "Q?",
            "retrieved_original_context_ids": [],
        },
        {
            "question_id": "q3",
            "experiment_id": "exp",
            "generated_answer": "",
            "question": "Q?",
            "retrieved_original_context_ids": ["c3"],
            "error": "generation failed",
        },
        {
            "question_id": "q4",
            "experiment_id": "exp",
            "generated_answer": "not available",
            "question": "Q?",
            "retrieved_original_context_ids": ["c4"],
        },
    ]

    evaluated = EvaluationOrchestrator()._evaluate_rows(
        rows,
        {
            "q1": {"id": "q1", "answer": "2602 million dollars"},
            "q2": {"id": "q2", "answer": "100"},
            "q3": {"id": "q3", "answer": "300"},
            "q4": {"id": "q4", "answer": "400"},
        },
        {"q1": ["c1", "c9"], "q2": ["c2"], "q3": ["c3"], "q4": ["c4"]},
        cfg,
    )

    # provider=deterministic_hash routes value to bow_token_overlap_similarity, not embedding_similarity
    assert evaluated[0]["bow_token_overlap_similarity"] > 0.0
    assert evaluated[0]["embedding_similarity"] is None
    assert evaluated[0]["hit_at_2"] == 1.0
    assert evaluated[0]["recall_at_2"] == 0.5
    assert evaluated[0]["context_precision_at_2"] == 0.5
    assert evaluated[0]["duplicate_count_at_2"] == 0
    assert evaluated[0]["total_latency_ms"] == 6
    assert evaluated[1]["abstention_rate"] == 1.0
    assert evaluated[1]["is_unknown"] == 1.0
    assert evaluated[2]["generation_failed"] is True
    # failed row: bow_token_overlap_similarity zeroed, embedding_similarity remains None
    assert evaluated[2]["bow_token_overlap_similarity"] == 0.0
    assert evaluated[2]["embedding_similarity"] is None
    assert evaluated[3]["id_alignment_ok"] is True


def test_required_output_files_are_written():
    workspace = Path(".tmp_test_pipeline2_required_outputs").resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    try:
        rag_path = workspace / "rag.jsonl"
        questions_path = workspace / "questions.jsonl"
        qa_path = workspace / "qa.jsonl"
        gold_path = workspace / "gold.jsonl"
        cfg_path = workspace / "eval.yaml"
        out_dir = workspace / "out"
        _write_jsonl(
            rag_path,
            [
                {
                    "question_id": "q1",
                    "experiment_id": "exp",
                    "generated_answer": "100",
                    "question": "Q?",
                    "retrieved_original_context_ids": ["c1"],
                    "retrieval_time_ms": 1,
                    "rerank_time_ms": 2,
                    "generation_time_ms": 3,
                }
            ],
        )
        _write_jsonl(questions_path, [{"id": "q1", "question": "Q?"}])
        _write_jsonl(qa_path, [{"id": "q1", "answer": "100"}])
        _write_jsonl(gold_path, [{"id": "q1", "context_id": "c1"}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: eval
  output_dir: "{out_dir.as_posix()}"
  retrieval_eval_field: "retrieved_original_context_ids"
inputs:
  rag_outputs:
    - "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
retrieval:
  ks: [1]
embedding_similarity:
  provider: "deterministic_hash"
  model_name: "unit-test"
  dimensions: 64
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        assert (run_dir / "per_question_metrics.jsonl").exists()
        assert (run_dir / "summary_metrics.json").exists()
        assert (run_dir / "eval_manifest.json").exists()
        assert (run_dir / "audit_report.json").exists()
        assert (run_dir / "audit_report.md").exists()
        # Deprecated outputs must NOT be written
        row = read_jsonl(run_dir / "per_question_metrics.jsonl")[0]
        summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        # provider=deterministic_hash routes value to bow_token_overlap_similarity
        assert row["bow_token_overlap_similarity"] == pytest.approx(1.0)
        assert row["embedding_similarity"] is None
        assert row["total_latency_ms"] == 6
        # Benchmark validity must be present
        assert "benchmark_validity" in summary
        assert summary["benchmark_validity"]["benchmark_validity_status"] in ("VALID", "WARNING", "INVALID")
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
