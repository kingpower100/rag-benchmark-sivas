import json
import shutil
from pathlib import Path

import pytest

from src.pipeline2.orchestrator import EvaluationOrchestrator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_pipeline2_writes_final_metric_outputs():
    workspace = Path(".tmp_test_pipeline2_integration").resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    try:
        rag_path = workspace / "rag.jsonl"
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
                    "question": "What is revenue?",
                    "retrieved_original_context_ids": ["c2", "c1"],
                    "raw_retrieved_original_context_ids": ["c2", "c2", "c1"],
                    "retrieval_time_ms": 5,
                    "generation_time_ms": 15,
                    "total_latency_ms": 20,
                    "input_tokens": 6,
                    "output_tokens": 2,
                    "total_tokens": 8,
                    "estimated_cost": 0.0,
                    "error": None,
                }
            ],
        )
        _write_jsonl(qa_path, [{"id": "q1", "question": "Q?", "answer": "100"}])
        _write_jsonl(gold_path, [{"id": "q1", "context_id": "c1"}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_eval
  output_dir: "{out_dir.as_posix()}"
inputs:
  rag_outputs:
    - "{rag_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
retrieval:
  k: 2
answer_quality:
  enable_numeric_accuracy: true
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        per_question = (run_dir / "per_question.jsonl").read_text(encoding="utf-8").strip()
        row = json.loads(per_question)
        assert row["hit_at_1"] == 0.0
        assert row["hit_at_3"] == 1.0
        assert row["recall_at_3"] == 1.0
        assert row["context_precision_at_3"] == 1 / 3
        assert row["mrr_at_3"] == 0.5
        assert row["ndcg_at_3"] > 0
        assert row["raw_duplicate_rate"] == 1 / 3
        assert row["retrieval_time_ms"] == 5
        assert row["generation_time_ms"] == 15
        assert row["input_tokens"] == 6
        assert row["output_tokens"] == 2
        assert row["retrieved_original_context_ids"] == ["c2", "c1"]
        assert row["gold_context_ids"] == ["c1"]
        assert row["id_alignment_ok"] is True
        assert row["numeric_accuracy"] == 1.0
        assert row["exact_match"] == 1.0
        assert row["numeric_parse_success"] == 1.0
        assert row["non_empty_answer_rate"] == 1.0
        summary = (run_dir / "summary_by_experiment.csv").read_text(encoding="utf-8")
        assert "mean_hit_at_3" in summary
        assert "mean_context_precision_at_3" in summary
        assert "pipeline_success_rate" in summary
        manifest = json.loads((run_dir / "eval_manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_path"] == str(cfg_path.resolve())
        assert manifest["config_hash"]
        assert manifest["row_counts"]["pipeline1_results"] == 1
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def test_pipeline2_missing_input_fails_fast():
    workspace = Path(".tmp_test_pipeline2_missing_input").resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    try:
        qa_path = workspace / "qa.jsonl"
        gold_path = workspace / "gold.jsonl"
        cfg_path = workspace / "eval.yaml"
        out_dir = workspace / "out"
        _write_jsonl(qa_path, [{"id": "q1", "question": "Q?", "answer": "100"}])
        _write_jsonl(gold_path, [{"id": "q1", "context_id": "c1"}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_eval
  output_dir: "{out_dir.as_posix()}"
inputs:
  rag_outputs:
    - "{(workspace / 'missing_results.jsonl').as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
""",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="missing"):
            EvaluationOrchestrator().run(str(cfg_path))
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
