import json
import shutil
from pathlib import Path

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
        _write_jsonl(questions_path, [{"id": "q1", "question": "What is revenue?"}])
        _write_jsonl(qa_path, [{"id": "q1", "question": "Q?", "answer": "100"}])
        _write_jsonl(gold_path, [{"id": "q1", "context_id": "c1"}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_eval
  output_dir: "{out_dir.as_posix()}"
  retrieval_eval_field: "retrieved_original_context_ids"
inputs:
  rag_outputs:
    - "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
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
        assert row["context_precision_at_3"] == 0.5
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
        cfg_path = workspace / "eval.yaml"
        out_dir = workspace / "out"
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_eval
  output_dir: "{out_dir.as_posix()}"
inputs:
  rag_outputs:
    - "{(workspace / 'missing_results.jsonl').as_posix()}"
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))
        audit = json.loads((run_dir / "audit_report.json").read_text(encoding="utf-8"))
        assert audit["audit_status"] == "skipped"
        assert audit["fake_run_detection"]["suspicious"] is True
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def test_pipeline2_evaluates_uid_only_qa_file():
    workspace = Path(".tmp_test_pipeline2_uid_qa").resolve()
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
                    "question_id": "UID0002",
                    "experiment_id": "sivas",
                    "generated_answer": "507",
                    "question": "Q?",
                    "retrieved_original_context_ids": ["chunk_a"],
                    "retrieved_file_names": ["sivas_manual_01.md"],
                    "retrieved_document_ids": ["doc-key-1"],
                    "error": None,
                }
            ],
        )
        _write_jsonl(questions_path, [{"id": "UID0002", "question": "Q?"}])
        _write_jsonl(
            qa_path,
            [
                {
                    "uid": "UID0002",
                    "question": "Q?",
                    "answer": "507",
                    "source_docs": "https://example.test/source",
                    "source_files": "sivas_manual_01.md",
                    "difficulty": "easy",
                }
            ],
        )
        _write_jsonl(gold_path, [{"id": "UID0002", "context_id": ["sivas_manual_01.md"]}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_uid_eval
  output_dir: "{out_dir.as_posix()}"
inputs:
  rag_outputs:
    - "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
retrieval:
  k: 1
  ks: [1]
answer_quality:
  enable_numeric_accuracy: true
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        row = json.loads((run_dir / "per_question.jsonl").read_text(encoding="utf-8").strip())
        assert row["ground_truth_answer"] == "507"
        assert row["numeric_accuracy"] == 1.0
        assert row["exact_match"] == 1.0
        assert row["answer_match_status"] == "match"
        assert row["retrieval_eval_ids"] == ["sivas_manual_01.md"]
        assert row["hit_at_1"] == 1.0
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def test_pipeline2_summary_counts_pipeline1_failures_in_denominator():
    workspace = Path(".tmp_test_pipeline2_fair_aggregation").resolve()
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
                    "error": None,
                },
                {
                    "question_id": "q2",
                    "experiment_id": "exp",
                    "generated_answer": "200",
                    "question": "Q?",
                    "retrieved_original_context_ids": ["c2"],
                    "error": None,
                },
                {
                    "question_id": "q3",
                    "experiment_id": "exp",
                    "generated_answer": "",
                    "question": "Q?",
                    "retrieved_original_context_ids": [],
                    "error": "generation failed",
                },
                {
                    "question_id": "q4",
                    "experiment_id": "exp",
                    "generated_answer": "",
                    "question": "Q?",
                    "retrieved_original_context_ids": [],
                    "error": "timeout",
                },
            ],
        )
        _write_jsonl(
            questions_path,
            [
                {"id": "q1", "question": "Q?"},
                {"id": "q2", "question": "Q?"},
                {"id": "q3", "question": "Q?"},
                {"id": "q4", "question": "Q?"},
            ],
        )
        _write_jsonl(
            qa_path,
            [
                {"id": "q1", "answer": "100"},
                {"id": "q2", "answer": "200"},
                {"id": "q3", "answer": "300"},
                {"id": "q4", "answer": "400"},
            ],
        )
        _write_jsonl(
            gold_path,
            [
                {"id": "q1", "context_id": "c1"},
                {"id": "q2", "context_id": "c2"},
                {"id": "q3", "context_id": "c3"},
                {"id": "q4", "context_id": "c4"},
            ],
        )
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_fair_eval
  output_dir: "{out_dir.as_posix()}"
  retrieval_eval_field: "retrieved_original_context_ids"
inputs:
  rag_outputs:
    - "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
retrieval:
  k: 1
  ks: [1]
answer_quality:
  enable_numeric_accuracy: true
leaderboard:
  sort_metric: "mean_numeric_accuracy"
  sort_ascending: false
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        per_question = [
            json.loads(line)
            for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = (run_dir / "summary_by_experiment.csv").read_text(encoding="utf-8")
        leaderboard = (run_dir / "leaderboard.csv").read_text(encoding="utf-8")
        failed = [row for row in per_question if row["pipeline1_error"]]
        assert len(per_question) == 4
        assert len(failed) == 2
        assert all(row["numeric_accuracy"] == 0.0 for row in failed)
        assert all(row["exact_match"] == 0.0 for row in failed)
        assert all(row["pipeline_success"] == 0.0 for row in failed)
        assert all(row["answer_match_status"] == "pipeline1_error" for row in failed)
        assert "mean_numeric_accuracy" in leaderboard

        import csv

        summary_row = next(csv.DictReader(summary.splitlines()))
        leaderboard_row = next(csv.DictReader(leaderboard.splitlines()))
        assert float(summary_row["mean_numeric_accuracy"]) == 0.5
        assert float(summary_row["mean_exact_match"]) == 0.5
        assert float(summary_row["mean_recall_at_1"]) == 0.5
        assert float(summary_row["mean_mrr_at_1"]) == 0.5
        assert float(summary_row["pipeline_success_rate"]) == 0.5
        assert float(leaderboard_row["mean_numeric_accuracy"]) == 0.5
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)


def test_pipeline2_evaluates_multiple_experiments_with_same_question_ids():
    workspace = Path(".tmp_test_pipeline2_multi_config").resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()
    try:
        rag_a_path = workspace / "rag_a.jsonl"
        rag_b_path = workspace / "rag_b.jsonl"
        questions_path = workspace / "questions.jsonl"
        qa_path = workspace / "qa.jsonl"
        gold_path = workspace / "gold.jsonl"
        cfg_path = workspace / "eval.yaml"
        out_dir = workspace / "out"
        _write_jsonl(
            rag_a_path,
            [
                {
                    "question_id": "q1",
                    "experiment_id": "exp_a",
                    "generated_answer": "100",
                    "question": "Q?",
                    "retrieved_original_context_ids": ["gold"],
                    "error": None,
                }
            ],
        )
        _write_jsonl(
            rag_b_path,
            [
                {
                    "question_id": "q1",
                    "experiment_id": "exp_b",
                    "generated_answer": "90",
                    "question": "Q?",
                    "retrieved_original_context_ids": ["wrong"],
                    "error": None,
                }
            ],
        )
        _write_jsonl(questions_path, [{"id": "q1", "question": "Q?"}])
        _write_jsonl(qa_path, [{"id": "q1", "answer": "100"}])
        _write_jsonl(gold_path, [{"id": "q1", "context_id": "gold"}])
        cfg_path.write_text(
            f"""
evaluation:
  eval_run_id: test_multi_config_eval
  output_dir: "{out_dir.as_posix()}"
  retrieval_eval_field: "retrieved_original_context_ids"
inputs:
  rag_outputs:
    - "{rag_a_path.as_posix()}"
    - "{rag_b_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{gold_path.as_posix()}"
retrieval:
  k: 1
  ks: [1]
answer_quality:
  enable_numeric_accuracy: true
leaderboard:
  sort_metric: "mean_recall_at_1"
  sort_ascending: false
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        per_question = [
            json.loads(line)
            for line in (run_dir / "per_question.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [(row["experiment_id"], row["question_id"]) for row in per_question] == [("exp_a", "q1"), ("exp_b", "q1")]

        import csv

        leaderboard_rows = list(csv.DictReader((run_dir / "leaderboard.csv").read_text(encoding="utf-8").splitlines()))
        assert [row["experiment_id"] for row in leaderboard_rows] == ["exp_a", "exp_b"]
        assert [float(row["mean_recall_at_1"]) for row in leaderboard_rows] == [1.0, 0.0]
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
