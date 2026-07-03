import json
import shutil
from pathlib import Path

from src.pipeline2.orchestrator import EvaluationOrchestrator


class FakeBertScorer:
    model_name = "fake-bert"
    tokenizer_name = "fake-bert"
    device = "cpu"

    def score(self, generated_answer: str, ground_truth_answer: str) -> dict[str, float]:
        if not generated_answer.strip() or not ground_truth_answer.strip():
            return {
                "official_bertscore_precision": 0.0,
                "official_bertscore_recall": 0.0,
                "official_bertscore_f1": 0.0,
            }
        value = 1.0 if generated_answer.strip() == ground_truth_answer.strip() else 0.25
        return {
            "official_bertscore_precision": value,
            "official_bertscore_recall": value,
            "official_bertscore_f1": value,
        }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_pipeline2_writes_final_metric_outputs(monkeypatch):
    monkeypatch.setattr("src.pipeline2.orchestrator.build_bert_score_scorer", lambda *args: FakeBertScorer())
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
        _write_jsonl(qa_path, [{"id": "q1", "question": "Q?", "answer": "100", "gold_kategorie": "Vertrieb"}])
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
bert_score:
  enabled: true
  model_name: fake-bert
embedding_similarity:
  provider: deterministic_hash
  model_name: unit-test
  dimensions: 64
  offline_mode: true
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        row = json.loads((run_dir / "per_question.jsonl").read_text(encoding="utf-8").strip())
        assert row["hit_at_1"] == 0.0
        assert row["hit_at_3"] == 1.0
        assert row["recall_at_3"] == 1.0
        assert row["context_precision_at_3"] == 0.5
        assert row["mrr_at_3"] == 0.5
        assert row["raw_duplicate_rate"] == 1 / 3
        assert row["official_bertscore_precision"] == 1.0
        assert row["official_bertscore_recall"] == 1.0
        assert row["official_bertscore_f1"] == 1.0
        assert row["hashed_embedding_cosine_similarity"] is not None
        assert row["embedding_similarity"] is None
        assert row["non_empty_answer_rate"] == 1.0

        summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        summary_row = summary["summary_by_experiment"][0]
        assert summary_row["mean_official_bertscore_precision"] == 1.0
        assert summary_row["mean_official_bertscore_recall"] == 1.0
        assert summary_row["mean_official_bertscore_f1"] == 1.0
        assert summary["metric_priority"]["primary_metrics"] == [
            "official_bertscore_f1",
            "hashed_embedding_cosine_similarity",
            "category_accuracy",
            "category_coverage",
        ]

        manifest = json.loads((run_dir / "eval_manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_path"] == str(cfg_path.resolve())
        assert manifest["row_counts"]["pipeline1_results"] == 1
        assert manifest["metric_runtime"]["bert_score"]["model_name"] == "fake-bert"
        assert manifest["metric_runtime"]["embedding_similarity"]["provider"] == "deterministic_hash"
        assert manifest["metric_priority"]["primary_metrics"][0] == "official_bertscore_f1"

        audit = json.loads((run_dir / "audit_report.json").read_text(encoding="utf-8"))
        assert audit["metric_runtime"]["bert_score"]["device_used"] == "cpu"
        audit_md = (run_dir / "audit_report.md").read_text(encoding="utf-8")
        assert "BERTScore F1" in audit_md
        assert "Embedding Similarity" in audit_md

        expected_files = {
            "per_question.jsonl",
            "per_question_metrics.jsonl",
            "per_question.csv",
            "summary_metrics.json",
            "eval_manifest.json",
            "audit_report.json",
            "audit_report.md",
        }
        assert expected_files <= {path.name for path in run_dir.iterdir()}
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
embedding_similarity:
  provider: deterministic_hash
  offline_mode: true
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


def test_pipeline2_evaluates_uid_only_qa_file(monkeypatch):
    monkeypatch.setattr("src.pipeline2.orchestrator.build_bert_score_scorer", lambda *args: FakeBertScorer())
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
        _write_jsonl(qa_path, [{"uid": "UID0002", "answer": "507", "difficulty": "easy"}])
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
bert_score:
  enabled: true
  model_name: fake-bert
embedding_similarity:
  provider: deterministic_hash
  model_name: unit-test
  dimensions: 64
  offline_mode: true
""",
            encoding="utf-8",
        )

        run_dir = EvaluationOrchestrator().run(str(cfg_path))

        row = json.loads((run_dir / "per_question.jsonl").read_text(encoding="utf-8").strip())
        assert row["ground_truth_answer"] == "507"
        assert row["answer_match_status"] == "match"
        assert row["retrieval_eval_ids"] == ["sivas_manual_01.md"]
        assert row["hit_at_1"] == 1.0
        assert row["official_bertscore_f1"] == 1.0
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
