from __future__ import annotations

import json
from pathlib import Path

from src.pipeline3.judge.response_parser import JudgeResponse
from src.pipeline3.metrics.ragas_metrics import RagasResults
from src.pipeline3.orchestrator import Pipeline3Orchestrator
from src.pipeline3.stages.judge_stage import JudgeRowResult, JudgeStageResult
from src.pipeline3.stages.loader_stage import LoaderResult


def test_pipeline3_writes_expected_six_output_files(tmp_path, monkeypatch):
    rag_path = tmp_path / "pipeline1_results.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    for path in (rag_path, questions_path, qa_path):
        path.write_text("{}\n", encoding="utf-8")

    config_path = tmp_path / "pipeline3.yaml"
    output_dir = tmp_path / "pipeline3_outputs"
    config_path.write_text(
        f"""
pipeline3:
  run_id: "contract"
  output_dir: "{output_dir.as_posix()}"
  overwrite: true
  save_csv: true
  version: "1.0.0"
  prompt_version: "v2"
inputs:
  pipeline1_results_path: "{rag_path.as_posix()}"
  questions_path: "{questions_path.as_posix()}"
  qa_path: "{qa_path.as_posix()}"
  gold_contexts_path: "{qa_path.as_posix()}"
ragas:
  enabled: true
  fail_on_ragas_error: true
  embeddings_device: "cuda"
  require_cuda: true
judge:
  model: "qwen2.5:14b"
  temperature: 0.0
llm_judge:
  enabled: true
""",
        encoding="utf-8",
    )

    rag_rows = [
        {
            "question_id": "q1",
            "experiment_id": "exp",
            "question": "What is configured?",
            "generated_answer": "CUDA is configured.",
            "retrieved_context_texts": ["CUDA is configured."],
        }
    ]
    qa_rows = [{"question_id": "q1", "answer": "CUDA is configured."}]
    questions_rows = [{"question_id": "q1", "frage": "What is configured?"}]

    monkeypatch.setattr(
        "src.pipeline3.orchestrator.load_inputs",
        lambda cfg, project_root: LoaderResult(
            rag_rows=rag_rows,
            qa_rows=qa_rows,
            questions_rows=questions_rows,
            rag_path=rag_path,
            questions_path=questions_path,
            qa_path=qa_path,
            gold_contexts_path=qa_path,
        ),
    )
    monkeypatch.setattr(
        "src.pipeline3.orchestrator.run_ragas_stage",
        lambda rows, qa_by_id, evaluator: RagasResults(
            rows=[
                {
                    "question_id": "q1",
                    "ragas_faithfulness": 1.0,
                    "ragas_answer_relevancy": 1.0,
                    "ragas_context_precision": 1.0,
                    "ragas_context_recall": 1.0,
                }
            ],
            enabled_metrics=[
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
        ),
    )

    judge_response = JudgeResponse(
        correctness=5,
        faithfulness=5,
        relevancy=5,
        completeness=5,
        hallucination=0,
        context_relevance=5,
        overall_score=5.0,
        reasoning="The answer is fully correct.",
    )
    monkeypatch.setattr(
        "src.pipeline3.orchestrator.run_judge_stage",
        lambda rows, qa_by_id, judge_cfg, llm_judge_cfg: JudgeStageResult(
            rows=[
                JudgeRowResult(
                    question_id="q1",
                    success=True,
                    response=judge_response,
                    raw_response=json.dumps(judge_response.raw),
                )
            ],
            total=1,
            successes=1,
            failures=0,
            failure_rate=0.0,
        ),
    )

    class _FakeOllamaClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_model_info(self):
            return {"digest": "sha256:test"}

    monkeypatch.setattr("src.pipeline3.orchestrator.OllamaClient", _FakeOllamaClient)

    run_dir = Pipeline3Orchestrator().run(str(config_path))

    expected_files = {
        "per_question_semantic_metrics.csv",
        "semantic_summary.csv",
        "judge_raw_outputs.json",
        "judge_failures.json",
        "pipeline3_report.md",
        "evaluation_manifest.json",
    }
    assert {path.name for path in run_dir.iterdir() if path.is_file()} == expected_files
    for name in expected_files:
        path = run_dir / name
        assert path.stat().st_size > 0, f"{name} should be non-empty"

    manifest = json.loads((run_dir / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ragas_stats"]["fail_on_ragas_error"] is True
    assert manifest["ragas_stats"]["embeddings_device"] == "cuda"
    assert manifest["ragas_stats"]["require_cuda"] is True
