import shutil
from pathlib import Path

import pytest

from src.pipeline2.aggregation.summarizer import summarize_by_experiment
from src.pipeline2.orchestrator import (
    EvaluationOrchestrator,
    _validate_eval_diagnostics,
    _validate_leakage_audit,
    _validate_three_way_alignment,
    build_fake_run_detection,
    build_three_way_alignment_report,
    build_eval_diagnostics,
    build_leakage_audit,
    compare_reported_vs_recomputed_metrics,
)
from src.pipeline1.utils.hashing import file_sha256
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def _cfg(retrieval_only: bool = False) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "evaluation": {
                "eval_run_id": "eval",
                "retrieval_only": retrieval_only,
                "retrieval_eval_field": "retrieved_file_names",
            },
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 1, "ks": [1]},
        }
    )


def _rag_row(qid: str = "q1", answer: str = "100", files=None) -> dict:
    return {
        "question_id": qid,
        "experiment_id": "exp",
        "question": "Q?",
        "generated_answer": answer,
        "retrieved_original_context_ids": ["chunk1"],
        "retrieved_file_names": ["doc.txt"] if files is None else files,
    }


def _diagnostics(rag_rows, qa_rows, gold_rows, qa_by_id, gold_by_id, cfg):
    questions_rows = [{"uid": row.get("question_id"), "question": "Q?"} for row in rag_rows if row.get("question_id")]
    alignment = build_three_way_alignment_report(questions_rows, qa_rows, gold_rows, rag_rows)
    return build_eval_diagnostics(rag_rows, questions_rows, qa_rows, gold_rows, qa_by_id, gold_by_id, alignment, cfg)


def test_empty_pipeline1_results_fail():
    cfg = _cfg()
    diagnostics = _diagnostics([], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="zero Pipeline 1 result rows"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_zero_qa_intersection_fails():
    cfg = _cfg()
    diagnostics = _diagnostics([_rag_row("q_missing")], [{"uid": "q1", "answer": "100"}], [{"id": "q_missing", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q_missing": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="zero matching question IDs.*QA"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_zero_gold_intersection_fails():
    cfg = _cfg()
    diagnostics = _diagnostics([_rag_row("q1")], [{"uid": "q1", "answer": "100"}], [], {"q1": {"uid": "q1", "answer": "100"}}, {}, cfg)

    with pytest.raises(ValueError, match="zero matching question IDs.*gold"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_generated_answer_for_all_rows_fails():
    cfg = _cfg()
    diagnostics = _diagnostics([_rag_row("q1", answer="")], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="no generated_answer"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_retrieved_field_for_all_rows_fails():
    cfg = _cfg()
    diagnostics = _diagnostics([_rag_row("q1", files=[])], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="no non-empty values"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_retrieval_eval_field_raises_during_evaluation():
    cfg = _cfg()
    row = _rag_row("q1")
    del row["retrieved_file_names"]

    with pytest.raises(ValueError, match="retrieval_eval_field='retrieved_file_names' is missing"):
        EvaluationOrchestrator()._evaluate_rows([row], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)


def test_normal_small_fixture_evaluates_correct_row_count_and_summary():
    cfg = _cfg()
    rows = [_rag_row("q1", "100"), _rag_row("q2", "200", ["wrong.txt"])]
    qa_by_id = {"q1": {"uid": "q1", "answer": "100"}, "q2": {"uid": "q2", "answer": "200"}}
    gold_by_id = {"q1": ["doc.txt"], "q2": ["doc.txt"]}

    diagnostics = _diagnostics(rows, list(qa_by_id.values()), [{"id": "q1", "context_id": ["doc.txt"]}, {"id": "q2", "context_id": ["doc.txt"]}], qa_by_id, gold_by_id, cfg)
    _validate_eval_diagnostics(diagnostics, cfg)
    evaluated = EvaluationOrchestrator()._evaluate_rows(rows, qa_by_id, gold_by_id, cfg)
    summary = summarize_by_experiment(evaluated)

    assert len(evaluated) == 2
    assert summary[0]["n_questions"] == len(evaluated)
    assert summary[0]["mean_hit_at_1"] == 0.5


def test_three_way_alignment_requires_exact_id_sets():
    report = build_three_way_alignment_report(
        [{"uid": "q1"}],
        [{"uid": "q1", "answer": "1"}],
        [{"id": "q2", "context_id": ["doc.txt"]}],
        [],
    )

    assert report["exact_set_equality"] is False
    assert report["missing_from_retrieval_evidence"] == ["q1"]
    assert report["missing_from_questions"] == ["q2"]
    with pytest.raises(ValueError, match="ID sets are not identical"):
        _validate_three_way_alignment(report)


def test_three_way_alignment_rejects_duplicates_in_all_inputs():
    report = build_three_way_alignment_report(
        [{"uid": "q1"}, {"id": "q1"}],
        [{"uid": "q1", "answer": "1"}],
        [{"id": "q1", "context_id": ["doc.txt"]}],
        [_rag_row("q1"), _rag_row("q1")],
    )

    assert report["duplicate_id_summary"]["questions"] == ["q1"]
    assert report["duplicate_id_summary"]["pipeline1_results"] == ["exp:q1"]
    with pytest.raises(ValueError, match="duplicate IDs"):
        _validate_three_way_alignment(report)


def test_reported_vs_recomputed_metric_comparison_flags_mismatch():
    comparison = compare_reported_vs_recomputed_metrics(
        [{"question_id": "q1", "hit_at_1": 0.0, "exact_match": 1.0}],
        [{"question_id": "q1", "hit_at_1": 1.0, "exact_match": 1.0}],
        [1],
    )

    assert comparison["comparison_count"] == 1
    assert comparison["failure_count"] == 1
    assert comparison["failed_examples"][0]["metric"] == "hit_at_1"


def test_reported_vs_recomputed_metric_comparison_reports_missing_metrics():
    comparison = compare_reported_vs_recomputed_metrics(
        [{"question_id": "q1", "generated_answer": "1"}],
        [{"question_id": "q1", "hit_at_1": 1.0}],
        [1],
    )

    assert comparison["message"] == "No reported metrics found to compare against."


def test_leakage_audit_flags_gold_terms_in_prompt_artifacts():
    run_dir = Path(".project_tmp") / "test_leakage_audit"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    try:
        results = run_dir / "results.jsonl"
        results.write_text("", encoding="utf-8")
        (run_dir / "prompt.txt").write_text("Loaded ground_truth_answer in prompt", encoding="utf-8")

        report = build_leakage_audit([results])

        assert report["critical_leakage_found"] is True
        with pytest.raises(ValueError, match="gold-data leakage"):
            _validate_leakage_audit(report)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_leakage_audit_skips_when_pipeline1_artifacts_are_missing():
    report = build_leakage_audit([Path("missing_run_dir") / "results.jsonl"])

    assert report["result"] == "skipped"
    assert report["message"] == "Leakage audit skipped: Pipeline 1 artifacts not found."
    _validate_leakage_audit(report)


def test_pipeline2_run_skips_real_audit_when_pipeline1_output_missing(capsys):
    eval_id = "test_missing_remote_pipeline1_output"
    cfg_dir = Path(".project_tmp") / "pipeline2_missing_output"
    out_dir = Path(".project_tmp") / "eval_runs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.yaml"
    cfg_path.write_text(
        f"""
evaluation:
  eval_run_id: "{eval_id}"
  output_dir: "{out_dir.as_posix()}"
  retrieval_eval_field: "retrieved_file_names"
inputs:
  rag_outputs:
    - ".project_tmp/missing_remote_results/results.jsonl"
  questions_path: "data/raw/questions_fixed.jsonl"
  qa_path: "data/raw/qa_ground_truth_fixed.jsonl"
  gold_contexts_path: "data/raw/qa_ground_truth_fixed.jsonl"
retrieval:
  k: 1
  ks: [1]
runtime:
  overwrite: true
  save_csv: false
""",
        encoding="utf-8",
    )

    run_dir = EvaluationOrchestrator().run(str(cfg_path))
    captured = capsys.readouterr()
    report = json_load(run_dir / "audit_report.json")

    assert "Real-run audit skipped: Pipeline 1 outputs not found on this machine." in captured.out
    assert report["audit_status"] == "skipped"
    assert (run_dir / "audit_report.md").exists()


def json_load(path):
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_fake_run_detection_valid_full_synthetic_run():
    run_dir = Path(".project_tmp") / "test_fake_valid_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    try:
        result_path = run_dir / "results.jsonl"
        result_rows = [
            {
                "question_id": "q1",
                "generated_answer": "10",
                "retrieved_file_names": ["doc1.txt"],
                "total_latency_ms": 10,
                "llm_model": "llm",
                "embedding_model": "embed",
                "retriever_type": "dense",
            },
            {
                "question_id": "q2",
                "generated_answer": "20",
                "retrieved_file_names": ["doc2.txt"],
                "total_latency_ms": 12,
                "llm_model": "llm",
                "embedding_model": "embed",
                "retriever_type": "dense",
            },
        ]
        result_path.write_text("\n".join(json_line(row) for row in result_rows) + "\n", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json_line(
                {
                    "run_id": "run1",
                    "artifacts": {"results.jsonl": {"sha256": file_sha256(result_path)}},
                    "run_stats": {"n_queries": 2},
                    "start_timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "end_timestamp_utc": "2026-01-01T00:00:01+00:00",
                }
            ),
            encoding="utf-8",
        )

        report = build_fake_run_detection(
            _cfg(),
            [result_path],
            result_rows,
            [{"uid": "q1"}, {"uid": "q2"}],
            result_rows,
            {"failure_count": 0, "failed_examples": []},
            {"critical_leakage_found": False, "findings": [], "result": "pass"},
            {},
        )

        assert report["suspicious"] is False
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_fake_run_detection_flags_empty_output_duplicate_rows_and_empty_retrievals():
    cfg = _cfg()
    rows = [
        {"question_id": "q1", "generated_answer": "same", "retrieved_file_names": [], "total_latency_ms": 0},
        {"question_id": "q1", "generated_answer": "same", "retrieved_file_names": [], "total_latency_ms": 0},
        {"question_id": "q2", "generated_answer": "same", "retrieved_file_names": [], "total_latency_ms": 0},
        {"question_id": "q3", "generated_answer": "same", "retrieved_file_names": [], "total_latency_ms": 0},
        {"question_id": "q4", "generated_answer": "same", "retrieved_file_names": [], "total_latency_ms": 0},
    ]

    report = build_fake_run_detection(
        cfg,
        [],
        rows,
        [{"uid": f"q{i}"} for i in range(1, 5)],
        rows,
        {"failure_count": 1, "failed_examples": [{"metric": "hit_at_1"}]},
        {"critical_leakage_found": False, "findings": [], "result": "pass"},
        {},
    )
    names = {item["name"] for item in report["suspicious_examples"]}

    assert "result_rows_do_not_match_questions" in names
    assert "duplicate_pipeline1_result_question_ids_within_experiment" in names
    assert "reported_metrics_differ_from_recomputed" in names
    assert "many_generated_answers_identical" in names
    assert "all_retrieval_lists_empty" in names
    assert "all_latencies_zero_or_missing" in names


def test_fake_run_detection_flags_manifest_hash_mismatch_and_bad_timestamps():
    run_dir = Path(".project_tmp") / "test_fake_hash_mismatch"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    try:
        result_path = run_dir / "results.jsonl"
        rows = [{"question_id": "q1", "generated_answer": "1", "retrieved_file_names": ["doc.txt"], "total_latency_ms": 1}]
        result_path.write_text(json_line(rows[0]) + "\n", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(
            json_line(
                {
                    "run_id": "run1",
                    "artifacts": {"results.jsonl": {"sha256": "wrong"}},
                    "run_stats": {"n_queries": 2},
                    "start_timestamp_utc": "2026-01-02T00:00:00+00:00",
                    "end_timestamp_utc": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        report = build_fake_run_detection(
            _cfg(),
            [result_path],
            rows,
            [{"uid": "q1"}],
            rows,
            {"failure_count": 0, "failed_examples": []},
            {"critical_leakage_found": False, "findings": [], "result": "pass"},
            {},
        )
        names = {item["name"] for item in report["suspicious_examples"]}

        assert "hash_mismatch_between_manifest_and_evaluated_file" in names
        assert "pipeline1_manifest_question_count_mismatch" in names
        assert "timestamps_impossible_or_missing" in names
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def json_line(row):
    import json

    return json.dumps(row, ensure_ascii=False)
