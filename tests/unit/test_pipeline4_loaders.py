from __future__ import annotations

import csv
import json
from pathlib import Path

from src.pipeline4.loaders import load_p2_summary, load_p3_summary


def test_load_p2_summary_reads_strict_audit_manifest(tmp_path: Path):
    run_dir = tmp_path / "p2_a"
    run_dir.mkdir()
    _write_p2_summary(run_dir, "a", 96)
    _write_jsonl(run_dir / "per_question.jsonl", [{"question_id": f"q{i:03d}"} for i in range(96)])
    _write_jsonl(run_dir / "per_question_metrics.jsonl", [{"question_id": f"q{i:03d}"} for i in range(96)])
    (run_dir / "eval_manifest.json").write_text(
        json.dumps(
            {
                "final_verdict": "valid",
                "strict_audit_pass": True,
                "qa_hash": "qa",
                "gold_contexts_hash": "gold",
                "fake_run_detection": {"suspicious": False, "checks": []},
                "row_counts": {
                    "pipeline1_results": 96,
                    "questions_rows": 96,
                    "evaluated_rows": 96,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_p2_summary(run_dir)

    assert summary.audit_manifest_present is True
    assert summary.final_verdict == "valid"
    assert summary.strict_audit_pass is True
    assert summary.expected_question_count == 96
    assert len(summary.question_ids) == 96
    assert summary.required_outputs_present is True


def test_load_p2_summary_marks_legacy_manifest_missing(tmp_path: Path):
    run_dir = tmp_path / "p2_legacy"
    run_dir.mkdir()
    _write_p2_summary(run_dir, "legacy", 96)

    summary = load_p2_summary(run_dir)

    assert summary.audit_manifest_present is False
    assert "eval_manifest.json" in summary.missing_required_outputs
    assert summary.required_outputs_present is False


def test_load_p3_summary_reads_row_level_coverage(tmp_path: Path):
    run_dir = tmp_path / "p3_a"
    run_dir.mkdir()
    _write_p3_manifest(run_dir, "a", 96)
    (run_dir / "semantic_summary.csv").write_text("run_id,n_questions\np3_a,96\n", encoding="utf-8")
    with (run_dir / "per_question_semantic_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id"])
        writer.writeheader()
        for i in range(96):
            writer.writerow({"question_id": f"q{i:03d}"})

    summary = load_p3_summary(run_dir)

    assert summary.validation_passed is True
    assert summary.summary_present is True
    assert summary.row_output_present is True
    assert summary.expected_question_count == 96
    assert len(summary.question_ids) == 96
    assert not summary.duplicate_question_ids


def test_load_p3_summary_detects_duplicate_question_ids(tmp_path: Path):
    run_dir = tmp_path / "p3_dup"
    run_dir.mkdir()
    _write_p3_manifest(run_dir, "dup", 2)
    (run_dir / "semantic_summary.csv").write_text("run_id,n_questions\np3_dup,2\n", encoding="utf-8")
    with (run_dir / "per_question_semantic_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id"])
        writer.writeheader()
        writer.writerow({"question_id": "q001"})
        writer.writerow({"question_id": "q001"})

    summary = load_p3_summary(run_dir)

    assert summary.duplicate_question_ids == ["q001"]


def _write_p2_summary(run_dir: Path, exp_id: str, n_questions: int) -> None:
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(
            {
                "summary_by_experiment": [
                    {
                        "experiment_id": exp_id,
                        "n_questions": n_questions,
                        "run_valid": True,
                        "generation_failure_rate": 0.0,
                        "mean_recall_at_5": 0.5,
                        "mean_mrr_at_5": 0.6,
                        "mean_ndcg_at_5": 0.7,
                        "mean_context_precision_at_5": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_p3_manifest(run_dir: Path, exp_id: str, n_questions: int) -> None:
    (run_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"p3_{exp_id}",
                "inputs": {"questions_rows": n_questions, "qa_sha256": "qa"},
                "validation": {"passed": True},
                "ragas_stats": {"nan_counts": {}},
                "summary": {
                    "n_questions": n_questions,
                    "judge_success_rate": 1.0,
                    "judge_failure_count": 0,
                    "mean_judge_correctness": 3.0,
                    "mean_judge_faithfulness": 3.0,
                    "mean_judge_context_relevance": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
