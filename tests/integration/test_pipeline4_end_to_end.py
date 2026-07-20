from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline4.orchestrator import Pipeline4Orchestrator


EXPERIMENT_IDS = [
    "91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0",
    "92_sivas_fixed512_faiss_dense_mistralsmall_prompt_v1",
    "93_sivas_fixed512_faiss_dense_mistralsmall_prompt_v2",
    "94_sivas_fixed512_faiss_dense_mistralsmall_prompt_v3",
    "95_sivas_fixed512_faiss_dense_mistralsmall_prompt_v4",
    "96_sivas_fixed512_faiss_dense_mistralsmall_prompt_v5",
]


@pytest.fixture
def p4_tmpdir():
    d = tempfile.mkdtemp()
    root = Path(d)
    _write_fixture_runs(root / "pipeline2", root / "pipeline3")
    yield root
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def p4_config_path(p4_tmpdir):
    cfg = {
        "pipeline2_runs_dir": str(p4_tmpdir / "pipeline2"),
        "pipeline3_runs_dir": str(p4_tmpdir / "pipeline3"),
        "output_dir": str(p4_tmpdir / "pipeline4_out"),
        "run_id": "test_e2e_run",
        "ranking_mode": "retrieval_only",
    }
    path = p4_tmpdir / "p4_test.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def p4_rag_config_path(p4_tmpdir):
    cfg = {
        "pipeline2_runs_dir": str(p4_tmpdir / "pipeline2"),
        "pipeline3_runs_dir": str(p4_tmpdir / "pipeline3"),
        "output_dir": str(p4_tmpdir / "pipeline4_rag_out"),
        "run_id": "test_e2e_rag_run",
        "ranking_mode": "overall_rag",
    }
    path = p4_tmpdir / "p4_rag_test.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


class TestPipeline4EndToEnd:
    def test_retrieval_mode_produces_all_output_files(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        assert (run_dir / "retrieval_leaderboard.csv").exists()
        assert (run_dir / "overall_rqi_leaderboard.csv").exists()
        assert (run_dir / "full_experiment_summary.csv").exists()
        assert (run_dir / "leaderboard.json").exists()
        assert (run_dir / "comparison_report.md").exists()
        assert (run_dir / "validation_report.json").exists()
        assert (run_dir / "pipeline4_manifest.json").exists()

    def test_retrieval_leaderboard_has_6_experiments(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "retrieval_leaderboard.csv").open(encoding="utf-8")))
        assert len(rows) == 6

    def test_retrieval_leaderboard_rank_1_exists(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "retrieval_leaderboard.csv").open(encoding="utf-8")))
        ranks = [int(r["rank"]) for r in rows]
        assert 1 in ranks

    def test_retrieval_scores_in_valid_range(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "retrieval_leaderboard.csv").open(encoding="utf-8")))
        for row in rows:
            score = float(row["retrieval_score"])
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {row['experiment_id']}"

    def test_full_summary_has_all_experiments(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "full_experiment_summary.csv").open(encoding="utf-8")))
        assert len(rows) == 6

    def test_manifest_has_required_keys(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        manifest = json.loads((run_dir / "pipeline4_manifest.json").read_text(encoding="utf-8"))
        for key in ("run_id", "ranking_mode", "inputs", "outputs", "summary",
                    "retrieval_score_weights", "rqi_weights"):
            assert key in manifest, f"Key '{key}' missing from manifest"

    def test_leaderboard_json_structure(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        data = json.loads((run_dir / "leaderboard.json").read_text(encoding="utf-8"))
        assert "retrieval_leaderboard" in data
        assert "rqi_leaderboard" in data
        assert "comparison_groups" in data
        assert len(data["retrieval_leaderboard"]) == 6

    def test_validation_report_structure(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
        assert "total_experiments" in report
        assert "excluded_experiments" in report
        assert report["total_experiments"] == 6

    def test_comparison_report_is_valid_markdown(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        text = (run_dir / "comparison_report.md").read_text(encoding="utf-8")
        assert text.startswith("# Pipeline 4")
        assert "Retrieval Leaderboard" in text

    def test_baseline_experiment_in_retrieval_leaderboard(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "retrieval_leaderboard.csv").open(encoding="utf-8")))
        exp_ids = [r["experiment_id"] for r in rows]
        assert "91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0" in exp_ids

    def test_rag_mode_no_rqi_ranking_when_p3_incomplete(self, p4_rag_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_rag_config_path))

        rows = list(csv.DictReader((run_dir / "overall_rqi_leaderboard.csv").open(encoding="utf-8")))
        full_rows = list(csv.DictReader((run_dir / "full_experiment_summary.csv").open(encoding="utf-8")))
        rqi_exp_ids = {r["experiment_id"] for r in rows}
        all_exp_ids_with_p3 = {
            r["experiment_id"] for r in full_rows if r["has_p3"] == "True"
        }
        assert rqi_exp_ids == all_exp_ids_with_p3 or len(rqi_exp_ids) <= len(all_exp_ids_with_p3)

    def test_baseline_retrieval_score_matches_formula(self, p4_config_path):
        orch = Pipeline4Orchestrator()
        run_dir = orch.run(str(p4_config_path))

        rows = list(csv.DictReader((run_dir / "retrieval_leaderboard.csv").open(encoding="utf-8")))
        by_id = {r["experiment_id"]: r for r in rows}
        baseline = by_id.get("91_sivas_fixed512_faiss_dense_mistralsmall_prompt_v0")
        assert baseline is not None

        recall = float(baseline["recall_at_5"])
        mrr = float(baseline["mrr_at_5"])
        ndcg = float(baseline["ndcg_at_5"])
        cp = float(baseline["context_precision_at_5"])
        score = float(baseline["retrieval_score"])

        expected = 0.35 * recall + 0.25 * mrr + 0.20 * ndcg + 0.20 * cp
        assert abs(score - expected) < 1e-5


def _write_fixture_runs(p2_root: Path, p3_root: Path) -> None:
    question_ids = [f"q{i:03d}" for i in range(96)]
    for idx, exp_id in enumerate(EXPERIMENT_IDS):
        recall = round(0.30 + idx * 0.05, 6)
        p2_dir = p2_root / f"p2_{exp_id}"
        p2_dir.mkdir(parents=True, exist_ok=True)
        summary_row = {
            "experiment_id": exp_id,
            "n_questions": 96,
            "run_valid": True,
            "generation_failure_rate": 0.0,
            "mean_recall_at_5": recall,
            "mean_mrr_at_5": 0.70,
            "mean_ndcg_at_5": 0.60,
            "mean_context_precision_at_5": 0.50,
            "unknown_rate": 0.10,
            "mean_embedding_similarity": 0.80,
            "mean_official_bertscore_f1": 0.75,
        }
        (p2_dir / "summary_metrics.json").write_text(
            json.dumps({"summary_by_experiment": [summary_row]}, indent=2),
            encoding="utf-8",
        )
        rows = [{"question_id": qid, "experiment_id": exp_id} for qid in question_ids]
        _write_jsonl(p2_dir / "per_question.jsonl", rows)
        _write_jsonl(p2_dir / "per_question_metrics.jsonl", rows)
        (p2_dir / "eval_manifest.json").write_text(
            json.dumps(
                {
                    "final_verdict": "valid",
                    "strict_audit_pass": True,
                    "qa_hash": "qa_hash_abc",
                    "gold_contexts_hash": "gold_hash_abc",
                    "fake_run_detection": {"suspicious": False, "checks": []},
                    "row_counts": {
                        "pipeline1_results": 96,
                        "questions_rows": 96,
                        "evaluated_rows": 96,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if idx < 3:
            _write_p3_fixture(p3_root / f"p3_{exp_id}", exp_id, question_ids)


def _write_p3_fixture(run_dir: Path, exp_id: str, question_ids: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "semantic_summary.csv").write_text("run_id,n_questions\np3_%s,96\n" % exp_id, encoding="utf-8")
    with (run_dir / "per_question_semantic_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id"])
        writer.writeheader()
        for qid in question_ids:
            writer.writerow({"question_id": qid})
    (run_dir / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"p3_{exp_id}",
                "judge_model": "qwen2.5:14b",
                "prompt_version": "v2",
                "inputs": {"questions_rows": 96, "rag_rows": 96, "qa_sha256": "qa_hash_abc"},
                "validation": {"passed": True, "errors": [], "warnings": [], "stats": {}},
                "ragas_stats": {"nan_counts": {}},
                "summary": {
                    "n_questions": 96,
                    "judge_success_rate": 1.0,
                    "judge_failure_count": 0,
                    "mean_judge_correctness": 3.0,
                    "mean_judge_faithfulness": 3.0,
                    "mean_judge_completeness": 3.0,
                    "mean_judge_hallucination": 1.0,
                    "mean_judge_context_relevance": 3.0,
                    "mean_judge_overall_score": 3.0,
                    "mean_ragas_faithfulness": 0.6,
                    "mean_ragas_answer_relevancy": 0.7,
                    "mean_ragas_context_recall": 0.8,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
