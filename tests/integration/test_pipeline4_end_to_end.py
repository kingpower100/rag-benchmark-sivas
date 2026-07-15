from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline4.orchestrator import Pipeline4Orchestrator


REAL_P2_DIR = Path("data/eval/runs/pipeline2")
REAL_P3_DIR = Path("data/eval/runs/pipeline3")


@pytest.fixture
def p4_tmpdir():
    d = tempfile.mkdtemp()
    yield Path(d)
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def p4_config_path(p4_tmpdir):
    cfg = {
        "pipeline2_runs_dir": str(REAL_P2_DIR),
        "pipeline3_runs_dir": str(REAL_P3_DIR),
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
        "pipeline2_runs_dir": str(REAL_P2_DIR),
        "pipeline3_runs_dir": str(REAL_P3_DIR),
        "output_dir": str(p4_tmpdir / "pipeline4_rag_out"),
        "run_id": "test_e2e_rag_run",
        "ranking_mode": "overall_rag",
    }
    path = p4_tmpdir / "p4_rag_test.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.mark.skipif(
    not REAL_P2_DIR.exists(),
    reason="Real P2 data not available at data/eval/runs/pipeline2",
)
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
