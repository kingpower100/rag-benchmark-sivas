from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from src.pipeline4.loaders import P2Summary, P3Summary
from src.pipeline4.reporting import (
    ExperimentRecord,
    build_records,
    write_full_summary,
    write_retrieval_leaderboard,
    write_rqi_leaderboard,
    write_leaderboard_json,
    write_comparison_report,
    write_validation_report,
)
from src.pipeline4.schemas import Pipeline4Config
from src.pipeline4.validation import (
    ComparisonGroup,
    ExperimentValidation,
    P2_VALID,
    P3_NOT_AVAILABLE,
    P3_VALID,
    build_comparison_groups,
    validate_all,
)


def _p2(exp_id="exp_a", recall=0.5, mrr=0.9, ndcg=0.6, cp=0.35, unknown=0.2) -> P2Summary:
    return P2Summary(
        experiment_id=exp_id,
        n_questions=96,
        run_valid=True,
        generation_failure_rate=0.0,
        mean_recall_at_5=recall,
        mean_mrr_at_5=mrr,
        mean_ndcg_at_5=ndcg,
        mean_context_precision_at_5=cp,
        unknown_rate=unknown,
        mean_embedding_similarity=0.88,
        mean_official_bertscore_f1=0.66,
        qa_hash="qa_hash_abc",
        gold_contexts_hash="qa_hash_abc",
        p2_run_dir=f"/fake/{exp_id}",
    )


def _p3(exp_id="exp_a") -> P3Summary:
    return P3Summary(
        run_id=f"p3_{exp_id}",
        experiment_id=exp_id,
        n_questions=96,
        judge_model="qwen2.5:14b",
        prompt_version="v2",
        qa_sha256="qa_hash_abc",
        judge_success_rate=1.0,
        judge_failure_count=0,
        mean_judge_correctness=2.39,
        mean_judge_faithfulness=2.83,
        mean_judge_completeness=1.45,
        mean_judge_hallucination=1.07,
        mean_judge_context_relevance=3.76,
        mean_judge_overall_score=2.68,
        mean_ragas_faithfulness=0.625,
        mean_ragas_answer_relevancy=0.700,
        ragas_faithfulness_nan_rate=0.01,
        ragas_answer_relevancy_nan_rate=0.0,
        p3_run_dir=f"/fake/p3/{exp_id}",
    )


def _cfg(mode="retrieval_only") -> Pipeline4Config:
    return Pipeline4Config(ranking_mode=mode)


def _make_records_and_groups(p2_list, p3_map, cfg):
    validations = validate_all(p2_list, p3_map, cfg.validation)
    groups = build_comparison_groups(p2_list, p3_map, validations, cfg.ranking_mode)
    records = build_records(p2_list, p3_map, validations, groups, cfg)
    return records, groups, validations


class TestBuildRecords:
    def test_records_count_matches_p2(self):
        p2_list = [_p2("a"), _p2("b")]
        p3_map = {"a": None, "b": None}
        cfg = _cfg()
        records, _, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        assert len(records) == 2

    def test_retrieval_score_computed(self):
        p2_list = [_p2("a")]
        p3_map = {"a": None}
        cfg = _cfg()
        records, _, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        assert records[0].retrieval_score > 0.0

    def test_rqi_none_without_p3(self):
        p2_list = [_p2("a")]
        p3_map = {"a": None}
        cfg = _cfg(mode="overall_rag")
        records, _, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        assert records[0].rqi is None

    def test_rqi_computed_with_p3(self):
        p2_list = [_p2("a")]
        p3_map = {"a": _p3("a")}
        cfg = _cfg(mode="overall_rag")
        records, _, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        assert records[0].rqi is not None
        assert 0.0 < records[0].rqi < 1.0

    def test_has_p3_flag(self):
        p2_list = [_p2("a"), _p2("b")]
        p3_map = {"a": _p3("a"), "b": None}
        cfg = _cfg()
        records, _, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        by_id = {r.experiment_id: r for r in records}
        assert by_id["a"].has_p3 is True
        assert by_id["b"].has_p3 is False


class TestWriteRetrievalLeaderboard:
    def test_csv_has_header_and_rows(self):
        p2_list = [_p2("a"), _p2("b")]
        p3_map = {"a": None, "b": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        from src.pipeline4.ranking import rank_retrieval
        scores = {r.experiment_id: r.retrieval_score for r in records}
        ranks = rank_retrieval(scores, groups)
        for rec in records:
            rec.retrieval_rank = ranks.get(rec.experiment_id)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "retrieval.csv"
            write_retrieval_leaderboard(records, out)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 2
        assert "rank" in rows[0]
        assert "retrieval_score" in rows[0]

    def test_ranked_by_score_descending(self):
        p2_list = [_p2("a", recall=0.3), _p2("b", recall=0.8)]
        p3_map = {"a": None, "b": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        from src.pipeline4.ranking import rank_retrieval
        scores = {r.experiment_id: r.retrieval_score for r in records}
        ranks = rank_retrieval(scores, groups)
        for rec in records:
            rec.retrieval_rank = ranks.get(rec.experiment_id)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "retrieval.csv"
            write_retrieval_leaderboard(records, out)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert rows[0]["experiment_id"] == "b"
        assert rows[1]["experiment_id"] == "a"


class TestWriteFullSummary:
    def test_all_experiments_included(self):
        p2_list = [_p2("a"), _p2("b", mrr=0.0)]
        p3_map = {"a": _p3("a"), "b": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.csv"
            write_full_summary(records, out)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert len(rows) == 2

    def test_p3_columns_empty_when_no_p3(self):
        p2_list = [_p2("a")]
        p3_map = {"a": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.csv"
            write_full_summary(records, out)
            rows = list(csv.DictReader(out.open(encoding="utf-8")))
        assert rows[0]["judge_model"] == ""
        assert rows[0]["mean_judge_correctness"] == ""


class TestLeaderboardJson:
    def test_json_has_expected_keys(self):
        p2_list = [_p2("a")]
        p3_map = {"a": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        from src.pipeline4.ranking import rank_retrieval
        scores = {r.experiment_id: r.retrieval_score for r in records}
        ranks = rank_retrieval(scores, groups)
        for rec in records:
            rec.retrieval_rank = ranks.get(rec.experiment_id)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "leaderboard.json"
            write_leaderboard_json(records, groups, cfg, out)
            data = json.loads(out.read_text(encoding="utf-8"))
        assert "retrieval_leaderboard" in data
        assert "rqi_leaderboard" in data
        assert "comparison_groups" in data
        assert "ranking_mode" in data


class TestComparisonReport:
    def test_report_is_markdown(self):
        p2_list = [_p2("a"), _p2("b")]
        p3_map = {"a": None, "b": None}
        cfg = _cfg()
        records, groups, _ = _make_records_and_groups(p2_list, p3_map, cfg)
        from src.pipeline4.ranking import rank_retrieval
        scores = {r.experiment_id: r.retrieval_score for r in records}
        ranks = rank_retrieval(scores, groups)
        for rec in records:
            rec.retrieval_rank = ranks.get(rec.experiment_id)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.md"
            write_comparison_report(records, groups, cfg, out)
            text = out.read_text(encoding="utf-8")
        assert text.startswith("# Pipeline 4")
        assert "Retrieval Leaderboard" in text
